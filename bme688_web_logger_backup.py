#!/usr/bin/env python3
"""
bme688_web_logger.py
BME688 Web Logger (scan mode) + rolling-window live classification
with:
- model-synced heater temps (from model bundle step_cols)
- LIVE features that EXACTLY match train_model.py feature_vector_from_window()
- sticky prediction (hold last confident label)
- plateau latch (don’t “regress to UNKNOWN” during steady-state)

Run:
  (venv) python3 bme688_web_logger.py --model models/model.joblib

Open:
  http://<pi-ip>:5000/
"""

import csv
import os
import time
import threading
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue, Empty
from statistics import median
from collections import deque

from flask import Flask, Response, jsonify, request, render_template_string

import numpy as np
import joblib

import board
import busio
import adafruit_bme680


# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Globals / defaults
# -------------------------
scan_lock = threading.Lock()
HEATER_TEMPS = [200, 240, 280, 320, 360]  # will be overridden by model bundle step_cols
HEATER_MS = 100
WARMUP_S = 120

SENSOR = None
SENSOR_START: float | None = None

state_lock = threading.Lock()
event_q: Queue = Queue(maxsize=2000)

model_lock = threading.Lock()
MODEL_PATH = "models/model.joblib"
MODEL = None
MODEL_LABELS: list[str] | None = None
MODEL_STEP_COLS: list[str] | None = None   # e.g. ['gas_200',...]
MODEL_EXPECTED_NFEAT: int | None = None


# -------------------------
# UI
# -------------------------
HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>BME688 Logger + Live Classifier</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; }
    label { display:block; font-size: 12px; color:#444; margin-bottom: 4px; }
    input { padding: 8px; border-radius: 10px; border: 1px solid #ccc; min-width: 220px; }
    button { padding: 10px 12px; border-radius: 12px; border: 1px solid #bbb; background: #f7f7f7; cursor: pointer; }
    button.primary { border-color: #444; }
    button.danger { border-color: #b00; }
    #status { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
    canvas { max-width: 100%; }
    .muted { color:#666; font-size: 13px; }
    .sep { height: 14px; }
    .pill { display:inline-block; padding:2px 8px; border:1px solid #bbb; border-radius:999px; font-size:12px; color:#333; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    @media (max-width: 900px){ .grid{grid-template-columns:1fr;} input{min-width: 180px;} }
  </style>
</head>
<body>
  <h2>BME688 Web Logger <span class="pill">Sticky + Plateau</span></h2>
  <p class="muted">
    Live classifier uses the SAME feature schema as training, and holds predictions via sticky + plateau latch.
  </p>

  <div class="grid">
    <div class="card">
      <h3 style="margin-top:0;">Logger</h3>
      <div class="row">
        <div>
          <label>Session name</label>
          <input id="session" value="session_live" />
        </div>
        <div>
          <label>Label</label>
          <input id="label" value="air" />
        </div>
      </div>
      <div class="row" style="margin-top:10px;">
        <div>
          <label>Interval (sec)</label>
          <input id="interval" type="number" step="0.1" value="1.0" />
        </div>
        <div>
          <label>Duration (minutes)</label>
          <input id="minutes" type="number" step="1" value="15" />
        </div>
      </div>

      <div class="sep"></div>

      <div class="row">
        <button class="primary" onclick="startLogging()">Start</button>
        <button class="danger" onclick="stopLogging()">Stop</button>
        <button onclick="preset('air',15)">Air</button>
        <button onclick="preset('John Varvatos XX',15)">John Varvatos XX</button>
        <button onclick="preset('recovery_air',15)">Recovery</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0;">Scan + Live</h3>
      <div class="row">
        <div>
          <label>Heater temps (°C)</label>
          <input id="temps" value="200,240,280,320,360" />
          <div class="muted">Forced to model step_cols (if present).</div>
        </div>
        <div>
          <label>Heater duration (ms)</label>
          <input id="hdur" type="number" value="100" />
        </div>
      </div>
      <div class="row" style="margin-top:10px;">
        <div>
          <label>Warm-up seconds</label>
          <input id="warmup" type="number" value="120" />
        </div>
        <button onclick="applyScan()">Apply scan settings</button>
      </div>

      <div class="sep"></div>

      <div class="row">
        <div>
          <label>Model path</label>
          <input id="model_path" value="models/model.joblib" />
        </div>
        <div>
          <label>Unknown threshold</label>
          <input id="unk_thr" type="number" step="0.01" value="0.70" />
        </div>
      </div>

      <div class="row" style="margin-top:10px;">
        <div>
          <label>Sticky hold (sec)</label>
          <input id="hold_s" type="number" step="1" value="45" />
        </div>
        <div>
          <label>Plateau std thr (ohms)</label>
          <input id="plat_std" type="number" step="1" value="120" />
        </div>
      </div>

      <div class="row" style="margin-top:10px;">
        <button class="primary" onclick="startDetector()">Start live</button>
        <button class="danger" onclick="stopDetector()">Stop live</button>
        <button onclick="reloadModel()">Reload model</button>
        <button onclick="syncFromModel()">Sync heaters</button>
      </div>
    </div>
  </div>

  <div class="sep"></div>

  <div class="card">
    <div id="status">Connecting…</div>
  </div>

  <div class="sep"></div>

  <div class="card">
    <canvas id="chart" height="120"></canvas>
  </div>

<script>
const status = document.getElementById("status");
const chartEl = document.getElementById("chart");
let es, chart;
let xs=[], ys=[];

function preset(lbl, mins){
  document.getElementById("label").value = lbl;
  document.getElementById("minutes").value = mins;
}

function startLogging(){
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      session:document.getElementById("session").value,
      label:document.getElementById("label").value,
      interval:parseFloat(document.getElementById("interval").value),
      duration:parseInt(document.getElementById("minutes").value)*60
    })
  });
}
function stopLogging(){ fetch('/api/stop',{method:'POST'}); }

function applyScan(){
  fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      temps:document.getElementById("temps").value,
      heater_ms:parseInt(document.getElementById("hdur").value),
      warmup_s:parseInt(document.getElementById("warmup").value)
    })
  }).then(r=>r.json()).then(j=>{
    if(j.temps){ document.getElementById("temps").value = j.temps.join(","); }
  });
}

function startDetector(){
  fetch('/api/live/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      unknown_thr:parseFloat(document.getElementById("unk_thr").value),
      hold_s:parseFloat(document.getElementById("hold_s").value),
      plateau_std_thr:parseFloat(document.getElementById("plat_std").value),
      model_path:document.getElementById("model_path").value
    })
  }).then(r=>r.json()).then(j=>{
    if(j.temps){ document.getElementById("temps").value = j.temps.join(","); }
  });
}
function stopDetector(){ fetch('/api/live/stop',{method:'POST'}); }

function reloadModel(){
  fetch('/api/model/reload',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model_path:document.getElementById("model_path").value})
  }).then(r=>r.json()).then(j=>{
    if(j.temps){ document.getElementById("temps").value = j.temps.join(","); }
  });
}

function syncFromModel(){
  fetch('/api/model/sync_heaters',{method:'POST'})
    .then(r=>r.json()).then(j=>{
      if(j.temps){ document.getElementById("temps").value = j.temps.join(","); }
    });
}

function setupChart(){
  chart=new Chart(chartEl.getContext('2d'),{
    type:'line',
    data:{labels:xs,datasets:[{label:'Gas median',data:ys}]},
    options:{animation:false}
  });
}

function connect(){
  es=new EventSource('/stream');
  es.onmessage=(e)=>{
    if(!e.data) return;
    const m=JSON.parse(e.data);
    status.textContent=JSON.stringify(m,null,2);
    if(m.gas_med){
      xs.push(xs.length);
      ys.push(m.gas_med);
      if(xs.length>300){xs.shift();ys.shift();}
      chart.update('none');
    }
  };
}
setupChart(); connect();
</script>
</body>
</html>
"""


# -------------------------
# State (logger)
# -------------------------
@dataclass
class State:
    running: bool = False
    session: str | None = None
    label: str | None = None
    interval: float = 1.0
    stop_at: float = 0.0

state = State()


# -------------------------
# Live classifier state (sticky + plateau)
# -------------------------
@dataclass
class Live:
    enabled: bool = False
    unknown_thr: float = 0.70
    window_s: int = 45
    stride_s: int = 5

    # sticky / plateau
    hold_s: float = 45.0
    plateau_std_thr: float = 120.0        # ohms std threshold inside window
    plateau_slope_thr: float = 8.0        # ohms/sample slope threshold
    plateau_min_frac: float = 0.70        # window fraction must be "flat-ish"

    # internal
    last_raw_label: str | None = None
    last_raw_conf: float | None = None

    sticky_label: str | None = None
    sticky_conf: float | None = None
    sticky_left: float = 0.0

    plateau_latched: bool = False
    plateau_label: str | None = None
    plateau_conf: float | None = None

    buf_med: deque = field(default_factory=lambda: deque(maxlen=900))
    buf_steps: deque = field(default_factory=lambda: deque(maxlen=900))
    t_since_pred: float = 0.0

live = Live()
live_lock = threading.Lock()


# -------------------------
# Model loading (bundle-aware)
# -------------------------
def _step_cols_to_temps(step_cols: list[str]) -> list[int]:
    temps = []
    for c in step_cols:
        if not c.startswith("gas_"):
            continue
        try:
            temps.append(int(c.split("_", 1)[1]))
        except Exception:
            pass
    return temps

def load_model(path: str):
    global MODEL, MODEL_LABELS, MODEL_STEP_COLS, MODEL_EXPECTED_NFEAT, MODEL_PATH, HEATER_TEMPS

    with model_lock:
        MODEL_PATH = path
        if not os.path.exists(path):
            MODEL = None
            MODEL_LABELS = None
            MODEL_STEP_COLS = None
            MODEL_EXPECTED_NFEAT = None
            return False, f"Model not found: {path}"

        try:
            obj = joblib.load(path)

            if isinstance(obj, dict) and ("model" in obj or "clf" in obj):
                m = obj.get("model", obj.get("clf"))
                step_cols = obj.get("step_cols", None)
            else:
                m = obj
                step_cols = None

            MODEL = m
            MODEL_LABELS = list(getattr(m, "classes_", [])) or None
            MODEL_EXPECTED_NFEAT = getattr(m, "n_features_in_", None)
            MODEL_STEP_COLS = step_cols if isinstance(step_cols, list) else None

            if MODEL_STEP_COLS:
                temps = _step_cols_to_temps(MODEL_STEP_COLS)
                if temps:
                    with scan_lock:
                        HEATER_TEMPS = temps

            return True, f"Loaded model: {path}"
        except Exception as e:
            MODEL = None
            MODEL_LABELS = None
            MODEL_STEP_COLS = None
            MODEL_EXPECTED_NFEAT = None
            return False, f"Failed to load model: {e}"


# -------------------------
# Sensor helpers
# -------------------------
def init_sensor():
    global SENSOR_START
    i2c = busio.I2C(board.SCL, board.SDA)
    for a in (0x77, 0x76):
        try:
            s = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=a)
            print(f"[OK] Found BME688 @0x{a:02X}")
            SENSOR_START = time.time()
            return s
        except Exception:
            pass
    raise RuntimeError("BME688 not found on 0x77 or 0x76")

def read_scan(sensor):
    with scan_lock:
        temps = list(HEATER_TEMPS)
        ms = int(HEATER_MS)

    vals = []
    for t in temps:
        sensor.gas_heater_temperature = int(t)
        sensor.gas_heater_duration = ms

        _ = sensor.temperature
        time.sleep(ms / 1000.0 + 0.05)
        _ = sensor.temperature

        vals.append(int(sensor.gas))

    return vals, int(median(vals))


# -------------------------
# Feature builder that EXACTLY matches train_model.py
# -------------------------
def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0

def build_window_features_exact(gas_med_series: list[float], steps_series: list[list[float]]) -> np.ndarray | None:
    """
    EXACT match to train_model.py feature_vector_from_window()

    feat = [
      gas0, gas_end, gas_min, gas_mean, gas_std, gas_slope,
      x_min, t_min, x_mean, x_std, x_slope_start, x_slope_end,
      *step_abs_meds (k),
      *step_norm_meds (k),
      x_end, x_range
    ]
    Total = 12 + k + k + 2 = 14 + 2k
    """
    if gas_med_series is None or len(gas_med_series) < 20:
        return None

    med = np.array(gas_med_series, dtype=float)
    if not np.all(np.isfinite(med)):
        return None

    steps = np.array(steps_series, dtype=float)  # shape (T,k)
    if steps.ndim != 2 or steps.shape[0] != len(med):
        return None
    if steps.shape[1] < 2:
        return None

    gas0 = float(med[0])
    gas_end = float(med[-1])
    gas_min = float(np.min(med))
    gas_mean = float(np.mean(med))
    gas_std = float(np.std(med))
    gas_slope = _slope(med)

    base = float(np.median(med[:10]))
    if not np.isfinite(base) or base <= 0:
        return None

    x = (med - base) / base
    x_min = float(np.min(x))
    t_min = float(int(np.argmin(x)))
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    x_slope_start = _slope(x[: min(30, len(x))])
    x_slope_end = _slope(x[max(0, len(x) - 30):])
    x_end = float(x[-1])
    x_range = float(np.max(x) - np.min(x))

    k = steps.shape[1]
    step_abs_meds = []
    step_norm_meds = []
    for j in range(k):
        v = steps[:, j]
        if not np.all(np.isfinite(v)):
            return None
        step_abs_meds.append(float(np.median(v)))
        step_norm_meds.append(float(np.median((v - base) / base)))

    feat = np.array([
        gas0, gas_end, gas_min, gas_mean, gas_std, gas_slope,
        x_min, t_min, x_mean, x_std, x_slope_start, x_slope_end,
        *step_abs_meds,
        *step_norm_meds,
        x_end, x_range
    ], dtype=float)

    return feat


# -------------------------
# Plateau detector (window-level)
# -------------------------
def detect_plateau(gas_med_series: list[float], slope_ohms_per_sample_thr: float, std_ohms_thr: float, min_frac: float) -> dict:
    """
    Plateau means the window is largely flat:
      - std below threshold
      - slope below threshold
    We also check "flat fraction": fraction of points within +/- (2*std_thr) of median.
    """
    med = np.array(gas_med_series, dtype=float)
    if len(med) < 20:
        return {"plateau": False}

    s = float(np.std(med))
    sl = float(_slope(med))
    med0 = float(np.median(med))

    band = 2.0 * float(std_ohms_thr)
    flat_frac = float(np.mean(np.abs(med - med0) <= band))

    plateau = (s <= float(std_ohms_thr)) and (abs(sl) <= float(slope_ohms_per_sample_thr)) and (flat_frac >= float(min_frac))

    return {
        "plateau": bool(plateau),
        "plateau_std": s,
        "plateau_slope": sl,
        "plateau_flat_frac": flat_frac
    }


# -------------------------
# Classification
# -------------------------
def classify_feat(feat: np.ndarray, unknown_thr: float):
    with model_lock:
        m = MODEL
        labels = MODEL_LABELS
        nfeat = MODEL_EXPECTED_NFEAT

    if m is None:
        return "NO_MODEL", 0.0

    if nfeat is not None and feat.shape[0] != int(nfeat):
        return f"FEATURE_MISMATCH({feat.shape[0]}!={nfeat})", 0.0

    try:
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(feat.reshape(1, -1))[0]
            if labels is None:
                labels = [str(i) for i in range(len(proba))]
            best_i = int(np.argmax(proba))
            best_label = str(labels[best_i])
            best_p = float(proba[best_i])
            if best_p < float(unknown_thr):
                return "UNKNOWN", best_p
            return best_label, best_p
        pred = m.predict(feat.reshape(1, -1))[0]
        return str(pred), 1.0
    except Exception as e:
        return f"MODEL_ERR:{e}", 0.0


# -------------------------
# Live update with sticky + plateau latch
# -------------------------
def live_update(gas_med: int, steps: list[int], stable: int, dt: float):
    out = {}
    with live_lock:
        if not live.enabled or not stable:
            out.update({
                "live_enabled": live.enabled,
                "live_raw_label": live.last_raw_label,
                "live_raw_conf": live.last_raw_conf,
                "live_pred": live.sticky_label,
                "live_conf": live.sticky_conf,
                "sticky_left_s": round(live.sticky_left, 1),
                "plateau_latched": live.plateau_latched,
                "unknown_thr": live.unknown_thr,
                "window_s": live.window_s,
            })
            return out

        live.buf_med.append(float(gas_med))
        live.buf_steps.append([float(v) for v in steps])

        live.t_since_pred += dt
        if live.sticky_left > 0:
            live.sticky_left = max(0.0, live.sticky_left - dt)

        if live.t_since_pred < float(live.stride_s):
            out.update({
                "live_enabled": live.enabled,
                "live_raw_label": live.last_raw_label,
                "live_raw_conf": live.last_raw_conf,
                "live_pred": live.sticky_label,
                "live_conf": live.sticky_conf,
                "sticky_left_s": round(live.sticky_left, 1),
                "plateau_latched": live.plateau_latched,
                "unknown_thr": live.unknown_thr,
                "window_s": live.window_s,
            })
            return out

        live.t_since_pred = 0.0

        need = max(20, int(live.window_s))
        if len(live.buf_med) < need:
            out.update({
                "live_enabled": live.enabled,
                "live_msg": f"BUILDING_WINDOW ({len(live.buf_med)}/{need})",
                "live_pred": live.sticky_label,
                "live_conf": live.sticky_conf,
                "sticky_left_s": round(live.sticky_left, 1),
                "plateau_latched": live.plateau_latched,
                "unknown_thr": live.unknown_thr,
                "window_s": live.window_s,
            })
            return out

        gas_series = list(live.buf_med)[-need:]
        step_series = list(live.buf_steps)[-need:]

        plat = detect_plateau(
            gas_series,
            slope_ohms_per_sample_thr=live.plateau_slope_thr,
            std_ohms_thr=live.plateau_std_thr,
            min_frac=live.plateau_min_frac
        )

        feat = build_window_features_exact(gas_series, step_series)
        if feat is None:
            live.last_raw_label = "BAD_WINDOW"
            live.last_raw_conf = 0.0
            raw_label, raw_conf = "BAD_WINDOW", 0.0
        else:
            raw_label, raw_conf = classify_feat(feat, live.unknown_thr)
            live.last_raw_label = raw_label
            live.last_raw_conf = float(raw_conf)

        # --- Sticky logic ---
        # If we got a confident non-UNKNOWN prediction, refresh sticky.
        if raw_label not in (None, "UNKNOWN") and isinstance(raw_label, str) and raw_label.startswith("FEATURE_MISMATCH") is False:
            if raw_label not in ("NO_MODEL", "BAD_WINDOW") and not raw_label.startswith("MODEL_ERR"):
                if raw_conf >= live.unknown_thr:
                    live.sticky_label = raw_label
                    live.sticky_conf = float(raw_conf)
                    live.sticky_left = float(live.hold_s)

        # --- Plateau latch ---
        # If plateau is detected and we have any sticky label, latch it.
        if plat.get("plateau") and live.sticky_label and live.sticky_left > 0:
            live.plateau_latched = True
            live.plateau_label = live.sticky_label
            live.plateau_conf = live.sticky_conf

        # Plateau unlatch when sticky expires
        if live.sticky_left <= 0:
            live.plateau_latched = False
            live.plateau_label = None
            live.plateau_conf = None

        # Final output label:
        # - if plateau latched: show plateau_label
        # - else if sticky active: show sticky_label
        # - else show raw_label (may be UNKNOWN)
        final_label = None
        final_conf = None
        if live.plateau_latched and live.plateau_label:
            final_label = live.plateau_label
            final_conf = live.plateau_conf
        elif live.sticky_left > 0 and live.sticky_label:
            final_label = live.sticky_label
            final_conf = live.sticky_conf
        else:
            final_label = raw_label
            final_conf = float(raw_conf)

        out.update({
            "live_enabled": live.enabled,
            "live_raw_label": raw_label,
            "live_raw_conf": float(raw_conf),
            "live_pred": final_label,
            "live_conf": float(final_conf) if final_conf is not None else None,
            "sticky_left_s": round(live.sticky_left, 1),
            "plateau_latched": live.plateau_latched,
            **plat,
            "unknown_thr": live.unknown_thr,
            "window_s": live.window_s,
            "feature_len": int(feat.shape[0]) if isinstance(feat, np.ndarray) else None,
        })
        return out


# -------------------------
# Sensor loop
# -------------------------
def sensor_loop():
    global SENSOR
    SENSOR = init_sensor()
    os.makedirs("logs_scans", exist_ok=True)

    last_emit = 0.0
    last_time = time.time()

    while True:
        with state_lock:
            running = state.running
            session = state.session
            label = state.label
            interval = state.interval
            stop_at = state.stop_at

        now = time.time()
        dt = max(0.0, now - last_time)
        last_time = now

        with scan_lock:
            warmup_s = int(WARMUP_S)
            temps = list(HEATER_TEMPS)

        stable = int(SENSOR_START is not None and (now - SENSOR_START) >= warmup_s)

        try:
            steps, med = read_scan(SENSOR)

            payload = {
                "running": running,
                "label": label,
                "gas_steps": steps,
                "gas_med": med,
                "stable": stable,
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "heater_temps": temps,
            }

            payload.update(live_update(med, steps, stable, dt))

            if running and stable:
                if stop_at and now >= stop_at:
                    with state_lock:
                        state.running = False
                    payload["running"] = False
                else:
                    fpath = f"logs_scans/{session}.csv"
                    new = not os.path.exists(fpath)
                    with open(fpath, "a", newline="") as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(["utc", "label", "gas_med", *[f"gas_{t}" for t in temps]])
                        w.writerow([payload["utc"], label, med, *steps])

        except Exception as e:
            payload = {"error": str(e), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

        if now - last_emit > 0.25:
            last_emit = now
            try:
                event_q.put_nowait(payload)
            except Exception:
                pass

        time.sleep(interval if running else 0.25)


# -------------------------
# Routes
# -------------------------
@app.get("/")
def index():
    return render_template_string(HTML)

@app.get("/favicon.ico")
def favicon():
    return ("", 204)

@app.post("/api/start")
def api_start():
    d = request.json or {}
    with state_lock:
        state.running = True
        state.session = d.get("session", "session_live")
        state.label = d.get("label", "air")
        state.interval = float(d.get("interval", 1.0))
        dur = float(d.get("duration", 0))
        state.stop_at = time.time() + dur if dur > 0 else 0.0
    return jsonify(ok=True)

@app.post("/api/stop")
def api_stop():
    with state_lock:
        state.running = False
    return jsonify(ok=True)

@app.post("/api/scan")
def api_scan():
    global HEATER_MS, WARMUP_S, HEATER_TEMPS
    d = request.json or {}

    req_temps = [int(x.strip()) for x in str(d.get("temps", "")).split(",") if x.strip()]
    heater_ms = int(d.get("heater_ms", HEATER_MS))
    warmup_s = int(d.get("warmup_s", WARMUP_S))

    with model_lock:
        step_cols = MODEL_STEP_COLS
    forced = _step_cols_to_temps(step_cols) if step_cols else None

    with scan_lock:
        HEATER_MS = heater_ms
        WARMUP_S = warmup_s
        if forced:
            HEATER_TEMPS = forced
        elif req_temps:
            HEATER_TEMPS = req_temps
        temps = list(HEATER_TEMPS)

    return jsonify(ok=True, temps=temps, heater_ms=HEATER_MS, warmup_s=WARMUP_S, forced_by_model=bool(forced))

@app.post("/api/model/reload")
def api_model_reload():
    d = request.json or {}
    mp = str(d.get("model_path", MODEL_PATH))
    ok, msg = load_model(mp)
    with scan_lock:
        temps = list(HEATER_TEMPS)
    return jsonify(ok=ok, msg=msg, temps=temps)

@app.post("/api/model/sync_heaters")
def api_model_sync_heaters():
    with model_lock:
        step_cols = MODEL_STEP_COLS
    forced = _step_cols_to_temps(step_cols) if step_cols else None
    if forced:
        with scan_lock:
            global HEATER_TEMPS
            HEATER_TEMPS = forced
    with scan_lock:
        temps = list(HEATER_TEMPS)
    return jsonify(ok=True, temps=temps, forced_by_model=bool(forced))

@app.post("/api/live/start")
def api_live_start():
    d = request.json or {}
    mp = str(d.get("model_path", MODEL_PATH))
    ok, msg = load_model(mp)

    with live_lock:
        live.enabled = True
        live.unknown_thr = float(d.get("unknown_thr", live.unknown_thr))
        live.hold_s = float(d.get("hold_s", live.hold_s))
        live.plateau_std_thr = float(d.get("plateau_std_thr", live.plateau_std_thr))

        live.buf_med.clear()
        live.buf_steps.clear()
        live.t_since_pred = 0.0

        live.last_raw_label = None
        live.last_raw_conf = None

        live.sticky_label = None
        live.sticky_conf = None
        live.sticky_left = 0.0

        live.plateau_latched = False
        live.plateau_label = None
        live.plateau_conf = None

    with scan_lock:
        temps = list(HEATER_TEMPS)

    return jsonify(ok=True, model_loaded=ok, model_msg=msg, temps=temps)

@app.post("/api/live/stop")
def api_live_stop():
    with live_lock:
        live.enabled = False
        live.last_raw_label = None
        live.last_raw_conf = None
        live.sticky_label = None
        live.sticky_conf = None
        live.sticky_left = 0.0
        live.plateau_latched = False
        live.plateau_label = None
        live.plateau_conf = None
    return jsonify(ok=True)

@app.get("/stream")
def stream():
    import json
    def gen():
        while True:
            try:
                m = event_q.get(timeout=15)
            except Empty:
                yield "data: {}\n\n"
                continue
            yield f"data: {json.dumps(m)}\n\n"
    return Response(gen(), mimetype="text/event-stream")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()

    load_model(args.model)  # not fatal

    t = threading.Thread(target=sensor_loop, daemon=True)
    t.start()
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
