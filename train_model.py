#!/usr/bin/env python3
"""
train_model.py
Window-based + plateau-aware training for BME688 scan logs.

- Reads logs_scans/*.csv
- Splits into labeled segments (contiguous blocks of same label)
- From each segment, generates multiple fixed-length windows:
    * early window
    * mid window
    * late/plateau window
  so the model learns both "drop" and "steady-state plateau" behavior.

- Builds FEATURES that include:
    * absolute gas_med level stats (so "air ~ 50k" vs "cologne ~ 6k" is learnable)
    * normalized shape stats (so it can generalize across drift)
    * heater-step absolute + normalized medians (fingerprint)

- Saves a model bundle to models/model.joblib:
    {
      "model": sklearn_model,
      "step_cols": ["gas_200", "gas_260", ...]  (in order),
      "feature_schema": {...},
      "classes": [...]
    }

Run:
  (venv) python3 train_model.py --logs logs_scans --out models/model.joblib --drop-label recovery_air
"""

import os
import glob
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Segment:
    label: str
    df: pd.DataFrame


def load_scans(folder: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files found in {folder}/")

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")
            continue

        required = {"utc", "label", "gas_med"}
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] skipping {p}: missing columns {sorted(missing)}")
            continue

        df = df.copy()
        df["source_file"] = os.path.basename(p)

        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["label"] != ""]
        df = df[df["label"].str.lower() != "label"]

        df["utc"] = pd.to_datetime(df["utc"], errors="coerce")
        df = df.dropna(subset=["utc", "label", "gas_med"])

        # numeric coercion
        for c in df.columns:
            if c.startswith("gas_") or c in ("gas_med", "temp_c", "humidity_rh", "pressure_hpa"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        step_cols = [c for c in df.columns if c.startswith("gas_") and c != "gas_med"]
        if len(step_cols) < 2:
            print(f"[WARN] skipping {p}: not enough heater-step columns")
            continue

        df = df.dropna(subset=["gas_med"] + step_cols)
        if len(df) < 40:
            print(f"[WARN] skipping {p}: too few rows after cleaning")
            continue

        dfs.append(df)

    if not dfs:
        raise SystemExit("No usable scan logs found.")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["source_file", "utc"]).reset_index(drop=True)
    return out


def split_segments(df: pd.DataFrame) -> list[Segment]:
    segs: list[Segment] = []
    for _, g in df.groupby("source_file", sort=False):
        g = g.sort_values("utc").reset_index(drop=True)
        labels = g["label"]
        seg_id = (labels != labels.shift(1)).cumsum()

        for _, block in g.groupby(seg_id, sort=False):
            lbl = str(block["label"].iloc[0]).strip()
            if not lbl:
                continue
            segs.append(Segment(label=lbl, df=block.reset_index(drop=True)))

    return segs


def slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def feature_vector_from_window(d: pd.DataFrame, step_cols: list[str]) -> np.ndarray | None:
    """
    IMPORTANT: must match bme688_web_logger.py feature builder.

    Feature count = 20 + 2*k? No: we lock it to exactly:
      12 core + 2*k heater (abs + norm) + 2 extras
    With k=3 heater steps => 12 + 6 + 2 = 20.
    """
    if len(d) < 20:
        return None

    med = d["gas_med"].to_numpy(dtype=float)
    if not np.all(np.isfinite(med)):
        return None

    gas0 = float(med[0])
    gas_end = float(med[-1])
    gas_min = float(np.min(med))
    gas_mean = float(np.mean(med))
    gas_std = float(np.std(med))
    gas_slope = slope(med)

    # baseline from first 10 samples
    base = float(np.median(med[:10]))
    if not np.isfinite(base) or base <= 0:
        return None

    x = (med - base) / base
    x_min = float(np.min(x))
    t_min = int(np.argmin(x))
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    x_slope_start = slope(x[: min(30, len(x))])
    x_slope_end = slope(x[max(0, len(x) - 30):])
    x_end = float(x[-1])
    x_range = float(np.max(x) - np.min(x))

    # heater step medians (absolute + normalized by base)
    step_abs_meds = []
    step_norm_meds = []
    for c in step_cols:
        v = d[c].to_numpy(dtype=float)
        if not np.all(np.isfinite(v)):
            return None
        step_abs_meds.append(float(np.median(v)))
        step_norm_meds.append(float(np.median((v - base) / base)))

    feat = np.array([
        gas0, gas_end, gas_min, gas_mean, gas_std, gas_slope,
        x_min, float(t_min), x_mean, x_std, x_slope_start, x_slope_end,
        # heater abs medians (k)
        *step_abs_meds,
        # heater norm medians (k)
        *step_norm_meds,
        # extras
        x_end, x_range
    ], dtype=float)

    return feat


def generate_windows(seg_df: pd.DataFrame, window_n: int, stride_n: int, mode: str) -> list[pd.DataFrame]:
    """
    mode:
      - "early_mid_late": always returns up to 3 windows (early/mid/late) if possible
      - "all": returns sliding windows across the whole segment
    """
    n = len(seg_df)
    if n < window_n:
        return []

    if mode == "all":
        outs = []
        for start in range(0, n - window_n + 1, stride_n):
            outs.append(seg_df.iloc[start:start + window_n])
        return outs

    # early/mid/late
    w1 = seg_df.iloc[0:window_n]
    mid_start = max(0, (n // 2) - (window_n // 2))
    w2 = seg_df.iloc[mid_start:mid_start + window_n]
    w3 = seg_df.iloc[n - window_n:n]

    outs = [w1]
    # avoid duplicates if segment is short-ish
    if mid_start > 0 and (mid_start + window_n) <= n and mid_start != 0 and mid_start != (n - window_n):
        outs.append(w2)
    if (n - window_n) != 0:
        outs.append(w3)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs_scans", help="Folder with scan CSVs")
    ap.add_argument("--out", default="models/model.joblib", help="Output model bundle path")
    ap.add_argument("--drop-label", action="append", default=[], help="Drop a label entirely (repeatable)")
    ap.add_argument("--min-windows-per-label", type=int, default=6, help="Require at least N windows per label")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--window-s", type=int, default=45, help="Window length in seconds (assumes ~1 scan/sec)")
    ap.add_argument("--stride-s", type=int, default=10, help="Stride in seconds for sliding windows")
    ap.add_argument("--window-mode", choices=["early_mid_late", "all"], default="early_mid_late",
                    help="Window generation mode")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = load_scans(args.logs)

    # determine step_cols from the first usable file (stable ordering)
    all_step_cols = sorted([c for c in df.columns if c.startswith("gas_") and c != "gas_med"])
    if len(all_step_cols) < 2:
        raise SystemExit("Not enough heater-step columns in loaded data.")
    step_cols = all_step_cols

    segments = split_segments(df)

    window_n = int(args.window_s)  # assume ~1 scan/sec
    stride_n = max(1, int(args.stride_s))

    X, y = [], []
    for seg in segments:
        if seg.label in set(args.drop_label):
            continue

        windows = generate_windows(seg.df, window_n=window_n, stride_n=stride_n, mode=args.window_mode)
        for w in windows:
            feat = feature_vector_from_window(w, step_cols=step_cols)
            if feat is None:
                continue
            X.append(feat)
            y.append(seg.label)

    if len(X) < 20:
        raise SystemExit(f"Not enough usable windows. Got {len(X)} windows.")

    X = np.vstack(X)
    y = np.array(y, dtype=str)

    counts = pd.Series(y).value_counts()
    print("\n=== Windows per label ===")
    print(counts.to_string())

    keep = set(counts[counts >= args.min_windows_per_label].index)
    mask = np.array([lbl in keep for lbl in y], dtype=bool)
    X, y = X[mask], y[mask]

    if len(set(y)) < 2:
        raise SystemExit("Need at least 2 labels (classes) after filtering.")

    # train/test
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=args.seed,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=1,
    )
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)

    print(f"\n=== Quick Accuracy (sanity check) === {acc:.3f}\n")
    print("=== Classification report ===")
    print(classification_report(yte, pred, digits=3))

    labels_sorted = sorted(list(set(y)))
    print("=== Confusion matrix ===")
    print(pd.DataFrame(
        confusion_matrix(yte, pred, labels=labels_sorted),
        index=[f"true:{l}" for l in labels_sorted],
        columns=[f"pred:{l}" for l in labels_sorted],
    ).to_string())

    bundle = {
        "model": clf,
        "step_cols": step_cols,
        "classes": list(getattr(clf, "classes_", [])),
        "feature_schema": {
            "version": 1,
            "window_s": args.window_s,
            "feature_count": int(X.shape[1]),
            "description": "abs gas stats + normalized shape + heater abs+norm + x_end+x_range",
        },
    }
    joblib.dump(bundle, args.out)

    print(f"\n[OK] Saved model bundle -> {args.out}")
    print(f"[OK] step_cols = {step_cols}")
    print(f"[OK] feature count = {X.shape[1]}\n")


if __name__ == "__main__":
    main()
