#!/usr/bin/env python3
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

        # Your format:
        # utc,label,gas_med,gas_200,gas_260,gas_320,(optional temp/humidity/pressure)
        required = {"utc", "label", "gas_med"}
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] skipping {p}: missing columns {sorted(missing)}")
            continue

        df = df.copy()
        df["source_file"] = os.path.basename(p)

        # Clean label
        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["label"] != ""]
        df = df[df["label"].str.lower() != "label"]  # safety (header artifacts)

        # Parse time (not strictly needed, but helps sorting)
        df["utc"] = pd.to_datetime(df["utc"], errors="coerce")
        df = df.dropna(subset=["utc", "label", "gas_med"])

        # Ensure numeric
        num_cols = [c for c in df.columns if c.startswith("gas_")] + ["gas_med", "temp_c", "humidity_rh", "pressure_hpa"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Need heater-step columns
        step_cols = [c for c in df.columns if c.startswith("gas_") and c != "gas_med"]
        if len(step_cols) < 2:
            print(f"[WARN] skipping {p}: not enough heater-step columns")
            continue

        df = df.dropna(subset=["gas_med"] + step_cols)
        if len(df) < 20:
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
    for fname, g in df.groupby("source_file", sort=False):
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
    return float(np.polyfit(x, y, 1)[0])


def segment_features(seg: Segment) -> tuple[np.ndarray, str] | None:
    d = seg.df

    step_cols = sorted([c for c in d.columns if c.startswith("gas_") and c != "gas_med"])
    if not step_cols:
        return None

    med = d["gas_med"].to_numpy(dtype=float)
    if len(med) < 20:
        return None

    # Normalize by segment baseline (first 10 samples median)
    base = float(np.median(med[:10]))
    if not np.isfinite(base) or base <= 0:
        return None
    x = (med - base) / base

    # Core shape features (how it drops/recovers)
    x_min = float(np.min(x))
    t_min = int(np.argmin(x))  # samples-to-min
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    x_slope_start = slope(x[:30])
    x_slope_end = slope(x[-30:])

    # Heater fingerprint features: median value per heater step (also normalized by base)
    step_feats = []
    for c in step_cols:
        v = d[c].to_numpy(dtype=float)
        step_feats.append(float(np.median((v - base) / base)))

    # Optional environment features
    temp = float(np.mean(d["temp_c"])) if "temp_c" in d.columns else 0.0
    hum = float(np.mean(d["humidity_rh"])) if "humidity_rh" in d.columns else 0.0
    pres = float(np.mean(d["pressure_hpa"])) if "pressure_hpa" in d.columns else 0.0

    feat = np.array(
        [x_min, t_min, x_mean, x_std, x_slope_start, x_slope_end, temp, hum, pres, *step_feats],
        dtype=float,
    )
    return feat, seg.label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs_scans", help="Folder with scan CSVs (default: logs_scans)")
    ap.add_argument("--out", default="models/model.joblib", help="Output model path")
    ap.add_argument("--min-segments-per-label", type=int, default=2, help="Require at least N segments per label")
    ap.add_argument("--drop-label", action="append", default=[], help="Drop a label entirely (can repeat)")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = load_scans(args.logs)
    segments = split_segments(df)

    X, y = [], []
    for seg in segments:
        lbl = seg.label
        if lbl in set(args.drop_label):
            continue
        out = segment_features(seg)
        if out is None:
            continue
        feat, label = out
        X.append(feat)
        y.append(label)

    if len(X) < 6:
        raise SystemExit(f"Not enough usable segments. Got {len(X)} segments.")

    X = np.vstack(X)
    y = np.array(y, dtype=str)

    # Show label counts by *segments* (not raw rows)
    counts = pd.Series(y).value_counts()
    print("\n=== Segments per label ===")
    print(counts.to_string())

    # Enforce minimum segments/label if requested
    keep_labels = set(counts[counts >= args.min_segments_per_label].index)
    mask = np.array([lbl in keep_labels for lbl in y], dtype=bool)
    X, y = X[mask], y[mask]

    if len(set(y)) < 2:
        raise SystemExit("Need at least 2 labels (classes) after filtering to train a classifier.")

    # Train/test split
    strat = y if len(set(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=strat
    )

    clf = RandomForestClassifier(
        n_estimators=500,
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
    print(pd.DataFrame(confusion_matrix(yte, pred, labels=labels_sorted),
                       index=[f"true:{l}" for l in labels_sorted],
                       columns=[f"pred:{l}" for l in labels_sorted]).to_string())

    joblib.dump(clf, args.out)
    print(f"\n[OK] Saved model -> {args.out}\n")


if __name__ == "__main__":
    main()
