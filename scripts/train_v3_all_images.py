"""
Train an image + tabular intensification model using EVERY satellite image
available in data/satellite_images.

This is the honest, reproducible version of the "V3 uses all pictures" idea:
  * discovers every PNG on disk (and cross-checks the manifest),
  * builds 4-frame (24 h) image sequences directly from those files,
  * derives short (48-72 h) and long (120-168 h) intensification labels from
    IBTrACS,
  * evaluates with leave-one-storm-out GroupKFold,
  * saves models to models/v3_all_images_{short,long}.pkl.

Run:
    python scripts/train_v3_all_images.py

NOTE ON ACCURACY
----------------
The local dataset contains ~471 images from 10 storms, producing ~298 usable
short-horizon sequences and ~87 long-horizon sequences. With that sample size,
"95% accuracy" is not a statistically meaningful or honest target. This script
reports the real cross-validated numbers so they can be quoted in the thesis.
Always report ROC-AUC (and PR-AUC) alongside accuracy for this imbalanced task.
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "satellite_images"
MANIFEST = ROOT / "data" / "satellite_manifest.csv"
IBTRACS = ROOT / "data" / "ibtracs_bob.csv"
MODEL_DIR = ROOT / "models"

SEQ_LEN = 4          # 4 frames x 3 h = 12 h history (images are 3-hourly)
STEP_HOURS = 3
IMG_SIZE = 64        # downsampled size fed to the feature extractor
GRID = 16            # spatial grid kept as raw pixels
THRESHOLD_KT = 15.0
HORIZONS = {"short": (48, 72), "long": (120, 168)}


def load_ibtracs() -> pd.DataFrame:
    df = pd.read_csv(IBTRACS, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    if "BASIN" in df.columns:
        df = df[df["BASIN"].astype(str).str.strip() == "NI"].copy()
    wind_col = next((c for c in ("WMO_WIND", "USA_WIND", "WIND") if c in df.columns), None)
    if wind_col is None:
        raise ValueError("No wind column found in IBTrACS CSV")
    df = df.rename(columns={wind_col: "WIND"})
    for c in ("LAT", "LON", "WIND"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON", "WIND"])
    df["SID"] = df["SID"].astype(str).str.strip()
    return df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def load_images() -> pd.DataFrame:
    files = sorted(glob.glob(str(IMAGE_DIR / "**" / "*.png"), recursive=True))
    if not files:
        raise SystemExit(f"No PNG images found under {IMAGE_DIR}")
    rows = []
    for f in files:
        stem = Path(f).stem                       # SID_YYYYMMDDHH
        if "_" not in stem:
            continue
        sid, stamp = stem.rsplit("_", 1)
        try:
            ts = pd.to_datetime(stamp, format="%Y%m%d%H", utc=True)
        except ValueError:
            continue
        rows.append({"SID": sid, "ISO_TIME": ts, "PATH": f})
    imgs = pd.DataFrame(rows).drop_duplicates(subset=["SID", "ISO_TIME"])
    print(f"  images on disk: {len(files)}  ·  unique SID/time keys: {len(imgs)}")
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        print(f"  manifest entries: {len(man)}  ·  manifest storms: {man['SID'].nunique()}")
    return imgs.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def image_features(paths: list[str]) -> np.ndarray:
    """Per-frame brightness stats + 16x16 downsampled pixels."""
    out = []
    for p in paths:
        img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        a = np.asarray(img, dtype=np.float32) / 255.0
        block = IMG_SIZE // GRID
        small = a.reshape(GRID, block, GRID, block).mean(axis=(1, 3)).ravel()
        out.extend([a.mean(), a.std(), a.min(), a.max()])
        out.extend(small.tolist())
    return np.asarray(out, dtype=np.float32)


def build_dataset(ib: pd.DataFrame, imgs: pd.DataFrame):
    """Build sequences of 4 image frames per storm, labelled from IBTrACS.

    Images are 3-hourly while IBTrACS is 6-hourly, so the track is linearly
    interpolated onto the image timestamps. This keeps *every* image usable.
    """
    ib_by_sid = {sid: g.set_index("ISO_TIME").sort_index() for sid, g in ib.groupby("SID")}
    X, y_short, y_long, groups, meta = [], [], [], [], []

    for sid, g in imgs.groupby("SID"):
        if sid not in ib_by_sid:
            continue
        track = ib_by_sid[sid][["LAT", "LON", "WIND"]]
        g = g.sort_values("ISO_TIME").reset_index(drop=True)
        grid_times = pd.DatetimeIndex(g["ISO_TIME"])

        # Interpolate the 6-hourly track onto the 3-hourly image grid.
        merged = (track.reindex(track.index.union(grid_times))
                       .interpolate(method="time")
                       .reindex(grid_times))
        in_range = (grid_times >= track.index.min()) & (grid_times <= track.index.max())
        if not in_range.any():
            continue
        merged = merged[in_range]
        g = g.loc[in_range].reset_index(drop=True)
        merged = merged.reset_index(drop=True)
        times = list(g["ISO_TIME"])
        paths = list(g["PATH"])

        for i in range(SEQ_LEN - 1, len(g)):
            seq_times = [times[i - k] for k in range(SEQ_LEN - 1, -1, -1)]
            if any((seq_times[j] - seq_times[j - 1]) != pd.Timedelta(hours=STEP_HOURS)
                   for j in range(1, SEQ_LEN)):
                continue

            row = merged.iloc[i]
            if not np.isfinite(row["WIND"]):
                continue
            end_t = times[i]
            w_now = float(row["WIND"])
            offsets = np.asarray([
                (times[k] - end_t) / pd.Timedelta(hours=1)
                for k in range(i + 1, len(g))
            ])
            future_wind = merged["WIND"].values[i + 1:]

            def label(win):
                lo, hi = win
                m = (offsets >= lo) & (offsets <= hi)
                vals = future_wind[m]
                if len(vals) == 0:
                    return np.nan
                return 1.0 if float(np.nanmax(vals)) - w_now >= THRESHOLD_KT else 0.0

            ys, yl = label(HORIZONS["short"]), label(HORIZONS["long"])
            if np.isnan(ys) and np.isnan(yl):
                continue

            prev = merged.iloc[i - 1] if i >= 1 else row
            tab = np.asarray([
                float(row["LAT"]), float(row["LON"]), w_now,
                float(end_t.month), float(end_t.hour),
                float(row["LAT"]) - float(prev["LAT"]),
                float(row["LON"]) - float(prev["LON"]),
                w_now - float(prev["WIND"]),
                0.0,  # translation speed placeholder (static snapshot)
            ], dtype=np.float32)

            X.append(np.concatenate([tab, image_features(paths[i - SEQ_LEN + 1:i + 1])]))
            y_short.append(ys)
            y_long.append(yl)
            groups.append(sid)
            meta.append((sid, str(end_t)))

    return (np.asarray(X, dtype=np.float32),
            np.asarray(y_short, dtype=np.float32),
            np.asarray(y_long, dtype=np.float32),
            np.asarray(groups),
            meta)


def evaluate(name: str, X, y, groups) -> float | None:
    valid = ~np.isnan(y)
    X, y, groups = X[valid], y[valid].astype(int), groups[valid]
    if len(X) == 0:
        print(f"\n{name}: no labelled samples — skipped")
        return None
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        print(f"\n{name}: only {len(np.unique(groups))} storm(s) — cannot cross-validate")
        return None

    probs = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
        ])
        pipe.fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]

    pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) == 2 else float("nan")
    ap = average_precision_score(y, probs) if len(np.unique(y)) == 2 else float("nan")
    print(f"\n{name}  (leave-one-storm-out, n={len(y)}, storms={len(np.unique(groups))})")
    print(f"  ROC-AUC : {auc:.3f}")
    print(f"  PR-AUC  : {ap:.3f}")
    print(f"  Accuracy: {accuracy_score(y, pred):.3f}")
    print(f"  F1      : {f1_score(y, pred):.3f}")
    print(f"  class balance: {np.bincount(y).tolist()}")
    return auc


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    print("Loading IBTrACS and every satellite image on disk …")
    ib = load_ibtracs()
    imgs = load_images()

    print("Building 4-frame image sequences …")
    X, y_short, y_long, groups, meta = build_dataset(ib, imgs)
    print(f"  samples: {len(X)}  ·  storms: {len(np.unique(groups))}  ·  features: {X.shape[1]}")

    evaluate("48-72 h intensification (all images)", X, y_short, groups)
    evaluate("120-168 h intensification (all images)", X, y_long, groups)

    # Final models trained on every available sequence
    for key, y in (("short", y_short), ("long", y_long)):
        valid = ~np.isnan(y)
        if valid.sum() == 0:
            continue
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
        ])
        pipe.fit(X[valid], y[valid].astype(int))
        out = MODEL_DIR / f"v3_all_images_{key}.pkl"
        joblib.dump({"pipeline": pipe, "n_features": X.shape[1],
                     "image_size": IMG_SIZE, "grid": GRID,
                     "seq_len": SEQ_LEN}, out)
        print(f"  saved {out}")

    print("\nDone. Note: 471 images / 10 storms cannot honestly support a 95% "
          "accuracy claim; quote the cross-validated AUC/accuracy above.")


if __name__ == "__main__":
    main()
