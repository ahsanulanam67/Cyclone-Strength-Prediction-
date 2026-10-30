"""
Train the Random Forest track predictor on IBTrACS Bay of Bengal data.
Saves model to models/track_predictor.pkl

Run:
    python scripts/train_track_predictor.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

INPUT_STEPS  = 4   # 4 x 6h = 24h history
OUTPUT_STEPS = 8   # 8 x 6h = 48h forecast

IBTRACS_PATH = ROOT / "data" / "ibtracs_bob.csv"
MODEL_PATH   = ROOT / "models" / "track_predictor.pkl"


def load_ibtracs():
    df = pd.read_csv(IBTRACS_PATH)
    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]
    # Keep Bay of Bengal storms
    if "BASIN" in df.columns:
        df = df[df["BASIN"].str.strip() == "NI"].copy()
    df = df.dropna(subset=["LAT", "LON", "WIND"]).copy()
    df["LAT"]   = pd.to_numeric(df["LAT"],  errors="coerce")
    df["LON"]   = pd.to_numeric(df["LON"],  errors="coerce")
    df["WIND"]  = pd.to_numeric(df["WIND"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON", "WIND"])

    # Parse ISO_TIME or similar
    time_col = next((c for c in df.columns if "TIME" in c or "DATE" in c), None)
    if time_col:
        df["DT"] = pd.to_datetime(df[time_col], errors="coerce")
        df["MONTH"] = df["DT"].dt.month.fillna(10).astype(int)
        df["HOUR"]  = df["DT"].dt.hour.fillna(0).astype(int)
    else:
        df["MONTH"] = 10
        df["HOUR"]  = 0

    sid_col = next((c for c in df.columns if "SID" in c or "ID" in c), None)
    if sid_col is None:
        raise ValueError("No storm ID column found in IBTrACS CSV")
    df["SID"] = df[sid_col]
    return df


def build_sequences(df):
    X_list, Y_list = [], []
    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("DT" if "DT" in grp.columns else grp.columns[0]).reset_index(drop=True)
        lats  = grp["LAT"].values
        lons  = grp["LON"].values
        winds = grp["WIND"].values
        months = grp["MONTH"].values
        hours  = grp["HOUR"].values

        n = len(grp)
        if n < INPUT_STEPS + OUTPUT_STEPS:
            continue

        for start in range(n - INPUT_STEPS - OUTPUT_STEPS + 1):
            inp = slice(start, start + INPUT_STEPS)
            out = slice(start + INPUT_STEPS, start + INPUT_STEPS + OUTPUT_STEPS)

            feat = []
            for k in range(INPUT_STEPS):
                feat += [lats[inp][k], lons[inp][k], winds[inp][k]]
                if k > 0:
                    feat += [lats[inp][k] - lats[inp][k-1],
                             lons[inp][k] - lons[inp][k-1],
                             winds[inp][k] - winds[inp][k-1]]
                else:
                    feat += [0.0, 0.0, 0.0]
            feat += [float(months[inp][-1]), float(hours[inp][-1])]

            base_lat = lats[inp][-1]
            base_lon = lons[inp][-1]
            tgt = []
            for k in range(OUTPUT_STEPS):
                tgt += [lats[out][k] - base_lat, lons[out][k] - base_lon]

            X_list.append(feat)
            Y_list.append(tgt)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def main():
    print("Loading IBTrACS data...")
    df = load_ibtracs()
    print(f"  {len(df)} observations, {df['SID'].nunique()} storms")

    print("Building sequences...")
    X, Y = build_sequences(df)
    print(f"  {len(X)} training sequences, {X.shape[1]} features, {Y.shape[1]} targets")

    if len(X) < 10:
        print("ERROR: Not enough sequences to train. Check ibtracs_bob.csv has Bay of Bengal storms.")
        sys.exit(1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Random Forest (300 trees)... this takes ~2-3 minutes")
    rf = RandomForestRegressor(n_estimators=300, max_depth=10,
                               n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(rf, n_jobs=1)
    model.fit(X_scaled, Y)

    # Quick MAE check
    preds = model.predict(X_scaled[:200])
    mae_24h = np.mean(np.abs(preds[:, :2] - Y[:200, :2])) * 111  # degrees → km
    mae_48h = np.mean(np.abs(preds[:, 14:16] - Y[:200, 14:16])) * 111
    print(f"  Train MAE  24h: {mae_24h:.1f} km   48h: {mae_48h:.1f} km")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model":        model,
        "scaler":       scaler,
        "input_steps":  INPUT_STEPS,
        "output_steps": OUTPUT_STEPS,
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Done! You can now run the dashboard.")


if __name__ == "__main__":
    main()
