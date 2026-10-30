"""
Track Prediction Module
=======================
Predicts future cyclone positions (lat/lon) for up to 48h ahead
using a Random Forest model trained on IBTrACS Bay of Bengal data.

Input:  last 4 observations (24h history) of lat, lon, wind
Output: 8 future positions (6h intervals, up to 48h ahead)
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "track_predictor.pkl"
_cache = {}


def _load():
    if "model" not in _cache:
        obj = joblib.load(MODEL_PATH)
        _cache["model"]   = obj["model"]
        _cache["scaler"]  = obj["scaler"]
        _cache["in_steps"]  = obj["input_steps"]
        _cache["out_steps"] = obj["output_steps"]
    return _cache


def predict_track(history: list[dict], n_steps: int = 8) -> list[dict]:
    """
    Predict future track positions.

    Parameters
    ----------
    history : list of dicts, each with keys lat, lon, wind, month, hour
              Must have at least 4 entries (most recent last).
    n_steps : int, number of 6h steps to predict (max 8 = 48h)

    Returns
    -------
    list of dicts: [{"lat": ..., "lon": ..., "hours_ahead": ...}, ...]
    """
    obj = _load()
    model   = obj["model"]
    scaler  = obj["scaler"]
    in_steps = obj["in_steps"]

    if len(history) < in_steps:
        # Pad with first entry if too short
        history = [history[0]] * (in_steps - len(history)) + list(history)
    history = history[-in_steps:]

    feat = []
    for k, h in enumerate(history):
        feat += [h["lat"], h["lon"], h["wind"]]
        if k > 0:
            feat += [history[k]["lat"] - history[k-1]["lat"],
                     history[k]["lon"] - history[k-1]["lon"],
                     history[k]["wind"] - history[k-1]["wind"]]
        else:
            feat += [0.0, 0.0, 0.0]
    feat += [float(history[-1].get("month", 10)),
             float(history[-1].get("hour", 0))]

    X = scaler.transform([feat])
    pred = model.predict(X)[0]  # shape (16,): [dlat0,dlon0, dlat1,dlon1, ...]

    cur_lat = history[-1]["lat"]
    cur_lon = history[-1]["lon"]
    results = []
    for i in range(min(n_steps, len(pred) // 2)):
        results.append({
            "lat":         round(cur_lat + float(pred[i*2]),   2),
            "lon":         round(cur_lon + float(pred[i*2+1]), 2),
            "hours_ahead": (i + 1) * 6,
        })
    return results
