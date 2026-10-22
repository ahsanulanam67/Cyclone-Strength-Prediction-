"""
preprocessing.py — IBTrACS data loading and feature engineering.

Functions
---------
load_ibtracs(url_or_path)
    Load and clean an IBTrACS CSV file.
add_track_features(df)
    Derive kinematic and temporal features from the storm track.
filter_bay_of_bengal(df)
    Restrict data to the Bay of Bengal domain.
"""

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ibtracs(url_or_path: str) -> pd.DataFrame:
    """
    Load an IBTrACS CSV file and return a cleaned DataFrame.

    The IBTrACS CSV format has a standard header on row 0 and a units/metadata
    row on row 1, so ``skiprows=[1]`` is used to drop the units row while
    keeping the column names.

    Parameters
    ----------
    url_or_path : str
        Local file path or HTTP URL to the IBTrACS CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns including SID, ISO_TIME (UTC datetime),
        LAT, LON, WIND, and any other columns present in the source file.
    """
    df = pd.read_csv(url_or_path, skiprows=[1], low_memory=False)

    # Coerce key numeric columns
    for col in ["LAT", "LON", "USA_WIND", "WMO_WIND", "USA_PRES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse ISO_TIME as UTC-aware datetime
    if "ISO_TIME" in df.columns:
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")

    # Build unified WIND column: prefer USA_WIND, fall back to WMO_WIND
    if "USA_WIND" in df.columns and "WMO_WIND" in df.columns:
        df["WIND"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    elif "USA_WIND" in df.columns:
        df["WIND"] = df["USA_WIND"]
    elif "WMO_WIND" in df.columns:
        df["WIND"] = df["WMO_WIND"]
    else:
        df["WIND"] = np.nan

    df["WIND"] = pd.to_numeric(df["WIND"], errors="coerce")

    # Drop rows missing essential identifiers
    df = df.dropna(subset=["SID", "ISO_TIME", "WIND"])

    # Sort for chronological per-storm ordering
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    return df


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived kinematic and temporal features to a storm-track DataFrame.

    New columns added
    -----------------
    MONTH : int (1–12)
    HOUR : int (0–23)
    LAT_CHANGE : float  — latitude change since previous observation (degrees)
    LON_CHANGE : float  — longitude change since previous observation (degrees)
    WIND_CHANGE : float — wind-speed change since previous observation (knots)
    TRACK_SPEED : float — estimated translation speed (km h⁻¹) between consecutive
                          observations derived from the haversine formula.

    The first observation of each storm gets NaN for all difference-based columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SID, ISO_TIME, LAT, LON, WIND.

    Returns
    -------
    pd.DataFrame
        Modified in-place copy with the new feature columns appended.
    """
    df = df.copy().sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # Temporal features
    if pd.api.types.is_datetime64_any_dtype(df["ISO_TIME"]):
        dt = df["ISO_TIME"]
        df["MONTH"] = dt.dt.month.astype(float)
        df["HOUR"] = dt.dt.hour.astype(float)
    else:
        df["MONTH"] = np.nan
        df["HOUR"] = np.nan

    # Per-storm difference features
    grp = df.groupby("SID", sort=False)
    df["LAT_CHANGE"] = grp["LAT"].diff()
    df["LON_CHANGE"] = grp["LON"].diff()
    df["WIND_CHANGE"] = grp["WIND"].diff()

    # Track speed: haversine distance / time gap in hours
    speeds = np.full(len(df), np.nan)
    for sid, group in df.groupby("SID", sort=False):
        idxs = group.index.tolist()
        lats = group["LAT"].values
        lons = group["LON"].values
        times = group["ISO_TIME"].values

        for k in range(1, len(idxs)):
            try:
                dist_km = _haversine_km(lats[k - 1], lons[k - 1], lats[k], lons[k])
                dt_hr = (times[k] - times[k - 1]) / np.timedelta64(1, "h")
                if dt_hr > 0:
                    speeds[idxs[k]] = dist_km / dt_hr
            except Exception:
                pass

    df["TRACK_SPEED"] = speeds
    return df


def filter_bay_of_bengal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose position falls within the Bay of Bengal domain.

    Domain: latitude 0°–25°N, longitude 80°–100°E.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain LAT and LON columns.

    Returns
    -------
    pd.DataFrame
        Filtered subset (index reset).
    """
    mask = (
        df["LAT"].between(0, 25) &
        df["LON"].between(80, 100)
    )
    return df[mask].reset_index(drop=True)
