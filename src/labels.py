"""
labels.py — Cyclone intensification label creation.

Generates binary TARGET labels for a given time-window horizon:
  TARGET = 1  if max(WIND in [t+start_hours, t+end_hours]) - WIND[t] >= threshold
  TARGET = 0  if future observations exist but the condition is not met
  TARGET = NaN if there are no future observations in the window
"""

import numpy as np
import pandas as pd


def create_labels(df: pd.DataFrame, start_hours: int, end_hours: int, threshold: float = 15.0) -> pd.DataFrame:
    """
    Add a TARGET column to *df* indicating whether cyclone intensification occurs
    within the forecast window [t + start_hours, t + end_hours].

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: SID (str), ISO_TIME (datetime64[ns, UTC] or tz-naive),
        WIND (float, knots).
    start_hours : int
        Lower bound of the future window in hours (inclusive).
    end_hours : int
        Upper bound of the future window in hours (inclusive).
    threshold : float
        Wind-speed increase (knots) required to label as intensification (default 15).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional float column TARGET (1.0, 0.0, or NaN).
    """
    df = df.copy()
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # Ensure ISO_TIME is datetime; strip timezone for arithmetic if needed
    if not pd.api.types.is_datetime64_any_dtype(df["ISO_TIME"]):
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)

    targets = np.full(len(df), np.nan)

    for sid, group in df.groupby("SID", sort=False):
        idx = group.index.tolist()
        times = group["ISO_TIME"].values  # numpy datetime64
        winds = group["WIND"].values

        for pos, i in enumerate(idx):
            t = times[pos]
            w_curr = winds[pos]

            # Compute hour offsets for all other rows in this storm
            dt_hours = (times - t) / np.timedelta64(1, "h")

            mask = (dt_hours >= start_hours) & (dt_hours <= end_hours)
            future_winds = winds[mask]

            if len(future_winds) == 0:
                targets[i] = np.nan
            else:
                max_future = float(np.nanmax(future_winds))
                targets[i] = 1.0 if (max_future - w_curr) >= threshold else 0.0

    df["TARGET"] = targets
    return df
