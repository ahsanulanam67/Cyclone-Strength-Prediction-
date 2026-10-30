"""
Storm Surge Risk Estimator
==========================
Heuristic storm surge risk index for Bay of Bengal coastal stations.
Based on published relationships between cyclone parameters and surge height.

This is a physics-based heuristic, NOT a trained ML model.
A proper ML surge model requires ADCIRC/FVCOM synthetic simulations
(identified as future work in the research paper).

Reference: Irish et al. (2008), Needham & Keim (2012)
"""

import math
import numpy as np

# Key coastal stations in Bay of Bengal (Bangladesh + India + Myanmar)
COASTAL_STATIONS = [
    {"name": "Cox's Bazar",   "lat": 21.43, "lon": 91.97, "exposure": 1.0},
    {"name": "Chittagong",    "lat": 22.33, "lon": 91.83, "exposure": 0.9},
    {"name": "Khepupara",     "lat": 21.98, "lon": 90.58, "exposure": 1.0},
    {"name": "Paradip",       "lat": 20.32, "lon": 86.61, "exposure": 0.8},
    {"name": "Visakhapatnam", "lat": 17.70, "lon": 83.30, "exposure": 0.7},
    {"name": "Chennai",       "lat": 13.08, "lon": 80.28, "exposure": 0.6},
    {"name": "Yangon",        "lat": 16.80, "lon": 96.15, "exposure": 0.7},
    {"name": "Kolkata",       "lat": 22.57, "lon": 88.36, "exposure": 0.5},
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, min(1, a))))


def estimate_surge_risk(
    storm_lat: float,
    storm_lon: float,
    wind_kt: float,
    track_forecast: list[dict] | None = None,
    msl_hpa: float | None = None,
) -> list[dict]:
    """
    Estimate storm surge risk at each coastal station.

    Uses:
    - Inverse barometer effect: ~1 cm surge per 1 hPa pressure drop below 1013 hPa
    - Wind stress: surge ~ (Vmax/Vref)^2 scaled by 1/distance^0.5
    - Approach angle: higher risk when track heads directly toward coast
    - Coastal exposure factor per station

    Returns list of dicts sorted by risk score (highest first).
    """
    # Inverse barometer contribution
    if msl_hpa is not None:
        ib_surge_m = max(0.0, (1013.0 - msl_hpa) * 0.01)
    else:
        # Estimate from wind using empirical relationship
        ib_surge_m = max(0.0, (wind_kt / 150.0) * 2.0)

    # Wind stress contribution (Vmax^2 relationship)
    wind_ms = wind_kt * 0.5144
    wind_surge_factor = (wind_ms / 50.0) ** 2  # normalized to Cat-4

    results = []
    for stn in COASTAL_STATIONS:
        dist_km = _haversine_km(storm_lat, storm_lon, stn["lat"], stn["lon"])

        # Only consider stations within 500 km
        if dist_km > 500:
            results.append({
                "station": stn["name"], "lat": stn["lat"], "lon": stn["lon"],
                "surge_m": 0.0, "risk": "NONE", "distance_km": round(dist_km),
                "hours_to_landfall": None,
            })
            continue

        # Distance decay
        dist_factor = max(0.1, 1.0 - dist_km / 500.0) ** 0.5

        # Approach angle from track forecast
        approach_factor = 1.0
        hours_to_landfall = None
        if track_forecast:
            # Find closest forecast point to this station
            min_d = dist_km
            for fp in track_forecast:
                fd = _haversine_km(fp["lat"], fp["lon"], stn["lat"], stn["lon"])
                if fd < min_d:
                    min_d = fd
                    hours_to_landfall = fp["hours_ahead"]
            # Closer approach = higher factor
            if min_d < dist_km:
                approach_factor = 1.0 + (dist_km - min_d) / dist_km

        # Final surge estimate
        surge_m = (ib_surge_m + wind_surge_factor * 3.0) * dist_factor * approach_factor * stn["exposure"]
        surge_m = round(min(surge_m, 10.0), 2)  # cap at 10m

        # Risk category (based on Bangladesh BWDB thresholds)
        if surge_m < 0.5:
            risk = "LOW"
        elif surge_m < 2.0:
            risk = "MODERATE"
        elif surge_m < 4.0:
            risk = "HIGH"
        else:
            risk = "EXTREME"

        results.append({
            "station":         stn["name"],
            "lat":             stn["lat"],
            "lon":             stn["lon"],
            "surge_m":         surge_m,
            "risk":            risk,
            "distance_km":     round(dist_km),
            "hours_to_landfall": hours_to_landfall,
        })

    return sorted(results, key=lambda x: x["surge_m"], reverse=True)
