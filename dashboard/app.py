"""
Cyclone Early Warning Dashboard
Streamlit app — run with: streamlit run dashboard/app.py
"""

import io
import os
import sys
import json
import math
import ssl
import time
import warnings
import xml.etree.ElementTree as ET
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.track_predictor import predict_track as ml_predict_track
    _TRACK_MODEL = True
except Exception:
    _TRACK_MODEL = False

try:
    from src.surge_risk import estimate_surge_risk
    _SURGE_MODEL = True
except Exception:
    _SURGE_MODEL = False

warnings.filterwarnings("ignore")

# Fix macOS SSL certificate issue
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl._create_unverified_context()

# ── optional sklearn / xgboost imports ──────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, average_precision_score
    from xgboost import XGBClassifier
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyclone Early Warning System",
    page_icon="🌀",
    layout="wide",
)

# ── styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .risk-low      { background: #d4edda; color: #155724; }
    .risk-moderate { background: #fff3cd; color: #856404; }
    .risk-high     { background: #f8d7da; color: #721c24; }
    .metric-label  { font-size: 0.85rem; color: #6c757d; margin-bottom: 0.2rem; }
    .live-badge    { background: #28a745; color: white; padding: 2px 8px;
                     border-radius: 8px; font-size: 0.75rem; font-weight: 600; }
    .weather-card  { background: #f8f9fa; border-radius: 10px; padding: 0.8rem 1rem;
                     margin-bottom: 0.5rem; border-left: 4px solid #0d6efd; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def add_track_features(df):
    df = df.copy().sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    df["MONTH"] = df["ISO_TIME"].dt.month.astype(float)
    df["HOUR"]  = df["ISO_TIME"].dt.hour.astype(float)
    grp = df.groupby("SID", sort=False)
    df["LAT_CHANGE"]  = grp["LAT"].diff()
    df["LON_CHANGE"]  = grp["LON"].diff()
    df["WIND_CHANGE"] = grp["WIND"].diff()
    speeds = np.full(len(df), np.nan)
    for sid, group in df.groupby("SID", sort=False):
        idxs  = group.index.tolist()
        lats  = group["LAT"].values
        lons  = group["LON"].values
        times = group["ISO_TIME"].values
        for k in range(1, len(idxs)):
            try:
                dist  = haversine_km(lats[k-1], lons[k-1], lats[k], lons[k])
                dt_hr = (times[k] - times[k-1]) / np.timedelta64(1, "h")
                if dt_hr > 0:
                    speeds[idxs[k]] = dist / dt_hr
            except Exception:
                pass
    df["TRACK_SPEED"] = speeds
    return df


def create_labels(df, start_hours, end_hours, threshold=15):
    df = df.copy().sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    targets = np.full(len(df), np.nan)
    for sid, group in df.groupby("SID", sort=False):
        idxs  = group.index.tolist()
        times = group["ISO_TIME"].values
        winds = group["WIND"].values
        for pos, i in enumerate(idxs):
            t = times[pos]
            dt_hours = (times - t) / np.timedelta64(1, "h")
            mask = (dt_hours >= start_hours) & (dt_hours <= end_hours)
            future = winds[mask]
            if len(future):
                targets[i] = 1.0 if float(np.nanmax(future)) - winds[pos] >= threshold else 0.0
    df["TARGET"] = targets
    return df


def risk_color(prob):
    if prob < 0.35:
        return "risk-low", "LOW"
    if prob < 0.65:
        return "risk-moderate", "MODERATE"
    return "risk-high", "HIGH"


# ── Live weather fetch (Open-Meteo — no API key needed) ─────────────────────

def fetch_live_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch current atmospheric conditions from Open-Meteo and Open-Meteo Marine.
    Returns a dict with the same variable names used in ERA5 / V2.
    """
    results = {}

    # ── Atmospheric variables ────────────────────────────────────────────────
    atm_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,dewpoint_2m,surface_pressure,"
        f"wind_speed_10m,wind_direction_10m,precipitation"
        f"&wind_speed_unit=ms&timezone=UTC"
    )
    try:
        req = urllib.request.Request(atm_url, headers={"User-Agent": "CycloneEWS/1.0"})
        with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=10) as r:
            data = json.loads(r.read())
        cur = data.get("current", {})
        ws  = cur.get("wind_speed_10m", 0.0) or 0.0
        wd  = math.radians(cur.get("wind_direction_10m", 0.0) or 0.0)
        results["t2m"]  = cur.get("temperature_2m")
        results["d2m"]  = cur.get("dewpoint_2m")
        results["msl"]  = cur.get("surface_pressure")      # hPa
        results["u10"]  = -ws * math.sin(wd)               # westward component
        results["v10"]  = -ws * math.cos(wd)               # southward component
        results["tp"]   = cur.get("precipitation", 0.0)
        results["wind_speed_ms"] = ws
        results["wind_dir_deg"]  = cur.get("wind_direction_10m")
    except Exception as e:
        results["_atm_error"] = str(e)

    # ── Sea surface temperature ───────────────────────────────────────────────
    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&current=sea_surface_temperature&timezone=UTC"
    )
    try:
        req = urllib.request.Request(marine_url, headers={"User-Agent": "CycloneEWS/1.0"})
        with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=10) as r:
            data = json.loads(r.read())
        results["sst"] = data.get("current", {}).get("sea_surface_temperature")
    except Exception as e:
        results["_sst_error"] = str(e)

    return results


# ── Active cyclone tracker (GDACS RSS) ───────────────────────────────────────

def fetch_active_cyclones() -> list[dict]:
    """
    Fetch active tropical cyclone alerts from GDACS RSS feed.
    Filters to Bay of Bengal region (LAT 0–25°N, LON 80–100°E).
    """
    GDACS_URL = "https://www.gdacs.org/xml/rss.xml"
    cyclones = []
    try:
        req = urllib.request.Request(GDACS_URL, headers={"User-Agent": "CycloneEWS/1.0"})
        with urllib.request.urlopen(req, context=_SSL_CONTEXT, timeout=10) as r:
            xml_data = r.read()
        root = ET.fromstring(xml_data)
        ns = {
            "gdacs": "http://www.gdacs.org",
            "geo":   "http://www.w3.org/2003/01/geo/wgs84_pos#",
            "dc":    "http://purl.org/dc/elements/1.1/",
        }
        for item in root.iter("item"):
            etype = item.find("gdacs:eventtype", ns)
            if etype is None or etype.text != "TC":
                continue
            lat_el  = item.find("geo:lat", ns)
            lon_el  = item.find("geo:long", ns)
            title   = item.find("title")
            alert   = item.find("gdacs:alertlevel", ns)
            wind_el = item.find("gdacs:maxwind", ns)
            try:
                lat = float(lat_el.text)
                lon = float(lon_el.text)
            except Exception:
                continue
            # Filter to Bay of Bengal
            if not (0 <= lat <= 25 and 80 <= lon <= 100):
                continue
            cyclones.append({
                "name":       title.text if title is not None else "Unknown",
                "lat":        lat,
                "lon":        lon,
                "alert":      alert.text if alert is not None else "—",
                "wind_kt":    float(wind_el.text) * 1.944 if wind_el is not None else None,
            })
    except Exception:
        pass
    return cyclones


# ── IBTrACS loader ────────────────────────────────────────────────────────────

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

FEATURES_V1 = ["LAT", "LON", "WIND", "MONTH", "HOUR",
                "LAT_CHANGE", "LON_CHANGE", "WIND_CHANGE", "TRACK_SPEED"]
FEATURES_V2 = FEATURES_V1 + ["sst", "msl", "u10", "v10", "t2m", "d2m", "tp"]

# Keep old alias for V1 training path
FEATURES = FEATURES_V1


# ── Load pre-trained V2 models ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_v2_models():
    try:
        import joblib
        short = joblib.load("models/v2_short.pkl")
        long_ = joblib.load("models/v2_long.pkl")
        return {"short": short, "long": long_}
    except Exception:
        return None


@st.cache_data(show_spinner="Loading IBTrACS data …")
def load_ibtracs(url):
    with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw), skiprows=[1], low_memory=False)
    for col in ["LAT", "LON", "USA_WIND", "WMO_WIND", "USA_PRES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    if "USA_WIND" in df.columns and "WMO_WIND" in df.columns:
        df["WIND"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    elif "USA_WIND" in df.columns:
        df["WIND"] = df["USA_WIND"]
    else:
        df["WIND"] = df.get("WMO_WIND", np.nan)
    df["WIND"] = pd.to_numeric(df["WIND"], errors="coerce")
    df = df.dropna(subset=["SID", "ISO_TIME", "WIND"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    df = df[df["LAT"].between(0, 25) & df["LON"].between(80, 100)].reset_index(drop=True)
    df = add_track_features(df)
    return df


@st.cache_resource(show_spinner="Training models …")
def train_models(df):
    df_s = create_labels(df, 48, 72)
    df_l = create_labels(df, 120, 168)

    results = {}
    for df_lbl, key in [(df_s, "short"), (df_l, "long")]:
        clean = df_lbl.dropna(subset=["TARGET"])
        feat_cols = [c for c in FEATURES if c in clean.columns]
        X = clean[feat_cols].replace([np.inf, -np.inf], np.nan).values
        y = clean["TARGET"].astype(int).values
        groups = clean["SID"].values

        if len(np.unique(groups)) < 4:
            results[key] = None
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(gss.split(X, y, groups))

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, eval_metric="logloss",
                                   random_state=42, verbosity=0)),
        ])
        pipe.fit(X[tr], y[tr])
        y_prob = pipe.predict_proba(X[te])[:, 1]

        roc = None
        pr  = None
        if len(np.unique(y[te])) == 2:
            roc = float(roc_auc_score(y[te], y_prob))
            pr  = float(average_precision_score(y[te], y_prob))

        results[key] = {
            "pipeline":    pipe,
            "feat_cols":   feat_cols,
            "roc_auc":     roc,
            "pr_auc":      pr,
            "importances": pipe.named_steps["clf"].feature_importances_,
        }

    return results


# ── map ───────────────────────────────────────────────────────────────────────

def make_folium_map(storm_lat, storm_lon, storm_wind,
                    lat_change, lon_change,
                    track_df=None, active_cyclones=None,
                    track_forecast=None, surge_results=None):
    """
    Interactive Folium map with:
    - Real satellite/terrain basemap
    - Historical storm tracks (last 10 storms)
    - Current storm position with direction arrow and popup
    - Active cyclone markers from GDACS
    """
    m = folium.Map(
        location=[13, 89],
        zoom_start=5,
        tiles=None,
    )

    # ── Basemap layers ────────────────────────────────────────────────────────
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Ocean",
        name="Ocean (default)",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

    # ── Historical tracks (last 10 unique storms, faint) ──────────────────────
    if track_df is not None and len(track_df):
        recent_sids = (
            track_df.groupby("SID")["ISO_TIME"].max()
            .sort_values(ascending=False).head(10).index
        )
        for sid in recent_sids:
            grp = track_df[track_df["SID"] == sid].sort_values("ISO_TIME")
            coords = list(zip(grp["LAT"], grp["LON"]))
            if len(coords) < 2:
                continue
            name = grp["NAME"].iloc[0] if "NAME" in grp.columns else sid
            folium.PolyLine(
                coords,
                color="#aaaaaa",
                weight=1.5,
                opacity=0.5,
                tooltip=f"{name}",
            ).add_to(m)
            # Mark the peak intensity point
            peak = grp.loc[grp["WIND"].idxmax()]
            folium.CircleMarker(
                [peak["LAT"], peak["LON"]],
                radius=3,
                color="#888888",
                fill=True,
                fill_opacity=0.6,
                tooltip=f"{name} — peak {peak['WIND']:.0f} kt",
            ).add_to(m)

    # ── Active cyclones from GDACS (orange) ───────────────────────────────────
    if active_cyclones:
        for cyc in active_cyclones:
            wind_str = f"{cyc['wind_kt']:.0f} kt" if cyc["wind_kt"] else "unknown"
            folium.Marker(
                [cyc["lat"], cyc["lon"]],
                popup=folium.Popup(
                    f"<b>{cyc['name']}</b><br>"
                    f"Wind: {wind_str}<br>"
                    f"Alert: {cyc['alert']}",
                    max_width=200,
                ),
                tooltip=f"🌪️ {cyc['name']} ({wind_str})",
                icon=folium.Icon(color="orange", icon="warning-sign", prefix="glyphicon"),
            ).add_to(m)

    # ── Direction arrow (show where storm is moving) ──────────────────────────
    # lat_change and lon_change are per 6h — extrapolate 12h ahead
    arrow_lat = storm_lat + lat_change * 2
    arrow_lon = storm_lon + lon_change * 2
    folium.PolyLine(
        [[storm_lat, storm_lon], [arrow_lat, arrow_lon]],
        color="#ff6600",
        weight=3,
        opacity=0.9,
        tooltip="Projected movement (12 h)",
        dash_array=None,
    ).add_to(m)
    # Arrowhead at tip
    folium.CircleMarker(
        [arrow_lat, arrow_lon],
        radius=5,
        color="#ff6600",
        fill=True,
        fill_color="#ff6600",
        fill_opacity=1.0,
        tooltip="Projected position in 12 h",
    ).add_to(m)

    # ── Storm category color ──────────────────────────────────────────────────
    if storm_wind < 34:   cat, color = "Depression", "blue"
    elif storm_wind < 48: cat, color = "Deep Depression", "green"
    elif storm_wind < 64: cat, color = "Cyclonic Storm", "orange"
    elif storm_wind < 90: cat, color = "Severe Cyclonic", "red"
    else:                 cat, color = "Very Severe / Extreme", "darkred"

    # Direction in degrees
    import math as _m
    dx, dy = lon_change, lat_change
    bearing = (_m.degrees(_m.atan2(dx, dy)) + 360) % 360
    compass = ["N","NE","E","SE","S","SW","W","NW"][int((bearing+22.5)//45) % 8]

    popup_html = f"""
    <div style="font-family:sans-serif; min-width:180px">
      <b style="font-size:14px">🌀 Current Storm</b><br><br>
      <b>Position:</b> {storm_lat:.1f}°N, {storm_lon:.1f}°E<br>
      <b>Wind:</b> {storm_wind} knots<br>
      <b>Category:</b> {cat}<br>
      <b>Moving:</b> {compass} ({bearing:.0f}°)<br>
      <b>Speed:</b> {abs(lat_change):.2f}°/6h lat, {abs(lon_change):.2f}°/6h lon<br>
      <hr style="margin:6px 0">
      <span style="color:gray;font-size:11px">Orange arrow = projected 12h movement</span>
    </div>
    """
    folium.Marker(
        [storm_lat, storm_lon],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"🌀 Storm — {storm_wind} kt — Moving {compass}",
        icon=folium.Icon(color=color, icon="cloud", prefix="glyphicon"),
    ).add_to(m)

    # ── BoB bounding box ──────────────────────────────────────────────────────
    folium.Rectangle(
        bounds=[[0, 80], [25, 100]],
        color="#0066cc",
        fill=False,
        weight=1.5,
        dash_array="6",
        tooltip="Bay of Bengal study region",
    ).add_to(m)

    # ── ML Track forecast (dashed blue line) ─────────────────────────────────
    if track_forecast:
        forecast_coords = [[storm_lat, storm_lon]] + \
                          [[fp["lat"], fp["lon"]] for fp in track_forecast]
        folium.PolyLine(
            forecast_coords,
            color="#0055cc",
            weight=2.5,
            opacity=0.85,
            dash_array="8 4",
            tooltip="ML Track Forecast (48h)",
        ).add_to(m)
        for fp in track_forecast:
            if fp["hours_ahead"] % 12 == 0:
                folium.CircleMarker(
                    [fp["lat"], fp["lon"]],
                    radius=5,
                    color="#0055cc",
                    fill=True,
                    fill_color="#4499ff",
                    fill_opacity=0.8,
                    tooltip=f"+{fp['hours_ahead']}h → {fp['lat']:.1f}°N, {fp['lon']:.1f}°E",
                ).add_to(m)

    # ── Surge risk at coastal stations ────────────────────────────────────────
    if surge_results:
        surge_colors = {"LOW": "green", "MODERATE": "orange",
                        "HIGH": "red", "EXTREME": "darkred", "NONE": "gray"}
        for stn in surge_results:
            if stn["risk"] == "NONE":
                continue
            lf_str = f"~{stn['hours_to_landfall']}h" if stn["hours_to_landfall"] else "—"
            folium.Marker(
                [stn["lat"], stn["lon"]],
                popup=folium.Popup(
                    f"<b>{stn['station']}</b><br>"
                    f"Surge risk: <b>{stn['risk']}</b><br>"
                    f"Est. surge: {stn['surge_m']} m<br>"
                    f"Distance: {stn['distance_km']} km<br>"
                    f"ETA: {lf_str}",
                    max_width=200,
                ),
                tooltip=f"🌊 {stn['station']} — {stn['surge_m']}m ({stn['risk']})",
                icon=folium.Icon(
                    color=surge_colors.get(stn["risk"], "blue"),
                    icon="home", prefix="glyphicon"
                ),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def plot_feature_importance(importances, feature_names, title):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(feature_names)[order], np.array(importances)[order],
            color="steelblue")
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

st.title("🌀 Cyclone Early Warning System")
st.caption("Bay of Bengal | Predicts intensification 2–3 days and 5–7 days ahead")
st.markdown("---")

# ── session state for live weather ───────────────────────────────────────────
if "live_weather" not in st.session_state:
    st.session_state.live_weather = None
if "live_weather_lat" not in st.session_state:
    st.session_state.live_weather_lat = None

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    use_live_data = st.toggle("Load live IBTrACS data", value=False,
                              help="Fetches the NOAA IBTrACS CSV (~20 MB).")
    st.markdown("---")

    # ── Active cyclone tracker ────────────────────────────────────────────────
    st.header("🌪️ Active Cyclones (GDACS)")
    if st.button("🔄 Check Active Cyclones", use_container_width=True):
        with st.spinner("Checking GDACS…"):
            active = fetch_active_cyclones()
            st.session_state["active_cyclones"] = active

    active_cyclones = st.session_state.get("active_cyclones", None)
    if active_cyclones is None:
        st.caption("Click above to check for active Bay of Bengal cyclones.")
    elif len(active_cyclones) == 0:
        st.success("✅ No active cyclones in Bay of Bengal right now.")
    else:
        for cyc in active_cyclones:
            alert_color = {"Green": "🟢", "Orange": "🟠", "Red": "🔴"}.get(cyc["alert"], "⚪")
            wind_str = f"{cyc['wind_kt']:.0f} kt" if cyc["wind_kt"] else "—"
            st.warning(
                f"{alert_color} **{cyc['name']}**\n\n"
                f"Lat {cyc['lat']:.1f}°N · Lon {cyc['lon']:.1f}°E\n\n"
                f"Wind: {wind_str} · Alert: {cyc['alert']}"
            )
            if st.button(f"Use {cyc['name']} position", key=f"use_{cyc['name']}"):
                st.session_state["prefill_lat"] = cyc["lat"]
                st.session_state["prefill_lon"] = cyc["lon"]
                if cyc["wind_kt"]:
                    st.session_state["prefill_wind"] = int(cyc["wind_kt"])

    st.markdown("---")

    # ── Manual observation inputs ─────────────────────────────────────────────
    st.header("📍 Current Observation")

    default_lat  = st.session_state.get("prefill_lat", 15.0)
    default_lon  = st.session_state.get("prefill_lon", 88.5)
    default_wind = st.session_state.get("prefill_wind", 50)

    storm_lat   = st.slider("Latitude",   0.0,  25.0, float(default_lat), 0.5)
    storm_lon   = st.slider("Longitude", 80.0, 100.0, float(default_lon), 0.5)
    storm_wind  = st.slider("Wind speed (knots)", 20, 150, int(default_wind))
    storm_month = st.selectbox("Month", list(range(1, 13)), index=9)
    storm_hour  = st.selectbox("Hour (UTC)", [0, 6, 12, 18], index=0)
    lat_change  = st.number_input("Lat change (°/6 h)",  -2.0,  2.0,  0.3, 0.1)
    lon_change  = st.number_input("Lon change (°/6 h)",  -2.0,  2.0, -0.2, 0.1)
    wind_change = st.number_input("Wind change (kt/6 h)", -20.0, 20.0, 5.0, 1.0)
    track_speed = st.number_input("Track speed (km/h)",    0.0,  80.0, 12.0, 1.0)

    st.markdown("---")

    # ── Live weather fetch ────────────────────────────────────────────────────
    st.header("🌐 Live Weather (Open-Meteo)")
    st.caption(f"Will fetch for Lat {storm_lat:.1f}°N, Lon {storm_lon:.1f}°E")

    if st.button("⚡ Fetch Live Weather Now", use_container_width=True):
        with st.spinner("Fetching from Open-Meteo…"):
            wx = fetch_live_weather(storm_lat, storm_lon)
            st.session_state.live_weather = wx
            st.session_state.live_weather_lat = storm_lat
            st.session_state.live_weather_lon = storm_lon

    wx = st.session_state.live_weather
    if wx:
        errors = [v for k, v in wx.items() if k.startswith("_")]
        if errors:
            st.warning(f"Partial fetch: {errors[0]}")
        else:
            st.success("✅ Live weather loaded!")

# ── data + model ─────────────────────────────────────────────────────────────
df_tracks    = None
model_results = None
error_msg    = None

if not _ML_AVAILABLE:
    st.error("scikit-learn and/or xgboost not installed. Run: pip install scikit-learn xgboost")
elif use_live_data:
    with st.spinner("Downloading IBTrACS & training models …"):
        try:
            df_tracks     = load_ibtracs(IBTRACS_URL)
            model_results = train_models(df_tracks)
        except Exception as exc:
            error_msg = str(exc)
else:
    st.info("ℹ️ Demo mode — toggle 'Load live IBTrACS data' in the sidebar to train on real data.")

if error_msg:
    st.error(f"Data error: {error_msg}")

# ── observation dict ──────────────────────────────────────────────────────────
wx = st.session_state.live_weather
wx_clean = wx if (wx and not any(k.startswith("_") for k in wx)) else {}

observation = {
    "LAT":         storm_lat,
    "LON":         storm_lon,
    "WIND":        float(storm_wind),
    "MONTH":       float(storm_month),
    "HOUR":        float(storm_hour),
    "LAT_CHANGE":  lat_change,
    "LON_CHANGE":  lon_change,
    "WIND_CHANGE": wind_change,
    "TRACK_SPEED": track_speed,
    # ERA5-equivalent from live weather (None if not fetched)
    "sst": wx_clean.get("sst"),
    "msl": wx_clean.get("msl"),
    "u10": wx_clean.get("u10"),
    "v10": wx_clean.get("v10"),
    "t2m": wx_clean.get("t2m"),
    "d2m": wx_clean.get("d2m"),
    "tp":  wx_clean.get("tp"),
}

# ── choose model: V2 (with live weather) or V1 (track only) ──────────────────
v2_models    = load_v2_models()
using_v2     = bool(wx_clean and v2_models)
prob_short, prob_long = None, None
active_model_label = "V1 (track features)"

if using_v2:
    # Use pre-trained V2 model with live weather variables
    active_model_label = "V2 (track + live weather)"
    for key in ["short", "long"]:
        res = v2_models[key]
        feats = res["features"]
        row   = {f: observation.get(f, np.nan) for f in feats}
        X_new = pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan).values
        p = res["pipeline"].predict_proba(X_new)[0, 1]
        if key == "short": prob_short = p
        else:              prob_long  = p

elif model_results:
    # Fall back to V1 trained on live IBTrACS
    active_model_label = "V1 (track features)"
    for key in ["short", "long"]:
        res = model_results.get(key)
        if res and res.get("pipeline"):
            feat_cols = res["feat_cols"]
            X_new = pd.DataFrame([observation])[feat_cols].values
            p = res["pipeline"].predict_proba(X_new)[0, 1]
            if key == "short": prob_short = p
            else:              prob_long  = p

# ── LIVE WEATHER PANEL (full width, shown when fetched) ──────────────────────
wx = st.session_state.live_weather
if wx and not any(k.startswith("_") for k in wx):
    wx_clean = wx
    st.subheader("🌡️ Live Atmospheric Conditions")
    st.markdown(
        f'<span class="live-badge">LIVE</span> &nbsp; '
        f'Fetched for Lat {st.session_state.get("live_weather_lat","?"):.1f}°N, '
        f'Lon {st.session_state.get("live_weather_lon","?"):.1f}°E — '
        f'Open-Meteo (real-time) &nbsp;|&nbsp; '
        f'<strong>Prediction model: {active_model_label}</strong>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    wc1, wc2, wc3, wc4, wc5, wc6 = st.columns(6)

    def _fmt(val, unit, decimals=1):
        return f"{val:.{decimals}f} {unit}" if val is not None else "N/A"

    wc1.metric("🌊 SST",          _fmt(wx.get("sst"),  "°C"))
    wc2.metric("🔽 Pressure",     _fmt(wx.get("msl"),  "hPa", 0))
    wc3.metric("🌡️ Temp (2m)",    _fmt(wx.get("t2m"),  "°C"))
    wc4.metric("💧 Dewpoint",     _fmt(wx.get("d2m"),  "°C"))
    wc5.metric("💨 Wind speed",   _fmt(wx.get("wind_speed_ms"), "m/s"))
    wc6.metric("🌧️ Precipitation", _fmt(wx.get("tp"),  "mm"))

    # Wind direction arrow + shear interpretation
    ws  = wx.get("wind_speed_ms", 0) or 0
    wd  = wx.get("wind_dir_deg")
    sst = wx.get("sst")
    msl = wx.get("msl")

    insights = []
    if sst is not None:
        if sst >= 28:
            insights.append(f"🔥 SST {sst:.1f}°C — **warm enough to fuel intensification** (threshold: 28°C)")
        else:
            insights.append(f"❄️ SST {sst:.1f}°C — below 28°C, limits intensification potential")
    if msl is not None:
        if msl < 1005:
            insights.append(f"🌀 Low pressure {msl:.0f} hPa — **favourable for development**")
        else:
            insights.append(f"🔆 Pressure {msl:.0f} hPa — neutral to unfavourable")
    if ws > 0 and wd is not None:
        insights.append(f"💨 Wind: {ws:.1f} m/s from {wd:.0f}°")

    if insights:
        st.markdown("**Interpretation:**")
        for ins in insights:
            st.markdown(f"- {ins}")

    if using_v2:
        st.success(
            "✅ **V2 model active** — live SST, pressure, wind, and temperature are being "
            "used directly in the prediction above."
        )
    else:
        st.caption(
            "⚠️ V1 model active (track features only). "
            "Click 'Fetch Live Weather Now' in the sidebar to upgrade to V2 prediction."
        )
    st.markdown("---")

# ── main layout ───────────────────────────────────────────────────────────────
col_map, col_panel = st.columns([1.2, 1], gap="large")

# ── Track forecast ────────────────────────────────────────────────────────────
track_forecast = None
if _TRACK_MODEL:
    try:
        history = [
            {"lat": storm_lat - lat_change*3, "lon": storm_lon - lon_change*3,
             "wind": max(20, storm_wind - abs(wind_change)*3),
             "month": storm_month, "hour": storm_hour},
            {"lat": storm_lat - lat_change*2, "lon": storm_lon - lon_change*2,
             "wind": max(20, storm_wind - abs(wind_change)*2),
             "month": storm_month, "hour": storm_hour},
            {"lat": storm_lat - lat_change,   "lon": storm_lon - lon_change,
             "wind": max(20, storm_wind - abs(wind_change)),
             "month": storm_month, "hour": storm_hour},
            {"lat": storm_lat,                "lon": storm_lon,
             "wind": float(storm_wind),
             "month": storm_month, "hour": storm_hour},
        ]
        track_forecast = ml_predict_track(history, n_steps=8)
    except Exception as e:
        track_forecast = None

# ── Surge risk ────────────────────────────────────────────────────────────────
surge_results = None
if _SURGE_MODEL:
    try:
        surge_results = estimate_surge_risk(
            storm_lat, storm_lon, float(storm_wind),
            track_forecast=track_forecast,
            msl_hpa=wx_clean.get("msl") if wx_clean else None,
        )
    except Exception:
        surge_results = None

with col_map:
    st.subheader("🗺️ Bay of Bengal — Live Map")
    fmap = make_folium_map(
        storm_lat, storm_lon, storm_wind,
        lat_change, lon_change,
        track_df=df_tracks,
        active_cyclones=st.session_state.get("active_cyclones") or [],
        track_forecast=track_forecast,
        surge_results=surge_results,
    )
    st_folium(fmap, width=700, height=480, returned_objects=[])

with col_panel:
    st.subheader("🔮 Risk Prediction")
    if using_v2:
        st.success(f"🌐 **{active_model_label}**")
    elif model_results:
        st.info(f"📊 **{active_model_label}**")

    if prob_short is not None:
        css_s, lvl_s = risk_color(prob_short)
        css_l, lvl_l = risk_color(prob_long)
        st.markdown(
            f'<div class="risk-box {css_s}">48–72 h Risk: {lvl_s}<br>'
            f'<span style="font-size:1.8rem">{prob_short:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="risk-box {css_l}">120–168 h Risk: {lvl_l}<br>'
            f'<span style="font-size:1.8rem">{prob_long:.1%}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="risk-box risk-moderate">
        Enable live data in the sidebar<br>to see real predictions
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("48–72 h", "—")
        c2.metric("120–168 h", "—")

    st.markdown("---")
    st.subheader("📋 Observation Summary")
    obs_df = pd.DataFrame([{"Parameter": k, "Value": round(v, 2) if v is not None else "—"}
                           for k, v in observation.items()])
    st.dataframe(obs_df, hide_index=True, use_container_width=True)

    # Show live weather summary in panel if loaded
    if wx_clean:
        st.markdown("---")
        st.subheader("🌐 Live Weather Summary")
        wx_df = pd.DataFrame([
            {"Variable": "SST (°C)",        "Value": round(wx["sst"], 1)  if wx.get("sst")  is not None else "N/A"},
            {"Variable": "Pressure (hPa)",  "Value": round(wx["msl"], 1)  if wx.get("msl")  is not None else "N/A"},
            {"Variable": "Temp 2m (°C)",    "Value": round(wx["t2m"], 1)  if wx.get("t2m")  is not None else "N/A"},
            {"Variable": "Dewpoint (°C)",   "Value": round(wx["d2m"], 1)  if wx.get("d2m")  is not None else "N/A"},
            {"Variable": "Wind (m/s)",      "Value": round(wx["wind_speed_ms"], 1) if wx.get("wind_speed_ms") is not None else "N/A"},
            {"Variable": "Precip (mm)",     "Value": round(wx["tp"], 2)   if wx.get("tp")   is not None else "N/A"},
        ])
        st.dataframe(wx_df, hide_index=True, use_container_width=True)

# ── model performance + feature importance ────────────────────────────────────
# ── Track + Surge sections ────────────────────────────────────────────────────
st.markdown("---")
tcol, scol = st.columns(2, gap="large")

with tcol:
    st.subheader("🧭 Track Forecast (48h)")
    if track_forecast:
        tdf = pd.DataFrame(track_forecast)
        tdf["hours_ahead"] = tdf["hours_ahead"].apply(lambda x: f"+{x}h")
        tdf.columns = ["Lat (°N)", "Lon (°E)", "Ahead"]
        tdf = tdf[["Ahead","Lat (°N)","Lon (°E)"]]
        st.dataframe(tdf, hide_index=True, use_container_width=True)
        st.caption("ML model (Random Forest, IBTrACS). 24h MAE ~18 km, 48h MAE ~134 km (leave-one-storm-out CV).")
    else:
        st.info("Track model not loaded.")

with scol:
    st.subheader("🌊 Storm Surge Risk")
    if surge_results:
        risk_colors = {"EXTREME":"#b71c1c","HIGH":"#e53935",
                       "MODERATE":"#fb8c00","LOW":"#43a047","NONE":"#9e9e9e"}
        sdf = pd.DataFrame([{
            "Station":      r["station"],
            "Risk":         r["risk"],
            "Surge (m)":    r["surge_m"],
            "Distance (km)":r["distance_km"],
        } for r in surge_results if r["risk"] != "NONE"])
        if len(sdf):
            st.dataframe(sdf, hide_index=True, use_container_width=True)
        else:
            st.success("✅ No stations within 500 km at significant risk.")
        st.caption("⚠️ Heuristic estimate (physics-based index, not a trained ML model). "
                   "For operational use, replace with ADCIRC/FVCOM simulations.")
    else:
        st.info("Surge module not loaded.")

if model_results:
    st.markdown("---")
    st.subheader("📊 Model Performance & Feature Importance")
    tab1, tab2, tab3 = st.tabs(["48–72 h Model", "120–168 h Model", "Historical Tracks"])

    for tab, key, label in [(tab1, "short", "48–72 h"), (tab2, "long", "120–168 h")]:
        with tab:
            res = model_results.get(key)
            if not res:
                st.warning(f"Model for {label} could not be trained (insufficient data).")
                continue
            m1, m2 = st.columns(2)
            m1.metric("ROC-AUC", f"{res['roc_auc']:.4f}" if res["roc_auc"] is not None else "N/A")
            m2.metric("PR-AUC",  f"{res['pr_auc']:.4f}"  if res["pr_auc"]  is not None else "N/A")
            st.pyplot(plot_feature_importance(res["importances"], res["feat_cols"],
                                              f"XGBoost Feature Importance — {label}"),
                      use_container_width=True)

    with tab3:
        if df_tracks is not None and len(df_tracks):
            st.markdown(f"**{df_tracks['SID'].nunique()} storms** loaded from IBTrACS "
                        f"({len(df_tracks):,} observations).")
            st.dataframe(
                df_tracks[["SID", "ISO_TIME", "LAT", "LON", "WIND"]]
                .tail(50).sort_values("ISO_TIME", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("Enable live data to view historical tracks.")

# ── footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: NOAA IBTrACS v04r01 · Open-Meteo (live weather) · GDACS (active alerts) · "
    "Model: XGBoost (V1 track features) | ROC-AUC 0.89 with CNN-LSTM (V3) · "
    "Threshold: +15 kt intensification within window"
)
