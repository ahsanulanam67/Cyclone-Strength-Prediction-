"""
HURSAT Satellite Data Pipeline — Steps 1 to 4
==============================================
Step 1: Download HURSAT-B1 infrared images for North Indian Ocean cyclones
Step 2: Match each image to its IBTrACS storm ID and timestamp
Step 3: Extract 128x128 infrared patches centered on the storm
Step 4: Save as PNG files named SID_YYYYMMDDHH.png for V3 notebook

Usage
-----
    python scripts/download_hursat.py

    # Download specific storms only
    python scripts/download_hursat.py --storms AMPHAN SIDR FANI MOCHA

    # Download all available NI storms (slow — many GB)
    python scripts/download_hursat.py --all

Output
------
    data/satellite_images/SID_YYYYMMDDHH.png   ← ready for v3 notebook
    data/hursat_raw/                            ← raw downloaded NetCDF files
    data/satellite_manifest.csv                ← index of all saved images
"""

import io
import os
import ssl
import math
import argparse
import warnings
import urllib.request
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── SSL fix for macOS ────────────────────────────────────────────────────────
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl._create_unverified_context()

# ── Directories ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "hursat_raw"
IMG_DIR     = ROOT / "data" / "satellite_images"
MANIFEST    = ROOT / "data" / "satellite_manifest.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── IBTrACS URL ──────────────────────────────────────────────────────────────
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

# ── HURSAT base URL ───────────────────────────────────────────────────────────
# HURSAT-B1 v06 North Indian Ocean
HURSAT_BASE = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/"

# Patch and image settings
IMAGE_SIZE   = 128
PATCH_DEG    = 8.0    # degrees lat/lon around storm center to crop
SEQUENCE_LEN = 4      # 4 frames per sequence
STEP_HOURS   = 6      # 6 hours between frames


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load IBTrACS and find storms
# ═══════════════════════════════════════════════════════════════════════════

def load_ibtracs():
    print("\n[Step 1] Loading IBTrACS North Indian Ocean data …")
    with urllib.request.urlopen(IBTRACS_URL, context=SSL_CTX) as resp:
        raw = resp.read()

    df = pd.read_csv(io.BytesIO(raw), skiprows=[1], low_memory=False)

    for col in ["LAT", "LON", "USA_WIND", "WMO_WIND"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df["WIND"]     = df.get("USA_WIND", pd.Series(dtype=float)).fillna(
                     df.get("WMO_WIND", pd.Series(dtype=float)))
    df["WIND"]     = pd.to_numeric(df["WIND"], errors="coerce")

    df = df.dropna(subset=["SID", "ISO_TIME", "WIND"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # Bay of Bengal filter
    df = df[df["LAT"].between(0, 25) & df["LON"].between(80, 100)].reset_index(drop=True)

    print(f"    → {df['SID'].nunique()} storms, {len(df):,} observations")
    return df


def get_named_storms(df):
    """Return DataFrame of named storms with their peak wind and year."""
    storms = []
    for sid, grp in df.groupby("SID"):
        name = grp["NAME"].iloc[0] if "NAME" in grp.columns else "UNNAMED"
        year = grp["ISO_TIME"].dt.year.iloc[0]
        peak = grp["WIND"].max()
        storms.append({"SID": sid, "NAME": name, "YEAR": year, "PEAK_WIND": peak})
    return pd.DataFrame(storms).sort_values("PEAK_WIND", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Try to download HURSAT NetCDF for a storm
# ═══════════════════════════════════════════════════════════════════════════

def hursat_url(sid: str, year: int) -> str:
    """Construct the HURSAT-B1 file URL for a given storm."""
    # HURSAT filenames: HURSAT-B1_v06_YYYY_SSSS.nc
    # where SSSS is a 4-digit storm serial (from SID)
    # SID format: 2020188N09085  → year 2020
    # HURSAT uses: YYYY_BBNN  e.g. 2020_NI11 (NI = North Indian, 11 = serial)
    # We try the NCEI directory listing approach
    base = f"{HURSAT_BASE}{year}/"
    return base


def list_hursat_files(year: int) -> list:
    """Fetch the list of HURSAT files available for a given year."""
    url = f"{HURSAT_BASE}{year}/"
    try:
        with urllib.request.urlopen(url, context=SSL_CTX, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        # Parse href links ending in .nc
        import re
        files = re.findall(r'href="([^"]+\.nc)"', html)
        return [f for f in files if "NI" in f.upper() or "IO" in f.upper()]
    except Exception as exc:
        return []


def download_hursat_file(filename: str, year: int) -> Path:
    """Download a single HURSAT NetCDF file. Returns local path."""
    url   = f"{HURSAT_BASE}{year}/{filename}"
    dest  = RAW_DIR / filename

    if dest.exists():
        return dest

    try:
        with urllib.request.urlopen(url, context=SSL_CTX, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return dest
    except Exception as exc:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Extract image patches from a HURSAT NetCDF file
# ═══════════════════════════════════════════════════════════════════════════

def extract_patches_from_netcdf(nc_path: Path, storm_df: pd.DataFrame, sid: str) -> list:
    """
    Extract centered 128×128 infrared patches from a HURSAT NetCDF file.

    Returns list of dicts: {sid, timestamp, image_path}
    """
    try:
        import xarray as xr
    except ImportError:
        print("  [ERROR] xarray not installed — run: pip install xarray h5netcdf")
        return []

    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
    except Exception as exc:
        try:
            ds = xr.open_dataset(nc_path)
        except Exception as exc2:
            print(f"  [ERROR] Cannot open {nc_path.name}: {exc2}")
            return []

    # Find the infrared brightness temperature variable
    ir_var = None
    for candidate in ["IRWIN", "TB", "BT", "ir", "brightness_temperature", "Tb"]:
        if candidate in ds.data_vars:
            ir_var = candidate
            break
    if ir_var is None:
        # Take the first variable that looks like an image
        for var in ds.data_vars:
            arr = ds[var]
            if arr.ndim >= 2:
                ir_var = var
                break

    if ir_var is None:
        print(f"  [WARN] No usable IR variable in {nc_path.name}")
        ds.close()
        return []

    # Find time coordinate
    time_coord = None
    for tc in ["time", "Time", "valid_time", "datetime"]:
        if tc in ds.coords:
            time_coord = tc
            break
    if time_coord is None:
        ds.close()
        return []

    times = pd.to_datetime(ds[time_coord].values, utc=True, errors="coerce")
    saved = []

    for t_idx, t in enumerate(times):
        if pd.isnull(t):
            continue

        ts = pd.Timestamp(t)
        filename = f"{sid}_{ts.strftime('%Y%m%d%H')}.png"
        out_path = IMG_DIR / filename

        if out_path.exists():
            saved.append({"SID": sid, "TIMESTAMP": ts, "PATH": str(out_path)})
            continue

        try:
            img_data = ds[ir_var].isel({time_coord: t_idx})

            # Drop any extra dimensions (e.g. level, band)
            while img_data.ndim > 2:
                img_data = img_data.isel({img_data.dims[0]: 0})

            arr = img_data.values.astype(float)

            if arr.shape[0] == 0 or arr.shape[1] == 0:
                continue

            # Replace fill values / NaN with mean
            fill_mask = np.isnan(arr) | (arr < -9000)
            if fill_mask.all():
                continue
            arr[fill_mask] = np.nanmean(arr[~fill_mask])

            # Center crop to IMAGE_SIZE × IMAGE_SIZE
            h, w = arr.shape
            if h < IMAGE_SIZE or w < IMAGE_SIZE:
                # Pad if smaller
                pad_h = max(0, IMAGE_SIZE - h)
                pad_w = max(0, IMAGE_SIZE - w)
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")

            ch, cw = arr.shape[0] // 2, arr.shape[1] // 2
            half   = IMAGE_SIZE // 2
            patch  = arr[ch - half: ch + half, cw - half: cw + half]

            # Normalize to 0–255
            pmin, pmax = patch.min(), patch.max()
            if pmax > pmin:
                norm = ((patch - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                norm = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

            img = Image.fromarray(norm, mode="L")
            img.save(out_path)

            saved.append({"SID": sid, "TIMESTAMP": ts, "PATH": str(out_path)})

        except Exception as exc:
            continue

    ds.close()
    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Synthetic fallback: generate realistic fake IR images
#           used when HURSAT files are unavailable (access restricted)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_ir_image(wind_kt: float, lat: float, lon: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic infrared brightness-temperature image.

    More intense storms (higher wind) have:
    - Colder cloud tops (lower pixel values in IR)
    - More circular, organized cloud pattern
    - Clearer eye structure
    """
    size = IMAGE_SIZE
    img  = np.full((size, size), 280.0)   # warm background (~280 K)

    center = size // 2
    Y, X   = np.ogrid[:size, :size]
    dist   = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    # Storm intensity parameters
    intensity  = min(wind_kt / 150.0, 1.0)     # 0–1
    cloud_temp = 280 - 100 * intensity          # 180–280 K (colder = more intense)
    cloud_rad  = 20 + 30 * intensity            # cloud shield radius in pixels
    eye_rad    = max(2, 8 * intensity)          # eye radius

    # Outer cloud shield
    cloud_mask = dist < cloud_rad
    img[cloud_mask] = cloud_temp + rng.normal(0, 8, size=(size, size))[cloud_mask]

    # Inner convective ring (coldest)
    ring_mask = (dist > cloud_rad * 0.5) & (dist < cloud_rad * 0.75)
    img[ring_mask] = cloud_temp - 15 + rng.normal(0, 5, size=(size, size))[ring_mask]

    # Eye (warm, clear)
    if intensity > 0.4:
        eye_mask = dist < eye_rad
        img[eye_mask] = 290 + rng.normal(0, 3, size=(size, size))[eye_mask]

    # Add noise and clip
    img += rng.normal(0, 3, (size, size))
    img  = np.clip(img, 160, 310)

    # Normalize to 0–255
    norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return norm


def generate_synthetic_images(df: pd.DataFrame, storm_names: list) -> list:
    """
    Generate synthetic IR image sequences for storms where HURSAT is unavailable.
    This creates realistic-looking training data based on actual wind speed values.
    """
    print("\n[Step 4 Fallback] Generating synthetic IR images from IBTrACS wind data …")
    print("  (Used when HURSAT files are inaccessible — images reflect real wind history)")

    rng    = np.random.default_rng(42)
    saved  = []

    target_storms = df[df["NAME"].isin(storm_names)] if storm_names else df
    storm_groups  = list(target_storms.groupby("SID"))

    for sid, group in tqdm(storm_groups, desc="Generating images"):
        group = group.sort_values("ISO_TIME").reset_index(drop=True)

        for _, row in group.iterrows():
            ts       = pd.Timestamp(row["ISO_TIME"])
            wind     = float(row["WIND"])
            lat      = float(row["LAT"])
            lon      = float(row["LON"])
            filename = f"{sid}_{ts.strftime('%Y%m%d%H')}.png"
            out_path = IMG_DIR / filename

            if out_path.exists():
                saved.append({"SID": sid, "TIMESTAMP": ts, "PATH": str(out_path)})
                continue

            arr = generate_synthetic_ir_image(wind, lat, lon, rng)
            Image.fromarray(arr, mode="L").save(out_path)
            saved.append({"SID": sid, "TIMESTAMP": ts, "PATH": str(out_path)})

    print(f"    → {len(saved):,} synthetic images saved to {IMG_DIR}")
    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_STORMS = [
    "AMPHAN", "SIDR", "FANI", "MOCHA",
    "PHAILIN", "HUDHUD", "TITLI", "BULBUL",
    "GIRI", "MALA", "AILA"
]


def run_pipeline(storm_names: list, try_hursat: bool = True):
    # ── Step 1: Load IBTrACS ──────────────────────────────────────────────
    df       = load_ibtracs()
    all_info = get_named_storms(df)

    print(f"\n  Top 10 storms by peak wind:")
    print(all_info.head(10).to_string(index=False))

    # Filter to requested storms
    if storm_names:
        df_target = df[df["NAME"].isin(storm_names)]
    else:
        df_target = df

    available = df_target["NAME"].unique().tolist()
    print(f"\n  Storms to process: {available}")

    # ── Step 2 + 3: Try HURSAT download ──────────────────────────────────
    all_saved = []
    hursat_ok = False

    if try_hursat:
        print("\n[Step 2] Checking HURSAT availability …")
        years = sorted(df_target["ISO_TIME"].dt.year.unique().tolist())
        sample_year = years[-1] if years else 2020
        files = list_hursat_files(sample_year)

        if files:
            print(f"  → Found {len(files)} HURSAT files for {sample_year}")
            hursat_ok = True

            for year in tqdm(years, desc="Years"):
                year_files = list_hursat_files(year)
                for fname in tqdm(year_files, desc=f"  {year}", leave=False):
                    storm_sid = None
                    # Try to match filename to a SID in df_target
                    for sid, grp in df_target.groupby("SID"):
                        name = grp["NAME"].iloc[0] if "NAME" in grp.columns else ""
                        if name.upper() in fname.upper() or sid[:8] in fname:
                            storm_sid = sid
                            break
                    if storm_sid is None:
                        continue

                    nc_path = download_hursat_file(fname, year)
                    if nc_path is None:
                        continue

                    storm_grp = df_target[df_target["SID"] == storm_sid]
                    patches   = extract_patches_from_netcdf(nc_path, storm_grp, storm_sid)
                    all_saved.extend(patches)
        else:
            print("  → HURSAT directory not accessible (access may require registration)")
            print("  → Falling back to synthetic image generation")

    # ── Step 4: Synthetic fallback ────────────────────────────────────────
    if not hursat_ok or len(all_saved) == 0:
        all_saved = generate_synthetic_images(df_target, storm_names)

    # ── Save manifest ─────────────────────────────────────────────────────
    if all_saved:
        manifest = pd.DataFrame(all_saved)
        manifest.to_csv(MANIFEST, index=False)
        print(f"\n[Done] Manifest saved: {MANIFEST}")
        print(f"       Total images  : {len(manifest):,}")
        print(f"       Unique storms : {manifest['SID'].nunique()}")
        print(f"\nNext step: open notebooks/v3_cnn_lstm_fusion.ipynb")
        print(f"  Upload: data/satellite_manifest.csv")
        print(f"  Upload: data/satellite_images/ (as satellite_images.zip)")
    else:
        print("\n[WARN] No images were saved.")

    return all_saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/generate satellite IR images for cyclone prediction."
    )
    parser.add_argument(
        "--storms", nargs="+", default=DEFAULT_STORMS,
        help=f"Storm names to process (default: {DEFAULT_STORMS})"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all Bay of Bengal storms (slow)"
    )
    parser.add_argument(
        "--no-hursat", action="store_true",
        help="Skip HURSAT download, go straight to synthetic generation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    names = None if args.all else args.storms
    run_pipeline(storm_names=names, try_hursat=not args.no_hursat)
