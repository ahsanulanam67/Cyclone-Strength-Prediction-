"""
GridSat-B1 Satellite Image Download Pipeline
=============================================
Downloads infrared brightness temperature images from NOAA GridSat-B1
directly from AWS S3 — no login or registration required.

Covers 1980–present — includes ALL your key storms:
SIDR (2007), GIRI (2010), PHAILIN (2013), HUDHUD (2014),
FANI (2019), AMPHAN (2020), MOCHA (2023)

Usage
-----
    python scripts/download_gridsat.py

Output
------
    data/satellite_images/SID_YYYYMMDDHH.png  ← ready for V3 notebook
"""

import io
import os
import ssl
import math
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl._create_unverified_context()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "satellite_images"
RAW_DIR = ROOT / "data" / "gridsat_raw"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── AWS S3 bucket ────────────────────────────────────────────────────────────
S3_BASE = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com"

# ── IBTrACS URL ───────────────────────────────────────────────────────────────
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

# Key storms to download — add more if needed
TARGET_STORMS = [
    "SIDR", "GIRI", "PHAILIN", "HUDHUD",
    "MALA", "FANI", "AMPHAN", "MOCHA",
    "TITLI", "BULBUL"
]

IMAGE_SIZE = 128


# ── Helpers ──────────────────────────────────────────────────────────────────

def gridsat_s3_key(timestamp: pd.Timestamp) -> str:
    """Return the S3 key for a GridSat-B1 file at a given timestamp."""
    # Files are 3-hourly: round to nearest 3h
    hour = (timestamp.hour // 3) * 3
    return (
        f"data/{timestamp.year}/"
        f"GRIDSAT-B1.{timestamp.year}.{timestamp.month:02d}.{timestamp.day:02d}"
        f".{hour:02d}.v02r01.nc"
    )


def download_gridsat_file(timestamp: pd.Timestamp) -> Path | None:
    """Download one GridSat-B1 NetCDF file. Returns local path or None."""
    key      = gridsat_s3_key(timestamp)
    url      = f"{S3_BASE}/{key}"
    filename = Path(key).name
    dest     = RAW_DIR / filename

    if dest.exists():
        return dest

    try:
        with urllib.request.urlopen(url, context=SSL_CTX, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return dest
    except Exception as exc:
        return None


def extract_bob_patch(nc_path: Path, storm_lat: float, storm_lon: float) -> np.ndarray | None:
    """
    Extract a 128×128 infrared patch centered on the storm from a GridSat file.
    GridSat ch4 = IR window channel ~11 microns — best for cloud-top temperature.
    """
    try:
        import xarray as xr
    except ImportError:
        return None

    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
    except Exception:
        try:
            ds = xr.open_dataset(nc_path)
        except Exception:
            return None

    # Find IR channel variable
    ir_var = None
    for candidate in ["ch4", "irwin_cdr", "temp_11_0um_nom", "IR_WV"]:
        if candidate in ds.data_vars:
            ir_var = candidate
            break
    if ir_var is None:
        for var in ds.data_vars:
            if ds[var].ndim >= 2:
                ir_var = var
                break
    if ir_var is None:
        ds.close()
        return None

    # Find lat/lon coordinates
    lat_name = next((c for c in ["lat", "latitude"] if c in ds.coords), None)
    lon_name = next((c for c in ["lon", "longitude"] if c in ds.coords), None)
    if lat_name is None or lon_name is None:
        ds.close()
        return None

    try:
        # Select patch around storm center (±8 degrees)
        patch_ds = ds[ir_var].sel(
            {lat_name: slice(storm_lat + 8, storm_lat - 8),
             lon_name: slice(storm_lon - 8, storm_lon + 8)}
        )

        # Drop extra dims
        arr = patch_ds.values
        while arr.ndim > 2:
            arr = arr[0]

        arr = arr.astype(float)

        # Handle fill values
        fill_mask = np.isnan(arr) | (arr < -9000) | (arr > 400)
        if fill_mask.all():
            ds.close()
            return None
        arr[fill_mask] = np.nanmean(arr[~fill_mask])

        # Resize to 128×128
        from PIL import Image as PILImage
        img = PILImage.fromarray(
            ((arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255).astype(np.uint8)
        ).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.LANCZOS)

        ds.close()
        return np.array(img)

    except Exception:
        ds.close()
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_ibtracs():
    print("Loading IBTrACS...")
    with urllib.request.urlopen(IBTRACS_URL, context=SSL_CTX) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw), skiprows=[1], low_memory=False)
    for col in ["LAT", "LON", "USA_WIND", "WMO_WIND"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df["WIND"] = df["USA_WIND"].fillna(df.get("WMO_WIND", pd.Series(dtype=float)))
    df["WIND"] = pd.to_numeric(df["WIND"], errors="coerce")
    df = df.dropna(subset=["SID", "ISO_TIME", "WIND"])
    df = df[df["LAT"].between(0, 25) & df["LON"].between(80, 100)]
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    return df


def run():
    df = load_ibtracs()

    # Filter to target storms
    df_target = df[df["NAME"].isin(TARGET_STORMS)].copy()
    print(f"Target storms found: {df_target['NAME'].unique().tolist()}")
    print(f"Total observations : {len(df_target)}")

    saved    = []
    failed   = 0
    manifest = []

    for (sid, name), group in tqdm(
        df_target.groupby(["SID", "NAME"]),
        desc="Downloading images"
    ):
        group = group.sort_values("ISO_TIME").reset_index(drop=True)

        for _, row in group.iterrows():
            ts       = pd.Timestamp(row["ISO_TIME"])
            lat      = float(row["LAT"])
            lon      = float(row["LON"])
            filename = f"{sid}_{ts.strftime('%Y%m%d%H')}.png"
            out_path = IMG_DIR / filename

            # Skip if already exists
            if out_path.exists():
                manifest.append({"SID": sid, "NAME": name,
                                  "TIMESTAMP": ts, "PATH": str(out_path)})
                saved.append(filename)
                continue

            # Download GridSat file
            nc_path = download_gridsat_file(ts)
            if nc_path is None:
                failed += 1
                continue

            # Extract patch
            patch = extract_bob_patch(nc_path, lat, lon)
            if patch is None:
                failed += 1
                continue

            # Save as PNG
            Image.fromarray(patch, mode="L").save(out_path)
            manifest.append({"SID": sid, "NAME": name,
                              "TIMESTAMP": ts, "PATH": str(out_path)})
            saved.append(filename)

    # Save manifest
    if manifest:
        mf_path = ROOT / "data" / "gridsat_manifest.csv"
        pd.DataFrame(manifest).to_csv(mf_path, index=False)
        print(f"\n✅ Done!")
        print(f"   Images saved : {len(saved)}")
        print(f"   Failed       : {failed}")
        print(f"   Manifest     : {mf_path}")
    else:
        print(f"\n❌ No images saved. Failed: {failed}")
        print("   The S3 bucket may use a different file structure.")
        print("   Check: https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/")


if __name__ == "__main__":
    run()
