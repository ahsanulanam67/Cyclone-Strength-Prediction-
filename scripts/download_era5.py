"""
ERA5 Download Helper
====================
Downloads ERA5 variables required for Version 2 of the cyclone prediction project.

Prerequisites
-------------
1. Create a free account at https://cds.climate.copernicus.eu
2. Accept the dataset licence for "ERA5 hourly data on single levels"
3. Install the CDS API client:
       pip install cdsapi
4. Create ~/.cdsapirc with your credentials:
       url: https://cds.climate.copernicus.eu/api/v2
       key: <your-uid>:<your-api-key>

Usage
-----
    python scripts/download_era5.py --year 2020
    python scripts/download_era5.py --year 2015 2016 2017 --outdir data/era5

The downloaded NetCDF file(s) can be uploaded directly to Google Colab
and used by notebooks/v2_era5_weather.ipynb.
"""

import argparse
import os
import sys

try:
    import cdsapi
except ImportError:
    print("ERROR: cdsapi is not installed.")
    print("       Run:  pip install cdsapi")
    sys.exit(1)


# ── ERA5 request configuration ───────────────────────────────────────────────

DATASET = "reanalysis-era5-single-levels"

# Variables needed for the cyclone intensification model.
VARIABLES = [
    "sea_surface_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
]

# Bay of Bengal domain (north, west, south, east).
AREA = [25, 80, 0, 100]

# 6-hourly synoptic times matching IBTrACS observation cadence.
TIMES = ["00:00", "06:00", "12:00", "18:00"]

MONTHS = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12",
]

DAYS = [f"{d:02d}" for d in range(1, 32)]


# ── download ─────────────────────────────────────────────────────────────────

def download_year(year: int, outdir: str) -> str:
    """
    Download all 12 months of ERA5 for *year* over the Bay of Bengal.

    Returns the path to the saved NetCDF file.
    """
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"era5_bay_of_bengal_{year}.nc")

    if os.path.exists(outfile):
        print(f"[{year}] Already exists — skipping: {outfile}")
        return outfile

    print(f"[{year}] Requesting ERA5 data …")
    print(f"        Variables : {len(VARIABLES)}")
    print(f"        Domain    : {AREA}  (N W S E)")
    print(f"        Times/day : {TIMES}")
    print(f"        Output    : {outfile}")
    print()
    print("NOTE: Large requests may take several minutes on the CDS queue.")

    client = cdsapi.Client()

    client.retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "variable":     VARIABLES,
            "year":         str(year),
            "month":        MONTHS,
            "day":          DAYS,
            "time":         TIMES,
            "area":         AREA,
            "format":       "netcdf",
        },
        outfile,
    )

    size_mb = os.path.getsize(outfile) / (1024 ** 2)
    print(f"[{year}] Download complete — {outfile}  ({size_mb:.1f} MB)")
    return outfile


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ERA5 data for the Bay of Bengal cyclone project."
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=int,
        default=[2020],
        metavar="YEAR",
        help="One or more years to download (default: 2020).",
    )
    parser.add_argument(
        "--outdir",
        default="data/era5",
        help="Directory to save NetCDF files (default: data/era5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ERA5 Download — Bay of Bengal Cyclone Project")
    print("=" * 60)
    print()

    downloaded = []
    failed = []

    for year in sorted(args.year):
        try:
            path = download_year(year, args.outdir)
            downloaded.append(path)
        except Exception as exc:
            print(f"[{year}] FAILED: {exc}")
            failed.append(year)

    print()
    print("=" * 60)
    print(f"  Downloaded : {len(downloaded)} file(s)")
    if failed:
        print(f"  Failed     : {failed}")
    print()
    print("Next step: upload these NetCDF files to Google Colab")
    print("and run notebooks/v2_era5_weather.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
