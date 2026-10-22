# ERA5 Data — Download Instructions

ERA5 NetCDF files are NOT included in this repository (too large for GitHub, ~500 MB).

## How to download

1. Create a free account at https://cds.climate.copernicus.eu
2. Accept the ERA5 dataset licence
3. Install the CDS API client and add your credentials:

```bash
pip install cdsapi
```

Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY-HERE
```

4. Run the download script:

```bash
python scripts/download_era5.py --year 2018 2019 2020
```

Files will be saved as:
```
data/era5/era5_bay_of_bengal_2018.nc/
data/era5/era5_bay_of_bengal_2019.nc/
data/era5/era5_bay_of_bengal_2020.nc/
```

Each folder contains 3 NetCDF files (accum, instant, wave) that are merged automatically by the notebook.

## Pre-merged CSV

`era5_ibtracs_2020_merged.csv` is included — this is the 2020 ERA5 data already matched to IBTrACS storm observations (147 rows). Sufficient to run V2 notebook without downloading ERA5.
