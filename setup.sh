#!/bin/bash
# One-command setup for Cyclone Early Warning System
# Run: bash setup.sh

echo "================================================"
echo "  Cyclone Early Warning System — Setup"
echo "================================================"

# Check Python version
PY=$(python3 --version 2>&1)
echo "Python: $PY"

# Install all packages
echo ""
echo "Installing packages..."
pip install \
  pandas "numpy>=1.24,<2.0" scikit-learn xgboost joblib \
  matplotlib seaborn streamlit folium streamlit-folium \
  xarray netCDF4 h5netcdf h5py \
  pillow requests certifi cdsapi

# Install PyTorch (for V3 CNN-LSTM)
echo ""
echo "Installing PyTorch (for V3 model)..."
pip install torch

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  Run the dashboard:"
echo "    streamlit run dashboard/app.py"
echo ""
echo "  Download ERA5 data (needs Copernicus account):"
echo "    python scripts/download_era5.py --year 2018 2019 2020"
echo ""
echo "  Download satellite images (free, no login):"
echo "    python scripts/download_gridsat.py"
echo "================================================"
