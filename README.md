# 🌀 AI-Based Cyclone Early Warning System
### Tropical Cyclone Intensification Prediction — Bay of Bengal

> **Master's Thesis Project** | Predicts whether an active cyclone will significantly intensify (wind ≥ +15 knots) at **2–3 days** and **5–7 days** ahead.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What This Project Does

```
A cyclone exists in the Bay of Bengal
         ↓
System fetches its live position (GDACS)
+ real-time weather (Open-Meteo: SST, pressure, wind)
         ↓
V2 XGBoost model predicts:
  → 48-72h:   "67% chance of intensification"
  → 120-168h: "81% chance of intensification"
         ↓
Dashboard shows LOW / MODERATE / HIGH risk
with interactive map, direction arrow, and weather panel
```

---

## 📊 Three Versions — Progressive Improvement

| Version | Data Used | ROC-AUC (48-72h) | ROC-AUC (120-168h) |
|---|---|---|---|
| V1 XGBoost | Track history only | 0.58 | 0.10 |
| V2 XGBoost | + ERA5 weather (SST, pressure, wind shear) | 0.67 | **0.80** |
| V3 CNN-LSTM | + Satellite IR image sequences | **0.89** | — |

---

## 🚀 Quick Start

```bash
git clone https://github.com/ahsanul-anam/cyclone-early-warning
cd cyclone-early-warning
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Open **http://localhost:8501**

---

## 📁 Project Structure

```
cyclone-early-warning/
├── dashboard/app.py                ← Streamlit live dashboard
├── notebooks/
│   ├── v1_ibtracs_baseline.ipynb
│   ├── v2_era5_weather.ipynb
│   └── v3_cnn_lstm_fusion.ipynb
├── scripts/
│   ├── download_era5.py
│   └── download_gridsat.py
├── src/
│   ├── labels.py
│   ├── preprocessing.py
│   └── evaluation.py
├── models/
│   ├── v2_short.pkl
│   └── v2_long.pkl
├── data/
│   ├── gridsat_manifest.csv
│   ├── satellite_images/           ← 445 GridSat-B1 IR patches
│   └── era5/                       ← ERA5 NetCDF (Git LFS)
├── RESEARCH_PAPER.md
└── requirements.txt
```

---

## 🛰️ Datasets

| Dataset | Source | Size | Login |
|---|---|---|---|
| IBTrACS v04r01 | NOAA NCEI | ~20 MB | No |
| ERA5 2018-2020 | Copernicus CDS | ~500 MB | Free |
| GridSat-B1 | NOAA AWS S3 | ~7 MB | No |
| Open-Meteo | open-meteo.com | API live | No |
| GDACS | gdacs.org | RSS live | No |

### Get ERA5 data
```bash
pip install cdsapi
# Add credentials to ~/.cdsapirc
python scripts/download_era5.py --year 2018 2019 2020
```

### Get satellite images
```bash
python scripts/download_gridsat.py
```

---

## 📄 Research Paper

See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for the full academic paper.

**Result:** ROC-AUC: 0.58 (V1) → 0.67 (V2) → **0.89 (V3)**

---

## ⚠️ Limitations

- Predicts intensification only — not track, landfall, or flood
- ERA5 is reanalysis (historical), not real-time forecast
- V3 not yet integrated in dashboard (research prototype)

---

*Built for master's thesis on AI-based disaster prediction, Bay of Bengal.*
