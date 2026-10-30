# 🌀 AI-Based Cyclone Early Warning System
### Tropical Cyclone Intensification + Track Prediction — Bay of Bengal

> **Master's Thesis Project** | Predicts whether an active cyclone will significantly intensify (wind ≥ +15 knots) at **2–3 days** and **5–7 days** ahead, forecasts the **48h track**, and estimates **storm surge risk** at 8 coastal stations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What This Project Does

```
A cyclone exists in the Bay of Bengal
         ↓
Fetch live position (GDACS RSS) + real-time weather (Open-Meteo)
         ↓
V2 XGBoost predicts intensification:
  → 48-72h:   "67% chance of intensification"
  → 120-168h: "81% chance of intensification"
         ↓
Random Forest predicts 48h track (8 × 6h waypoints)
  → 24h MAE ~18 km, 48h MAE ~134 km (leave-one-storm-out CV)
         ↓
Physics heuristic estimates storm surge at 8 coastal stations
  → LOW / MODERATE / HIGH / EXTREME
         ↓
Streamlit dashboard: interactive Folium map, risk panel, weather panel
```

---

## 📊 Three Model Versions — Progressive Improvement

| Version | Data Used | ROC-AUC (48-72h) | ROC-AUC (120-168h) |
|---|---|---|---|
| V1 XGBoost | Track history only (IBTrACS) | 0.58 | 0.10 |
| V2 XGBoost | + ERA5 weather (SST, pressure, wind) | 0.67 | **0.80** |
| V3 CNN-LSTM | + Satellite IR image sequences (GridSat-B1) | **0.89** | — |

**Key finding:** Adding ERA5 environmental features improved long-range (5–7 day) AUC from 0.10 → 0.80. Adding satellite IR images pushed short-range AUC to 0.89.

---

## 🚀 Quick Start

```bash
git clone https://github.com/ahsanulanam67/Cyclone-Strength-Prediction-
cd Cyclone-Strength-Prediction-
pip install -r requirements.txt

# Train the track predictor (first time only, ~3 min)
python scripts/train_track_predictor.py

# Launch the dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501**

One-command install (installs all packages including PyTorch for V3):
```bash
bash setup.sh
```

---

## 📁 Project Structure

```
Cyclone-Strength-Prediction-/
├── dashboard/
│   └── app.py                      ← Streamlit live dashboard
├── notebooks/
│   ├── v1_ibtracs_baseline.ipynb   ← V1 XGBoost baseline
│   ├── v2_era5_weather.ipynb       ← V2 ERA5-augmented model
│   └── v3_cnn_lstm_fusion.ipynb    ← V3 CNN-LSTM satellite model
├── scripts/
│   ├── train_track_predictor.py    ← Train 48h track forecast model
│   ├── download_era5.py            ← Download ERA5 reanalysis data
│   ├── download_gridsat.py         ← Download GridSat-B1 satellite images
│   └── download_hursat.py
├── src/
│   ├── track_predictor.py          ← Random Forest track forecast (48h)
│   ├── surge_risk.py               ← Storm surge heuristic (8 stations)
│   ├── labels.py
│   ├── preprocessing.py
│   └── evaluation.py
├── models/
│   ├── v2_short.pkl                ← V2 model for 48-72h (329 KB)
│   ├── v2_long.pkl                 ← V2 model for 120-168h (248 KB)
│   └── track_predictor.pkl         ← Track model — NOT in repo (>100MB)
│                                      Run: python scripts/train_track_predictor.py
├── data/
│   ├── ibtracs_bob.csv             ← IBTrACS NI basin (405 storms, 62k rows)
│   ├── satellite_manifest.csv
│   ├── satellite_images/           ← 445 GridSat-B1 IR patches
│   └── era5/                       ← ERA5 NetCDF (excluded, ~500MB)
│       └── README_DOWNLOAD.md      ← Download instructions
├── docs/
│   ├── research_paper.pdf          ← Full academic paper (PDF)
│   └── defense_questions.pdf       ← Thesis defense Q&A (25+ questions)
├── requirements.txt
└── setup.sh                        ← One-command installer
```

---

## 🛰️ Datasets

| Dataset | Source | Size | Login |
|---|---|---|---|
| IBTrACS v04r01 | NOAA NCEI | ~27 MB | No |
| ERA5 2018-2020 | Copernicus CDS | ~500 MB | Free |
| GridSat-B1 | NOAA AWS S3 | ~7 MB | No |
| Open-Meteo | open-meteo.com | API live | No |
| GDACS | gdacs.org | RSS live | No |

### ERA5 data (needed for V2 training only)
```bash
pip install cdsapi
# Create ~/.cdsapirc with your Copernicus API key
python scripts/download_era5.py --year 2018 2019 2020
```
See `data/era5/README_DOWNLOAD.md` for full instructions.

### Satellite images (needed for V3 training only)
```bash
python scripts/download_gridsat.py
```

---

## 🗺️ Dashboard Features

| Feature | How it works |
|---|---|
| Live cyclone detection | GDACS RSS feed — shows active BoB storms as orange markers |
| Intensification forecast | V2 XGBoost — probability at 48-72h and 120-168h |
| 48h track forecast | Random Forest (src/track_predictor.py) — dashed blue line on map |
| Storm surge risk | Physics heuristic (src/surge_risk.py) — 8 coastal stations |
| Live weather panel | Open-Meteo API — SST, pressure, wind, temperature |
| Interactive map | Folium + Esri Ocean basemap — zoom, pan, click popups |

---

## 📄 Docs

- [Research Paper (PDF)](docs/research_paper.pdf) — full academic write-up
- [Defense Q&A (PDF)](docs/defense_questions.pdf) — 25+ thesis defense questions with answers

---

## ⚠️ Limitations

- **Intensification only** — predicts significant wind increase, not exact wind speed
- **Track forecast accuracy degrades at 48h** — 24h MAE ~18 km, 48h MAE ~134 km (leave-one-storm-out CV); no steering wind features yet
- **Storm surge is a physics heuristic** — not a trained ML model; does not account for bathymetry, storm size, or tidal interaction. Future work: ADCIRC/FVCOM numerical model
- **ERA5 vs. live weather distribution shift** — models trained on ERA5 reanalysis, served with Open-Meteo real-time data; variable definitions differ slightly
- **Small training set** — 2018–2020, North Indian Ocean only (~21 storms with ERA5 overlap)
- **V3 CNN-LSTM not in dashboard** — AUC 0.89 achieved in notebook; PyTorch export to dashboard is pending

---

## 🔭 Future Work

- Add 500 hPa steering winds to track predictor (biggest accuracy improvement)
- Calibrate intensification probabilities (isotonic regression) for reliable "% chance"
- Integrate V3 CNN-LSTM into dashboard via TorchScript/ONNX export
- Expand training to all basins (multi-basin transfer learning → NI fine-tune)
- Replace surge heuristic with ADCIRC/FVCOM numerical surge model
- Add SHAP feature importance plots for model interpretability

---

*Built for a master's thesis on AI-based tropical cyclone early warning, Bay of Bengal.*
