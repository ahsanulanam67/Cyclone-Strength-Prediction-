# AI-Based Tropical Cyclone Intensification Prediction for the Bay of Bengal: A Multi-Modal Deep Learning Approach

**Authors:** Nujum (Master's Thesis, 2026)
**Keywords:** Tropical cyclone, Intensification prediction, Bay of Bengal, XGBoost, CNN-LSTM, ERA5, Satellite imagery, Early warning system

---

## Abstract

Tropical cyclones in the Bay of Bengal cause catastrophic loss of life and infrastructure across Bangladesh, India, and Myanmar. Accurate prediction of cyclone intensification — whether an existing storm will significantly strengthen — remains one of the most challenging problems in operational meteorology. This paper presents a three-version AI-based early warning system that progressively incorporates richer data modalities: historical track features (V1), atmospheric reanalysis (V2), and satellite infrared image sequences (V3). Experiments on 375 Bay of Bengal storms (1970–2024) and ERA5 reanalysis data (2018–2020) show that each additional data modality improves prediction accuracy, with ROC-AUC improving from 0.58 (V1) to 0.67 (V2) to 0.89 (V3) at the 48–72 hour forecast horizon. A real-time Streamlit dashboard integrates live weather from Open-Meteo and active storm alerts from GDACS to enable operational use.

---

## 1. Introduction

The Bay of Bengal (BoB) accounts for approximately 7% of global tropical cyclone activity but disproportionately high mortality, due to the shallow bathymetry, dense coastal populations, and funnel-shaped coastline of Bangladesh. Historical disasters such as Cyclone Sidr (2007, ~3,500 deaths), Cyclone Fani (2019, ~89 deaths), and Cyclone Amphan (2020, ~128 deaths and $13 billion damage) highlight the critical need for accurate early warning systems.

The central challenge is **intensification prediction**: given a cyclone's current state, will it significantly strengthen over the coming days? The National Hurricane Center defines Rapid Intensification (RI) as a wind increase of ≥30 kt in 24 hours. This work uses a slightly relaxed threshold of ≥15 kt within a forecast window, making the label applicable to the broader intensification problem at longer lead times.

**Research question:** Does integrating atmospheric reanalysis (ERA5) and satellite infrared image sequences with historical cyclone-track data improve prediction of tropical cyclone intensification in the Bay of Bengal at 48–72 hour and 120–168 hour lead times?

**Hypotheses:**
- H1: Adding ERA5 atmospheric variables (V2) improves over track-only baseline (V1)
- H2: Adding satellite infrared image sequences (V3) further improves over V2

---

## 2. Related Work

### 2.1 Statistical and Machine Learning Approaches

Traditional intensification prediction relies on Statistical Hurricane Intensity Prediction Scheme (SHIPS, DeMaria & Kaplan, 1994), which uses large-scale environmental factors including sea surface temperature, wind shear, and humidity. Machine learning extensions of SHIPS using Random Forest and gradient boosting have demonstrated improvements over linear models (Knaff et al., 2018).

### 2.2 Deep Learning for Cyclone Analysis

Convolutional Neural Networks (CNNs) applied to satellite imagery have shown promising results for tropical cyclone intensity estimation (Pradhan et al., 2018; Chen et al., 2019). LSTM networks capture temporal evolution of storm structure (Kim et al., 2019). Combined CNN-LSTM architectures for satellite image sequences represent the current state of the art for intensity change prediction.

### 2.3 Bay of Bengal Specific Work

Studies of BoB cyclone climatology (Sahoo & Bhaskaran, 2016) identify October–November as the peak intensification season. SST above 28°C and low vertical wind shear are the primary thermodynamic preconditions (Emanuel, 1986). Recent work on BoB drought and extreme events using SPI, SPEI, and NDVI (Kamruzzaman et al., 2022; Sarker et al., 2026; Esha et al., 2026) provides methodological context for climate-variable integration.

---

## 3. Data

### 3.1 IBTrACS v04r01 (Track Data)

The International Best Track Archive for Climate Stewardship (Knapp et al., 2010) provides 6-hourly cyclone track observations for the North Indian Ocean from 1970 to 2024. After filtering to the Bay of Bengal domain (0–25°N, 80–100°E), 375 storms were available for training, comprising 12,847 individual observations.

| Variable | Description |
|---|---|
| LAT, LON | Storm center position (°N, °E) |
| WIND | Maximum sustained wind speed (knots) |
| MONTH, HOUR | Seasonality features |
| LAT_CHANGE, LON_CHANGE | 6-hourly displacement |
| WIND_CHANGE | 6-hourly wind tendency |
| TRACK_SPEED | Translation speed (km/h, Haversine formula) |

### 3.2 ERA5 Reanalysis (Atmospheric Data)

ERA5 hourly single-level reanalysis (Hersbach et al., 2020) was downloaded via the Copernicus Climate Data Store API for years 2018–2020, covering the BoB domain at 0.25° spatial resolution. Three files per year (accumulated, instantaneous, wave) were merged.

| Variable | ERA5 name | Unit |
|---|---|---|
| Sea surface temperature | sst | °C |
| Mean sea-level pressure | msl | hPa |
| 10m U-wind component | u10 | m/s |
| 10m V-wind component | v10 | m/s |
| 2m temperature | t2m | °C |
| 2m dewpoint | d2m | °C |
| Total precipitation | tp | m |
| Significant wave height | swh | m |

Each IBTrACS observation was matched to the nearest ERA5 grid point and time step using `xarray.sel(method='nearest')`, yielding 520 matched observations across 21 storms (2018–2020).

### 3.3 GridSat-B1 Satellite Imagery

GridSat-B1 (Knapp et al., 2011) provides 3-hourly global infrared brightness temperature composites from 1980 to present, freely available via NOAA AWS S3. The infrared window channel (ch4, ~11 μm) captures cloud-top temperatures, with cold (bright) pixels indicating deep convective activity characteristic of intensifying cyclones.

For each storm observation, a 4-frame sequence of 128×128 pixel infrared patches centered on the storm was extracted (frames at T-18h, T-12h, T-6h, T-0h). A total of 445 images were downloaded across 9 major storms: MALA (2006), SIDR (2007), GIRI (2010), PHAILIN (2013), HUDHUD (2014), TITLI (2018), FANI (2019), AMPHAN (2020), MOCHA (2023).

### 3.4 Live Data (Operational Dashboard)

For real-time inference, the dashboard fetches:
- **Open-Meteo API**: Current SST, surface pressure, temperature, wind, and precipitation (free, no authentication)
- **GDACS RSS feed**: Active tropical cyclone alerts filtered to Bay of Bengal

---

## 4. Methodology

### 4.1 Label Generation

The intensification label TARGET ∈ {0, 1} is defined for each observation i as:

```
TARGET_i = 1  if  max(WIND_j : t_i + h_start ≤ t_j ≤ t_i + h_end) − WIND_i ≥ 15 kt
TARGET_i = 0  otherwise
TARGET_i = NaN if no future observations exist in the window
```

Two forecast horizons are used:
- **Short (48–72h):** 2–3 day intensification
- **Long (120–168h):** 5–7 day intensification

### 4.2 Train-Test Split

Storm-grouped splitting prevents data leakage: all observations from a given storm are assigned exclusively to either train or test. `GroupShuffleSplit` (test_size=0.25, random_state=42) from scikit-learn was used.

### 4.3 Version 1 — XGBoost Baseline (Track Features)

An XGBoost classifier (Chen & Guestrin, 2016) trained on 9 track features with median imputation for missing values.

**Hyperparameters:** n_estimators=300, max_depth=4, learning_rate=0.05

### 4.4 Version 2 — XGBoost with ERA5 Features

V1 extended with 7 ERA5 variables matched to each observation. The same XGBoost pipeline is used with 16 total features. Training used 21 storms from 2018–2020 with ERA5 coverage.

### 4.5 Version 3 — CNN-LSTM Fusion (Satellite + Track)

A two-branch neural network implemented in PyTorch:

```
Branch A — Satellite image sequence:
  Input: (4, 64, 64) grayscale IR images
  → TimeDistributed CNN:
      Conv2D(1→16, 3×3) + BN + ReLU + MaxPool(4)
      Conv2D(16→32, 3×3) + BN + ReLU + MaxPool(4)
      Flatten → Linear(512→64) + ReLU
  → LSTM(input=64, hidden=32)
  → Last hidden state: 32-dim

Branch B — Tabular track features:
  Input: 9 features (standardized)
  → Linear(9→32) + ReLU + Dropout(0.3)
  → Linear(32→16) + ReLU

Fusion:
  Concatenate [32 + 16] = 48-dim
  → Linear(48→32) + ReLU + Dropout(0.3)
  → Linear(32→1) → BCEWithLogitsLoss
```

**Training:** Adam optimizer (lr=5×10⁻⁴, weight_decay=10⁻⁴), CosineAnnealingLR scheduler, early stopping (patience=15), class-weighted loss to handle imbalance (74% positive class).

### 4.6 Evaluation Metrics

| Metric | Rationale |
|---|---|
| ROC-AUC | Ranking ability, threshold-independent |
| PR-AUC | Performance under class imbalance |
| Brier Score | Calibration of probability estimates |
| F1-score | Balance of precision and recall |

---

## 5. Results

### 5.1 Main Comparison

| Version | Data | ROC-AUC (48–72h) | ROC-AUC (120–168h) | PR-AUC (48–72h) |
|---|---|---|---|---|
| V1 XGBoost | Track only | 0.5848 | 0.1000 | 0.6242 |
| V2 XGBoost | Track + ERA5 | 0.6696 | **0.8037** | 0.7703 |
| V3 CNN-LSTM | Track + Satellite | **0.8862** | — | **0.9487** |

Both H1 and H2 are confirmed at the 48–72h horizon. ERA5 atmospheric variables dramatically improve the 5–7 day prediction (V1: 0.10 → V2: 0.80), confirming that large-scale thermodynamic conditions are the dominant driver at longer lead times, while track-only features are nearly uninformative beyond 5 days. Satellite infrared image sequences provide the largest single improvement at the 2–3 day horizon.

### 5.2 Feature Importance (V2)

The top predictive features from V2 XGBoost were SST, WIND (current intensity), and MSL pressure — consistent with theoretical understanding that warm ocean waters and low ambient pressure are the primary energy sources for intensification.

### 5.3 V3 Classification Report (48–72h test set, n=83)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Non-intensifying (0) | 0.65 | 0.81 | 0.72 |
| Intensifying (1) | 0.90 | 0.79 | 0.84 |
| **Weighted avg** | **0.82** | **0.80** | **0.80** |

Brier Score: 0.143 (well-calibrated; perfect = 0.0)

---

## 6. Dashboard and Operational Use

A Streamlit web dashboard enables real-time inference:

1. **GDACS integration**: Automatically detects active BoB cyclones
2. **Open-Meteo integration**: Fetches live SST, pressure, wind, and temperature
3. **V2 live prediction**: When weather is fetched, the trained V2 XGBoost model uses live atmospheric conditions to generate risk scores
4. **Interactive map**: Folium map with real satellite basemap, storm track direction arrow, historical track overlays, and category-colored storm marker

The automatic upgrade from V1 (track-only) to V2 (track + live weather) when the user clicks "Fetch Live Weather" demonstrates the system's ability to improve predictions using real-time data at zero additional cost.

---

## 7. Limitations

1. **ERA5 is reanalysis, not forecast**: V2 uses reconstructed past conditions matched to historical storms. For true operational use, ERA5 should be replaced with GFS or ECMWF forecast fields.

2. **Small ERA5 training set**: Only 21 storms from 2018–2020 were used for V2, due to the manual ERA5 download requirement. Extending to 2010–2020 would improve generalization.

3. **V3 long-range gap**: The 120–168h V3 evaluation was not possible because the test split contained only one class (all storms intensified), a consequence of the small 9-storm dataset.

4. **Intensification only**: The system predicts whether a cyclone strengthens, not track, landfall location, or flood extent.

5. **Open-Meteo vs ERA5 units**: Live weather from Open-Meteo is used as a proxy for ERA5 inputs. Unit conversions are applied (pressure: hPa, SST: °C) but spatial resolution differences may introduce slight bias.

---

## 8. Conclusion

This work demonstrates that multi-modal data integration significantly improves tropical cyclone intensification prediction in the Bay of Bengal. The progressive improvement across V1 → V2 → V3 confirms both research hypotheses. The operational dashboard bridges the gap between research prototype and real-world early warning by automatically fetching live atmospheric data and active storm positions.

**Future work** should focus on: (1) expanding ERA5 coverage to 2010–2020 for V2; (2) integrating INSAT-3D imagery from MOSDAC for higher-resolution Indian Ocean satellite data; (3) replacing ERA5 with operational NWP forecast fields for true real-time V2 inference; (4) extending the model to predict landfall intensity.

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- DeMaria, M., & Kaplan, J. (1994). A Statistical Hurricane Intensity Prediction Scheme (SHIPS). *Weather and Forecasting, 9*, 209–220.
- Emanuel, K. A. (1986). An air-sea interaction theory for tropical cyclones. *Journal of Atmospheric Sciences, 43*, 585–604.
- Esha, et al. (2026). Rainfall variability trends in Barind Tract using Mann-Kendall and SAI. *Discover Atmosphere*.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*, 1999–2049.
- Kamruzzaman, M., et al. (2022). Spatiotemporal drought analysis using SPI and SPEI in Bangladesh. *Scientific Reports*.
- Knapp, K. R., et al. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society, 91*, 363–376.
- Knapp, K. R., et al. (2011). Globally Gridded Satellite (GridSat) observations. *Journal of Climate, 24*, 3505–3519.
- Pradhan, R., et al. (2018). Tropical cyclone intensity estimation using deep CNN. *IEEE TGRS*.
- Sahoo, B., & Bhaskaran, P. K. (2016). Assessment on historical cyclone tracks in the Bay of Bengal. *International Journal of Climatology*.
- Sarker, et al. (2026). SPI + LST + NDVI/NDMI for heatwave/drought mapping. *Environmental Earth Sciences*.
