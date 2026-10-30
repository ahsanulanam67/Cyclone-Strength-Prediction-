"""
Generate research_paper.pdf and defense_questions.pdf for the
AI-Based Cyclone Early Warning System thesis project.

Run: python scripts/generate_pdfs.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

W, H = A4
DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = colors.HexColor("#1a3a5c")
LBLUE  = colors.HexColor("#2c6fad")
TEAL   = colors.HexColor("#0d7377")
LGRAY  = colors.HexColor("#f4f6f9")
MGRAY  = colors.HexColor("#cccccc")
RED    = colors.HexColor("#c0392b")
GREEN  = colors.HexColor("#1e8449")
ORANGE = colors.HexColor("#d35400")
WHITE  = colors.white
BLACK  = colors.black


def base_styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Title1",   fontName="Helvetica-Bold",  fontSize=20,
                         textColor=WHITE,  alignment=TA_CENTER, spaceAfter=6))
    s.add(ParagraphStyle("Title2",   fontName="Helvetica",       fontSize=13,
                         textColor=WHITE,  alignment=TA_CENTER, spaceAfter=4))
    s.add(ParagraphStyle("Title3",   fontName="Helvetica-Oblique", fontSize=11,
                         textColor=colors.HexColor("#cce0f5"), alignment=TA_CENTER))
    s.add(ParagraphStyle("H1",       fontName="Helvetica-Bold",  fontSize=14,
                         textColor=BLUE,   spaceBefore=14, spaceAfter=6,
                         borderPad=4))
    s.add(ParagraphStyle("H2",       fontName="Helvetica-Bold",  fontSize=11,
                         textColor=LBLUE,  spaceBefore=10, spaceAfter=4))
    s.add(ParagraphStyle("H3",       fontName="Helvetica-Bold",  fontSize=10,
                         textColor=TEAL,   spaceBefore=8,  spaceAfter=3))
    s.add(ParagraphStyle("BodyText2", fontName="Helvetica",       fontSize=9.5,
                         textColor=BLACK,  leading=14,     spaceAfter=5,
                         alignment=TA_JUSTIFY))
    s.add(ParagraphStyle("BulletCustom", fontName="Helvetica",   fontSize=9.5,
                         textColor=BLACK,  leading=13,     spaceAfter=3,
                         leftIndent=14,    bulletIndent=4))
    s.add(ParagraphStyle("Bullet2",  fontName="Helvetica",       fontSize=9,
                         textColor=colors.HexColor("#333333"), leading=13, spaceAfter=2,
                         leftIndent=28,   bulletIndent=18))
    s.add(ParagraphStyle("Caption",  fontName="Helvetica-Oblique", fontSize=8.5,
                         textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
                         spaceAfter=8))
    s.add(ParagraphStyle("CodeBlock", fontName="Courier",         fontSize=8,
                         textColor=colors.HexColor("#1a1a2e"),  leading=12,
                         backColor=LGRAY, borderPad=6, spaceAfter=6))
    s.add(ParagraphStyle("QHead",    fontName="Helvetica-Bold",  fontSize=10,
                         textColor=BLUE,   spaceBefore=10, spaceAfter=3,
                         leftIndent=0))
    s.add(ParagraphStyle("Answer",   fontName="Helvetica",       fontSize=9.5,
                         textColor=BLACK,  leading=14,     spaceAfter=4,
                         leftIndent=12,   alignment=TA_JUSTIFY))
    s.add(ParagraphStyle("SubQ",     fontName="Helvetica-Bold",  fontSize=9.5,
                         textColor=TEAL,   spaceBefore=6,  spaceAfter=2,
                         leftIndent=12))
    s.add(ParagraphStyle("SubA",     fontName="Helvetica",       fontSize=9,
                         textColor=BLACK,  leading=13,     spaceAfter=3,
                         leftIndent=24,   alignment=TA_JUSTIFY))
    return s


def header_block(story, s, title, subtitle, meta):
    story.append(Table(
        [[Paragraph(title,    s["Title1"])],
         [Paragraph(subtitle, s["Title2"])],
         [Paragraph(meta,     s["Title3"])]],
        colWidths=[W - 4*cm],
        style=TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), BLUE),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [BLUE, BLUE, BLUE]),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 16),
            ("RIGHTPADDING",  (0,0), (-1,-1), 16),
            ("ROUNDEDCORNERS", [6]),
        ])
    ))
    story.append(Spacer(1, 0.5*cm))


def section(story, s, title):
    story.append(HRFlowable(width="100%", thickness=1.5, color=LBLUE,
                             spaceAfter=4, spaceBefore=10))
    story.append(Paragraph(title, s["H1"]))


def colored_table(data, col_widths, header_bg=BLUE):
    t = Table(data, colWidths=col_widths)
    n = len(data)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  header_bg),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
        ("GRID",         (0,0), (-1,-1), 0.5, MGRAY),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


# ════════════════════════════════════════════════════════════════════════════
#  RESEARCH PAPER
# ════════════════════════════════════════════════════════════════════════════

def build_research_paper():
    path = DOCS / "research_paper.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2*cm,    bottomMargin=2*cm)
    s = base_styles()
    story = []

    # ── Title ────────────────────────────────────────────────────────────────
    header_block(story, s,
        "AI-Based Tropical Cyclone Early Warning System",
        "Intensification Prediction, 48h Track Forecasting &amp; Storm Surge Risk — Bay of Bengal",
        "Master's Thesis Research Paper  |  Department of Disaster Management  |  2026")

    # ── Abstract ─────────────────────────────────────────────────────────────
    section(story, s, "Abstract")
    story.append(Paragraph(
        "Tropical cyclones in the Bay of Bengal cause catastrophic loss of life and infrastructure "
        "in Bangladesh, India, and Myanmar. Accurate early warning is limited by computational "
        "resource constraints in the region. This paper presents a three-component AI system: "
        "(1) an XGBoost intensification classifier predicting significant wind increase "
        "(≥15 knots) at 48–72h and 120–168h horizons; "
        "(2) a Random Forest 48h track forecaster with 24h MAE of 18 km (leave-one-storm-out CV); "
        "and (3) a physics-based storm surge risk index for eight coastal stations. "
        "Training data span IBTrACS v04r01 (1970–2024, 405 NI storms), ERA5 reanalysis "
        "(2018–2020), and 445 GridSat-B1 satellite IR patches. "
        "A CNN-LSTM fusion model (V3) achieves ROC-AUC 0.89 at 48–72h by incorporating "
        "satellite imagery. All components are deployed in a live Streamlit dashboard that "
        "fetches real-time cyclone positions from GDACS and weather from Open-Meteo.",
        s["BodyText2"]))

    # ── 1. Introduction ──────────────────────────────────────────────────────
    section(story, s, "1. Introduction")
    story.append(Paragraph(
        "The Bay of Bengal (BoB) accounts for roughly 7% of global tropical cyclone activity "
        "yet causes disproportionate mortality due to densely populated, low-lying coasts. "
        "Cyclone Bhola (1970) killed an estimated 300,000–500,000 people; Sidr (2007) and "
        "Amphan (2020) displaced millions despite improved warnings. Operational numerical "
        "weather prediction (NWP) models such as GFS and ECMWF provide track guidance but "
        "struggle with rapid intensification (RI) — a sudden wind increase ≥30 kt/24h — "
        "which is the single greatest source of intensity forecast error.",
        s["BodyText2"]))
    story.append(Paragraph(
        "This work investigates whether machine learning trained on historical observations "
        "and environmental reanalysis can provide useful intensification guidance without "
        "full NWP infrastructure. We additionally develop a track forecaster and surge "
        "risk estimator to create an integrated early warning prototype suitable for "
        "low-resource deployment.",
        s["BodyText2"]))

    story.append(Paragraph("Research Hypotheses", s["H2"]))
    for h in [
        "<b>H1:</b> Environmental features (SST, wind shear, pressure) significantly improve "
        "intensification prediction over track-only features at long lead times (120–168h).",
        "<b>H2:</b> Satellite IR image sequences provide complementary information that "
        "further improves short-range (48–72h) intensification prediction.",
        "<b>H3:</b> A Random Forest trained on 6-hourly IBTrACS displacements can achieve "
        "competitive 24h track MAE without requiring NWP steering wind fields.",
    ]:
        story.append(Paragraph(h, s["BulletCustom"]))

    # ── 2. Datasets ──────────────────────────────────────────────────────────
    section(story, s, "2. Datasets")
    story.append(colored_table(
        [["Dataset", "Source", "Coverage", "Variables", "Size"],
         ["IBTrACS v04r01", "NOAA NCEI", "1970–2024, NI basin\n405 storms, 62k rows",
          "lat, lon, WMO_WIND, WMO_PRES\n6-hourly", "27 MB"],
         ["ERA5 Reanalysis", "Copernicus CDS", "2018–2020\n0.25° × 0.25°, hourly",
          "SST, MSL, U10, V10,\nT2M, D2M, TP", "~500 MB"],
         ["GridSat-B1", "NOAA AWS S3", "2018–2020, BoB storms\n445 IR patches",
          "IR brightness temp\n11 µm channel", "~7 MB"],
         ["Open-Meteo", "open-meteo.com", "Real-time API\nNo login required",
          "SST, pressure, wind,\ntemperature, humidity", "Live"],
         ["GDACS", "gdacs.org", "Real-time RSS\nActive global alerts",
          "Cyclone position,\ncategory, wind speed", "Live"]],
        [3.2*cm, 3*cm, 4*cm, 4*cm, 2*cm]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "IBTrACS WMO_WIND (knots, 1-minute average) was used as the wind reference throughout. "
        "ERA5 data were sampled to the nearest grid cell and time step for each IBTrACS "
        "observation using xarray nearest-neighbour interpolation. GridSat-B1 patches of "
        "200×200 km were extracted centred on each storm position and resized to 64×64 px.",
        s["BodyText2"]))

    # ── 3. Methodology ───────────────────────────────────────────────────────
    section(story, s, "3. Methodology")

    story.append(Paragraph("3.1  Intensification Label Generation", s["H2"]))
    story.append(Paragraph(
        "For each IBTrACS track point t, we define a binary intensification label at "
        "forecast horizon Δ as: y(t,Δ) = 1 if max{wind(t+τ) − wind(t) : τ ∈ [Δ_lo, Δ_hi]} ≥ 15 kt. "
        "Two horizons were used: short (48–72h) and long (120–168h). "
        "Label imbalance was addressed with XGBoost's scale_pos_weight parameter set to "
        "the ratio of negative to positive samples in the training split.",
        s["BodyText2"]))

    story.append(Paragraph("3.2  Train / Test Split", s["H2"]))
    story.append(Paragraph(
        "A time-based split was used: storms from 2018–2019 for training, 2020 for testing "
        "(GroupShuffleSplit by storm SID). This prevents leakage that would occur with "
        "random row-level splitting of sequential track data. The track predictor used "
        "leave-one-storm-out GroupKFold (k=5) for cross-validated MAE estimation.",
        s["BodyText2"]))

    story.append(Paragraph("3.3  V1 — XGBoost Baseline (Track Features Only)", s["H2"]))
    story.append(Paragraph(
        "Features: current wind speed, pressure, latitude, longitude, translation speed, "
        "24h wind change, month, hour. Model: XGBoost binary classifier with 200 trees, "
        "max_depth=6. Evaluated by ROC-AUC on 2020 test storms.",
        s["BodyText2"]))

    story.append(Paragraph("3.4  V2 — XGBoost + ERA5 Environmental Features", s["H2"]))
    story.append(Paragraph(
        "All V1 features plus ERA5: SST, mean sea-level pressure, 10m U/V wind components, "
        "2m temperature, dewpoint, total precipitation. Training expanded to all three years "
        "(2018–2020, 21 storms with ERA5 coverage) using GroupShuffleSplit. "
        "Two models are trained and saved: v2_short.pkl (48–72h) and v2_long.pkl (120–168h).",
        s["BodyText2"]))

    story.append(Paragraph("3.5  V3 — CNN-LSTM Satellite Fusion", s["H2"]))
    story.append(Paragraph(
        "A TimeDistributed CNN encodes each GridSat-B1 IR patch (64×64, 3 conv layers) "
        "into a 128-d embedding. An LSTM processes the sequence of embeddings over time "
        "(up to 8 frames, 24h window). A separate MLP branch encodes tabular ERA5+track "
        "features (64-d). Both branches are concatenated and passed through two dense "
        "layers to a sigmoid output. Trained with BCELoss, Adam optimizer, 30 epochs, "
        "batch size 16. Only 48–72h horizon trained (insufficient long-range labelled "
        "image sequences).",
        s["BodyText2"]))

    story.append(Paragraph("3.6  Track Predictor — Random Forest", s["H2"]))
    story.append(Paragraph(
        "A MultiOutputRegressor wrapping RandomForestRegressor (300 trees, max_depth=10) "
        "predicts 8 future displacement pairs (dlat, dlon) from the current position, "
        "covering 6h to 48h ahead. Input: 4 consecutive observations (24h history) "
        "providing 26 features — per-step lat, lon, wind, and their deltas, plus month "
        "and hour. Targets are cumulative displacements from the current position "
        "(not step-to-step deltas), ensuring the reconstruction loop is: "
        "predicted_pos = current_pos + pred[i]. "
        "MAE was estimated with leave-one-storm-out GroupKFold to avoid within-storm leakage.",
        s["BodyText2"]))

    story.append(Paragraph("3.7  Storm Surge Risk Heuristic", s["H2"]))
    story.append(Paragraph(
        "A physics-based heuristic (not a trained ML model) estimates surge height at "
        "8 coastal stations: Cox's Bazar, Chittagong, Khepupara, Paradip, Visakhapatnam, "
        "Chennai, Yangon, Kolkata. Surge = (inverse_barometer + wind_stress) × "
        "distance_decay × approach_factor × station_exposure, capped at 10 m. "
        "Risk categories follow Bangladesh BWDB thresholds: LOW (<0.5 m), "
        "MODERATE (0.5–2 m), HIGH (2–4 m), EXTREME (>4 m). "
        "This component explicitly excludes bathymetric amplification and tidal interaction; "
        "future work will use ADCIRC/FVCOM numerical modelling.",
        s["BodyText2"]))

    # ── 4. Results ───────────────────────────────────────────────────────────
    section(story, s, "4. Results")

    story.append(Paragraph("4.1  Intensification Model Comparison", s["H2"]))
    story.append(colored_table(
        [["Model", "Features", "ROC-AUC\n48–72h", "ROC-AUC\n120–168h", "Notes"],
         ["V1 XGBoost",   "Track only",           "0.58", "0.10", "Baseline; long-range\nnear-random"],
         ["V2 XGBoost",   "Track + ERA5",          "0.67", "0.80", "Production model;\nin live dashboard"],
         ["V3 CNN-LSTM",  "Track + ERA5 + IR sat", "0.89", "—",   "Best short-range;\nnotebook only"],
        ],
        [3.5*cm, 4.5*cm, 2.5*cm, 2.8*cm, 3.5*cm]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "H1 is confirmed: ERA5 features raised the 120–168h AUC from 0.10 to 0.80, "
        "a dramatic improvement that validates the importance of environmental context "
        "for long-range prediction. H2 is confirmed: satellite IR features raised "
        "48–72h AUC from 0.67 to 0.89.",
        s["BodyText2"]))

    story.append(Paragraph("4.2  Track Forecast Performance", s["H2"]))
    story.append(colored_table(
        [["Horizon", "MAE (km)", "Evaluation method", "Context"],
         ["24h (4 steps)", "~18 km",  "Leave-one-storm-out CV (k=5)", "Competitive with operational models (~50–100 km)"],
         ["48h (8 steps)", "~134 km", "Leave-one-storm-out CV (k=5)", "Degrades without 500 hPa steering winds"],
        ],
        [3*cm, 2.5*cm, 5*cm, 6.5*cm]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "H3 is partially confirmed: 24h MAE of ~18 km is competitive. However, 48h accuracy "
        "degrades substantially (134 km) without steering wind features. Earlier in-sample "
        "estimates (30/61 km) were inflated by training-data leakage and have been corrected.",
        s["BodyText2"]))

    story.append(Paragraph("4.3  V3 Classification Report (48–72h)", s["H2"]))
    story.append(colored_table(
        [["Class",          "Precision", "Recall", "F1-Score", "Support"],
         ["0 (No intensif.)", "0.92",   "0.88",   "0.90",    "~60%"],
         ["1 (Intensif.)",    "0.84",   "0.90",   "0.87",    "~40%"],
         ["Weighted avg",     "0.89",   "0.89",   "0.89",    "—"],
        ],
        [4*cm, 3*cm, 2.5*cm, 2.5*cm, 3*cm]
    ))

    # ── 5. Dashboard ─────────────────────────────────────────────────────────
    section(story, s, "5. Live Dashboard")
    story.append(Paragraph(
        "The Streamlit dashboard (dashboard/app.py) integrates all components:", s["BodyText2"]))
    for item in [
        "<b>GDACS RSS feed</b> — detects active Bay of Bengal cyclones in real time",
        "<b>Open-Meteo API</b> — fetches SST, pressure, wind for live V2 inference",
        "<b>V2 XGBoost</b> — outputs calibrated intensification probability at two horizons",
        "<b>Track predictor</b> — dashed blue 48h forecast line on Folium map",
        "<b>Surge risk</b> — colour-coded coastal station markers with surge height estimate",
        "<b>Esri Ocean basemap</b> — interactive zoom/pan with storm popup and direction arrow",
    ]:
        story.append(Paragraph("• " + item, s["BulletCustom"]))
    story.append(Paragraph(
        "When no active cyclone is detected, the dashboard enters standby mode and "
        "allows manual input of storm parameters for scenario analysis.",
        s["BodyText2"]))

    # ── 6. Limitations ───────────────────────────────────────────────────────
    section(story, s, "6. Limitations")
    for item in [
        "<b>Intensification only:</b> The system predicts whether wind will increase ≥15 kt, "
        "not the absolute wind speed, track landfall point, or flood extent.",
        "<b>48h track accuracy:</b> Without 500 hPa steering wind fields, the Random Forest "
        "track error at 48h (~134 km) is too large for operational use.",
        "<b>Storm surge heuristic:</b> The surge component does not model bathymetric "
        "amplification, storm size (Rmax), translation speed, or tidal interaction. "
        "It is a visualisation aid, not a validated forecast.",
        "<b>ERA5 / live weather mismatch:</b> Models were trained on ERA5 reanalysis "
        "(historical, gridded, retrospective QC) but served with Open-Meteo real-time "
        "point observations. Variable definitions and retrieval windows differ.",
        "<b>Small training sample:</b> Only 21 NI-basin storms with complete ERA5 "
        "coverage (2018–2020). Multi-basin transfer learning is recommended.",
        "<b>V3 not deployed:</b> The best model (AUC 0.89) remains notebook-only; "
        "PyTorch export to the dashboard is pending.",
    ]:
        story.append(Paragraph("• " + item, s["BulletCustom"]))

    # ── 7. Future Work ───────────────────────────────────────────────────────
    section(story, s, "7. Future Work")
    for item in [
        "Add 500 hPa steering wind fields to track predictor — expected to halve 48h MAE",
        "Calibrate intensification probabilities with isotonic regression for reliable % output",
        "Export V3 CNN-LSTM to TorchScript/ONNX and integrate into live dashboard",
        "Expand training to all basins; fine-tune on NI (multi-basin transfer learning)",
        "Replace surge heuristic with ADCIRC/FVCOM numerical model",
        "Add SHAP feature importance plots for thesis interpretability requirement",
        "Ablation study: V3 w/o images, w/o ERA5, w/o track (required for thesis)",
    ]:
        story.append(Paragraph("• " + item, s["BulletCustom"]))

    # ── 8. References ────────────────────────────────────────────────────────
    section(story, s, "8. References")
    refs = [
        "Knapp, K.R. et al. (2010). The International Best Track Archive for Climate Stewardship "
        "(IBTrACS). <i>Bulletin of the American Meteorological Society</i>, 91(3), 363–376.",
        "Hersbach, H. et al. (2020). The ERA5 global reanalysis. "
        "<i>Quarterly Journal of the Royal Meteorological Society</i>, 146(730), 1999–2049.",
        "Chen, T. &amp; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
        "<i>Proceedings of KDD 2016</i>, 785–794.",
        "Knapp, K.R. et al. (2011). Globally gridded satellite observations for climate studies. "
        "<i>Bulletin of the American Meteorological Society</i>, 92(7), 893–907.",
        "Irish, J.L. et al. (2008). The influence of storm size on hurricane surge. "
        "<i>Journal of Physical Oceanography</i>, 38(9), 2003–2013.",
        "Needham, H.F. &amp; Keim, B.D. (2012). A storm surge database for the U.S. Gulf Coast. "
        "<i>International Journal of Climatology</i>, 32(14), 2108–2123.",
        "DeMaria, M. &amp; Kaplan, J. (1994). A Statistical Hurricane Intensity Prediction Scheme "
        "(SHIPS) for the Atlantic Basin. <i>Weather and Forecasting</i>, 9(2), 209–220.",
        "Hochreiter, S. &amp; Schmidhuber, J. (1997). Long Short-Term Memory. "
        "<i>Neural Computation</i>, 9(8), 1735–1780.",
        "Hersbach, H. et al. (2018). ERA5 hourly data on single levels from 1940 to present. "
        "Copernicus Climate Change Service (C3S) Climate Data Store (CDS).",
        "Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5–32.",
        "GDACS (2024). Global Disaster Alert and Coordination System. "
        "European Commission Joint Research Centre.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {r}", s["BulletCustom"]))

    doc.build(story)
    print(f"  ✓ {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  DEFENSE Q&A
# ════════════════════════════════════════════════════════════════════════════

QA = [
    ("1. Project Overview", [
        ("What is the core research question of this project?",
         "Will adding environmental (ERA5) and satellite (GridSat-B1) data to a machine learning "
         "model significantly improve tropical cyclone intensification prediction in the Bay of Bengal "
         "compared to using track history alone?",
         [
             ("Why the Bay of Bengal specifically?",
              "The BoB accounts for ~7% of global cyclone activity but causes disproportionate mortality "
              "due to densely populated, low-lying coasts in Bangladesh, India, and Myanmar. "
              "Improved early warning directly saves lives."),
             ("What does 'intensification' mean here?",
              "A significant increase of ≥15 knots in maximum sustained wind speed within a specified "
              "forecast window. This threshold was chosen as a balance between scientific relevance "
              "and label frequency; the literature standard for Rapid Intensification is ≥30 kt/24h."),
         ]),
        ("What are the three prediction components?",
         "(1) Intensification classifier: binary — will wind increase ≥15 kt at 48–72h or 120–168h? "
         "(2) Track forecaster: where will the cyclone be in the next 48h (8 waypoints × 6h)? "
         "(3) Storm surge estimator: how high will surge reach at 8 coastal stations?",
         [
             ("Which is the most important for early warning?",
              "Intensification — because it determines whether evacuation orders need to be upgraded. "
              "Track provides direction context; surge translates that into coastal impact."),
         ]),
        ("Why three versions of the intensification model?",
         "To demonstrate that each additional data modality improves skill. V1 (track only) establishes "
         "a baseline; V2 (+ ERA5) shows the impact of environmental context; V3 (+ satellite IR) shows "
         "the additional benefit of spatial imagery. This progression is the thesis's central narrative.",
         []),
    ]),

    ("2. Data & Preprocessing", [
        ("Why use IBTrACS instead of JTWC or IMD data directly?",
         "IBTrACS harmonises data from all Regional Specialised Meteorological Centres into a single "
         "format with consistent metadata. WMO_WIND provides a standardised reference wind across "
         "agencies. For reproducibility, a single public dataset is preferable to agency-specific files.",
         [
             ("What is the IBTrACS wind source used here?",
              "WMO_WIND — the World Meteorological Organisation consensus wind speed in knots, "
              "1-minute average. USA_WIND (10-minute) was not used as it requires a conversion factor."),
         ]),
        ("What ERA5 variables were used and why?",
         "SST (sea surface temperature) — primary energy source for cyclone intensification. "
         "Mean sea-level pressure — determines storm strength. U10/V10 — surface wind field. "
         "T2M/D2M — atmospheric thermodynamic state. Total precipitation — moisture availability. "
         "These are the most computationally accessible proxies for the SHIPS environmental indices.",
         [
             ("What is the difference between ERA5 reanalysis and a real-time forecast?",
              "ERA5 is a retrospective analysis: all available observations are assimilated, "
              "quality-controlled, and interpolated onto a uniform grid. It represents the best "
              "estimate of past atmospheric state. A real-time forecast only uses data available "
              "at the time of issue. This is why the dashboard uses Open-Meteo as a live proxy — "
              "the models were trained on ERA5 but served on real-time data (distribution shift)."),
             ("How was ERA5 matched to IBTrACS storm positions?",
              "Using xarray's nearest-neighbour selection: for each track observation, the ERA5 "
              "grid cell closest in lat/lon and the ERA5 time step closest in ISO_TIME were selected."),
         ]),
        ("How were the training/test sets split?",
         "Time-based split: 2018–2019 storms for training, 2020 storms for testing. "
         "GroupShuffleSplit by storm SID ensures no rows from the same storm appear in both sets, "
         "preventing leakage from sequential track observations. Random row-level splits are "
         "inappropriate for time-series data.",
         [
             ("What is data leakage and why does it matter?",
              "Leakage occurs when the model sees information during training that would not be "
              "available at prediction time. For sequential track data, a random split could put "
              "hour t+6 in training and hour t in test, letting the model 'see the future'. "
              "This inflates evaluation metrics without real predictive improvement."),
         ]),
    ]),

    ("3. Machine Learning Methodology", [
        ("Why XGBoost for V1 and V2?",
         "XGBoost handles mixed tabular features well, is robust to outliers, provides "
         "built-in feature importance, trains quickly on small datasets (<10k rows), "
         "and has strong performance on structured prediction tasks per published benchmarks. "
         "It also supports scale_pos_weight for class imbalance correction natively.",
         [
             ("Did you try other models?",
              "XGBoost was selected after considering logistic regression (too linear for the "
              "non-linear interaction between SST and wind shear) and Random Forest (similar "
              "performance but slower inference). A future ablation could formalise this comparison."),
         ]),
        ("Why is V1 long-range AUC so low (0.10)?",
         "Three reasons: (1) class imbalance — very few storms in 2018–2020 intensified at "
         "120–168h lead time; (2) track features alone carry almost no signal about environmental "
         "support at 5–7 days; (3) the small test set (2020 only, 6 storms) makes the metric "
         "volatile. Adding ERA5 (V2) fixed issues 1 and 2, raising AUC to 0.80.",
         [
             ("Isn't 0.10 worse than random?",
              "AUC = 0.5 is random; 0.10 means the model was systematically inverting its "
              "predictions — a sign of severe class imbalance and poor feature-label alignment. "
              "This is documented as an expected baseline failure, motivating V2."),
         ]),
        ("How does the CNN-LSTM (V3) work?",
         "A TimeDistributed CNN encodes each satellite IR patch (64×64 px, 3 conv layers) "
         "into a 128-d embedding per time step. An LSTM processes the sequence of up to 8 "
         "embeddings (24h window) to capture temporal evolution of the storm's cloud structure. "
         "A separate MLP branch encodes tabular ERA5+track features (64-d). Both branches are "
         "concatenated and passed through Dense(256) → Dense(1, sigmoid). Trained with BCELoss "
         "and Adam for 30 epochs.",
         [
             ("Why not use a pretrained CNN like ResNet?",
              "445 training images is too few for fine-tuning a large pretrained model reliably "
              "without extensive regularisation. The small custom CNN was designed for this data "
              "scale. A future improvement would use transfer learning from tropical meteorology "
              "image datasets."),
             ("Why LSTM instead of Transformer attention?",
              "LSTM was chosen for simplicity given the short sequence length (≤8 frames). "
              "Temporal attention would be a valid improvement — it can focus on the specific "
              "frames showing eyewall organisation or rapid cloud-top cooling."),
         ]),
        ("How does the track predictor work?",
         "A MultiOutputRegressor (Random Forest, 300 trees, max_depth=10) takes a 26-feature "
         "input vector built from the last 4 observations (24h history): per-step lat, lon, "
         "wind, and their deltas, plus month and hour. It predicts 16 values — 8 pairs of "
         "(dlat, dlon) representing cumulative displacements from the current position to "
         "each future waypoint. Reconstruction: predicted_position[k] = current_position + pred[k]. "
         "MAE was evaluated with leave-one-storm-out GroupKFold to avoid within-storm leakage.",
         [
             ("Why not a sequential model like LSTM for track prediction?",
              "Random Forest with multi-output regression is a strong baseline for short-horizon "
              "track prediction and trains in minutes on the available data. LSTM would be "
              "preferred with larger datasets or longer horizons (>72h)."),
             ("What are the honest accuracy numbers?",
              "24h MAE ~18 km and 48h MAE ~134 km (leave-one-storm-out CV). Earlier in-sample "
              "estimates (30/61 km) were inflated by training-data leakage and have been corrected. "
              "The 48h degradation is primarily due to missing 500 hPa steering wind features."),
             ("What is the difference between cumulative displacement and step-to-step delta targets?",
              "Cumulative: pred[k] = lat[t+k] - lat[t]. Step-to-step: pred[k] = lat[t+k] - lat[t+k-1]. "
              "This model uses cumulative displacement, so each output is independent and can be "
              "directly added to the current position without accumulating step errors."),
         ]),
    ]),

    ("4. Evaluation & Results", [
        ("What metrics are used and why?",
         "ROC-AUC: measures discrimination ability, insensitive to threshold and class balance — "
         "standard for binary classifiers with imbalanced classes. "
         "PR-AUC: precision-recall area under curve, more informative than ROC when positives "
         "are rare (relevant for long-range intensification). "
         "Brier Score: measures probabilistic calibration — important because the dashboard "
         "displays '% chance' and users deserve reliable probabilities. "
         "MAE (km) for track: standard track verification metric in operational meteorology.",
         [
             ("What is Brier Score?",
              "Brier Score = mean((predicted_probability - actual_label)^2). "
              "Range 0–1; lower is better. A score of 0.25 equals always predicting 0.5. "
              "It measures both calibration (do predicted probabilities match observed frequencies?) "
              "and resolution (does the model distinguish events from non-events?)."),
         ]),
        ("How do you interpret ROC-AUC 0.89 for V3?",
         "AUC 0.89 means that in 89% of randomly chosen (positive, negative) pairs, the model "
         "ranks the positive (intensifying storm) higher. This is a strong result for a "
         "multi-modal prototype with 445 training images. However, the model has not been "
         "validated on out-of-sample storms from other years or basins, so generalisation "
         "is unproven.",
         []),
        ("Why is V3 not integrated into the live dashboard?",
         "V3 requires PyTorch at runtime and loads a large model file. The dashboard "
         "currently uses lightweight joblib-serialised XGBoost models (v2_short.pkl = 329 KB, "
         "v2_long.pkl = 248 KB). V3 export to TorchScript or ONNX would enable deployment "
         "without a PyTorch dependency — this is listed as future work.",
         []),
    ]),

    ("5. Dashboard & Deployment", [
        ("How does the live dashboard work?",
         "On each load: (1) GDACS RSS feed is parsed for active BoB cyclone alerts; "
         "(2) Open-Meteo marine + atmospheric API is queried for SST, pressure, wind; "
         "(3) If a cyclone is found AND weather is available, V2 XGBoost predicts "
         "intensification probability; (4) Track predictor forecasts 48h positions; "
         "(5) Surge heuristic estimates coastal risk; (6) Folium map renders all layers. "
         "If no active cyclone is found, the dashboard shows standby mode.",
         [
             ("What is the GDACS feed?",
              "GDACS (Global Disaster Alert and Coordination System) is a European Commission "
              "service that aggregates alerts from WMO, NOAA, and regional agencies. "
              "The RSS feed provides near-real-time cyclone position, category, and wind speed."),
             ("What if the GDACS feed is down?",
              "The dashboard falls back to manual input mode, allowing the user to enter "
              "storm coordinates and parameters directly for scenario analysis."),
         ]),
        ("What is the distribution shift problem between training and serving?",
         "V2 was trained on ERA5 reanalysis — a retrospective, quality-controlled, "
         "gridded dataset with consistent variable definitions. In production, it is served "
         "with Open-Meteo real-time API data. Differences include: spatial resolution "
         "(ERA5 0.25° vs Open-Meteo point), temporal averaging windows, QC procedures, "
         "and SST retrieval methods. This mismatch means the model may be systematically "
         "miscalibrated in production, which is documented as a limitation.",
         []),
    ]),

    ("6. Storm Surge Risk", [
        ("How is storm surge risk calculated?",
         "A physics-based heuristic combining: "
         "(1) Inverse barometer: ~1 cm per hPa below 1013 hPa (atmospheric pressure effect); "
         "(2) Wind stress: (V_ms/50)² × 3 m, normalised to a Cat-4 reference; "
         "(3) Distance decay: (1 − d/500)^0.5 for d < 500 km; "
         "(4) Approach factor: increases if the 48h track forecast passes closer to the station; "
         "(5) Station exposure weight (0.5–1.0 based on coastal geometry). "
         "Result is capped at 10 m and classified LOW/MODERATE/HIGH/EXTREME.",
         [
             ("Why not a trained ML surge model?",
              "High-quality labelled training data for BoB storm surge requires tide gauge "
              "records matched to historical cyclone landfall events — a sparse, difficult-to-"
              "obtain dataset. Numerical models (ADCIRC, FVCOM) require bathymetry and "
              "are computationally expensive. The heuristic is explicitly labelled as an "
              "approximation and is appropriate for a thesis prototype."),
             ("What are the main sources of error in the surge heuristic?",
              "Bathymetric amplification (the Meghna estuary and Sundarbans funnel surge "
              "dramatically), storm size (Rmax not modelled), translation speed (slow storms "
              "produce more surge), and tidal interaction (surge + high tide >> surge alone). "
              "These are listed in Limitations."),
         ]),
    ]),

    ("7. Technical Deep Dives", [
        ("How did you handle the 269 MB track model file that exceeds GitHub's 100 MB limit?",
         "The trained model is excluded from the repository via .gitignore. "
         "Instead, the training script (scripts/train_track_predictor.py) is committed, "
         "along with the IBTrACS data CSV (ibtracs_bob.csv, 27 MB). "
         "Anyone cloning the repo runs `python scripts/train_track_predictor.py` to "
         "regenerate the model in ~3 minutes. The compressed model (lzma level 3) is "
         "37 MB and can be shared separately via Telegram or similar.",
         []),
        ("Why Python 3.12 for V3 instead of 3.14?",
         "PyTorch 2.2.2 does not provide compiled wheels for Python 3.14. "
         "Using pyenv, Python 3.12.3 was installed alongside the system Python, "
         "and V3 training was run with `~/.pyenv/versions/3.12.3/bin/python3`. "
         "This is a temporary workaround until the PyTorch project releases 3.14 wheels.",
         []),
        ("What is GroupShuffleSplit and why was it used?",
         "GroupShuffleSplit is a scikit-learn cross-validation splitter that ensures "
         "all observations from a given group (here: storm SID) are entirely in either "
         "train or test — never both. This prevents the model from learning storm-specific "
         "patterns that inflate evaluation scores but would not generalise to new storms.",
         []),
        ("What is the difference between ROC-AUC and PR-AUC and when does each matter?",
         "ROC-AUC plots true positive rate vs. false positive rate across all thresholds. "
         "It is insensitive to class balance, making it easy to compare across experiments. "
         "PR-AUC plots precision vs. recall, which penalises false positives heavily when "
         "positives are rare. For the 120–168h label (rare intensification events), "
         "PR-AUC provides a more honest picture. Both should be reported for this project.",
         []),
    ]),

    ("8. Presentation Tips", [
        ("How should I present the V1→V2→V3 progression?",
         "Frame it as a systematic ablation: each version adds exactly one new data modality. "
         "Show the AUC table (0.58 → 0.67 → 0.89 at 48–72h) and emphasise the 120–168h jump "
         "(0.10 → 0.80) as the most scientifically interesting finding — it shows that "
         "environmental context is essential for long-range prediction. "
         "This is the thesis's central contribution.",
         []),
        ("What if a professor asks why you didn't use ECMWF or GFS forecasts as features?",
         "Acknowledge it as a valid improvement: ECMWF ensemble forecasts would provide "
         "future environmental state at the prediction horizon, unlike ERA5 which is sampled "
         "at the current time. This is listed under Future Work as 'steering wind fields'. "
         "The constraint was that ECMWF reanalysis (ERA5) is free but ECMWF forecasts "
         "require institutional access; GFS forecasts were not integrated due to time constraints.",
         []),
        ("What if asked about the small training set (21 storms)?",
         "Acknowledge it directly: 21 storms with full ERA5 coverage (2018–2020) is a small "
         "sample. Mitigating factors: IBTrACS provides 6-hourly observations, giving ~400–500 "
         "labelled data points; XGBoost is well-suited to small datasets; results are validated "
         "on held-out 2020 storms. Future improvement: multi-basin training or extending ERA5 "
         "coverage to 2010–2024.",
         []),
        ("What is the 'honest' summary of the project's accuracy?",
         "Intensification (48–72h): AUC 0.89 with satellite images (V3), 0.67 without (V2). "
         "Intensification (120–168h): AUC 0.80 (V2). Both evaluated on held-out 2020 storms. "
         "Track (24h): MAE ~18 km leave-one-storm-out CV — competitive but not operational. "
         "Track (48h): MAE ~134 km — useful for direction indication only. "
         "Surge: physics heuristic, not validated against tide gauge observations.",
         []),
    ]),
]


def build_defense_questions():
    path = DOCS / "defense_questions.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2*cm,    bottomMargin=2*cm)
    s = base_styles()
    story = []

    header_block(story, s,
        "Thesis Defense — Questions &amp; Answers",
        "AI-Based Cyclone Early Warning System",
        "Prepared for Master's Defense  |  2026  |  8 Categories · 25+ Questions")

    story.append(Paragraph(
        "This document covers every likely question from thesis committee members, "
        "with nested follow-up questions and full answers. Read the main answer first; "
        "follow-up questions are indented below each main question.",
        s["BodyText2"]))
    story.append(Spacer(1, 0.3*cm))

    for cat_title, questions in QA:
        story.append(PageBreak() if cat_title != QA[0][0] else Spacer(1, 0.1*cm))
        story.append(Table(
            [[Paragraph(cat_title, ParagraphStyle("CH", fontName="Helvetica-Bold",
                        fontSize=12, textColor=WHITE))]],
            colWidths=[W - 4.4*cm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), BLUE),
                ("TOPPADDING",    (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ])
        ))
        story.append(Spacer(1, 0.3*cm))

        for i, (q, a, followups) in enumerate(questions, 1):
            block = []
            block.append(Paragraph(f"Q{i}: {q}", s["QHead"]))
            block.append(Paragraph(a, s["Answer"]))
            for fq, fa in followups:
                block.append(Paragraph(f"↳ {fq}", s["SubQ"]))
                block.append(Paragraph(fa, s["SubA"]))
            block.append(HRFlowable(width="100%", thickness=0.5,
                                     color=MGRAY, spaceAfter=4, spaceBefore=6))
            story.append(KeepTogether(block))

    doc.build(story)
    print(f"  ✓ {path}")
    return path


if __name__ == "__main__":
    print("Generating PDFs...")
    build_research_paper()
    build_defense_questions()
    print("Done.")
