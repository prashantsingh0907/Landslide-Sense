# LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Tests Status](https://img.shields.io/badge/Tests-11%20Passed%20(100%25)-brightgreen.svg)]()
[![Academic Status](https://img.shields.io/badge/Academic%20Milestone-7th%20Sem%20MTE%20%2F%20IPR--I-indigo.svg)]()

> **ACADEMIC EVALUATION PROTOTYPE NOTICE:**  
> Developed for the **7th-Semester Mid-Term Evaluation (MTE)** and **Interim Project Report-I (IPR-I)** for the B.Tech Computer Science and Engineering program at **Galgotias University**, Greater Noida.  
> This software is an engineering research prototype designed to demonstrate supervised classification pipelines and transparent geotechnical risk scoring. It is **NOT** an operational emergency warning or civil defense dissemination system.

---

## 1. Project Overview

Rainfall-induced landslides represent a severe catastrophic threat across the Himalayas and Western Ghats of India. **LandslideSense** bridges the gap between static geological susceptibility and dynamic meteorological triggers by combining:
1. **Precipitation Telemetry:** Short-term intensity (24-hour rainfall) and multi-day cumulative antecedent saturation (3-day and 7-day rainfall).
2. **Topographic Geomorphology:** Slope angle (degrees) and terrain elevation (AMSL) derived from Digital Elevation Models (DEMs).
3. **Surface Conditioning:** Soil moisture saturation, vegetation root cohesion (NDVI), and land cover classification.
4. **Dual Prediction Architecture:**
   * **Mode 1 (Machine Learning):** A leak-free, scikit-learn supervised pipeline (`RandomForestClassifier`) with stratified splitting, schema validation, and mathematically sound metric evaluation.
   * **Mode 2 (Demonstration Mode):** A transparent, fully documented heuristic risk index based on empirical geotechnical thresholds (Caine 1980; Guzzetti et al. 2007) ensuring the application remains interactive and honest when unverified or incomplete field datasets are loaded.

---

## 2. Project Directory Structure

```
Landslide Sense/
├── app.py                             # Streamlit interactive web dashboard
├── requirements.txt                   # Core Python dependencies
├── README.md                          # Project documentation and guide
├── .gitignore                         # Version control ignore rules
├── data/
│   ├── raw/                           # Raw downloaded geodatabases / shapefiles (.gitkeep)
│   ├── processed/                     # Preprocessed tabular datasets (.gitkeep)
│   ├── templates/
│   │   └── landslide_training_template.csv   # Standard CSV schema template
│   └── demo/
│       └── synthetic_validation_sample.csv   # Synthetic verification dataset (30 rows)
├── src/
│   ├── __init__.py                    # Python package init
│   ├── data_loader.py                 # CSV ingestion, schema audit & leakage prevention
│   ├── preprocessing.py              # Leak-free ColumnTransformer pipeline
│   ├── risk_assessment.py             # Input bounds validation & heuristic scoring
│   ├── train_model.py                 # Stratified Random Forest training & persistence
│   └── evaluation.py                  # Confusion matrix, ROC-AUC & Plotly visualizers
├── models/
│   ├── .gitkeep                       # Directory placeholder
│   └── landslide_random_forest.joblib # Serialized fitted model bundle
├── reports/                           # Output evaluation reports (.gitkeep)
├── docs/
│   ├── IPR_I_progress_notes.md        # University IPR-I compliant report
│   ├── data_sources.md                # Official geoscientific data audit (GSI, ISRO, NASA, USGS)
│   ├── literature_review_notes.md     # Peer-reviewed papers with verified DOIs (IEEE format)
│   └── implementation_log.md          # Chronological engineering & test record
└── tests/
    ├── __init__.py
    ├── test_data_loader.py            # Unit tests for loader, schema audit, leakage check
    └── test_risk_assessment.py         # Unit tests for validation, heuristic & ML inference
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.11, 3.12, or 3.14 installed on your workstation.
* PowerShell or standard terminal.

### Step 1: Install Dependencies
Open a terminal in the project directory and run:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests
Verify all validation bounds, loader checks, and model training pipelines:
```bash
python -m pytest tests/ -v
```
*(All 11 tests should execute and report **PASSED**).*

### Step 3: Launch the Streamlit Web Dashboard
Run the following command to launch the local web server:
```bash
python -m streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 4. Key Application Features

### A. Home Dashboard
* Geotechnical problem framing (effective stress collapse and pore-water pressure).
* High-level architectural pipeline visualization.
* Milestone progress tracker for Galgotias University MTE / IPR-I.

### B. Interactive Landslide Risk Assessment
* Input fields for **24h, 3-day, and 7-day rainfall**, **slope gradient**, **elevation**, **soil moisture**, **NDVI**, and **land cover**.
* Dynamic physical input validation (detects invalid slopes $>90^\circ$ or inconsistent rainfall $24\text{h} > 3\text{d}$).
* Transparent handling of missing optional inputs without fabricating data.
* Visual Plotly radial gauge chart categorized into **Low**, **Medium**, and **High** risk.
* Physical factor contribution breakdown table explaining failure mechanisms.
* Tailored geotechnical precautionary advisories.

### C. Supervised ML Training Pipeline (Mode 1)
* Dataset schema audit verifying target column (`landslide` $\in \{0, 1\}$), missingness, and class balance.
* Stratified train/test split preventing distribution skew.
* Leak-free scikit-learn `ColumnTransformer` (median imputer + standard scaler + one-hot encoder).
* Feature importance bar chart and model serialization to `models/landslide_random_forest.joblib`.
* Mathematically guarded evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix).

### D. Transparent Heuristic Demonstration Engine (Mode 2)
* Operates seamlessly when no validated training dataset is available.
* Documented formula combining empirical rainfall trigger indices and topographic susceptibility.
* Clearly labeled output preventing confusion between heuristics and trained probabilities.

### E. Official Data Sources & Audit Explorer
* Thorough review of official geoscientific portals:
  * **Geological Survey of India (GSI)** Landslide Hazard Portal
  * **ISRO / NRSC** Landslide Atlas of India
  * **NASA Earth Data** Global Precipitation Measurement (GPM IMERG)
  * **USGS** Landslide Inventories across the United States
* Downloadable standard CSV training template (`data/templates/landslide_training_template.csv`).

---

## 5. Verification & Test Summary

Automated testing executed via `pytest`:
* `test_load_dataset_template` — PASSED
* `test_audit_schema_valid_synthetic_dataset` — PASSED
* `test_audit_missing_target` — PASSED
* `test_audit_non_binary_target` — PASSED
* `test_prepare_training_features_leakage_prevention` — PASSED
* `test_input_validation_valid` — PASSED
* `test_input_validation_inconsistent_rainfall` — PASSED
* `test_input_validation_out_of_bounds_slope` — PASSED
* `test_heuristic_calculation_missing_optional_factors` — PASSED
* `test_heuristic_extreme_conditions` — PASSED
* `test_end_to_end_ml_training_and_inference` — PASSED

**Result:** 11 passed in 10.86s (100% pass rate).

---

## 6. Academic Presentation Checklist for MTE

When presenting this prototype to your project supervisor and evaluators:
1. **Highlight Scientific Honesty:** Emphasize that the system implements a dual-mode engine—demonstrating a transparent heuristic fallback rather than training a fabricated model on invented samples.
2. **Demonstrate Input Boundary Checks:** In the Risk Assessment form, show how entering an impossible value (e.g., 24h rainfall higher than 3-day rainfall) triggers a physical inconsistency error.
3. **Demonstrate Both Risk Extremes:**
   * *Baseline Stable:* Rainfall 10mm, slope 15° $\rightarrow$ **Low Risk** (Green gauge).
   * *Critical Hazard:* Rainfall 110mm/24h, 220mm/3d, slope 42°, soil moisture 80% $\rightarrow$ **High Risk** (Red gauge + emergency warning).
4. **Demonstrate Leak-Free Model Training:** On the *Prediction Engine* page, train the Random Forest pipeline on the synthetic test set, view the confusion matrix, ROC curve, and feature importances.
5. **Explain the Real Data Acquisition Roadmap:** Open the *Data Sources* page and explain how GSI/ISRO spatial inventory polygons will be intersected with DEM and NASA GPM precipitation rasters for 8th-semester deployment.

---

## 7. License & Credits

Developed as part of the B.Tech Computer Science Capstone Project at Galgotias University.  
Academic research and educational prototype use only.
