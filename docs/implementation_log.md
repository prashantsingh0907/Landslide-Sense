# Implementation Log & Engineering Record
**Project:** LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System  
**Institution:** Galgotias University, Greater Noida  
**Milestone:** 7th Semester Mid-Term Evaluation (MTE) / Interim Project Report-I (IPR-I)

---

## 1. Environment & Dependency Provisioning

| Component | Target Specification | Actual Installed Version | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 AMD64 | Windows 11 (NT kernel) | Verified host platform |
| **Python Runtime** | Python 3.11+ | Python 3.14.0 | Supported prebuilt wheels |
| **Web Framework** | Streamlit | Streamlit 1.65.0 | Tested on port 8501 (HTTP 200) |
| **Data Processing** | pandas / numpy | pandas 3.0.6, numpy 2.5.4 | Active |
| **Machine Learning** | scikit-learn | scikit-learn 1.9.1 | Active |
| **Visualization** | Plotly | Plotly 7.1.0 | Active |
| **Model Persistence** | joblib | joblib 1.6.0 | Active |
| **Test Runner** | pytest | pytest 9.1.1 | 11 unit tests passing |

---

## 2. Chronological Engineering Milestones

### Step 1: Project Structuring & Data Templates
* Created modular repository layout: `src/`, `data/templates/`, `data/demo/`, `data/raw/`, `data/processed/`, `models/`, `reports/`, `docs/`, `tests/`.
* Created standard CSV schema template: `data/templates/landslide_training_template.csv` with core columns: `rainfall_24h`, `rainfall_3d`, `rainfall_7d`, `slope`, `elevation`, `soil_moisture`, `ndvi`, `land_cover`, `landslide`.
* Created synthetic demo dataset: `data/demo/synthetic_validation_sample.csv` (30 rows, clearly designated for pipeline verification).

### Step 2: Risk Assessment Engine (`src/risk_assessment.py`)
* Implemented `validate_environmental_inputs`:
  * Enforces physical non-negativity and domain bounds ($0 \le \beta \le 90^\circ$, $-500 \le \text{elevation} \le 9000$ m).
  * Validates temporal rainfall consistency ($24\text{h} \le 3\text{d} \le 7\text{d}$ cumulative sum).
  * Safely annotates missing optional inputs (`soil_moisture`, `ndvi`, `land_cover`) without fabricating values.
* Implemented `calculate_heuristic_risk` (Demonstration Mode):
  * Calibrated dynamic trigger index (max 40 pts) based on Caine & Guzzetti rainfall trigger envelopes.
  * Calibrated topographic susceptibility index (max 35 pts) peaking at critical colluvial gradient ($30^\circ-48^\circ$).
  * Calibrated surface conditioning index (max 25 pts) for moisture saturation, root cohesion (NDVI), and land cover.
  * Outputs normalized risk index (0–100), risk tier (Low, Medium, High), and precautionary advisories.

### Step 3: Data Loader & Schema Auditor (`src/data_loader.py`)
* Implemented `audit_dataset_schema`:
  * Verifies presence of binary target column `landslide` ($y \in \{0, 1\}$).
  * Audits missing cell counts and duplicate rows.
  * Audits class balance (requires minimum 2 samples per class to allow train/test partitioning).
  * Prevents target leakage by systematically stripping identifier columns (`location_id`, coordinates, dates) from predictor matrix $X$.

### Step 4: Leak-Free Preprocessing Pipeline (`src/preprocessing.py`)
* Constructed scikit-learn `ColumnTransformer`:
  * Numerical transformer: `SimpleImputer(strategy='median')` + `StandardScaler()`.
  * Categorical transformer: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
  * Ensures transformations are fitted strictly on training data splits.

### Step 5: Supervised Model Trainer (`src/train_model.py`)
* Implemented `train_landslide_model`:
  * Performs stratified train/test split.
  * Trains `RandomForestClassifier(n_estimators=100, class_weight='balanced')`.
  * Persists fitted pipeline, feature importances, and evaluation metrics into `models/landslide_random_forest.joblib`.
* Implemented `predict_with_ml_model` for inference with automatic imputation of missing optional parameters.

### Step 6: Rigorous Evaluation & Visualizations (`src/evaluation.py`)
* Implemented `evaluate_model`:
  * Computes Accuracy, Precision, Recall, F1-Score.
  * Computes ROC-AUC strictly when both positive and negative classes are present in the test partition.
* Implemented Plotly rendering routines: Confusion Matrix heatmap, ROC curve, and Feature Importance bar chart.

### Step 7: Streamlit Academic Web Application (`app.py`)
* Developed full academic interface with six dedicated navigation sections:
  1. 🏠 Home Dashboard
  2. ⚠️ Landslide Risk Assessment (Interactive Input Form)
  3. 🤖 Prediction Engine & Training (Dual-Mode Interface)
  4. 📊 Data Explorer & Distribution Analysis
  5. 📚 Data Sources & Acquisition Audit
  6. 📖 About & Geotechnical Methodology
* Embedded mandatory academic prototype disclaimers across all pages.
* Launched dev server at `http://localhost:8501`.

---

## 3. Verification & Test Suite Execution

Executed comprehensive automated test suite via pytest:
```bash
python -m pytest tests/ -v
```

### Test Results Summary:
* `tests/test_data_loader.py::test_load_dataset_template` — **PASSED**
* `tests/test_data_loader.py::test_audit_schema_valid_synthetic_dataset` — **PASSED**
* `tests/test_data_loader.py::test_audit_missing_target` — **PASSED**
* `tests/test_data_loader.py::test_audit_non_binary_target` — **PASSED**
* `tests/test_data_loader.py::test_prepare_training_features_leakage_prevention` — **PASSED**
* `tests/test_risk_assessment.py::test_input_validation_valid` — **PASSED**
* `tests/test_risk_assessment.py::test_input_validation_inconsistent_rainfall` — **PASSED**
* `tests/test_risk_assessment.py::test_input_validation_out_of_bounds_slope` — **PASSED**
* `tests/test_risk_assessment.py::test_heuristic_calculation_missing_optional_factors` — **PASSED**
* `tests/test_risk_assessment.py::test_heuristic_extreme_conditions` — **PASSED**
* `tests/test_risk_assessment.py::test_end_to_end_ml_training_and_inference` — **PASSED**

**Overall Outcome:** 11 passed in 10.86 seconds (100% success rate, 0 failures, 0 warnings).
