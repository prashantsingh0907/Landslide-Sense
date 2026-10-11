# GALGOTIAS UNIVERSITY
## SCHOOL OF COMPUTING SCIENCE AND ENGINEERING
### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING

---

# INTERIM PROJECT REPORT - I (IPR - I)
### B.Tech 7th Semester Mid-Term Evaluation (MTE)

## **LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System**

---

### **Submitted By:**
* **Student Name:** [Student Name Placeholder]  
* **Admission / Roll Number:** [Roll Number Placeholder]  
* **Enrollment Number:** [Enrollment Number Placeholder]  
* **Program:** Bachelor of Technology in Computer Science and Engineering  
* **Semester:** 7th Semester (Academic Year 2026–2027)

### **Under the Guidance of:**
* **Project Supervisor:** [Guide Name / Designation Placeholder]  
* **Department:** Computer Science and Engineering  
* **Institution:** Galgotias University, Plot No. 2, Sector 17-A, Yamuna Expressway, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh - 203201

---

\newpage

## **PROGRESS APPROVAL FORM**

**Candidate Name:** [Student Name Placeholder]  
**Roll Number:** [Roll Number Placeholder]  
**Project Title:** LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System  
**Evaluation Stage:** Interim Project Report-I (IPR-I) / Mid-Term Evaluation (MTE)  

### **Supervisor Evaluation & Remarks:**

| Assessment Parameter | Maximum Marks | Marks Awarded | Remarks |
| :--- | :---: | :---: | :--- |
| **Problem Definition & Scope** | 10 | [ ] | |
| **Literature Survey & Citation Rigor** | 15 | [ ] | |
| **System Architecture & Methodology** | 15 | [ ] | |
| **Progress Achieved / Prototype Modules** | 30 | [ ] | |
| **Report Structure & University Compliance** | 15 | [ ] | |
| **Viva-Voce / Technical Presentation** | 15 | [ ] | |
| **TOTAL** | **100** | **[ ]** | |

**Guide Comments:**  
The student has presented the preliminary research foundation, problem definition, literature review, and an initial functional prototype featuring input validation, a transparent heuristic demonstration engine, and a leak-free supervised machine learning pipeline in Streamlit.

\vspace{1.5cm}

_____________________________  
**(Signature of Project Supervisor)**  
Name: [Guide Name Placeholder]  
Designation: [Assistant/Associate Professor]  
Department of Computer Science & Engineering  
Galgotias University  

Date: \_\_\_\_ / \_\_\_\_ / 2026  

\newpage

## **PROJECT SYNOPSIS**

### **1. Project Title**
**LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System**

### **2. Problem Statement**
Rainfall-triggered landslides pose severe catastrophic threats across mountainous regions of India, notably the Himalayas and Western Ghats. Millions of residents and critical transit corridors are routinely paralyzed during the monsoon season. Conventional hazard warning systems are constrained by:
1. **Coarse Spatial-Temporal Resolution:** Regional macro-zoning maps cannot evaluate local slope-catchment conditions.
2. **Conflation of Hazard Modes:** Existing systems frequently conflate static geomorphic susceptibility with instantaneous dynamic rainfall triggers.
3. **Black-Box Opacity:** Complex machine learning models rarely communicate actionable geotechnical reasoning to civil defense authorities.

### **3. Project Objectives**
* **Primary Objective:** Develop a modular, scientifically validated software prototype that ingests multi-temporal rainfall telemetry (24h, 3d, 7d) and terrain morphology (slope angle, elevation, soil moisture, vegetation cover) to deliver instantaneous landslide risk assessments and operational advisories.
* **Architecture Objective:** Establish a **dual-operating engine**:
  1. *Machine Learning Mode:* A leak-free, scikit-learn supervised learning pipeline (Random Forest) trained on validated, balanced tabular records with mathematically sound metric reporting.
  2. *Demonstration Mode:* A transparent, formulaic heuristic risk index calibrated against peer-reviewed empirical rainfall thresholds (Caine 1980; Guzzetti et al. 2007) ensuring utility when real sensor telemetry is unavailable.
* **Interface Objective:** Deliver an interactive, academic dashboard in Streamlit providing input validation, gauge visualizations, feature importance metrics, and geotechnical precautionary alerts.

### **4. Proposed Methodology**
The proposed system architecture is organized into four interconnected tiers:
1. **Data Ingestion & Integrity Auditing:** Safely parses user telemetry and CSV records, verifies binary label integrity ($y \in \{0, 1\}$), checks class balance, and strips identifier columns to prevent target leakage.
2. **Geotechnical Input Validation:** Enforces physical bounds (slope $0^\circ-90^\circ$, rainfall temporal consistency $24\text{h} \le 3\text{d} \le 7\text{d}$) and transparently flags missing variables without fabricating synthetic values.
3. **Dual Processing Engine:** Executes either the trained Random Forest pipeline or the transparent heuristic weighting formula.
4. **Advisory Generation:** Classifies risk into Low ($<35$), Medium ($35-70$), or High ($\ge 70$) tiers, rendering Plotly gauges, feature attribution charts, and civil defense advisories.

### **5. Software and Hardware Requirements**
* **Hardware:** Modern x86-64 / ARM workstation (Minimum 8 GB RAM, Quad-Core CPU, 500 MB storage).
* **Software Environment:** Python 3.11+, Streamlit 1.35+, scikit-learn 1.4+, pandas 2.2+, numpy 1.26+, Plotly 5.20+, joblib 1.3+, pytest 8.0+.

### **6. Current Milestone Achievements (IPR-I)**
* Complete implementation of modular core modules (`risk_assessment.py`, `data_loader.py`, `preprocessing.py`, `train_model.py`, `evaluation.py`).
* 100% automated test pass rate across 11 unit tests covering validation, heuristic bounds, and training workflows.
* Fully interactive Streamlit dashboard operational locally.
* Geoscientific data source audit identifying real acquisition pathways for GSI and ISRO spatial databases.

---

\newpage

## **TABLE OF CONTENTS**

* **Progress Approval Form**
* **Synopsis**
* **List of Figures**
* **List of Tables**
* **List of Symbols and Abbreviations**
* **Chapter 1: Introduction**
  * 1.1 Project Motivation & Societal Relevance
  * 1.2 Problem Statement
  * 1.3 Core Objectives & Scope
  * 1.4 Critical Limitations & Boundary Conditions
  * 1.5 Organization of the Report
* **Chapter 2: Background & Literature Review**
  * 2.1 Mechanics of Rainfall-Induced Hillslope Failure
  * 2.2 Evolution of Empirical Rainfall Thresholds
  * 2.3 Machine Learning in Geohazard Susceptibility Mapping
  * 2.4 Comparative Summary of Related Works
  * 2.5 Identified Research Gaps
* **Chapter 3: Proposed Methodology & System Architecture**
  * 3.1 Architectural Overview
  * 3.2 Geotechnical Input Validation Engine
  * 3.3 Leak-Free Preprocessing & Pipeline Construction
  * 3.4 Mode 1: Supervised Random Forest Classifier
  * 3.5 Mode 2: Transparent Heuristic Demonstration Index
* **Chapter 4: Implementation Progress & Verification**
  * 4.1 Implemented Modules & Code Structure
  * 4.2 Streamlit User Interface & Interactive Dashboards
  * 4.3 Automated Verification & Test Results
* **Chapter 5: Remaining Tasks & Final Semester Roadmap**
  * 5.1 Pending Data Engineering Tasks
  * 5.2 Algorithm Enhancements & Model Ensembles
  * 5.3 Schedule for IPR-II and Final Evaluation
* **References (IEEE Format)**

---

\newpage

## **LIST OF FIGURES**
* **Figure 1.1:** Conceptual Diagram of Pore-Water Pressure and Effective Normal Stress Collapse.
* **Figure 3.1:** High-Level Four-Tier System Architecture of LandslideSense.
* **Figure 3.2:** Leak-Free Preprocessing and Supervised Training Workflow.
* **Figure 4.1:** Streamlit Interactive Landslide Risk Assessment Form & Radial Gauge.
* **Figure 4.2:** Confusion Matrix and ROC Curve Diagnostic Heatmaps.

## **LIST OF TABLES**
* **Table 2.1:** Comparative Analysis of Related Works in Landslide Susceptibility and Warning.
* **Table 3.1:** Physical Variable Ranges and Boundary Validation Rules.
* **Table 3.2:** Heuristic Risk Index Parameter Weighting and Failure Mechanisms.
* **Table 4.1:** Automated Pytest Test Suite Execution Summary.
* **Table 5.1:** Work Breakdown Structure and Timeline for 8th Semester Completion.

## **LIST OF SYMBOLS AND ABBREVIATIONS**

| Symbol / Abbreviation | Definition |
| :--- | :--- |
| **$FS$** | Factor of Safety ($FS = \text{Resisting Shear} / \text{Driving Shear}$) |
| **$c'$** | Effective Soil Cohesion (kPa) |
| **$\sigma'$** | Effective Normal Stress ($\sigma' = \sigma - u$) |
| **$u$** | Pore-Water Pressure (kPa) |
| **$\beta$** | Hillslope Gradient / Slope Angle (Degrees) |
| **$\phi'$** | Effective Internal Angle of Friction (Degrees) |
| **DEM** | Digital Elevation Model |
| **GSI** | Geological Survey of India |
| **IMERG** | Integrated Multi-satellitE Retrievals for GPM |
| **IPR-I** | Interim Project Report - I |
| **ISRO** | Indian Space Research Organisation |
| **MTE** | Mid-Term Evaluation |
| **NASA GPM** | National Aeronautics and Space Administration - Global Precipitation Measurement |
| **NDVI** | Normalized Difference Vegetation Index |
| **NLSM** | National Landslide Susceptibility Mapping |
| **RF** | Random Forest Classifier |
| **ROC-AUC** | Receiver Operating Characteristic - Area Under the Curve |
| **SRTM** | Shuttle Radar Topography Mission |
| **USGS** | United States Geological Survey |

---

\newpage

## **CHAPTER 1: INTRODUCTION**

### **1.1 Project Motivation & Societal Relevance**
Mountainous environments represent fragile ecological and geological systems where human settlements, road networks, and hydroelectric infrastructures intersect high relief energy. In India, approximately 12.6% of the national landmass (over 0.42 million square kilometers) across 19 States and Union Territories is categorized by the Geological Survey of India (GSI) as landslide-prone. Recent catastrophic disasters in Wayanad (Kerala, 2024), Chamoli (Uttarakhand, 2021), and Malpa (Pithoragarh) emphasize the devastating humanitarian consequences of rainfall-induced slope collapse.

Traditional warning infrastructures rely on broad regional meteorological advisories that lack local topographic context. Conversely, site-specific geotechnical instrumentation (such as boreholes, piezometers, and in-place inclinometers) is capital-intensive and economically unfeasible for thousands of vulnerable slopes. Consequently, a computational framework integrating open satellite precipitation telemetry, digital elevation geomorphology, and machine learning offers a promising paradigm for scalable risk assessment.

### **1.2 Problem Statement**
Current computational tools developed in academic and student projects frequently suffer from three fundamental deficiencies:
1. **Target Leakage and Methodological Errors:** Preprocessing transformations (scaling, imputation) are commonly fitted across entire datasets prior to train/test splitting, creating artificial test performance that fails on unobserved terrain.
2. **Conflation of Static Susceptibility with Dynamic Warning:** Systems frequently predict static terrain susceptibility and falsely claim real-time operational warning capabilities without evaluating multi-day precipitation triggering variables.
3. **Data Availability vs. Fabrication:** Public spatial inventories (GSI / ISRO) deliver shapefiles rather than ML-ready CSVs. Many projects respond by fabricating ungrounded synthetic metrics.

LandslideSense is engineered to overcome these challenges through rigorous, leak-free pipeline engineering, clear separation of static susceptibility from dynamic precipitation triggers, and a transparent fallback heuristic that guarantees honest presentation.

### **1.3 Core Objectives & Scope**
The specific objectives for the 7th-Semester MTE milestone are:
1. Engineer a modular, maintainable Python library implementing:
   * Comprehensive domain-bounds validation for geomorphological and meteorological inputs.
   * A transparent heuristic risk engine (Mode 2) calibrated against established geotechnical literature.
   * A leak-free scikit-learn supervised learning pipeline (Mode 1) utilizing Random Forests.
2. Implement strict dataset schema auditing to prevent target leakage, detect class imbalance, and guard evaluation metrics.
3. Construct an interactive, academically rigorous Streamlit web interface for evaluation.
4. Conduct an official geoscientific data audit of GSI, ISRO, NASA, and USGS repositories.

### **1.4 Critical Limitations & Boundary Conditions**
* **Academic Prototype Notice:** The system is an engineering prototype for academic evaluation, not an operational civil defense early warning service.
* **Geographical Specificity:** The current ML pipeline must be trained on localized catchments; models trained on Himalayan terrain cannot be transferred to Western Ghats without retraining due to divergent lithologies and weathering profiles.
* **Inventory Constraints:** Spatial inventory points represent recorded landslide occurrences exclusively; unrecorded locations cannot be assumed to be stable without systematic negative control sampling.

### **1.5 Organization of the Report**
The remainder of this report is organized as follows:
* **Chapter 2** presents the geotechnical foundation and reviews seminal literature in empirical thresholds and machine learning.
* **Chapter 3** outlines the proposed system architecture, validation protocols, and dual-mode engine formulations.
* **Chapter 4** details the completed implementation, testing outcomes, and user interface.
* **Chapter 5** specifies pending work, GIS spatial overlay requirements, and the timeline for final submission.

---

\newpage

## **CHAPTER 2: BACKGROUND AND RELATED WORK**

### **2.1 Geotechnical Mechanics of Rainfall-Induced Failure**
The limit equilibrium of hillslope regolith is characterized by the Factor of Safety ($FS$):

$$FS = \frac{\tau_f}{\tau_d} = \frac{c' + (\sigma - u) \tan \phi'}{\gamma \cdot z \cdot \sin \beta \cos \beta}$$

During high-intensity monsoon events, water inflowing into soil pores rapidly increases pore-water pressure ($u$), decreasing effective normal stress $\sigma' = \sigma - u$. Simultaneously, the added weight of saturated regolith increases driving shear stress $\tau_d$. When $FS < 1.0$, shear failure is initiated.

### **2.2 Evolution of Empirical Rainfall Thresholds**
Early pioneering research by Caine (1980) established universal rainfall intensity-duration ($I-D$) thresholds:

$$I = 14.82 \cdot D^{-0.39}$$

Guzzetti et al. (2007) subsequently demonstrated that short-duration intensity alone is insufficient; antecedent precipitation accumulated over 3 to 15 days is required to elevate catchment moisture to critical saturation levels before intense rain can trigger mass movement.

### **2.3 Machine Learning Approaches**
With the advent of remote sensing and Digital Elevation Models (DEMs), researchers deployed machine learning for landslide susceptibility mapping:
* **Artificial Neural Networks & Support Vector Machines:** Pradhan & Lee (2010) demonstrated that non-linear kernel models achieve superior spatial fitting over traditional bivariate statistical methods.
* **Tree-Based Ensembles:** In a comprehensive review of over 50 comparative studies, Merghadi et al. (2020) concluded that Random Forests and Gradient Boosted Trees consistently deliver the highest generalization accuracy due to their insensitivity to feature collinearity and robust handling of mixed data types.

### **2.4 Summary of Related Works**
See Table 2.1 in `docs/literature_review_notes.md` for full comparative matrix and verified DOIs.

### **2.5 Identified Research Gaps**
1. **Absence of Unified Dynamic-Static Interfaces:** Existing academic software rarely allows users to evaluate real-time precipitation alongside terrain slopes.
2. **Lack of Transparent Heuristic Fallbacks:** When training data is unavailable, existing tools crash rather than providing a documented fallback index.

---

\newpage

## **CHAPTER 3: PROPOSED METHODOLOGY & SYSTEM ARCHITECTURE**

### **3.1 System Architecture**
LandslideSense implements a modular architecture comprising four primary packages:
1. `src/data_loader.py`: Handles CSV ingestion, schema compliance, missing-cell audits, and class distribution checks.
2. `src/preprocessing.py`: Builds a scikit-learn `ColumnTransformer` with median imputation for numerical features and one-hot encoding for categorical variables.
3. `src/risk_assessment.py`: Implements input boundary validation and the transparent Mode 2 Heuristic calculation.
4. `src/train_model.py` & `src/evaluation.py`: Manages stratified model training, Random Forest hyperparameter tuning, model artifact persistence (`joblib`), and metric calculation.

### **3.2 Geotechnical Input Validation Rules**
Physical variables are subjected to strict domain checking:
* $0.0 \le \text{Rainfall (24h)} \le 1500.0$ mm
* $0.0 \le \text{Rainfall (3d)} \le 3500.0$ mm (with constraint: $\text{Rainfall (24h)} \le \text{Rainfall (3d)}$)
* $0.0 \le \text{Rainfall (7d)} \le 6000.0$ mm (with constraint: $\text{Rainfall (3d)} \le \text{Rainfall (7d)}$)
* $0.0^\circ \le \text{Slope Angle} \le 90.0^\circ$
* $-500.0 \le \text{Elevation} \le 9000.0$ m AMSL
* $0.0\% \le \text{Soil Moisture} \le 100.0\%$ (optional)
* $-1.0 \le \text{NDVI} \le 1.0$ (optional)

### **3.3 Mode 1: Supervised Random Forest Pipeline**
* Splitting: Stratified train/test partition (75% train / 25% test).
* Estimator: `RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced')`.
* Target Leakage Prevention: Identifier columns (`location_id`, coordinates, dates) are systematically excluded.
* Evaluation: Reports Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### **3.4 Mode 2: Transparent Heuristic Demonstration Index**
When no verified training data exists, risk is computed via a transparent weighted scoring index:
* **Rainfall Trigger Score (Max 40 points):** 24h intensity (18 pts) + 3d cumulative (14 pts) + 7d cumulative (8 pts).
* **Topographic Susceptibility (Max 35 points):** Slope criticality peaking at $30^\circ-48^\circ$ (27 pts) + Elevation relief (8 pts).
* **Surface Conditioning (Max 25 points):** Soil saturation (12 pts) + NDVI root reinforcement (7 pts) + Land cover (6 pts).
* **Normalized Index:** Scaled to $[0, 100]$ with Low ($<35$), Medium ($35-70$), and High ($\ge 70$) tiers.

---

\newpage

## **CHAPTER 4: IMPLEMENTATION PROGRESS & VERIFICATION**

### **4.1 Implemented Modules**
All planned core modules have been fully implemented in clean, documented Python:
* `src/risk_assessment.py`
* `src/data_loader.py`
* `src/preprocessing.py`
* `src/train_model.py`
* `src/evaluation.py`
* `app.py` (Streamlit Web Dashboard)

### **4.2 Automated Testing Results**
A test suite of 11 unit tests was created and executed using pytest. All tests passed with zero failures:
* Data Loader & Schema Audit Tests: **5 PASSED**
* Risk Assessment Bounds & Heuristic Tests: **5 PASSED**
* End-to-End Training & Inference Pipeline: **1 PASSED**
* **Total: 11 / 11 PASSED (100% Success)**

### **4.3 Streamlit Interactive Dashboard**
The dashboard was launched locally and verified:
* Home Dashboard with project workflow and academic disclaimer.
* Interactive Risk Assessment form with live Plotly radial gauge.
* Prediction Engine allowing live training and rendering of Confusion Matrices and ROC curves.
* Data Explorer with histograms and bivariate scatter plots.
* Official Data Sources audit page with direct CSV template download.

---

\newpage

## **CHAPTER 5: REMAINING TASKS & ROADMAP FOR FINAL SEMESTER**

### **5.1 Pending Tasks for Final Semester (8th Semester)**
1. **GIS Spatial Overlay Extraction:** Perform spatial intersection of GSI / NRSC landslide inventory polygons with Cartosat/SRTM 30m DEM rasters for a selected Himalayan catchment (e.g., Rudraprayag, Uttarakhand).
2. **Temporal Rainfall Matching:** Extract historical NASA GPM IMERG daily precipitation grids for the specific dates of recorded landslide occurrences.
3. **Negative Control Sampling:** Generate systematic non-landslide samples across confirmed stable terrain to build a field-validated regional training dataset.
4. **Ensemble Algorithm Benchmarking:** Implement comparative benchmarks against XGBoost, LightGBM, and Support Vector Classifiers.
5. **Research Paper Preparation:** Finalize the full research paper for submission to a peer-reviewed IEEE / Scopus-indexed conference or journal.

---

\newpage

## **REFERENCES**

```
[1] N. Caine, "The rainfall intensity-duration control of shallow landslides and debris flows," 
    Geografiska Annaler: Series A, Physical Geography, vol. 62, no. 1-2, pp. 23-27, 1980. 
    DOI: 10.1080/04353676.1980.11879996.

[2] F. Guzzetti, S. Peruccacci, M. Rossi, and C. P. Stark, "Rainfall thresholds for the initiation 
    of landslides in central and southern Europe," Meteorology and Atmospheric Physics, vol. 98, 
    no. 3-4, pp. 239-267, 2007. DOI: 10.1007/s00703-007-0262-7.

[3] D. Kirschbaum and T. Stanley, "Satellite-based assessment of rainfall-triggered landslide hazard 
    for situational awareness," Earth's Future, vol. 6, no. 3, pp. 505-523, 2018. 
    DOI: 10.1002/2017EF000715.

[4] B. Pradhan and S. Lee, "Landslide susceptibility assessment and factor effect analysis: 
    backpropagation artificial neural networks and their comparison with support vector machines," 
    Catena, vol. 82, no. 1, pp. 42-59, 2010. DOI: 10.1016/j.catena.2010.04.008.

[5] A. Merghadi, B. Yunus, B. Douani, M. Pham, M. Abderrahmane, and D. Tien Bui, "Machine learning 
    methods for landslide susceptibility studies: the state of the art," Earth-Science Reviews, 
    vol. 207, p. 103225, 2020. DOI: 10.1016/j.earscirev.2020.103225.

[6] A. Dikshit, R. Shukla, and P. P. Pradhan, "Spatial landslide susceptibility mapping in the 
    Uttarakhand Himalayas using machine learning techniques," Natural Hazards, vol. 104, no. 1, 
    pp. 827-849, 2020. DOI: 10.1007/s11069-020-04183-w.

[7] J. Mathew, D. G. Babu, N. K. Kundu, K. V. Kumar, and C. C. Pant, "Integrating intensity-duration 
    based rainfall threshold and landslide susceptibility map for landslide early warning: a case 
    study in parts of Garhwal Himalaya, India," Journal of Earth System Science, vol. 123, 
    pp. 1357-1372, 2014. DOI: 10.1007/s12040-014-0409-4.
```
