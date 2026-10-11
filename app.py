"""
LandslideSense - Streamlit Web Dashboard
B.Tech Computer Science Capstone Project - 7th Semester MTE
Galgotias University
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.risk_assessment import (
    validate_environmental_inputs,
    calculate_heuristic_risk,
)
from src.data_loader import (
    load_dataset_from_csv,
    audit_dataset_schema,
)
from src.train_model import (
    train_landslide_model,
    load_trained_model,
    predict_with_ml_model,
    MODEL_PATH,
)
from src.evaluation import (
    create_confusion_matrix_chart,
    create_roc_curve_chart,
    create_feature_importance_chart,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="LandslideSense | Capstone Prototype",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Academic Dashboard Theme Styling
st.markdown("""
<style>
    /* Main container refinements */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        border-left: 6px solid #0284c7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #f8fafc;
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
        font-size: 0.95rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
        color: #991b1b;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-active {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-pending {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def render_academic_disclaimer():
    """Renders academic disclaimer box required on evaluation interfaces."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ ACADEMIC EVALUATION PROTOTYPE NOTICE:</strong>
        This software is an engineering prototype developed for the 7th-Semester Mid-Term Evaluation (MTE) /
        Interim Project Report-I (IPR-I) at Galgotias University. It demonstrates algorithmic risk scoring and
        supervised learning pipelines. It is <strong>NOT</strong> an operational civil defense early warning service
        and must not be used for life-critical disaster evacuation decisions.
    </div>
    """, unsafe_allow_html=True)


def make_gauge_chart(score: float, level: str) -> go.Figure:
    """Builds a gauge chart for the risk score."""
    if level == "Low":
        bar_color = "#22c55e"
    elif level == "Medium":
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Landslide Risk Index: <b>{level.upper()}</b>", 'font': {'size': 18}},
        number={'suffix': "/100", 'font': {'size': 26, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.15)'},
                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=15),
    )
    return fig


# ---------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/mountain.png", width=64)
    st.markdown("### LandslideSense")
    st.caption("Capstone Project | Galgotias University")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "⚠️ Landslide Risk Assessment",
            "🤖 Prediction Engine & Training",
            "📊 Data Explorer",
            "📚 Data Sources & Acquisition Audit",
            "📖 About & Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System State")
    trained_bundle = load_trained_model()
    if trained_bundle:
        st.markdown(
            f"<span class='status-badge-active'>● ML Model Loaded</span><br>"
            f"<small>Trained on: {trained_bundle['dataset_name']}<br>"
            f"Samples: {trained_bundle['training_samples']} train / {trained_bundle['test_samples']} test</small>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span class='status-badge-pending'>○ No Trained ML Model</span><br>"
            "<small>Demonstration Mode active (transparent heuristic fallback).</small>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("Environment: Python 3.14 | Streamlit | scikit-learn")


# ---------------------------------------------------------
# PAGE 1: Home Dashboard
# ---------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>LandslideSense: ML-Based Landslide Prediction & Risk Assessment</h1>
        <p>B.Tech Computer Science Capstone Project — 7th Semester Mid-Term Evaluation (MTE) & IPR-I</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Primary Trigger Factor",
            value="Antecedent Rainfall",
            delta="24h, 3-Day & 7-Day Cumulative",
        )
    with col2:
        st.metric(
            label="Geotechnical Susceptibility",
            value="Topography & Regolith",
            delta="Slope Angle (Peak 30°-48°)",
        )
    with col3:
        st.metric(
            label="Prototype Operating State",
            value="Dual-Mode Capable",
            delta="ML Mode + Heuristic Demo",
        )

    st.markdown("### 1. Problem Context & Physical Phenomenon")
    st.markdown("""
    Landslides in mountainous terrains (such as the Himalayas and Western Ghats of India) are complex geotechnical
    phenomena governed by **static conditioning factors** (slope gradient, lithology, elevation, land cover) and
    **dynamic triggering factors** (short-term intense precipitation and cumulative antecedent rainfall saturation).

    When cumulative rainfall infiltrates permeable regolith overlying impermeable bedrock, pore-water pressure rises.
    Under the **Mohr-Coulomb failure criterion**, elevated pore pressure reduces the effective normal stress ($\sigma'$),
    diminishing the soil's internal shear resistance until gravitational shear stress along the slope induces failure.
    """)

    st.markdown("### 2. System Workflow Architecture")
    c_wf1, c_wf2, c_wf3, c_wf4 = st.columns(4)
    with c_wf1:
        st.markdown("""
        **1. Input Telemetry**
        - Cumulative rainfall (24h, 3d, 7d)
        - Digital Elevation Model (DEM) slope & height
        - Optional Soil Moisture & NDVI
        """)
    with c_wf2:
        st.markdown("""
        **2. Safe Preprocessing**
        - Bounds & consistency validation
        - Missing factor tracking
        - Leak-free ColumnTransformer pipeline
        """)
    with c_wf3:
        st.markdown("""
        **3. Dual Engine**
        - **ML Mode:** Random Forest on verified labelled CSVs
        - **Demo Mode:** Transparent, formulaic heuristic index
        """)
    with c_wf4:
        st.markdown("""
        **4. Risk Tiers & Advisory**
        - Normalized Risk Index (0-100)
        - Low / Medium / High Classification
        - Targeted geotechnical precautions
        """)

    st.markdown("### 3. Current Milestone Progress (IPR-I / MTE)")
    st.info("""
    **Interim Project Report-I Status:**
    - ✅ **Implemented & Verified:** Modular core packages (`risk_assessment.py`, `data_loader.py`, `preprocessing.py`, `train_model.py`, `evaluation.py`).
    - ✅ **Heuristic Demonstration Engine:** Transparent formula-based risk scoring with detailed physical contribution breakdowns.
    - ✅ **Data Schema & Leak-Free Pipeline:** Strict target validation, identifier exclusion, stratified train/test splitting, and evaluation metric guards.
    - ⚠️ **Data Limitation & Pending Work:** Official Indian spatial inventories (GSI / ISRO Landslide Atlas) provide spatial hazard polygons rather than ready-made temporal telemetry CSVs. Real sensor telemetry integration and spatial raster overlay are targeted for final 8th-semester deployment.
    """)


# ---------------------------------------------------------
# PAGE 2: Landslide Risk Assessment Form
# ---------------------------------------------------------
elif page == "⚠️ Landslide Risk Assessment":
    st.markdown("""
    <div class="main-header">
        <h1>Interactive Landslide Risk Assessment</h1>
        <p>Evaluate real-time or historical environmental parameters against susceptibility thresholds</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    # Engine Selection
    col_mode, col_info = st.columns([2, 3])
    with col_mode:
        selected_mode = st.radio(
            "Evaluation Operating Mode",
            ["Demonstration Mode (Formulaic Heuristic)", "Machine Learning Mode (Fitted Pipeline)"],
            help="Select whether to evaluate using the transparent heuristic or the trained Random Forest model."
        )
    with col_info:
        if selected_mode.startswith("Machine"):
            if trained_bundle is None:
                st.warning("⚠️ No trained ML model found. Go to 'Prediction Engine & Training' to train a pipeline first, or use Demonstration Mode.")
            else:
                st.success(f"✅ ML Model Active: Trained on '{trained_bundle['dataset_name']}' with {trained_bundle['training_samples']} samples.")
        else:
            st.info("ℹ️ Demonstration Mode Active: Uses a documented, transparent weighted heuristic. Does not fabricate ML probabilities.")

    st.markdown("### Environmental & Topographic Input Form")
    with st.form("risk_assessment_form"):
        st.markdown("##### 🌧️ Precipitation Telemetry (Hydrological Trigger)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r24 = st.number_input(
                "Rainfall - Previous 24 Hours (mm)",
                min_value=0.0,
                max_value=1200.0,
                value=45.0,
                step=1.0,
                help="Short-duration rainfall intensity triggering shallow rapid slip."
            )
        with c2:
            r3d = st.number_input(
                "Rainfall - Previous 3 Days Cumulative (mm)",
                min_value=0.0,
                max_value=2500.0,
                value=110.0,
                step=1.0,
                help="Intermediate antecedent rainfall saturating regolith."
            )
        with c3:
            r7d = st.number_input(
                "Rainfall - Previous 7 Days Cumulative (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=180.0,
                step=1.0,
                help="Prolonged antecedent precipitation raising groundwater table."
            )

        st.markdown("##### ⛰️ Terrain & Geomorphology (Static Conditioning)")
        c4, c5 = st.columns(2)
        with c4:
            slope = st.number_input(
                "Slope Angle (Degrees)",
                min_value=0.0,
                max_value=90.0,
                value=35.0,
                step=0.5,
                help="Critical failure threshold typically between 25° and 45°."
            )
        with c5:
            elevation = st.number_input(
                "Terrain Elevation (Metres AMSL)",
                min_value=-500.0,
                max_value=9000.0,
                value=1650.0,
                step=50.0,
                help="Altitude proxy for orographic precipitation and relief energy."
            )

        st.markdown("##### 🌿 Surface & In-Situ Parameters (Optional / Conditional)")
        enable_optional = st.checkbox("Include In-Situ / Remote Sensing Observations", value=True)

        if enable_optional:
            c6, c7, c8 = st.columns(3)
            with c6:
                soil_m_available = st.checkbox("Soil moisture sensor available", value=True)
                sm_val = st.slider("Soil Moisture Saturation (%)", 0.0, 100.0, 65.0, 1.0) if soil_m_available else None
            with c7:
                ndvi_available = st.checkbox("Satellite NDVI available", value=True)
                ndvi_val = st.slider("Vegetation Index (NDVI)", -0.2, 1.0, 0.35, 0.01) if ndvi_available else None
            with c8:
                lc_val = st.selectbox(
                    "Land Cover Category",
                    ["Forest", "Shrubland", "Agricultural", "Barren/Sparse", "Settlement", "None / Unknown"],
                    index=1
                )
                if lc_val == "None / Unknown":
                    lc_val = None
        else:
            sm_val, ndvi_val, lc_val = None, None, None

        submitted = st.form_submit_button("Compute Risk Assessment", use_container_width=True)

    if submitted:
        # Validate inputs
        is_valid, validation_errors, unavailable = validate_environmental_inputs(
            r24, r3d, r7d, slope, elevation, sm_val, ndvi_val, lc_val
        )

        if not is_valid:
            st.error("❌ Input Validation Error:\n" + "\n".join([f"- {e}" for e in validation_errors]))
        else:
            st.markdown("---")
            st.markdown("### Assessment Results & Geotechnical Advisory")

            # Execute assessment based on mode
            if selected_mode.startswith("Machine"):
                if trained_bundle is None:
                    st.error("Cannot run Machine Learning mode because no model is loaded. Switching to Demonstration Mode.")
                    result = calculate_heuristic_risk(r24, r3d, r7d, slope, elevation, sm_val, ndvi_val, lc_val)
                else:
                    sample_dict = {
                        "rainfall_24h": r24,
                        "rainfall_3d": r3d,
                        "rainfall_7d": r7d,
                        "slope": slope,
                        "elevation": elevation,
                        "soil_moisture": sm_val,
                        "ndvi": ndvi_val,
                        "land_cover": lc_val,
                    }
                    result = predict_with_ml_model(trained_bundle, sample_dict)
            else:
                result = calculate_heuristic_risk(r24, r3d, r7d, slope, elevation, sm_val, ndvi_val, lc_val)

            # Display top results
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.plotly_chart(make_gauge_chart(result["risk_score"], result["risk_level"]), use_container_width=True)

            with res_col2:
                st.markdown(f"#### Risk Classification: <span style='color:{result['alert_color']}; font-weight:700;'>{result['risk_level'].upper()} RISK</span>", unsafe_allow_html=True)
                st.markdown(f"**Operating Mode:** `{result['mode']}`")
                if "landslide_probability" in result:
                    st.markdown(f"**Model Probability:** `{result['landslide_probability']:.2%}`")
                st.markdown(f"**Calculated Risk Index:** `{result['risk_score']} / 100`")

                # Precautionary Alert Box
                if result["risk_level"] == "High":
                    st.error(f"⚠️ **Precautionary Advisory:** {result['precautionary_message']}")
                elif result["risk_level"] == "Medium":
                    st.warning(f"⚠️ **Precautionary Advisory:** {result['precautionary_message']}")
                else:
                    st.success(f"✅ **Precautionary Advisory:** {result['precautionary_message']}")

            # Transparency regarding unavailable factors
            unavail_items = result.get("unavailable_factors", [])
            if unavail_items:
                st.info("ℹ️ **Data Availability Notice:** The following environmental factors were not provided and were accounted for safely without fabricating missing measurements:\n" +
                        "\n".join([f"- {item}" for item in unavail_items]))

            # Factor breakdown (Demonstration Mode)
            if "factor_breakdown" in result:
                st.markdown("#### Physical Contribution Breakdown")
                breakdown_df = pd.DataFrame(result["factor_breakdown"])
                st.dataframe(
                    breakdown_df[["factor", "value", "score_contrib", "max_possible", "mechanism"]],
                    column_config={
                        "factor": "Parameter",
                        "value": "Input Value",
                        "score_contrib": "Points Contributed",
                        "max_possible": "Max Possible",
                        "mechanism": "Geotechnical Failure Mechanism",
                    },
                    use_container_width=True,
                    hide_index=True,
                )


# ---------------------------------------------------------
# PAGE 3: Prediction Engine & Model Training
# ---------------------------------------------------------
elif page == "🤖 Prediction Engine & Training":
    st.markdown("""
    <div class="main-header">
        <h1>Prediction Engine & Model Training Pipeline</h1>
        <p>Rigorous, leak-free supervised learning on verified labelled datasets</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    st.markdown("""
    ### Operating Principles
    1. **Strict Target Validation:** Requires binary target column `landslide` (0 = stable observation, 1 = recorded landslide event).
    2. **Target Leakage Prevention:** Identifier columns (`location_id`, coordinates, timestamps) are automatically stripped prior to training.
    3. **Stratified Split & Fitting:** Preserves class balance in train/test splits; preprocessors are fitted strictly on the training partition.
    4. **Honest Evaluation:** Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) are computed strictly on the held-out test split without fabrication.
    """)

    st.markdown("---")
    st.markdown("### Train Model on Dataset")

    dataset_choice = st.radio(
        "Select Training Data Source",
        [
            "Synthetic Validation Sample (30 observations, for local pipeline testing)",
            "Landslide Training Template (10 sample rows, boundary test)",
            "Upload Custom Labelled CSV",
        ],
        index=0,
    )

    df_to_train = None
    dataset_name = ""

    if dataset_choice.startswith("Synthetic"):
        synth_path = ROOT_DIR / "data" / "demo" / "synthetic_validation_sample.csv"
        df_to_train, err = load_dataset_from_csv(synth_path)
        dataset_name = "Synthetic Validation Sample"
        st.caption("ℹ️ Explicitly designated synthetic data for pipeline validation. Does not represent actual field telemetry.")
    elif dataset_choice.startswith("Landslide Training Template"):
        tpl_path = ROOT_DIR / "data" / "templates" / "landslide_training_template.csv"
        df_to_train, err = load_dataset_from_csv(tpl_path)
        dataset_name = "Template Sample"
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            df_to_train, err = load_dataset_from_csv(uploaded_file)
            dataset_name = uploaded_file.name
            if err:
                st.error(f"Failed to parse uploaded CSV: {err}")

    if df_to_train is not None:
        st.markdown(f"#### Dataset Schema Audit: `{dataset_name}`")
        audit = audit_dataset_schema(df_to_train)

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a1.metric("Total Rows", audit["total_rows"])
        col_a2.metric("Total Columns", audit["total_cols"])
        col_a3.metric("Missing Values", audit["total_missing_cells"])
        col_a4.metric("Suitability", "SUITABLE" if audit["is_suitable_for_training"] else "UNSUITABLE")

        if audit["class_distribution"]:
            st.write("**Class Balance:**", audit["class_distribution"])

        if not audit["is_suitable_for_training"]:
            st.error("⚠️ Dataset is not currently suitable for model training:\n" +
                     "\n".join([f"- {r}" for r in audit["unsuitability_reasons"]]))
        else:
            st.success("✅ Dataset meets all schema and class balance criteria for pipeline training.")

            col_t1, col_t2 = st.columns([1, 2])
            with col_t1:
                test_pct = st.slider("Test Split Ratio", 0.15, 0.40, 0.25, 0.05)
            with col_t2:
                train_clicked = st.button("Train Random Forest Pipeline", type="primary", use_container_width=True)

            if train_clicked:
                with st.spinner("Executing leak-free training pipeline..."):
                    try:
                        bundle = train_landslide_model(df_to_train, dataset_name=dataset_name, test_size=test_pct)
                        st.success(f"🎉 Model successfully trained and saved to `models/landslide_random_forest.joblib`!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

    # Display Current Model Diagnostics
    if trained_bundle is not None:
        st.markdown("---")
        st.markdown("### Current Active Model Diagnostics")
        st.write(f"**Model Type:** `{trained_bundle['model_type']}`")
        st.write(f"**Trained On:** `{trained_bundle['dataset_name']}` at `{trained_bundle['train_timestamp']}`")
        st.write(f"**Partition Sizes:** Training: {trained_bundle['training_samples']} samples | Test: {trained_bundle['test_samples']} samples")

        metrics = trained_bundle["evaluation_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['precision']:.2%}")
        m3.metric("Recall", f"{metrics['recall']:.2%}")
        m4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "N/A")

        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            cm_fig = create_confusion_matrix_chart(metrics["confusion_matrix"])
            st.plotly_chart(cm_fig, use_container_width=True)

        with c_diag2:
            roc_fig = create_roc_curve_chart(metrics["roc_curve"], metrics["roc_auc"])
            st.plotly_chart(roc_fig, use_container_width=True)

        if "encoded_feature_names" in trained_bundle and "feature_importances" in trained_bundle:
            st.markdown("#### Feature Importance Analysis")
            imp_fig = create_feature_importance_chart(
                trained_bundle["encoded_feature_names"],
                trained_bundle["feature_importances"]
            )
            st.plotly_chart(imp_fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 4: Data Explorer
# ---------------------------------------------------------
elif page == "📊 Data Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset Explorer & Quality Inspection</h1>
        <p>Inspect distributions, missingness, and structural health of environmental datasets</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    src_choice = st.selectbox(
        "Select Dataset to Inspect",
        [
            "Synthetic Validation Sample (data/demo/synthetic_validation_sample.csv)",
            "Landslide Training Template (data/templates/landslide_training_template.csv)",
            "Upload Custom File",
        ]
    )

    df_view = None
    if "Synthetic" in src_choice:
        df_view, _ = load_dataset_from_csv(ROOT_DIR / "data" / "demo" / "synthetic_validation_sample.csv")
    elif "Template" in src_choice:
        df_view, _ = load_dataset_from_csv(ROOT_DIR / "data" / "templates" / "landslide_training_template.csv")
    else:
        up = st.file_uploader("Upload CSV to inspect", type=["csv"], key="explorer_uploader")
        if up:
            df_view, _ = load_dataset_from_csv(up)

    if df_view is not None:
        st.markdown("#### Dataset Preview")
        st.dataframe(df_view.head(15), use_container_width=True)

        c_st1, c_st2, c_st3 = st.columns(3)
        c_st1.metric("Rows", len(df_view))
        c_st2.metric("Columns", len(df_view.columns))
        c_st3.metric("Missing Values", int(df_view.isnull().sum().sum()))

        st.markdown("#### Column Data Types & Missing Summary")
        summary_df = pd.DataFrame({
            "Data Type": df_view.dtypes.astype(str),
            "Null Count": df_view.isnull().sum(),
            "Null %": (df_view.isnull().sum() / len(df_view) * 100).round(1),
            "Unique Values": df_view.nunique(),
        })
        st.dataframe(summary_df, use_container_width=True)

        # Numerical distributions
        num_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown("#### Numerical Feature Distributions")
            selected_num = st.selectbox("Select feature to plot histogram", num_cols)
            fig_hist = px.histogram(
                df_view,
                x=selected_num,
                color="landslide" if "landslide" in df_view.columns else None,
                barmode="overlay",
                marginal="box",
                title=f"Distribution of {selected_num}",
                color_discrete_sequence=["#0284c7", "#ef4444"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        if "rainfall_24h" in df_view.columns and "slope" in df_view.columns:
            st.markdown("#### Bivariate Trigger vs Susceptibility Scatter")
            fig_scatter = px.scatter(
                df_view,
                x="slope",
                y="rainfall_24h",
                color="landslide" if "landslide" in df_view.columns else None,
                title="Slope Angle vs 24h Rainfall",
                labels={"slope": "Slope (degrees)", "rainfall_24h": "24h Rainfall (mm)"},
                color_discrete_sequence=["#10b981", "#ef4444"]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: Data Sources & Acquisition Audit
# ---------------------------------------------------------
elif page == "📚 Data Sources & Acquisition Audit":
    st.markdown("""
    <div class="main-header">
        <h1>Official Data Sources & Real Data Acquisition Audit</h1>
        <p>Scientific audit of primary landslide inventories, precipitation products, and access methods</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    st.markdown("""
    ### Official Geoscientific Repositories
    An essential requirement of academic project integrity is verifying accessibility and format rather than
    assuming web portals provide instantaneous ML-ready CSVs.
    """)

    sources = [
        {
            "Agency": "Geological Survey of India (GSI)",
            "Portal": "Landslide Hazard Portal",
            "URL": "https://gsi.gov.in/landslide-hazard/",
            "Coverage": "India (Himalayas, Western Ghats, NE Hills)",
            "Data Format": "Spatial GIS Shapefiles, Macro Landslide Susceptibility Maps (NLSM 1:50k), Inventory polygons.",
            "Access / Status": "Restricted public view; requires academic clearance / Bhukosh registration for raw shapefile export. Not a direct tabular CSV.",
            "Scientific Role": "Ground-truth inventory for landslide polygons; requires spatial intersection with DEM.",
        },
        {
            "Agency": "ISRO / NRSC",
            "Portal": "Landslide Atlas of India",
            "URL": "https://www.isro.gov.in/ISRO_EN/Landslide_Atlas_India.html",
            "Coverage": "India (17 States and 2 UTs, 1998–2022)",
            "Data Format": "High-resolution satellite inventory atlas (PDF / Web GIS on Bhuvan).",
            "Access / Status": "Published atlas document freely accessible; spatial GIS layers viewable on Bhuvan GIS. Tabular attribute extraction requires GIS polygon sampling.",
            "Scientific Role": "Historical validation of spatial susceptibility zones and landslide frequency across Indian districts.",
        },
        {
            "Agency": "NASA Earth Data / GPM",
            "Portal": "Global Precipitation Measurement (GPM)",
            "URL": "https://gpm.nasa.gov/data/directory",
            "Coverage": "Global (0.1° × 0.1° spatial resolution, 30-min / daily intervals)",
            "Data Format": "NetCDF-4 / HDF5 / GeoTIFF gridded precipitation products (IMERG).",
            "Access / Status": "Freely accessible via NASA EarthData Login and OpenDAP/HTTP endpoints.",
            "Scientific Role": "Dynamic hydrological triggering series; provides 24h, 3d, and 7d cumulative precipitation telemetry.",
        },
        {
            "Agency": "USGS (United States Geological Survey)",
            "Portal": "Landslide Inventories across the US (Ver 3.0)",
            "URL": "https://www.usgs.gov/data/landslide-inventories-across-united-states-ver-30-february-2025",
            "Coverage": "United States ONLY (Not India)",
            "Data Format": "Geodatabase, Shapefile, GeoPackage, and point/polygon inventories.",
            "Access / Status": "Public domain (US Government).",
            "Scientific Role": "High-quality benchmark for methodological testing; must be explicitly acknowledged as United States terrain if utilized.",
        },
    ]

    for s in sources:
        with st.expander(f"📍 {s['Agency']} — {s['Portal']}", expanded=True):
            st.markdown(f"**Official URL:** [{s['URL']}]({s['URL']})")
            st.markdown(f"**Geographical Coverage:** {s['Coverage']}")
            st.markdown(f"**Data Format & Delivery:** {s['Data Format']}")
            st.markdown(f"**Access Constraints & Status:** {s['Access / Status']}")
            st.markdown(f"**Scientific Application in LandslideSense:** {s['Scientific Role']}")

    st.markdown("---")
    st.markdown("### How to Acquire and Format Ground-Truth Data")
    st.markdown("""
    To train a scientifically valid machine learning model for an Indian catchment (e.g., Wayanad or Chamoli):
    1. **Step 1:** Download historical landslide inventory polygons from GSI Bhukosh or NRSC Bhuvan.
    2. **Step 2:** Extract digital elevation parameters (slope angle, aspect, elevation) from SRTM 30m or Cartosat DEM.
    3. **Step 3:** Query NASA GPM IMERG gridded rainfall for the dates of known landslide events and preceding 7 days.
    4. **Step 4 (Crucial):** Generate non-landslide negative control samples (label `0`) by randomly sampling stable terrain locations under non-trigger rainfall regimes, ensuring class balance.
    5. **Step 5:** Export to the LandslideSense CSV template format.
    """)

    # Download Template Button
    template_path = ROOT_DIR / "data" / "templates" / "landslide_training_template.csv"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Download Standard CSV Training Template",
            data=csv_data,
            file_name="landslide_training_template.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------
# PAGE 6: About & Academic Methodology
# ---------------------------------------------------------
elif page == "📖 About & Methodology":
    st.markdown("""
    <div class="main-header">
        <h1>About LandslideSense & Scientific Methodology</h1>
        <p>Theoretical framework, geotechnical principles, and capstone documentation</p>
    </div>
    """, unsafe_allow_html=True)

    render_academic_disclaimer()

    st.markdown("### 1. Conceptual Distinctions")
    st.markdown("""
    A frequent failure mode in student machine learning projects is conflating distinct hazard concepts:
    - **Susceptibility:** Where landslides are likely to occur based solely on static landscape characteristics (slope, geology, aspect). Susceptibility does not specify *when* an event will occur.
    - **Hazard / Risk Classification:** Combines spatial susceptibility with temporal trigger likelihood (e.g., monsoon precipitation intensity) to classify instantaneous vulnerability into risk tiers (Low, Medium, High).
    - **Event Prediction / Early Warning:** A real-time operational warning with explicit temporal lead time. Operational warning requires verified sensor telemetry, calibrated physical or empirical rainfall thresholds, and community dissemination protocols.
    """)

    st.markdown("### 2. Geotechnical Foundation")
    st.markdown(r"""
    The engineering basis of rainfall-induced landslides rests upon the **limit equilibrium** of hillslope regolith:

    $$\text{Factor of Safety } (FS) = \frac{c' + (\gamma \cdot z \cdot \cos^2\beta - u) \tan\phi'}{\gamma \cdot z \cdot \sin\beta \cos\beta}$$

    Where:
    - $c'$ = Effective soil cohesion (enhanced by root systems / NDVI)
    - $\gamma$ = Unit weight of soil
    - $z$ = Regolith depth
    - $\beta$ = Slope angle
    - $u$ = Pore-water pressure (drastically increased by cumulative infiltration)
    - $\phi'$ = Internal friction angle

    When high antecedent rainfall and intense 24h precipitation cause $u$ to rise, the numerator decreases while gravitational driving force increases, causing $FS < 1.0$ (slope collapse).
    """)

    st.markdown("### 3. Capstone Project Details")
    st.markdown("""
    - **Institution:** Galgotias University, Greater Noida
    - **Course:** B.Tech Computer Science and Engineering
    - **Evaluation:** 7th Semester Mid-Term Evaluation (MTE) / Interim Project Report-I (IPR-I)
    - **Primary Modules Completed:**
      - Input validation & bounds checking
      - Dual-mode prediction engine (Heuristic demonstration + Random Forest)
      - Schema auditor & target leakage prevention
      - Interactive Streamlit evaluation dashboard
    - **Roadmap for 8th Semester / Final Submission:**
      - Spatial GIS raster overlay pipeline for GSI / ISRO polygons
      - Integration of live Open-Meteo or NASA GPM API precipitation feeds
      - Ensemble comparative evaluation (XGBoost, LightGBM, Random Forest, Logistic Regression)
    """)
