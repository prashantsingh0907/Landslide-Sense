"""
Model Training Pipeline for LandslideSense
Orchestrates data splitting, leak-free pipeline assembly, Random Forest fitting,
validation metric calculation, and artifact persistence.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import (
    audit_dataset_schema,
    prepare_training_features,
)
from src.preprocessing import build_preprocessor
from src.evaluation import evaluate_model

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "landslide_random_forest.joblib"


def train_landslide_model(
    df: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end model training workflow:
    1. Schema audit & suitability verification
    2. Leak-free feature separation
    3. Stratified train/test split (when feasible)
    4. Preprocessor + Random Forest pipeline fitting
    5. Test-split evaluation
    6. Artifact serialization
    """
    # 1. Audit schema
    audit = audit_dataset_schema(df)
    if not audit["is_suitable_for_training"]:
        reasons_str = "; ".join(audit["unsuitability_reasons"])
        raise ValueError(f"Dataset unsuitable for training: {reasons_str}")

    # 2. Extract predictors and target
    X, y, num_features, cat_features = prepare_training_features(df)

    # 3. Stratified split verification
    min_class_count = min(audit["class_distribution"].values())
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 4. Assemble pipeline
    preprocessor = build_preprocessor(num_features, cat_features)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=2,
        class_weight="balanced",
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Fit strictly on training split
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on held-out test split
    eval_results = evaluate_model(pipeline, X_test, y_test)

    # 6. Extract transformed feature names and importances
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_rf = pipeline.named_steps["classifier"]
    
    try:
        encoded_feature_names = list(fitted_preprocessor.get_feature_names_out())
        # Clean prefix names e.g. "num__rainfall_24h" -> "rainfall_24h"
        clean_feature_names = [f.split("__")[-1] for f in encoded_feature_names]
    except Exception:
        clean_feature_names = list(num_features) + list(cat_features)

    importances = [float(x) for x in fitted_rf.feature_importances_]

    # Package metadata bundle
    bundle = {
        "pipeline": pipeline,
        "dataset_name": dataset_name,
        "train_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "all_input_features": list(X.columns),
        "encoded_feature_names": clean_feature_names,
        "feature_importances": importances,
        "evaluation_metrics": eval_results,
        "class_distribution_train": {
            "Stable (0)": int(np.sum(y_train == 0)),
            "Landslide (1)": int(np.sum(y_train == 1)),
        },
        "model_type": "RandomForestClassifier(n_estimators=100, class_weight='balanced')",
    }

    # 7. Persist to models/
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    return bundle


def load_trained_model(
    model_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Loads saved model bundle if available."""
    target_path = model_path or MODEL_PATH
    if not target_path.exists():
        return None
    try:
        bundle = joblib.load(target_path)
        return bundle
    except Exception:
        return None


def predict_with_ml_model(
    model_bundle: Dict[str, Any],
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes inference using a fitted pipeline.
    Validates input fields, handles missing features via pipeline imputation,
    and returns probabilities and risk tiers.
    """
    pipeline: Pipeline = model_bundle["pipeline"]
    expected_cols = model_bundle["all_input_features"]

    # Build single-row DataFrame matching training feature schema
    row_dict = {}
    missing_fields = []
    for col in expected_cols:
        val = input_data.get(col)
        if val is None or val == "":
            missing_fields.append(col)
            row_dict[col] = np.nan
        else:
            row_dict[col] = val

    df_row = pd.DataFrame([row_dict], columns=expected_cols)

    # Predict probability of class 1 (landslide)
    probabilities = pipeline.predict_proba(df_row)[0]
    landslide_prob = float(probabilities[1])
    predicted_class = int(pipeline.predict(df_row)[0])

    # Convert probability to risk percentage score [0, 100]
    risk_score = round(landslide_prob * 100.0, 1)

    if risk_score < 35.0:
        risk_level = "Low"
        alert_color = "#22c55e"
        precautionary_message = (
            "MODEL ASSESSMENT: Estimated failure probability is low. "
            "Normal geotechnical stability indicated by evaluated features."
        )
    elif risk_score < 70.0:
        risk_level = "Medium"
        alert_color = "#f59e0b"
        precautionary_message = (
            "MODEL ASSESSMENT: Moderate landslide probability. "
            "Environmental inputs align with past observed instability conditions. Caution advised."
        )
    else:
        risk_level = "High"
        alert_color = "#ef4444"
        precautionary_message = (
            "MODEL ASSESSMENT: HIGH LANDSLIDE PROBABILITY. "
            "Current feature configuration closely matches verified historical landslide events."
        )

    return {
        "mode": "Machine Learning (Random Forest Fitted Pipeline)",
        "risk_score": risk_score,
        "landslide_probability": round(landslide_prob, 4),
        "predicted_class": predicted_class,
        "risk_level": risk_level,
        "alert_color": alert_color,
        "precautionary_message": precautionary_message,
        "missing_features_imputed": missing_fields,
        "model_trained_on": model_bundle["dataset_name"],
        "train_timestamp": model_bundle["train_timestamp"],
    }
