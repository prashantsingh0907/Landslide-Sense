"""
Data Loader and Schema Validator for LandslideSense
Handles dataset ingestion, structural integrity checks, missing data auditing,
and training suitability verification.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Core feature definitions
EXPECTED_TARGET = "landslide"

DEFAULT_NUMERICAL_FEATURES = [
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "slope",
    "elevation",
    "soil_moisture",
    "ndvi",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "land_cover",
]

IDENTIFIER_COLUMNS = [
    "location_id",
    "id",
    "site_id",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "date",
    "timestamp",
    "year",
]


def load_dataset_from_csv(
    file_or_path: Union[str, Path, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely loads a CSV from a file path or uploaded file buffer.
    Returns (DataFrame, error_message).
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if not path.exists():
                return None, f"File not found: {path}"
            df = pd.read_csv(path)
        else:
            # File-like object (e.g., Streamlit UploadedFile)
            df = pd.read_csv(file_or_path)
        
        # Strip whitespace from column names
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df, None
    except Exception as e:
        return None, f"Failed to parse CSV: {str(e)}"


def audit_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects dataset structure, checks schema compliance, class distribution,
    missingness, and determines suitability for ML training.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    columns = list(df.columns)
    
    # Missing value count per column
    missing_counts = df.isnull().sum().to_dict()
    total_missing_cells = int(df.isnull().sum().sum())
    
    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Column identification
    target_present = EXPECTED_TARGET in df.columns
    id_cols_found = [c for c in columns if c in IDENTIFIER_COLUMNS]
    
    candidate_numerical = []
    candidate_categorical = []
    for col in columns:
        if col == EXPECTED_TARGET or col in id_cols_found:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidate_numerical.append(col)
        else:
            candidate_categorical.append(col)

    unsuitability_reasons: List[str] = []
    class_distribution: Dict[str, int] = {}
    is_suitable = True

    # Check minimum row count
    if total_rows < 10:
        unsuitability_reasons.append(
            f"Insufficient samples: dataset contains only {total_rows} rows (minimum 10 required for train/test split)."
        )
        is_suitable = False

    # Check target column
    if not target_present:
        unsuitability_reasons.append(
            f"Target column '{EXPECTED_TARGET}' is missing. Labelled training requires binary target 'landslide'."
        )
        is_suitable = False
    else:
        # Check target values
        valid_target_rows = df[EXPECTED_TARGET].dropna()
        unique_targets = set(valid_target_rows.unique())
        
        # Check if values are 0 and 1
        is_binary = unique_targets.issubset({0, 1, 0.0, 1.0})
        if not is_binary:
            unsuitability_reasons.append(
                f"Target column '{EXPECTED_TARGET}' contains invalid non-binary values: {unique_targets}. Must be 0 (no landslide) or 1 (landslide)."
            )
            is_suitable = False
        else:
            counts = valid_target_rows.value_counts().to_dict()
            class_distribution = {
                "Stable (0)": int(counts.get(0, counts.get(0.0, 0))),
                "Landslide (1)": int(counts.get(1, counts.get(1.0, 0))),
            }
            if class_distribution["Stable (0)"] < 2:
                unsuitability_reasons.append(
                    f"Severe class deficit: Class 0 (Stable) has only {class_distribution['Stable (0)']} samples (minimum 2 required)."
                )
                is_suitable = False
            if class_distribution["Landslide (1)"] < 2:
                unsuitability_reasons.append(
                    f"Severe class deficit: Class 1 (Landslide) has only {class_distribution['Landslide (1)']} samples (minimum 2 required)."
                )
                is_suitable = False

    # Check available feature columns
    if len(candidate_numerical) + len(candidate_categorical) == 0:
        unsuitability_reasons.append("No valid feature columns found beyond identifiers and target.")
        is_suitable = False

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": columns,
        "target_present": target_present,
        "missing_counts": missing_counts,
        "total_missing_cells": total_missing_cells,
        "duplicate_count": duplicate_count,
        "id_cols_found": id_cols_found,
        "candidate_numerical": candidate_numerical,
        "candidate_categorical": candidate_categorical,
        "class_distribution": class_distribution,
        "is_suitable_for_training": is_suitable,
        "unsuitability_reasons": unsuitability_reasons,
    }


def prepare_training_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separates predictors X and target y, stripping identifier columns to avoid target leakage.
    Returns (X, y, num_features, cat_features).
    """
    if EXPECTED_TARGET not in df.columns:
        raise ValueError(f"Target column '{EXPECTED_TARGET}' not present in dataframe.")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[EXPECTED_TARGET]).copy()
    y = clean_df[EXPECTED_TARGET].astype(int)

    # Exclude target and identifier columns from training features
    id_cols = [c for c in clean_df.columns if c in IDENTIFIER_COLUMNS]
    drop_cols = [EXPECTED_TARGET] + id_cols
    X = clean_df.drop(columns=drop_cols)

    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]

    return X, y, num_features, cat_features
