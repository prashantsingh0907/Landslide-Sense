"""
Unit tests for data loader and schema auditing.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import (
    load_dataset_from_csv,
    audit_dataset_schema,
    prepare_training_features,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT_DIR / "data" / "templates" / "landslide_training_template.csv"
SYNTHETIC_PATH = ROOT_DIR / "data" / "demo" / "synthetic_validation_sample.csv"


def test_load_dataset_template():
    """Verify that template CSV loads properly."""
    df, err = load_dataset_from_csv(TEMPLATE_PATH)
    assert err is None
    assert df is not None
    assert len(df) == 10
    assert "landslide" in df.columns


def test_audit_schema_valid_synthetic_dataset():
    """Verify that the synthetic validation dataset passes audit criteria."""
    df, err = load_dataset_from_csv(SYNTHETIC_PATH)
    assert err is None
    audit = audit_dataset_schema(df)
    assert audit["is_suitable_for_training"] is True
    assert audit["target_present"] is True
    assert audit["class_distribution"]["Stable (0)"] >= 2
    assert audit["class_distribution"]["Landslide (1)"] >= 2
    assert "location_id" in audit["id_cols_found"]
    assert "rainfall_24h" in audit["candidate_numerical"]


def test_audit_missing_target():
    """Verify rejection when target column is missing."""
    df = pd.DataFrame({
        "rainfall_24h": [10.0, 20.0, 30.0],
        "slope": [15.0, 25.0, 35.0]
    })
    audit = audit_dataset_schema(df)
    assert audit["is_suitable_for_training"] is False
    assert any("Target column 'landslide' is missing" in r for r in audit["unsuitability_reasons"])


def test_audit_non_binary_target():
    """Verify rejection when target column has non-binary labels."""
    df = pd.DataFrame({
        "rainfall_24h": [10.0] * 12,
        "slope": [20.0] * 12,
        "landslide": [0, 1, 2, 0, 1, 3, 0, 1, 0, 1, 0, 1]
    })
    audit = audit_dataset_schema(df)
    assert audit["is_suitable_for_training"] is False
    assert any("non-binary values" in r for r in audit["unsuitability_reasons"])


def test_prepare_training_features_leakage_prevention():
    """Ensure identifier columns and target are removed from predictor set X."""
    df, err = load_dataset_from_csv(SYNTHETIC_PATH)
    X, y, num_cols, cat_cols = prepare_training_features(df)
    assert "landslide" not in X.columns
    assert "location_id" not in X.columns
    assert len(y) == len(X)
    assert "rainfall_24h" in num_cols
    assert "land_cover" in cat_cols
