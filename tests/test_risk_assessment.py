"""
Unit tests for risk assessment engine, heuristic calculations, and pipeline integration.
"""

import pytest
from src.risk_assessment import (
    validate_environmental_inputs,
    calculate_heuristic_risk,
)
from src.data_loader import load_dataset_from_csv
from src.train_model import (
    train_landslide_model,
    predict_with_ml_model,
)
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SYNTHETIC_PATH = ROOT_DIR / "data" / "demo" / "synthetic_validation_sample.csv"


def test_input_validation_valid():
    """Verify clean inputs pass validation."""
    valid, errors, unavailable = validate_environmental_inputs(
        rainfall_24h=30.0,
        rainfall_3d=80.0,
        rainfall_7d=120.0,
        slope=25.0,
        elevation=1200.0,
        soil_moisture=50.0,
        ndvi=0.45,
        land_cover="Forest",
    )
    assert valid is True
    assert len(errors) == 0
    assert len(unavailable) == 0


def test_input_validation_inconsistent_rainfall():
    """Verify that 24h rain > 3d rain is flagged as inconsistent."""
    valid, errors, _ = validate_environmental_inputs(
        rainfall_24h=100.0,
        rainfall_3d=50.0,  # 24h cannot exceed 3-day sum
        rainfall_7d=120.0,
        slope=25.0,
        elevation=1200.0,
    )
    assert valid is False
    assert any("Physical inconsistency" in err for err in errors)


def test_input_validation_out_of_bounds_slope():
    """Verify slope angle > 90 is flagged."""
    valid, errors, _ = validate_environmental_inputs(
        rainfall_24h=10.0,
        rainfall_3d=20.0,
        rainfall_7d=30.0,
        slope=95.0,
        elevation=1200.0,
    )
    assert valid is False
    assert any("Slope angle out of physical range" in err for err in errors)


def test_heuristic_calculation_missing_optional_factors():
    """Ensure heuristic calculation works when optional inputs are absent."""
    result = calculate_heuristic_risk(
        rainfall_24h=20.0,
        rainfall_3d=50.0,
        rainfall_7d=80.0,
        slope=18.0,
        elevation=800.0,
        soil_moisture=None,
        ndvi=None,
        land_cover=None,
    )
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert len(result["unavailable_factors"]) == 3
    assert len(result["factor_breakdown"]) >= 5


def test_heuristic_extreme_conditions():
    """Extreme rain, critical slope, high moisture should trigger High risk."""
    result = calculate_heuristic_risk(
        rainfall_24h=120.0,
        rainfall_3d=220.0,
        rainfall_7d=350.0,
        slope=38.0,
        elevation=2100.0,
        soil_moisture=85.0,
        ndvi=0.1,
        land_cover="Barren/Sparse",
    )
    assert result["risk_level"] == "High"
    assert result["risk_score"] >= 70.0
    assert "CRITICAL HAZARD WARNING" in result["precautionary_message"]


def test_end_to_end_ml_training_and_inference():
    """Verify that training pipeline fits, produces valid metrics, and predicts on new sample."""
    df, err = load_dataset_from_csv(SYNTHETIC_PATH)
    assert err is None
    
    # Train model
    bundle = train_landslide_model(df, dataset_name="Synthetic Test", test_size=0.25)
    assert bundle is not None
    assert "evaluation_metrics" in bundle
    metrics = bundle["evaluation_metrics"]
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics

    # Inference test
    sample_input = {
        "rainfall_24h": 65.0,
        "rainfall_3d": 140.0,
        "rainfall_7d": 210.0,
        "slope": 36.0,
        "elevation": 1600.0,
        "soil_moisture": 68.0,
        "ndvi": 0.28,
        "land_cover": "Agricultural",
    }
    pred = predict_with_ml_model(bundle, sample_input)
    assert "landslide_probability" in pred
    assert 0.0 <= pred["landslide_probability"] <= 1.0
    assert pred["risk_level"] in ["Low", "Medium", "High"]
