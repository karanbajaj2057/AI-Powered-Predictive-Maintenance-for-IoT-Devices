"""Unit tests for preprocessing and feature engineering pipelines."""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import generate_mock_telemetry_data
from src.feature_engineering import extract_features
from src.preprocessing import (
    build_preprocessor,
    prepare_train_test_split,
    fit_and_transform_pipeline,
)
from src.config import (
    TARGET_COLUMN,
    FAILURE_SUBTYPE_COLUMNS,
    ID_COLUMNS,
    FINAL_FEATURE_COLUMNS,
)


def test_leakage_safety_in_feature_extraction():
    """Verify target and failure mode columns are stripped from input features."""
    df_mock = generate_mock_telemetry_data(n_samples=20)

    # Confirm raw data contains targets and subtypes
    assert TARGET_COLUMN in df_mock.columns
    for subtype in FAILURE_SUBTYPE_COLUMNS:
        assert subtype in df_mock.columns

    # Extract features
    features_df = extract_features(df_mock)

    # Ensure target, failure subtypes, and ID columns are NEVER in feature set
    assert TARGET_COLUMN not in features_df.columns
    for subtype in FAILURE_SUBTYPE_COLUMNS:
        assert subtype not in features_df.columns
    for id_col in ID_COLUMNS:
        assert id_col not in features_df.columns

    # Verify expected feature columns exist
    for col in FINAL_FEATURE_COLUMNS:
        assert col in features_df.columns


def test_engineered_features_calculation():
    """Verify calculation correctness for engineered features."""
    sample = pd.DataFrame([{
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }])
    feat = extract_features(sample)

    assert pytest.approx(feat.iloc[0]["temp_diff"], 0.01) == 10.5
    assert pytest.approx(feat.iloc[0]["power_factor"], 0.01) == 60000.0
    assert pytest.approx(feat.iloc[0]["tool_wear_torque"], 0.01) == 4000.0


def test_train_test_split_isolation():
    """Verify stratified split separates training and testing without data overlap."""
    df_mock = generate_mock_telemetry_data(n_samples=50)
    X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_split(df_mock, test_size=0.2)

    assert len(X_train_raw) == 40
    assert len(X_test_raw) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10
    # Ensure mutually exclusive indices
    assert set(X_train_raw.index).isdisjoint(set(X_test_raw.index))


def test_fit_and_transform_pipeline():
    """Verify that preprocessing fits on train and transforms both train and test correctly."""
    df_mock = generate_mock_telemetry_data(n_samples=40)
    X_train_raw, X_test_raw, _, _ = prepare_train_test_split(df_mock, test_size=0.25)

    X_train_proc, X_test_proc, preprocessor = fit_and_transform_pipeline(
        X_train_raw, X_test_raw
    )

    # 3 categories for Type ('H', 'L', 'M') + 8 continuous features = 11 output dimensions
    assert X_train_proc.shape[0] == 30
    assert X_test_proc.shape[0] == 10
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()
