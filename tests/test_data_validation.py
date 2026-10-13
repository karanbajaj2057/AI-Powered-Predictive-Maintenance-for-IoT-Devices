"""Unit tests for dataset schema validation, integrity checks, and error reporting."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import (
    validate_dataset,
    load_and_validate_dataset,
    generate_mock_telemetry_data,
)
from src.feature_engineering import extract_features
from src.config import TARGET_COLUMN, NUMERICAL_FEATURES, FAILURE_SUBTYPE_COLUMNS


@pytest.fixture
def valid_mock_df():
    """Small synthetic test fixture mimicking valid UCI AI4I 2020 schema."""
    return generate_mock_telemetry_data(n_samples=25, random_state=123)


def test_missing_dataset_file():
    """Verify FileNotFoundError is raised with descriptive message when dataset is missing."""
    fake_path = Path("non_existent_dataset_path.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_and_validate_dataset(fake_path)
    assert "Dataset not found" in str(exc_info.value)
    assert "UCI Machine Learning Repository" in str(exc_info.value)


def test_empty_dataset_rejected():
    """Verify empty DataFrame (0 rows) is rejected."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(empty_df)
    assert "empty" in str(exc_info.value).lower()


def test_missing_required_columns(valid_mock_df):
    """Verify ValueError is raised when required measurement or target columns are missing."""
    corrupted_df = valid_mock_df.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "Torque [Nm]" in str(exc_info.value)
    assert "Missing required columns" in str(exc_info.value)


def test_missing_target_column(valid_mock_df):
    """Verify ValueError is raised when target column is missing."""
    corrupted_df = valid_mock_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert TARGET_COLUMN in str(exc_info.value)


def test_invalid_target_values(valid_mock_df):
    """Verify target column with non-binary values is rejected."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[0, TARGET_COLUMN] = 5
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "non-binary" in str(exc_info.value).lower()


def test_target_missing_values(valid_mock_df):
    """Verify target column with NaN values is rejected."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[0, TARGET_COLUMN] = np.nan
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "missing (nan) values" in str(exc_info.value).lower()


def test_target_single_class_rejected(valid_mock_df):
    """Verify degenerate target with only one class is rejected."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df[TARGET_COLUMN] = 0
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "single class" in str(exc_info.value).lower()


def test_non_numeric_sensor_measurement(valid_mock_df):
    """Verify non-numeric string values in numeric sensor columns are rejected."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df["Rotational speed [rpm]"] = corrupted_df["Rotational speed [rpm]"].astype(str)
    corrupted_df.loc[0, "Rotational speed [rpm]"] = "corrupted_rpm_string"
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "non-numeric" in str(exc_info.value).lower()


def test_negative_physical_measurement(valid_mock_df):
    """Verify negative values in physical measurement columns are rejected."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[0, "Tool wear [min]"] = -10
    with pytest.raises(ValueError) as exc_info:
        validate_dataset(corrupted_df)
    assert "negative" in str(exc_info.value).lower()


def test_valid_dataset_validation(valid_mock_df):
    """Verify a conforming dataset passes validation and flags leakage columns."""
    report = validate_dataset(valid_mock_df)
    assert report["is_valid"] is True
    assert report["total_records"] == 25
    assert len(report["leakage_columns_flagged"]) == len(FAILURE_SUBTYPE_COLUMNS)


def test_unexpected_columns_isolation(valid_mock_df):
    """Verify extra columns are detected in report and NEVER included in model features."""
    extra_df = valid_mock_df.copy()
    extra_df["Arbitrary_Extra_Col"] = 999
    report = validate_dataset(extra_df)

    assert "Arbitrary_Extra_Col" in report["unexpected_cols"]

    # Verify extract_features ignores and drops arbitrary columns
    extracted = extract_features(extra_df)
    assert "Arbitrary_Extra_Col" not in extracted.columns
