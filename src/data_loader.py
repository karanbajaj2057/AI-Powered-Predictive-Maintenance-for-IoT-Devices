"""Data loading utilities for AI-Predictive-Maintenance-IoT."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    FAILURE_SUBTYPE_COLUMNS,
    ID_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


import logging
import warnings

logger = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates that a DataFrame conforms to the UCI AI4I 2020 Predictive Maintenance schema.

    Checks performed:
    1. Rejects empty DataFrames.
    2. Verifies presence of all required sensor and target columns.
    3. Confirms numerical sensor features can be converted to numeric and satisfy physical boundaries.
    4. Validates target column ('Machine failure') contains strictly binary labels (0 and 1).
    5. Identifies missing/null values across all columns.
    6. Identifies and isolates unexpected columns without allowing them to enter model features.
    7. Confirms failure subtype columns (TWF, HDF, PWF, OSF, RNF) are flagged for exclusion.

    Returns:
        Dict[str, Any]: Validation summary report.

    Raises:
        ValueError: If dataset violates schema, contains invalid types, or corrupt target labels.
    """
    if df is None or df.empty or len(df) == 0:
        raise ValueError(
            "Dataset validation failed: The dataset is empty (0 rows).\n"
            "The official UCI AI4I 2020 dataset should contain 10,000 records."
        )

    # 1. Check required columns
    required_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset validation failed: Missing required columns: {missing_cols}.\n"
            f"Expected required columns: {required_cols}.\n"
            f"Columns present in file: {list(df.columns)}.\n"
            "Please check that your CSV has not been renamed or modified."
        )

    # 2. Check and validate numerical sensor features
    for col in NUMERICAL_FEATURES:
        # Check conversion to numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        n_invalid = converted.isna().sum() - df[col].isna().sum()
        if n_invalid > 0:
            raise ValueError(
                f"Dataset validation failed: Column '{col}' contains {n_invalid} "
                "non-numeric values that cannot be parsed as numbers. "
                "All sensor telemetry measurements must be numeric."
            )

        # Check physical boundary sanity
        min_val = converted.dropna().min()
        if "temperature" in col.lower() and min_val <= 0:
            raise ValueError(
                f"Dataset validation failed: Temperature column '{col}' contains values <= 0 K ({min_val}). "
                "Temperatures must be positive absolute values in Kelvin (typically 290K - 320K)."
            )
        if ("speed" in col.lower() or "torque" in col.lower() or "wear" in col.lower()) and min_val < 0:
            raise ValueError(
                f"Dataset validation failed: Measurement column '{col}' contains negative values ({min_val}). "
                "Speed, torque, and tool wear cannot be negative."
            )

    # 3. Check categorical feature 'Type'
    invalid_types = set(df["Type"].dropna().unique()) - {"L", "M", "H"}
    if invalid_types:
        warnings.warn(
            f"Categorical column 'Type' contains unexpected variants: {invalid_types}. "
            "Expected product quality variants are 'L', 'M', or 'H'."
        )

    # 4. Check target column ('Machine failure')
    target_series = df[TARGET_COLUMN]
    if target_series.isna().any():
        n_na = target_series.isna().sum()
        raise ValueError(
            f"Dataset validation failed: Target column '{TARGET_COLUMN}' contains {n_na} missing (NaN) values. "
            "Every sample must have a valid binary target label."
        )

    unique_targets = set(target_series.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Dataset validation failed: Target column '{TARGET_COLUMN}' contains invalid non-binary values: {unique_targets}.\n"
            "Target labels must be binary: 0 (no failure) or 1 (machine failure)."
        )

    if len(unique_targets) < 2:
        raise ValueError(
            f"Dataset validation failed: Target column '{TARGET_COLUMN}' contains only a single class: {unique_targets}.\n"
            "Supervised classification requires both normal (0) and failure (1) samples."
        )

    # 5. Check null values across entire dataframe
    null_counts = df[required_cols].isna().sum().to_dict()
    has_nulls = any(cnt > 0 for cnt in null_counts.values())
    if has_nulls:
        warnings.warn(f"Dataset contains null values in required columns: {null_counts}")

    # 6. Detect and isolate unexpected columns
    known_cols = set(ID_COLUMNS + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN] + FAILURE_SUBTYPE_COLUMNS)
    unexpected_cols = [c for c in df.columns if c not in known_cols]
    if unexpected_cols:
        logger.info(
            f"Notice: Dataset contains extra/unexpected columns: {unexpected_cols}. "
            "These will be ignored and excluded from model input features."
        )

    # Calculate class balance
    target_dist = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()

    return {
        "is_valid": True,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_cols": [],
        "unexpected_cols": unexpected_cols,
        "null_counts": null_counts,
        "target_distribution": target_dist,
        "leakage_columns_flagged": [c for c in FAILURE_SUBTYPE_COLUMNS if c in df.columns],
    }


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the UCI AI4I 2020 dataset from disk without validation.
    """
    target_path = data_path or RAW_DATA_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Dataset not found at: {target_path}.\n"
            "Please download the dataset from the UCI Machine Learning Repository:\n"
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            "and place 'ai4i2020.csv' in the 'data/raw/' directory."
        )

    df = pd.read_csv(target_path)
    return df


def load_and_validate_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and rigorously validates the UCI AI4I 2020 dataset.

    Returns:
        pd.DataFrame: The loaded, validated DataFrame.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
        ValueError: If the dataset fails validation criteria.
    """
    df = load_raw_data(data_path)
    report = validate_dataset(df)
    return df


def generate_mock_telemetry_data(n_samples: int = 50, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic synthetic DataFrame mimicking the UCI AI4I 2020 dataset.
    Used for unit testing, offline dry-runs, and telemetry demonstrations when raw data is absent.
    """
    rng = np.random.RandomState(random_state)
    types = rng.choice(["L", "M", "H"], size=n_samples, p=[0.6, 0.3, 0.1])
    air_temp = rng.normal(300.0, 2.0, size=n_samples).round(1)
    process_temp = (air_temp + rng.normal(10.0, 1.0, size=n_samples)).round(1)
    rot_speed = rng.normal(1538.0, 179.0, size=n_samples).astype(int)
    torque = rng.normal(40.0, 9.9, size=n_samples).round(1)
    tool_wear = rng.uniform(0, 240, size=n_samples).astype(int)

    # Synthetic target with ~3.4% failure rate reflecting UCI dataset distribution
    probs = 1 / (1 + np.exp(-(0.02 * tool_wear + 0.05 * torque - 10)))
    machine_failure = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)

    df = pd.DataFrame({
        "UDI": np.arange(1, n_samples + 1),
        "Product ID": [f"{t}1000{i}" for i, t in enumerate(types)],
        "Type": types,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Machine failure": machine_failure,
        "TWF": np.zeros(n_samples, dtype=int),
        "HDF": np.zeros(n_samples, dtype=int),
        "PWF": np.zeros(n_samples, dtype=int),
        "OSF": np.zeros(n_samples, dtype=int),
        "RNF": np.zeros(n_samples, dtype=int),
    })
    return df
