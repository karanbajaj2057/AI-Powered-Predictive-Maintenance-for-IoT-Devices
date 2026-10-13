"""Feature engineering pipeline for predictive maintenance telemetry."""

import pandas as pd
from typing import List
from src.config import (
    BASE_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    FINAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    FAILURE_SUBTYPE_COLUMNS,
    ID_COLUMNS,
)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features from sensor inputs.

    Engineered features:
    - temp_diff: Difference between process and ambient air temperature [K]
    - power_factor: Mechanical power proxy (Rotational speed * Torque)
    - tool_wear_torque: Interaction between cumulative tool wear and mechanical torque

    Returns a new DataFrame containing base and engineered features.
    """
    df_feat = df.copy()

    # Safety: Exclude target and failure modes if present in incoming DataFrame
    cols_to_drop = [TARGET_COLUMN] + FAILURE_SUBTYPE_COLUMNS + ID_COLUMNS
    for col in cols_to_drop:
        if col in df_feat.columns:
            df_feat = df_feat.drop(columns=[col])

    # Validate essential input features exist
    required_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    missing = [c for c in required_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns for feature extraction: {missing}")

    # Engineered feature 1: Temperature difference
    df_feat["temp_diff"] = (
        df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    ).round(2)

    # Engineered feature 2: Power factor (Speed * Torque proxy)
    df_feat["power_factor"] = (
        df_feat["Rotational speed [rpm]"] * df_feat["Torque [Nm]"]
    ).round(2)

    # Engineered feature 3: Tool wear * Torque interaction
    df_feat["tool_wear_torque"] = (
        df_feat["Tool wear [min]"] * df_feat["Torque [Nm]"]
    ).round(2)

    return df_feat[FINAL_FEATURE_COLUMNS]
