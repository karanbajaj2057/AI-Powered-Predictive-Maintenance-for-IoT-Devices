"""Configuration and constants for AI-Predictive-Maintenance-IoT."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "ai4i2020.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "predictive_maintenance_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
ALERTS_DIR = OUTPUTS_DIR / "alerts"

# Dataset Columns (UCI AI4I 2020)
ID_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# TARGET SAFETY:
# Binary failure target:
TARGET_COLUMN = "Machine failure"

# Failure subtypes to EXCLUDE from input features (leakage prevention):
FAILURE_SUBTYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Base features permitted for model inputs (excluding target and subtypes)
BASE_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Engineered feature names
ENGINEERED_FEATURE_COLUMNS = [
    "temp_diff",           # Process temperature [K] - Air temperature [K]
    "power_factor",        # Rotational speed [rpm] * Torque [Nm]
    "tool_wear_torque",    # Tool wear [min] * Torque [Nm]
]

# All input features expected by model after feature engineering
FINAL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

# Training Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Alert Engine Thresholds
ALERT_THRESHOLDS = {
    "WARNING": 0.40,   # Probability >= 0.40 and < 0.75
    "CRITICAL": 0.75,  # Probability >= 0.75
}
