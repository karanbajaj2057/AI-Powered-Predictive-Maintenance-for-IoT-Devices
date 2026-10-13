"""Data preprocessing and transformation pipeline."""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ENGINEERED_FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
)
from src.feature_engineering import extract_features


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an unfitted scikit-learn ColumnTransformer.
    - OneHotEncoder on categorical features ('Type')
    - StandardScaler on numerical features and engineered features
    """
    numeric_cols = NUMERICAL_FEATURES + ENGINEERED_FEATURE_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits raw DataFrame into training and test partitions BEFORE preprocessing.
    Guarantees zero data leakage.

    Returns:
        (X_train_raw, X_test_raw, y_train, y_test)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    y = df[TARGET_COLUMN].copy()
    X_raw = df.copy()

    # Stratified split to maintain rare failure event distribution
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train_raw, X_test_raw, y_train, y_test


def fit_and_transform_pipeline(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Applies feature engineering, fits the preprocessor ON TRAINING DATA ONLY,
    and transforms both train and test splits.
    """
    # 1. Feature Engineering (deterministic row-level transformations)
    X_train_feat = extract_features(X_train_raw)
    X_test_feat = extract_features(X_test_raw)

    # 2. Fit preprocessor ONLY on training set
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_feat)

    # 3. Transform test set using the fitted preprocessor
    X_test_processed = preprocessor.transform(X_test_feat)

    return X_train_processed, X_test_processed, preprocessor
