"""Inference and prediction pipeline."""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_PATH, PREPROCESSOR_PATH
from src.feature_engineering import extract_features


def load_artifacts(
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """
    Loads saved model and preprocessor objects from disk.
    """
    m_path = model_path or MODEL_PATH
    p_path = preprocessor_path or PREPROCESSOR_PATH

    if not os.path.exists(m_path):
        raise FileNotFoundError(f"Model artifact not found at: {m_path}. Run training first.")
    if not os.path.exists(p_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at: {p_path}. Run training first.")

    model = joblib.load(m_path)
    preprocessor = joblib.load(p_path)
    return model, preprocessor


def predict_record(
    input_df: pd.DataFrame,
    model: Optional[Any] = None,
    preprocessor: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Runs model inference on an input DataFrame containing raw sensor columns.

    Returns DataFrame augmented with:
    - failure_prediction (0 or 1)
    - failure_probability (0.0 to 1.0)
    """
    if model is None or preprocessor is None:
        loaded_model, loaded_preproc = load_artifacts()
        model = model or loaded_model
        preprocessor = preprocessor or loaded_preproc

    # Extract engineered features
    features_df = extract_features(input_df)

    # Transform features using fitted preprocessor
    features_proc = preprocessor.transform(features_df)

    # Infer predictions and probabilities
    preds = model.predict(features_proc)
    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(features_proc)
        if proba_matrix.shape[1] > 1:
            classes = list(getattr(model, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            probs = proba_matrix[:, pos_idx]
        else:
            classes = list(getattr(model, "classes_", [0]))
            probs = proba_matrix[:, 0] if classes[0] == 1 else np.zeros(len(features_proc))
    else:
        probs = preds.astype(float)

    output_df = input_df.copy()
    output_df["failure_prediction"] = preds
    output_df["failure_probability"] = probs.round(4)

    return output_df
