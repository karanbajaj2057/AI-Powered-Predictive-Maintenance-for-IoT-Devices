"""Unit tests for inference and prediction pipeline."""

import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

from src.data_loader import generate_mock_telemetry_data
from src.preprocessing import fit_and_transform_pipeline, prepare_train_test_split
from src.predict import predict_record


def test_predict_record_with_mock_model():
    """Verify inference pipeline calculates predictions and probabilities in valid bounds."""
    df_mock = generate_mock_telemetry_data(n_samples=30)
    X_train_raw, X_test_raw, y_train, _ = prepare_train_test_split(df_mock, test_size=0.3)

    X_train_proc, _, preprocessor = fit_and_transform_pipeline(X_train_raw, X_test_raw)

    # Train lightweight deterministic classifier for test verification
    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(X_train_proc, y_train)

    test_samples = X_test_raw.head(5)
    pred_df = predict_record(test_samples, model=dummy_model, preprocessor=preprocessor)

    assert "failure_prediction" in pred_df.columns
    assert "failure_probability" in pred_df.columns
    assert len(pred_df) == 5
    assert pred_df["failure_prediction"].isin([0, 1]).all()
    assert ((pred_df["failure_probability"] >= 0.0) & (pred_df["failure_probability"] <= 1.0)).all()
