"""Model training and evaluation module."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
)
import json
from src.data_loader import load_and_validate_dataset
from src.preprocessing import prepare_train_test_split, fit_and_transform_pipeline


def train_model(
    data_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Executes the training and evaluation pipeline:
    1. Loads dataset
    2. Performs stratified train/test split (no data leakage)
    3. Fits preprocessor strictly on training data
    4. Trains balanced Random Forest classifier
    5. Evaluates on holdout test set
    6. Persists trained model and preprocessor to disk

    Returns evaluation metrics dictionary.
    """
    df = load_and_validate_dataset(data_path)

    # 1. Split data
    X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_split(df)

    # 2. Preprocess & transform (fit on train only)
    X_train_proc, X_test_proc, preprocessor = fit_and_transform_pipeline(
        X_train_raw, X_test_raw
    )

    # 3. Model instantiation & fitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_proc, y_train)

    # 4. Evaluation on holdout test partition
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # 5. Persist artifacts
    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        metrics_path = OUTPUTS_DIR / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print("Initiating model training...")
    try:
        results = train_model()
        print("Training succeeded. Test metrics:")
        for k, v in results.items():
            if k != "classification_report":
                print(f"  {k}: {v}")
    except FileNotFoundError as err:
        print(f"Training stopped: {err}")
