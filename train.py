"""Reproduce Sentinel's AI4I evaluation as documented on https://predictivemaintenanceiot.lovable.app

Run directly:
    python train.py
"""

from pathlib import Path
import json
import os
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

csv = DATA / "ai4i2020.csv"
if not csv.exists():
    archive = DATA / "dataset.zip"
    print("Downloading UCI AI4I 2020 dataset from official repository...")
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        archive,
    )
    with zipfile.ZipFile(archive) as z:
        z.extract("ai4i2020.csv", DATA)
    if archive.exists():
        archive.unlink()
    print("Dataset extracted successfully.")

df = pd.read_csv(csv)
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]
X = df[FEATURES].copy()
X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2})
y = df["Machine failure"]

# SAFETY: Never use failure-mode columns (TWF, HDF, PWF, OSF, RNF) or identifiers (UDI, Product ID) as inputs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "averagePrecision": float(average_precision_score(y_test, prob)),
    "confusion": confusion_matrix(y_test, pred).tolist(),
    "trainRows": len(X_train),
    "testRows": len(X_test),
    "totalRows": len(df),
    "failures": int(y.sum()),
    "featureImportance": model.feature_importances_.tolist(),
    "baselineAccuracy": float((y_test == 0).mean()),
}

joblib.dump(model, MODELS / "model.joblib")
joblib.dump(model, MODELS / "predictive_maintenance_model.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Browser-safe export of the actual trained forest for virtual simulation
trees = []
for estimator in model.estimators_:
    t = estimator.tree_
    values = t.value[:, 0, :]
    probs = values[:, 1] / values.sum(axis=1)
    trees.append({
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "probability": probs.tolist(),
    })
(OUT / "forest.json").write_text(json.dumps(trees, separators=(",", ":")), encoding="utf-8")

# Save genuine holdout predictions for inspection and plotting
result = X_test.copy()
result["actual_failure"] = y_test
result["predicted_failure"] = pred
result["failure_probability"] = prob
result.to_csv(OUT / "holdout_predictions.csv", index=False)

print("=" * 60)
print("Sentinel Model Training & Evaluation Complete")
print("=" * 60)
print(json.dumps(metrics, indent=2))
