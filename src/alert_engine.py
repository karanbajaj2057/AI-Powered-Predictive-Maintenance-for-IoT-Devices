"""Alert generation engine for predictive maintenance telemetry."""

import pandas as pd
from typing import Dict, Any, List
from src.config import ALERT_THRESHOLDS


def evaluate_alert(record: pd.Series) -> Dict[str, Any]:
    """
    Evaluates risk and produces an advisory alert for a single telemetry record.

    Advisory Safety Notice:
    Alert messages describe probabilistic risk indicators only and do not claim
    certain failure. No physical actuation or machinery control is executed.
    """
    prob = float(record.get("failure_probability", 0.0))
    pred = int(record.get("failure_prediction", 0))

    reasons = []
    # Heuristic telemetry context checks
    if record.get("Tool wear [min]", 0) > 200:
        reasons.append("Tool wear exceeds high runtime boundary (>200 min)")
    if record.get("Torque [Nm]", 0) > 60:
        reasons.append("High mechanical torque observed (>60 Nm)")
    if (record.get("Process temperature [K]", 0) - record.get("Air temperature [K]", 0)) < 8.5:
        reasons.append("Narrow thermal differential indicates possible heat dissipation constraint")

    if prob >= ALERT_THRESHOLDS["CRITICAL"]:
        severity = "CRITICAL"
        message = (
            f"Advisory CRITICAL: Estimated failure probability is elevated at {prob:.1%}. "
            f"Immediate technical inspection recommended. Contributing factors: {'; '.join(reasons) or 'Multiple multivariate sensor anomalies'}."
        )
    elif prob >= ALERT_THRESHOLDS["WARNING"]:
        severity = "WARNING"
        message = (
            f"Advisory WARNING: Estimated failure probability increased to {prob:.1%}. "
            f"Schedule preventative diagnostic review. Contributing factors: {'; '.join(reasons) or 'Sensor deviations detected'}."
        )
    else:
        severity = "NORMAL"
        message = f"Telemetry parameters within normal operational baseline (estimated risk: {prob:.1%})."

    return {
        "severity": severity,
        "failure_prediction": pred,
        "failure_probability": prob,
        "message": message,
        "contributing_factors": reasons,
    }


def generate_alerts(df_predictions: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generates alerts for a batch of prediction records.
    """
    alerts = []
    for idx, row in df_predictions.iterrows():
        alert = evaluate_alert(row)
        alert["record_index"] = idx
        alerts.append(alert)
    return alerts
