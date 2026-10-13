"""Unit tests for the alert generation engine."""

import pytest
import pandas as pd
from src.alert_engine import evaluate_alert, generate_alerts


def test_alert_severity_thresholds():
    """Verify alert severity bands (NORMAL, WARNING, CRITICAL)."""
    # 1. Normal condition
    rec_normal = pd.Series({"failure_probability": 0.15, "failure_prediction": 0})
    alert_norm = evaluate_alert(rec_normal)
    assert alert_norm["severity"] == "NORMAL"
    assert "operational baseline" in alert_norm["message"].lower()

    # 2. Warning condition
    rec_warn = pd.Series({"failure_probability": 0.55, "failure_prediction": 1})
    alert_warn = evaluate_alert(rec_warn)
    assert alert_warn["severity"] == "WARNING"
    assert "preventative" in alert_warn["message"].lower()

    # 3. Critical condition
    rec_crit = pd.Series({"failure_probability": 0.88, "failure_prediction": 1, "Tool wear [min]": 220})
    alert_crit = evaluate_alert(rec_crit)
    assert alert_crit["severity"] == "CRITICAL"
    assert "immediate technical inspection" in alert_crit["message"].lower()


def test_alert_safety_language():
    """Ensure alert engine messages are advisory and do not state failure is certain."""
    rec = pd.Series({"failure_probability": 0.99, "failure_prediction": 1})
    alert = evaluate_alert(rec)

    msg = alert["message"].lower()
    # Ensure probabilistic / advisory wording
    assert "advisory" in msg or "estimated" in msg
    assert "certain" not in msg
    assert "guaranteed" not in msg


def test_batch_generate_alerts():
    """Verify batch alert processing."""
    df_preds = pd.DataFrame([
        {"failure_probability": 0.1, "failure_prediction": 0},
        {"failure_probability": 0.5, "failure_prediction": 1},
        {"failure_probability": 0.9, "failure_prediction": 1},
    ])
    alerts = generate_alerts(df_preds)
    assert len(alerts) == 3
    assert [a["severity"] for a in alerts] == ["NORMAL", "WARNING", "CRITICAL"]
