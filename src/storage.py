"""Persistence and logging storage for model outputs and alerts."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import PREDICTIONS_DIR, ALERTS_DIR


def save_predictions_log(
    df_predictions: pd.DataFrame,
    filename: Optional[str] = None,
) -> Path:
    """
    Persists inference results to outputs/predictions/.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"predictions_{ts}.csv"
    output_path = PREDICTIONS_DIR / fname

    df_predictions.to_csv(output_path, index=False)
    return output_path


def save_alerts_log(
    alerts: List[Dict[str, Any]],
    filename: Optional[str] = None,
) -> Path:
    """
    Persists structured alert events to outputs/alerts/.
    """
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"alerts_{ts}.json"
    output_path = ALERTS_DIR / fname

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "total_alerts": len(alerts),
                "alerts": alerts,
            },
            f,
            indent=2,
        )

    return output_path
