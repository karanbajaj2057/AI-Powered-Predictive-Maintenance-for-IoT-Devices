"""Telemetry stream simulator for IoT predictive maintenance.

NOTICE: This simulator provides educational simulation only.
It streams simulated records or replays dataset rows and does NOT connect
to physical IoT hardware, sensors, or machinery actuators.
"""

import time
from typing import Generator, Optional
import pandas as pd
from pathlib import Path

from src.data_loader import load_raw_data, generate_mock_telemetry_data
from src.config import RAW_DATA_PATH


class TelemetrySimulator:
    """
    Simulates streaming sensor telemetry records.
    Can replay historical records or generate synthetic data streams.
    """

    def __init__(self, data_path: Optional[Path] = None, use_mock_if_missing: bool = True):
        self.data_path = data_path or RAW_DATA_PATH
        self.use_mock_if_missing = use_mock_if_missing
        self.data = self._load_source()

    def _load_source(self) -> pd.DataFrame:
        try:
            df = load_raw_data(self.data_path)
            print(f"[SIMULATED TELEMETRY] Replaying {len(df)} records from {self.data_path}")
            return df
        except FileNotFoundError:
            if self.use_mock_if_missing:
                print("[SIMULATED TELEMETRY] Raw file absent; generating synthetic telemetry stream.")
                return generate_mock_telemetry_data(n_samples=100)
            raise

    def stream_records(
        self,
        batch_size: int = 1,
        interval_seconds: float = 0.0,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yields batches of simulated sensor telemetry.
        """
        total = len(self.data)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = self.data.iloc[start_idx:end_idx].copy()
            batch["is_simulated_telemetry"] = True
            yield batch
            if interval_seconds > 0:
                time.sleep(interval_seconds)


if __name__ == "__main__":
    sim = TelemetrySimulator()
    print("Testing stream simulator (first 3 records):")
    for i, batch in enumerate(sim.stream_records(batch_size=1)):
        print(f"Sample {i+1}:\n", batch[["Type", "Air temperature [K]", "Torque [Nm]", "Tool wear [min]"]])
        if i >= 2:
            break
