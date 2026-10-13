"""Main CLI entry point for AI-Predictive-Maintenance-IoT.

This module provides command-line control for dataset preparation,
model training, telemetry streaming, and inference runs.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow execution from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console output on Windows with unicode paths
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import RAW_DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH


def run_training():
    """Triggers model training workflow."""
    print("=" * 60)
    print("Initiating Model Training Workflow")
    print("=" * 60)
    from src.train import train_model

    try:
        metrics = train_model()
        print("\nTraining completed successfully! Test partition metrics:")
        for metric_name, value in metrics.items():
            if metric_name != "classification_report":
                print(f"  - {metric_name}: {value}")
    except (FileNotFoundError, ValueError) as err:
        print(f"\n[BLOCKED] Training aborted: {err}")
        sys.exit(1)


def run_validation():
    """Validates the raw dataset against UCI AI4I schema."""
    print("=" * 60)
    print("Validating Raw Dataset Schema and Integrity")
    print("=" * 60)
    from src.data_loader import load_raw_data, validate_dataset

    try:
        df = load_raw_data()
        report = validate_dataset(df)
        print("\n[PASS] Dataset validation successful!")
        print(f"  - Total records:      {report['total_records']}")
        print(f"  - Total columns:      {report['total_columns']}")
        print(f"  - Target distribution:{report['target_distribution']}")
        print(f"  - Null counts:        {report['null_counts']}")
        if report['unexpected_cols']:
            print(f"  - Non-feature cols:   {report['unexpected_cols']}")
        print(f"  - Leakage prevented:  {report['leakage_columns_flagged']}")
    except (FileNotFoundError, ValueError) as err:
        print(f"\n[BLOCKED] Dataset validation failed: {err}")
        sys.exit(1)


def run_simulation(n_steps: int = 5):
    """Triggers simulated telemetry stream."""
    print("=" * 60)
    print(f"Running Simulated Telemetry Stream ({n_steps} cycles)")
    print("=" * 60)
    from src.simulator import TelemetrySimulator

    simulator = TelemetrySimulator()
    for i, batch in enumerate(simulator.stream_records(batch_size=1)):
        record = batch.iloc[0]
        print(
            f"Cycle {i+1:02d} | Type: {record['Type']} | "
            f"Air Temp: {record['Air temperature [K]']}K | "
            f"Torque: {record['Torque [Nm]']}Nm | "
            f"Tool Wear: {record['Tool wear [min]']}min"
        )
        if i + 1 >= n_steps:
            break


def run_prediction_pipeline():
    """Runs prediction on simulated batch and persists logs."""
    print("=" * 60)
    print("Executing Batch Prediction Pipeline")
    print("=" * 60)
    from src.simulator import TelemetrySimulator
    from src.predict import predict_record
    from src.alert_engine import generate_alerts
    from src.storage import save_predictions_log, save_alerts_log

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        print("[BLOCKED] Model artifacts not found. Please run training first (`python main.py --train`).")
        sys.exit(1)

    sim = TelemetrySimulator()
    batch = next(sim.stream_records(batch_size=10))
    preds = predict_record(batch)
    alerts = generate_alerts(preds)

    pred_path = save_predictions_log(preds)
    alert_path = save_alerts_log(alerts)

    print(f"Predictions saved to: {pred_path}")
    print(f"Alert logs saved to:  {alert_path}")
    print(f"Generated {len(alerts)} alert records.")


def print_system_status():
    """Prints current repository asset status."""
    print("=" * 60)
    print("AI-Predictive-Maintenance-IoT - Status Overview")
    print("=" * 60)
    print(f"Dataset (`{RAW_DATA_PATH.name}`):   {'[FOUND]' if os.path.exists(RAW_DATA_PATH) else '[MISSING] (data/raw/ai4i2020.csv)'}")
    print(f"Model Checkpoint:              {'[FOUND]' if os.path.exists(MODEL_PATH) else '[NOT TRAINED]'}")
    print(f"Preprocessor Checkpoint:       {'[FOUND]' if os.path.exists(PREPROCESSOR_PATH) else '[NOT FITTED]'}")
    print("\nAvailable commands:")
    print("  python main.py --validate      Validate raw dataset schema and integrity")
    print("  python main.py --train         Train model on dataset")
    print("  python main.py --simulate      Simulate telemetry stream")
    print("  python main.py --predict       Run inference and alert generation")
    print("  streamlit run dashboard/app.py Launch Streamlit monitoring dashboard")
    print("  pytest tests/                  Run unit test suite")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Predictive-Maintenance-IoT: CLI Controller and Execution Entry Point"
    )
    parser.add_argument("--validate", action="store_true", help="Validate raw dataset against required schema and integrity checks")
    parser.add_argument("--train", action="store_true", help="Train predictive maintenance model on raw dataset")
    parser.add_argument("--simulate", action="store_true", help="Stream simulated telemetry records to stdout")
    parser.add_argument("--predict", action="store_true", help="Run model inference and generate alert logs")
    parser.add_argument("--steps", type=int, default=5, help="Number of telemetry cycles for simulation (default: 5)")
    parser.add_argument("--status", action="store_true", help="Display environment and artifact readiness status")

    args = parser.parse_args()

    if args.validate:
        run_validation()
    elif args.train:
        run_training()
    elif args.simulate:
        run_simulation(n_steps=args.steps)
    elif args.predict:
        run_prediction_pipeline()
    elif args.status:
        print_system_status()
    else:
        print_system_status()


if __name__ == "__main__":
    main()
