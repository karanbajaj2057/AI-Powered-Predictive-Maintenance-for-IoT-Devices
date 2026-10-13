# Outputs Directory

## Purpose
This directory contains all runtime logs and predictions generated during model inference, batch evaluation, and telemetry simulation runs.

## Subdirectories
- `predictions/`: CSV logs containing model inference scores, failure probabilities, and predictions.
- `alerts/`: JSON logs containing structured advisory alert events emitted by `alert_engine.py`.

## Git Tracking Policy
- Output runtime logs are transient and excluded from GitHub tracking via `.gitignore`.
- Only folder documentation files are tracked.
