# Predictions Output Directory

## Purpose
Stores inference results, predicted failure probabilities, and binary predictions from batch runs or streaming simulations.

## Expected Files
- `predictions_YYYYMMDD_HHMMSS.csv`: Tabular predictions with input sensor features, `failure_prediction`, and `failure_probability`.

## Generation
Generated when executing prediction routines via:
```bash
python main.py --predict
```
or through `src/storage.py::save_predictions_log`.

## Git Tracking
- Excluded from GitHub commits via `.gitignore`.
