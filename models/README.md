# Models Directory

## Purpose
This directory stores trained scikit-learn model checkpoints, serialization objects, and fitted preprocessing transformers.

## Expected Files
- `predictive_maintenance_model.joblib`: Serialized trained classifier (e.g., Random Forest).
- `preprocessor.joblib`: Serialized fitted `ColumnTransformer` (OneHotEncoder + StandardScaler).

## Generation
Generated automatically by running the training pipeline:
```bash
python main.py --train
```
or executing `src/train.py`.

## Git Tracking Policy
- **Binary model weights and serialized transformers should not be committed to GitHub** (excluded via `.gitignore`).
- Only this documentation file is tracked in git.
- For reproducibility, artifacts should be recreated deterministically via code or versioned using tools like DVC or MLflow in production environments.
