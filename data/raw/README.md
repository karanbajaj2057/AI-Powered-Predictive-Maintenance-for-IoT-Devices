# Raw Data Directory

## Purpose
This folder holds the original, unmodified dataset files.

## Expected Files
- `ai4i2020.csv`: The UCI AI4I 2020 Predictive Maintenance Dataset.

## Ingestion Rule
- Never modify files in this directory directly. All cleaning, transformation, and feature extraction must be handled reproducibly via scripts in `src/`.
- This folder is ignored by git (except documentation) to prevent tracking large binaries or proprietary dumps.
