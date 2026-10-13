# Data Directory

This directory manages all dataset stages for the **AI-Predictive-Maintenance-IoT** project.

## Subdirectories
- `raw/`: Raw, unaltered source datasets (e.g., `ai4i2020.csv` from the UCI Machine Learning Repository).
- `processed/`: Cleaned, encoded, and split datasets ready for model training and validation.

## Acquisition Instructions
Download the official **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:
- URL: `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`
- Save the CSV as: `data/raw/ai4i2020.csv`

## GitHub Tracking Policy
- **Do not commit raw CSV data** or large processed datasets to GitHub (excluded via `.gitignore`).
- Only documentation and `.gitkeep`/`README.md` files in this directory are tracked.
