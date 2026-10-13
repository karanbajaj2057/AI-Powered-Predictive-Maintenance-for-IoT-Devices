# Model Evaluation Framework & Measured Results

## 1. Experimental Setup & Leakage Controls
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic observations).
- **Split:** Stratified 80/20 train/test split (8,000 training observations, 2,000 held-out test observations).
- **Deterministic Seed:** `random_state = 42`.
- **Target Label:** `Machine failure` (binary 0 = normal, 1 = failure).
- **Failure-Mode Exclusion:** Failure-mode indicator columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`) are **strictly excluded** from training to prevent target leakage.

---

## 2. Model Architecture & Hyperparameters
- **Classifier:** scikit-learn `RandomForestClassifier`
- **Number of Estimators:** `80`
- **Maximum Depth:** `12`
- **Class Weights:** `class_weight='balanced'` (penalizes misclassification of the rare minority failure class)
- **Decision Threshold:** `0.50`

---

## 3. Measured Held-Out Evaluation Results (2,000 Observations)

These metrics represent the verified holdout performance on the 2,000 held-out test observations:

| Metric | Score | Benchmark / Interpretation |
| :--- | :---: | :--- |
| **Always-Healthy Baseline Accuracy** | `96.60%` | Predicting "0" for all samples yields 96.60% accuracy but **0% failure recall**. |
| **Random Forest Model Accuracy** | `97.55%` | Correct overall predictions on the test set. |
| **Precision** | `61.45%` | `51 / (51 + 32)`: When the model predicts failure, it is correct 61.45% of the time. |
| **Recall (Sensitivity)** | `75.00%` | `51 / (51 + 17)`: 75.0% of all real failures in the test set are successfully detected. |
| **F1-Score** | `67.55%` | Harmonic mean of precision and recall. |
| **Average Precision (PR-AUC)** | `77.04%` | Area under the precision-recall curve (vs 3.4% random chance). |

---

## 4. Confusion Matrix Breakdown

```
                    Predicted: No Failure (0)    Predicted: Failure (1)
Actual: No Failure (0)       1,900 (TN)                    32 (FP)
Actual: Failure (1)            17 (FN)                     51 (TP)
```

- **True Negatives (TN = 1,900):** Normal operating conditions correctly identified.
- **False Positives (FP = 32):** False alarms (machines flagged as risky but healthy).
- **False Negatives (FN = 17):** Critical missed failures that continued operating.
- **True Positives (TP = 51):** Successful proactive detections preventing unplanned breakdown.

> [!IMPORTANT]
> In industrial operations, **17 missed failures matter**. A real-world production deployment requires cost-sensitive threshold tuning, continuous calibration, and human-in-the-loop inspection.

---

## 5. Feature Importance Analysis (Gini Impurity)

Relative contribution of each sensor telemetry input to tree split purity:

| Rank | Sensor Feature | Importance (%) | Physical Role |
| :---: | :--- | :---: | :--- |
| 1 | **Torque [Nm]** | `35.77%` | Direct indicator of mechanical load, overstrain, and resistance. |
| 2 | **Rotational speed [rpm]** | `27.13%` | Spindle velocity and drive motor operating conditions. |
| 3 | **Tool wear [min]** | `21.57%` | Cumulative tool blade degradation and wear boundary limit. |
| 4 | **Air temperature [K]** | `8.74%` | Ambient environmental thermal baseline. |
| 5 | **Process temperature [K]** | `5.38%` | Operational chamber/motor heat generation. |
| 6 | **Product type (L/M/H)** | `1.41%` | Machine quality variant (Low, Medium, High). |

---

## 6. Output Artifact Locations
- **Metrics Summary:** [`outputs/metrics.json`](file:///C:/Users/ḍēl̥l̥/.gemini/antigravity/scratch/AI-Predictive-Maintenance-IoT/outputs/metrics.json)
- **Held-Out Predictions:** [`outputs/holdout_predictions.csv`](file:///C:/Users/ḍēl̥l̥/.gemini/antigravity/scratch/AI-Predictive-Maintenance-IoT/outputs/holdout_predictions.csv)
- **Forest JSON Structure:** [`outputs/forest.json`](file:///C:/Users/ḍēl̥l̥/.gemini/antigravity/scratch/AI-Predictive-Maintenance-IoT/outputs/forest.json)
- **Model Checkpoint:** [`models/model.joblib`](file:///C:/Users/ḍēl̥l̥/.gemini/antigravity/scratch/AI-Predictive-Maintenance-IoT/models/model.joblib)
