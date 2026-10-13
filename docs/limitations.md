# Project Scope and Engineering Limitations

## 1. Nature of the Dataset
- **Synthetic Origin:** The UCI AI4I 2020 dataset was generated synthetically by modeling physical machinery equations. While realistic, it does not capture environmental noise, sensor degradation drift, electrical interference, or anomalous mechanical states found in physical factory environments.
- **Tabular Snapshot Nature:** The dataset consists of independent tabular rows rather than continuous, high-frequency time-series waveforms (e.g., raw vibration accelerometer streams).

## 2. Simulated Telemetry Boundary
- **No Physical Hardware:** The telemetry simulator (`src/simulator.py`) replays existing dataset records or emits synthetically generated sample rows. It does not interface with physical PLCs, microcontrollers, edge gateways, Modbus/OPC-UA protocols, or physical sensors.
- **No Actuation or Control:** The system generates informational and advisory alerts only. It has zero capability and zero authorization to perform machinery actuation, emergency stops, or hardware throttling.

## 3. Absence of Remaining Useful Life (RUL) Modeling
- **Classification vs. RUL:** The implemented model is a **binary failure risk classifier** (\(P(\text{failure} \mid \mathbf{x})\)).
- **Time-to-Failure Limitation:** The model does **NOT** estimate the exact continuous time, hours, or cycles remaining before equipment failure occurs (Remaining Useful Life). RUL estimation requires time-series survival analysis, degradation regression modeling, or Wiener/gamma degradation processes which are not part of this baseline classification prototype.

## 4. Probabilistic Interpretation
- All alert messages emitted by `alert_engine.py` represent probabilistic risk indicators derived from sensor correlations, and should never be interpreted as certainty of mechanical breakdown.
