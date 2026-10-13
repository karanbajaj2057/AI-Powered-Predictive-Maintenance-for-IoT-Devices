"""Streamlit Dashboard for AI-Predictive-Maintenance-IoT."""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    ALERT_THRESHOLDS,
)
from src.data_loader import generate_mock_telemetry_data, load_raw_data
from src.predict import predict_record
from src.alert_engine import evaluate_alert

st.set_page_config(
    page_title="AI Predictive Maintenance IoT",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AI-Powered Predictive Maintenance IoT Prototype")
st.markdown(
    "Educational telemetry monitoring and probabilistic equipment failure risk assessment."
)

st.sidebar.header("System Controls & Status")

# Check status of assets
data_exists = os.path.exists(RAW_DATA_PATH)
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)

st.sidebar.markdown(f"**Dataset status:** {'🟢 Available' if data_exists else '🟡 Missing (Mock active)'}")
st.sidebar.markdown(f"**Model status:** {'🟢 Trained' if model_exists else '🟡 Untrained'}")

# Mode selection
mode = st.sidebar.radio(
    "Select View",
    ["Telemetry Monitor & Live Inference", "Dataset Explorer", "System Documentation & Architecture"],
)

if mode == "Telemetry Monitor & Live Inference":
    st.subheader("📡 Simulated IoT Telemetry & Real-Time Risk Assessment")
    st.caption("Note: All data displayed is simulated telemetry. No physical machine connection exists.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        machine_type = st.selectbox("Machine Type (Variant)", ["L", "M", "H"], index=0)
    with col2:
        air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
    with col3:
        process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
    with col4:
        rot_speed = st.slider("Rotational Speed [rpm]", 1100, 2900, 1500, 10)

    col5, col6 = st.columns(2)
    with col5:
        torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
    with col6:
        tool_wear = st.slider("Tool Wear [min]", 0, 260, 50, 1)

    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    st.markdown("### Telemetry Snapshot")
    st.dataframe(input_df, use_container_width=True)

    if model_exists:
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            pred_df = predict_record(input_df, model=model, preprocessor=preprocessor)
            alert = evaluate_alert(pred_df.iloc[0])

            prob = alert["failure_probability"]
            severity = alert["severity"]

            st.markdown("### Advisory Diagnostic Output")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Failure Probability", f"{prob:.1%}")
            m_col2.metric("Failure Prediction Flag", "Failure Risk (1)" if alert["failure_prediction"] == 1 else "Normal (0)")
            m_col3.metric("Severity Level", severity)

            if severity == "CRITICAL":
                st.error(alert["message"])
            elif severity == "WARNING":
                st.warning(alert["message"])
            else:
                st.success(alert["message"])

            if alert["contributing_factors"]:
                st.info("Key Observation Factors: " + "; ".join(alert["contributing_factors"]))

        except Exception as e:
            st.error(f"Error during model inference: {e}")
    else:
        st.warning(
            "Model artifacts not found on disk. Run `python main.py --train` after placing the dataset in `data/raw/`."
        )

elif mode == "Dataset Explorer":
    st.subheader("📊 Dataset Overview")
    if data_exists:
        df = load_raw_data()
        st.write(f"Total rows: {len(df):,}")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("Raw dataset `ai4i2020.csv` is not present locally. Displaying simulated sample:")
        mock_df = generate_mock_telemetry_data(n_samples=20)
        st.dataframe(mock_df, use_container_width=True)

elif mode == "System Documentation & Architecture":
    st.subheader("📖 Architecture & Operational Boundaries")
    st.markdown(r"""
    - **Telemetry Model:** Binary classification estimating \(P(\text{failure} \mid \mathbf{x})\).
    - **Safety Principles:** Strictly advisory notifications; no physical equipment controls.
    - **RUL Notice:** Remaining Useful Life is not estimated in this tabular prototype.
    """)
