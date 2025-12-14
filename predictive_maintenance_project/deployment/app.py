import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="wash9968/predictive-maintainace-prediction", filename="best_predict_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Engine Failure Prediction App")
st.write("""
This application predicts the likelihood of a engine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
engine_rpm = st.number_input("Engine rpm", min_value=0,  value=750)
lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0,  value=3.162035)
lub_oil_temp = st.number_input("lub oil temp", min_value=0,  value=76.817350)
coolant_pressure = st.number_input("Coolant pressure", min_value=0.0,  value=2.166883)
coolant_temp = st.number_input("Coolant temp", min_value=0,  value=78.346662)
fuel_pressure = st.number_input("Fuel pressure", min_value=0.0,  value=6.201720)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant pressure": coolant_pressure,
    "Coolant temp": coolant_temp,
    "Fuel pressure": fuel_pressure
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
