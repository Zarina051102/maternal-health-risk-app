import streamlit as st
import numpy as np
import joblib
from catboost import CatBoostClassifier

# ------------------------------
# LOAD MODEL + SCALER (cached)
# Loads CatBoost model and Scaler only once to improve performance.
# ------------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("maternal_risk_model.cbm")   # Load trained CatBoost model
    scaler = joblib.load("scaler.pkl")            # Load StandardScaler
    return model, scaler

model, scaler = load_model()

# Risk labels mapping (model output → human-readable text)
risk_levels = {
    0: "High Risk",
    1: "Low Risk",
    2: "Mid Risk"
}

# ------------------------------
# USER INTERFACE
# ------------------------------
st.title("Maternal Health Risk Prediction")
st.write("Enter maternal health data to predict the risk level:")

# ------------------------------
# TEST EXAMPLES FOR QUICK CHECK
# ------------------------------
st.write("### Test Examples")

if st.button("Test LOW RISK example"):
    X = np.array([[35, 120, 60, 6.1, 98.0, 76]])   # Low-risk example
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    st.success(f"Predicted LOW example → {risk_levels[pred]}")

if st.button("Test MID RISK example"):
    X = np.array([[28, 130, 80, 10.5, 99.5, 88]])   # Mid-risk example
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    st.info(f"Predicted MID example → {risk_levels[pred]}")

if st.button("Test HIGH RISK example"):
    X = np.array([[25, 140, 85, 15.0, 100.0, 90]])  # High-risk example
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    st.warning(f"Predicted HIGH example → {risk_levels[pred]}")

# ------------------------------
# MAIN INPUT FIELDS
# ------------------------------
age = st.number_input("Age", min_value=10, max_value=60, value=25)
sbp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
dbp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
bs = st.number_input("Blood Sugar", min_value=4.0, max_value=20.0, value=7.0)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.0)
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=80)

# ------------------------------
# PREDICTION BLOCK
# ------------------------------
if st.button("Predict"):
    try:
        X = np.array([[age, sbp, dbp, bs, body_temp, heart_rate]])   # User input → array
        X_scaled = scaler.transform(X)                               # Apply scaling
        prediction = int(model.predict(X_scaled)[0])                 # Model output (0/1/2)

        st.success(f"Result: {risk_levels[prediction]}")             # Display text label

    except Exception as e:
        st.error("Error! Please check that all input fields are valid.")
        st.write(e)

