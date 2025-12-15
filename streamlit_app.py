import streamlit as st
import numpy as np
import joblib
from catboost import CatBoostClassifier

# ------------------------------
# LOAD MODEL + SCALER (cached)
# ------------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("maternal_risk_model.cbm")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()

# Risk labels
risk_levels = {
    0: "High Risk",
    1: "Low Risk",
    2: "Mid Risk"
}


# ------------------------------
# UI
# ------------------------------
st.title("Maternal Health Risk Prediction")
st.write("Введите данные матери для прогнозирования риска:")

st.write("### Тестовые примеры")
if st.button("Test LOW RISK example"):
    X = np.array([[35, 120, 60, 6.1, 98.0, 76]])
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    st.success(f"Predicted LOW example → {risk_levels[pred]}")

if st.button("Test HIGH RISK example"):
    X = np.array([[25, 140, 85, 15.0, 100.0, 90]])
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    st.warning(f"Predicted HIGH example → {risk_levels[pred]}")


# Inputs
age = st.number_input("Age", min_value=10, max_value=60, value=25)
sbp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
dbp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
bs = st.number_input("Blood Sugar", min_value=4.0, max_value=20.0, value=7.0)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.0)
heart_rate = st.number_input("Heart Rate", min_value=50, max_value=200, value=80)

if st.button("Predict"):
    try:
        X = np.array([[age, sbp, dbp, bs, body_temp, heart_rate]])
        X_scaled = scaler.transform(X)
        prediction = int(model.predict(X_scaled)[0])
        st.success(f"Result: {risk_levels[prediction]}")
    except Exception as e:
        st.error("Ошибка! Проверьте корректность данных.")
        st.write(e)


 
