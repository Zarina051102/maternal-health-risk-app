import streamlit as st
import joblib
import numpy as np

st.title("Maternal Health Risk Prediction")

# Загружаем модели
model = joblib.load("maternal_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.write("Введите данные матери для прогнозирования риска:")

age = st.number_input("Age", min_value=10, max_value=60, value=25)
bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
bs = st.number_input("Blood Sugar", min_value=50, max_value=300, value=100)
body_temp = st.number_input("Body Temperature", min_value=30.0, max_value=45.0, value=37.0)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=90)

if st.button("Predict"):
    try:
        X = np.array([[float(age), float(bp), float(bs), float(body_temp), float(heart_rate)]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        st.success(f"Result: {risk_levels[prediction]}")

    except Exception as e:
        st.error("Ошибка в данных. Проверьте, что все поля заполнены корректно.")
        st.write(e)

