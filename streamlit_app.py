import streamlit as st
import joblib
import numpy as np

st.title("Maternal Health Risk Prediction")

# Загружаем модели
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("maternal_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()


risk_levels = {
    0: "High Risk",
    1: "Low Risk",
    2: "Mid Risk"
}

st.write("Введите данные матери для прогнозирования риска:")

st.write("### Тестовые примеры")

if st.button("Test LOW RISK"):
    X = np.array([[20, 95, 60, 6.2, 98.0, 65]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.write("Predicted:", risk_levels[pred])

if st.button("Test MID RISK"):
    X = np.array([[28, 110, 75, 7.8, 99.1, 82]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.write("Predicted:", risk_levels[pred])

if st.button("Test HIGH RISK"):
    X = np.array([[45, 170, 110, 250, 103.5, 150]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.write("Predicted:", risk_levels[pred])



# --- ОСНОВНОЙ ВВОД ДАННЫХ --------------------------
age = st.number_input("Age", min_value=10, max_value=60, value=25)
sbp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=200, value=120)
dbp = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, value=80)
bs = st.number_input("Blood Sugar", min_value=50, max_value=300, value=100)
body_temp = st.number_input("Body Temperature", min_value=30.0, max_value=45.0, value=37.0)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=90)

if st.button("Predict"):
    try:
        X = np.array([[age, sbp, dbp, bs, body_temp, heart_rate]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        st.success(f"Result: {risk_levels[prediction]}")
    except Exception as e:
        st.error("Ошибка в данных. Проверьте, что все поля заполнены корректно.")
        st.write(e)

