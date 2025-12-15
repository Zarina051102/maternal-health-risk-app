# Maternal Health Risk Prediction App

This web application predicts maternal health risk levels using a Random Forest model.

## Features
- User-friendly web interface
- Real-time risk prediction
- Machine learning-based decision support

## Technologies
- Python
- Streamlit
- Scikit-learn

## How to run locally
```bash
streamlit run app.py


---

# 🟢 ШАГ 3. ВАЖНОЕ ИЗМЕНЕНИЕ В app.py

❗ Streamlit Cloud **НЕ любит .pkl**, созданные локально.  
👉 МЫ ОБУЧАЕМ МОДЕЛЬ ПРЯМО В APP.

### ✅ app.py (ВСТАВЬ ВЕСЬ ФАЙЛ)

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Загружаем данные
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Предобработка
le = LabelEncoder()
df["RiskLevel"] = le.fit_transform(df["RiskLevel"])

X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Интерфейс
st.title("🤰 Maternal Health Risk Prediction")

age = st.number_input("Age", 10, 60, 25)
sbp = st.number_input("Systolic BP", 80, 200, 120)
dbp = st.number_input("Diastolic BP", 50, 130, 80)
glucose = st.number_input("Blood Glucose", 50.0, 300.0, 90.0)
temp = st.number_input("Body Temperature", 35.0, 42.0, 36.6)
heart = st.number_input("Heart Rate", 40, 200, 75)

if st.button("Predict Risk"):
    X_input = np.array([[age, sbp, dbp, glucose, temp, heart]])
    X_input_scaled = scaler.transform(X_input)
    pred = model.predict(X_input_scaled)[0]

    if pred == 0:
        st.error("🔴 High Risk")
    elif pred == 1:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")
