from flask import Flask, render_template, request
import numpy as np
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("maternal_risk_model(2).pkl")
scaler = joblib.load("scaler(2).pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["Age"])
    sbp = float(request.form["SystolicBP"])
    dbp = float(request.form["DiastolicBP"])
    bs = float(request.form["BS"])
    temp = float(request.form["BodyTemp"])
    heart = float(request.form["HeartRate"])

    input_data = np.array([[age, sbp, dbp, bs, temp, heart]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]

    label_map = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
    result = label_map[prediction]

    # Цвет
    if result == "Low Risk":
        color = "low"
    elif result == "Mid Risk":
        color = "mid"
    else:
        color = "high"

    return render_template("result.html", result=result, color=color)


if __name__ == "__main__":
    app.run(debug=True)

