from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "✅ Credit Risk Prediction API is Live"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    income = data["income"]
    credit_score = data["credit_score"]
    existing_emi = data["existing_emi"]

    input_data = np.array([[income, credit_score, existing_emi]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    result = "High Risk ❌" if prediction == 1 else "Low Risk ✅"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
