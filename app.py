from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "credit_model.pkl")

# Load model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Make sure credit_model.pkl exists in models/"
    )

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return {"status": "Credit Risk API is running successfully"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["income"],
        data["credit_score"],
        data["emi"],
        data["past_defaults"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "credit_risk": "High Risk" if prediction == 1 else "Low Risk"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
