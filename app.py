from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "credit_model.pkl")
os.makedirs("models", exist_ok=True)

# Auto-create model if missing (Render-safe)
if not os.path.exists(MODEL_PATH):
    X = np.array([
        [50000, 750, 5000, 0],
        [30000, 620, 12000, 1],
        [80000, 820, 2000, 0],
        [25000, 580, 15000, 1],
        [60000, 700, 7000, 0]
    ])
    y = np.array([0, 1, 0, 1, 0])

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return {"status": "Credit Risk API running"}

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
