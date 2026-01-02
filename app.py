from flask import Flask, request, render_template_string
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ---------- MODEL SETUP ----------
MODEL_PATH = "models/credit_model.pkl"
os.makedirs("models", exist_ok=True)

# Auto-create model if missing (Render-safe)
if not os.path.exists(MODEL_PATH):
    X = np.array([
        [25, 50000, 1, 3, 0, 0, 12000, 12.5, 0.25, 0, 5, 1.6],
        [40, 30000, 0, 10, 1, 2, 25000, 18.0, 0.60, 1, 15, 1.5],
        [32, 80000, 2, 7, 2, 1, 15000, 10.0, 0.18, 0, 10, 1.4],
        [28, 28000, 0, 2, 1, 3, 20000, 20.0, 0.72, 1, 3, 1.5],
        [45, 90000, 2, 15, 2, 0, 10000, 8.0, 0.11, 0, 20, 1.3],
    ])

    y = np.array([0, 1, 0, 1, 0])  # 0 = Low Risk, 1 = High Risk

    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

# ---------- UI ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FinRisk Form</title>
    <style>
        body { font-family: Arial; background:#f5f7fb; }
        .card {
            width: 420px;
            margin: 40px auto;
            padding: 25px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h2 { text-align:center; }
        label { font-weight:600; font-size:14px; }
        input, select {
            width:100%;
            padding:8px;
            margin:6px 0 14px 0;
        }
        button {
            width:100%;
            padding:10px;
            background:#0d6efd;
            color:white;
            border:none;
            border-radius:6px;
            font-size:16px;
        }
        .result {
            margin-top:15px;
            padding:10px;
            text-align:center;
            font-weight:bold;
        }
    </style>
</head>

<body>
<div class="card">
    <h2>FinRisk Form</h2>

    <form method="POST">
        <label>Person Age</label>
        <input type="number" name="age" required>

        <label>Person Income</label>
        <input type="number" name="income" required>

        <label>Home Ownership</label>
        <select name="home">
            <option value="0">Rent</option>
            <option value="1">Mortgage</option>
            <option value="2">Own</option>
        </select>

        <label>Employment Length (years)</label>
        <input type="number" name="emp_length" required>

        <label>Loan Intent</label>
        <select name="intent">
            <option value="0">Personal</option>
            <option value="1">Education</option>
            <option value="2">Medical</option>
            <option value="3">Debt Consolidation</option>
        </select>

        <label>Loan Grade</label>
        <select name="grade">
            <option value="0">A</option>
            <option value="1">B</option>
            <option value="2">C</option>
        </select>

        <label>Loan Amount</label>
        <input type="number" name="amount" required>

        <label>Interest Rate (%)</label>
        <input type="number" step="0.1" name="rate" required>

        <label>Loan Percent Income</label>
        <input type="number" step="0.01" name="percent_income" required>

        <label>Default on File (0 or 1)</label>
        <input type="number" name="default" required>

        <label>Credit History Length</label>
        <input type="number" name="credit_len" required>

        <label>Credit to Employment Ratio</label>
        <input type="number" step="0.1" name="ratio" required>

        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <div class="result" style="background:{{ color }}">
            {{ result }}
        </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    color = "#d1e7dd"

    if request.method == "POST":
        values = [
            int(request.form["age"]),
            int(request.form["income"]),
            int(request.form["home"]),
            int(request.form["emp_length"]),
            int(request.form["grade"]),
            int(request.form["intent"]),
            int(request.form["amount"]),
            float(request.form["rate"]),
            float(request.form["percent_income"]),
            int(request.form["default"]),
            int(request.form["credit_len"]),
            float(request.form["ratio"]),
        ]

        prediction = model.predict([values])[0]
        result = "HIGH RISK LOAN" if prediction == 1 else "LOW RISK LOAN"
        color = "#f8d7da" if prediction == 1 else "#d1e7dd"

    return render_template_string(HTML_TEMPLATE, result=result, color=color)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
