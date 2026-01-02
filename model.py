import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Sample training dataset
data = {
    "income": [20000, 40000, 60000, 80000, 100000, 30000, 70000, 90000],
    "credit_score": [450, 550, 650, 750, 800, 500, 700, 780],
    "existing_emi": [8000, 6000, 3000, 2000, 1000, 7000, 2500, 1500],
    "default": [1, 1, 0, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[["income", "credit_score", "existing_emi"]]
y = df["default"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model & scaler saved successfully")
