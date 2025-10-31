# main_phase4.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import joblib

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\keert\OneDrive\Desktop\FinSight\data\synthetic_fin_data.csv")

# Feature/Target Split
X = data.drop(columns=["credit_score"])
y = data["credit_score"]

# -----------------------------
# 2️⃣ Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 4️⃣ Improved Credit Scoring Model (XGBoost)
# -----------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 5️⃣ Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
tolerance = np.mean(np.abs((y_pred - y_test) / y_test) <= 0.1) * 100

print("📊 Improved Credit Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Within ±10% Tolerance: {tolerance:.2f}%")

# -----------------------------
# 6️⃣ Fraud Detection (Isolation Forest)
# -----------------------------
# Detect unusual spending / income patterns
fraud_detector = IsolationForest(contamination=0.05, random_state=42)
fraud_detector.fit(X_scaled)

# Assign fraud labels: -1 = anomaly, 1 = normal
data["fraud_flag"] = fraud_detector.predict(X_scaled)

fraud_ratio = (data["fraud_flag"] == -1).mean() * 100
print(f"🚨 Fraudulent pattern ratio detected: {fraud_ratio:.2f}%")

# -----------------------------
# 7️⃣ Save Models
# -----------------------------
joblib.dump(model, "models/credit_model.pkl")
joblib.dump(fraud_detector, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Models saved successfully in /models folder!")
