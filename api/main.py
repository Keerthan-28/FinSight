from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ---- Initialize App ----
app = FastAPI(title="AI-Powered Financial Inclusion API")

# ---- Load Models ----
credit_model = joblib.load("../models/credit_model.pkl")
fraud_model = joblib.load("../models/fraud_model.pkl")

# ---- Input Schema ----
class UserData(BaseModel):
    upi_txn_count: int
    avg_txn_value: float
    bill_payments: int
    wallet_balance: float
    sms_count: int
    data_usage_gb: float
    ecom_purchases: int
    refund_count: int
    income_bracket: int
    age: int
    txn_variance: float

# ---- Credit Score Prediction ----
@app.post("/predict_credit")
def predict_credit(user: UserData):
    features = np.array([
        [
            user.upi_txn_count, user.avg_txn_value, user.bill_payments,
            user.wallet_balance, user.sms_count, user.data_usage_gb,
            user.ecom_purchases, user.refund_count, user.income_bracket,
            user.age, user.txn_variance
        ]
    ])
    credit_score = credit_model.predict(features)[0]
    return {"predicted_credit_score": round(float(credit_score), 2)}

# ---- Fraud Detection ----
@app.post("/detect_fraud")
def detect_fraud(user: UserData):
    features = np.array([
        [
            user.upi_txn_count, user.avg_txn_value, user.bill_payments,
            user.wallet_balance, user.sms_count, user.data_usage_gb,
            user.ecom_purchases, user.refund_count, user.income_bracket,
            user.age, user.txn_variance
        ]
    ])
    fraud_flag = fraud_model.predict(features)[0]
    return {"is_fraud": bool(fraud_flag)}

# ---- Health Check ----
@app.get("/")
def root():
    return {"message": "AI Financial Inclusion API is running 🚀"}