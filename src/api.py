import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import MODEL_PATH

# 1️⃣ Define the Data Structure (DTO)
# This acts like a TypeScript Interface.
# If the frontend sends wrong data types, FastAPI returns a 422 Error automatically.
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# 2️⃣ Initialize App & Load Model
app = FastAPI(title="Churn Prediction API")

# We load the model globally so it stays in memory (low latency)
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model_pipeline = None
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}. API will fail requests.")

# 3️⃣ Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

# 4️⃣ Prediction Endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert Pydantic object to Pandas DataFrame
    # The pipeline expects a DataFrame with specific column names
    input_data = pd.DataFrame([data.dict()])

    # Get Probability
    # predict_proba returns [[prob_0, prob_1]]
    probability = model_pipeline.predict_proba(input_data)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "churn_probability": float(probability),
        "is_churn_risk": bool(prediction),
        "message": "High Risk - Intervention Needed" if prediction else "Low Risk"
    }