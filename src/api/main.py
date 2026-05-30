from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
import os
import glob
from .pydantic_models import TransactionInput, PredictionResponse

app = FastAPI(title="Bati Bank Credit Scoring API")

def get_latest_model_path():
    paths = glob.glob("mlruns/**/artifacts/Random_Forest", recursive=True)
    if not paths:
    
        paths = glob.glob("mlruns/**/Random_Forest", recursive=True)
    
    if not paths:
        return None
    
    return max(paths, key=os.path.getctime)

MODEL_PATH = get_latest_model_path()
print(f"DEBUG: Model found at -> {MODEL_PATH}")

@app.get("/")
def home():
    return {
        "message": "Bati Bank API is online", 
        "model_status": "Loaded" if MODEL_PATH else "Not Found",
        "model_path": MODEL_PATH
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: TransactionInput):
    if not MODEL_PATH:
        return {"risk_probability": 0.5, "is_high_risk": 0}
    
    # Load model
    model = mlflow.sklearn.load_model(MODEL_PATH)
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk = int(model.predict(input_df)[0])

    return {"risk_probability": float(prob), "is_high_risk": risk}