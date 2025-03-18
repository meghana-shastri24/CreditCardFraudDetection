from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("app/saved_models/random_forest_best_model.joblib")
    except FileNotFoundError:
        raise RuntimeError("Model file not found. Train the model first.")

@app.post("/predict")
def predict(request: PredictionRequest):
    features = pd.DataFrame([request.features])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {"prediction": prediction, "probability": probability}
