from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(openapi_url="/openapi.json")

class PredictionRequest(BaseModel):
    features: list

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("saved_models/random_forest_best_model.joblib")
    except FileNotFoundError:
        raise RuntimeError("Model file not found. Train the model first.")

@app.get("/home", summary="Home Endpoint", description="Welcome message for the API")
def read_home():
    return {"message": "Welcome to the prediction API!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    features = pd.DataFrame([request.features])
    prediction = model.predict(features)[0]
    if prediction == 0:
        prediction = "Legitimate Transaction"
    else:
        prediction = "Fraudulent Transaction"
    probability = model.predict_proba(features)[0][1]
    return {"prediction": prediction, "confidence": probability}
    
