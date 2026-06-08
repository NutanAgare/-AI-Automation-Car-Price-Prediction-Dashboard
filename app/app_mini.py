# app_mini.py - FINAL version with correct features

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="AI Automation - Car Price Prediction")

MODEL_PATH = "F:/Project/AI_Automative_Projects/model/model.pkl"
model = pickle.load(open(MODEL_PATH, "rb"))

class CarInput(BaseModel):
    brand: int
    model: int
    model_year: int
    milage: float
    fuel_type: int
    engine: float
    transmission: int
    ext_col: int
    int_col: int
    accident: int        #0/1
    clean_title: int     # 0/1

@app.get("/")
async def health_check():
    return {"status": "AI Automation API is running", "version": "1.0"}

@app.get("/model-info")
async def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "status": "loaded",
        "features": [
            "brand", "model", "model_year", "milage", "fuel_type",
            "engine", "transmission", "ext_col", "int_col",
            "accident", "clean_title"
        ]
    }

@app.post("/predict")
async def predict_price(car_data: CarInput):
    # Pydantic model -> dict
    data_dict = car_data.dict()

    # Create DataFrame with exact column order used during training
    feature_cols = [
        "brand", "model", "model_year", "milage", "fuel_type",
        "engine", "transmission", "ext_col", "int_col",
        "accident", "clean_title"
    ]

    input_data = pd.DataFrame([data_dict])[feature_cols]

    prediction = model.predict(input_data)[0]

    return JSONResponse(content={
        "predicted_price": float(prediction),
        "status": "success"
    })