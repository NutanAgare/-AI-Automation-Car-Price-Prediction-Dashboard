from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import logging
from datetime import datetime

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Automation - Car Price Prediction")

# Load model
model = pickle.load(open("F:/Project/AI_Automative_Projects/model/model.pkl", "rb"))

# -------- SIMPLE ENCODING MAP (IMPORTANT FIX) --------
def simple_encode(df):
    df = df.copy()

    mappings = {
        "brand": {"toyota": 1, "honda": 2, "bmw": 3},
        "model": {"corolla": 1, "civic": 2, "x5": 3},
        "fuel_type": {"petrol": 1, "diesel": 2},
        "transmission": {"manual": 1, "automatic": 2},
        "ext_col": {"white": 1, "black": 2, "red": 3},
        "int_col": {"black": 1, "beige": 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    return df


# -------- PREDICT API --------
@app.post("/predict")
def predict_price(car_data: dict):
    try:
        input_data = pd.DataFrame([car_data])

        # encode categorical data
        input_data = simple_encode(input_data)

        prediction = model.predict(input_data)[0]

        return JSONResponse({
            "predicted_price": float(prediction),
            "status": "success"
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "failed"
        }, status_code=500)


# -------- HEALTH CHECK --------
@app.get("/")
def home():
    return {"status": "API running"}

# -------- MODEL INFO --------
@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestRegressor",
        "time": str(datetime.now())
    }