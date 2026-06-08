# app.py - FINAL BACKEND WITH LOGGING + FAULTS + CORRECT FEATURES

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
from datetime import datetime
import os

# ===================== Logging Setup =====================
os.makedirs("F:/Project/AI_Automative_Projects/app", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("F:/Project/AI_Automative_Projects/app/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("AI Automation API (FULL) started")

# ===================== FastAPI App =====================
app = FastAPI(title="AI Automation - Car Price Prediction (Full)")

# ===================== Load Model =====================
MODEL_PATH = "F:/Project/AI_Automative_Projects/model/model.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# ===================== Input Schema (same as app_mini) =====================
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
    accident: int
    clean_title: int

FEATURE_COLS = [
    "brand", "model", "model_year", "milage", "fuel_type",
    "engine", "transmission", "ext_col", "int_col",
    "accident", "clean_title"
]

# ===================== Health Check =====================
@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "AI Automation API is running", "version": "1.0"}

# ===================== Model Info =====================
@app.get("/model-info", tags=["Model"])
async def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "status": "loaded" if model is not None else "model_not_loaded",
        "features": FEATURE_COLS
    }

# ===================== Prediction Endpoint =====================
@app.post("/predict", tags=["Prediction"])
async def predict_price(car_data: CarInput):
    """
    Predict used car price based on encoded features.
    """
    logger.info(f"Prediction request received: {car_data}")
    try:
        if model is None:
            logger.error("Model not loaded")
            return JSONResponse(
                content={"error": "Model not loaded", "status": "failed"},
                status_code=500
            )

        data_dict = car_data.dict()
        input_data = pd.DataFrame([data_dict])[FEATURE_COLS]

        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction made: {prediction}")

        return JSONResponse(content={
            "predicted_price": float(prediction),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return JSONResponse(content={
            "error": str(e),
            "status": "failed"
        }, status_code=500)

# ===================== Fault Management =====================
fault_log = []

def detect_fault(data: dict):
    faults = []
    try:
        # Simple example rules:
        if "milage" in data and data["milage"] is not None and data["milage"] < 0:
            faults.append("Negative milage detected")

        if "engine" in data and data["engine"] is not None and data["engine"] <= 0:
            faults.append("Engine size must be positive")

        if any(v is None for v in data.values()):
            faults.append("Missing values detected")

        if faults:
            fault_log.append({
                "timestamp": str(datetime.now()),
                "faults": faults,
                "data": data
            })
            logger.warning(f"Faults detected: {faults}")

        return faults
    except Exception as e:
        logger.error(f"Fault detection error: {str(e)}")
        return ["Fault detection error"]

@app.post("/check-fault", tags=["Fault"])
async def check_fault(data: dict):
    faults = detect_fault(data)
    return {"faults": faults, "is_healthy": len(faults) == 0}

@app.get("/fault-logs", tags=["Fault"])
async def get_fault_logs():
    return {"fault_logs": fault_log, "total_faults": len(fault_log)}

print("=== FULL API Ready ===")
print("Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000")
print("Docs: http://127.0.0.1:8000/docs")