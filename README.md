# AI Automation – Car Price Prediction Dashboard

## Project Overview
An end-to-end AI/ML web application that predicts the price of used cars using machine learning. The project includes a FastAPI backend and an interactive Streamlit dashboard with model performance, EDA insights, fault management, and real-time streaming concepts.

## Project Objective
To build a production-ready car price prediction system that demonstrates:
- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- API deployment with FastAPI
- Interactive web dashboard with Streamlit
- Exploratory Data Analysis (EDA) and visualization
- Conceptual integration of automotive testing tools

## Tools & Technologies
- **Languages**: Python
- **ML**: RandomForestRegressor, Scikit-learn
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Conceptual Tools**: Vector CANoe, ECU-TEST, VeriStand, CANalyzer

## Project Flow
1. Load and clean `used_cars.csv` dataset (4009 rows, 12 columns).
2. Train RandomForestRegressor model for price prediction.
3. Save model as `model.pkl` and deploy via FastAPI (`/predict`).
4. Build Streamlit dashboard with multiple pages.
5. Connect Streamlit frontend to FastAPI backend for real-time predictions.
6. Add EDA insights (overview, statistics, correlation heatmap, histograms).
7. Conceptually include automotive tools for ECU testing and CAN data simulation.

## Modules
1. **Home** – Dashboard introduction and navigation.
2. **Price Prediction** – User input → FastAPI prediction → displayed price.
3. **Model Performance** – MAE, MSE, R² Score, Accuracy metrics.
4. **Fault Management** – Detect invalid/unrealistic input data.
5. **Real-time Streaming** – Concept for live CAN data processing.
6. **EDA Insights** – Dataset overview, statistics, correlation heatmap, histograms.

## Key Features
- Interactive car price prediction with real API calls.
- Model performance metrics: MAE, MSE, R².
- EDA visualizations: correlation heatmap, histograms.
- Fault detection concept for invalid data.
- Real-time streaming concept for automotive data.
- FastAPI backend with REST endpoints.
- Streamlit frontend with multi-page navigation.

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run FastAPI Backend
```bash
uvicorn app:app --reload
```
Backend runs on: `http://localhost:8000`

### Step 3: Run Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
Frontend runs on: `http://localhost:8501`

### Step 4: Use the Dashboard
- Open the Streamlit URL.
- Navigate to **Price Prediction**.
- Enter car details and click **Predict**.
- View predicted price from FastAPI.

## API Endpoints
- `/` – Health check.
- `/model-info` – Returns model type and feature names.
- `/predict` – Takes encoded features and returns predicted price.
- `/check-fault` – Checks if input data has faults.
- `/fault-logs` – Returns fault logs.

## Project Conclusion
This project delivers an end-to-end AI/ML web application for car price prediction with FastAPI backend and Streamlit frontend. It demonstrates data preprocessing, model training, API deployment, and interactive visualization, with conceptual integration of automotive testing tools.

## Future Work
1. Add cross-validation and better train/test split.
2. Try advanced models (XGBoost, LightGBM, Neural Networks).
3. Use real automotive data from CANoe/CANalyzer.
4. Implement actual real-time streaming with live CAN data.
5. Add detailed fault detection rules and reporting.
6. Integrate with real ECU testing tools (VeriStand, ECU-TEST) for production.

## Author
Nutan Agare   
© 2026 AI Automation Project

---

## License
This project is for educational purposes.
