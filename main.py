import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load trained LSTM model
model = keras.models.load_model("stock-backend-/model/lstm_model.h5")

# ✅ Load scalers
scaler = joblib.load("model/scaler.pkl")  # Used for input features (5 features)
scaler_close = joblib.load("model/scaler_close.pkl")  # Used only for 'Close' price prediction

# ✅ Load historical stock data
df = pd.read_csv("model/NFLX.csv")  # Ensure this file exists in the correct path

def get_last_49_days():
    """Fetch last 49 days of stock data for prediction."""
    last_49 = df[['Open', 'High', 'Low', 'Volume', 'Close']].values[-49:]  # Last 49 rows
    
    # Ensure we have 49 days of data
    if last_49.shape[0] < 49:
        raise ValueError("Not enough historical data available for prediction.")
    
    last_49_scaled = scaler.transform(last_49)  # Scale using feature scaler
    return last_49_scaled

# ✅ Define request model for validation
class StockInput(BaseModel):
    features: list[float]  # Expecting a list of 5 float values

@app.post("/predict")
async def predict(data: StockInput):
    try:
        user_input = np.array(data.features)

        # ✅ Ensure correct input shape
        if user_input.shape != (5,):
            raise HTTPException(status_code=400, detail="Input data must have exactly 5 features.")

        user_input = user_input.reshape(1, -1)  # Shape (1, 5)

        # ✅ Scale input using `scaler.pkl`
        scaled_input = scaler.transform(user_input)

        # ✅ Get last 49 days of historical data (already scaled)
        historical_data = get_last_49_days()

        # ✅ Append new scaled input to maintain 50 timesteps
        model_input = np.vstack([historical_data, scaled_input]).reshape(1, 50, 5)

        # ✅ Predict using LSTM model
        scaled_prediction = model.predict(model_input)  # Shape (1, 1)

        # ✅ Ensure prediction shape is correct
        if scaled_prediction.shape != (1, 1):
            raise HTTPException(status_code=500, detail="Unexpected output shape from model.")

        # ✅ Inverse transform prediction using `scaler_close.pkl` (Only 1 feature)
        actual_prediction = scaler_close.inverse_transform(scaled_prediction.reshape(-1, 1))

        return {"predicted_close_price": float(actual_prediction[0][0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))  # Return error message

