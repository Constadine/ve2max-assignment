from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Define the columns for regression
regression_columns = ['current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor',
                      'temp', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 
                      'speed', 'deg', 'hour', 'day_of_week', 'month']

# Define the columns for forecast
forecast_columns = regression_columns

app = FastAPI()

# Load the trained models and scaler
regression_model = joblib.load('regression_model.pkl')
scaler = joblib.load('scaler.pkl')
prophet_model = joblib.load('prophet_model.pkl')

# Define a Pydantic model for the request body
class RegressionRequest(BaseModel):
    current: float
    voltage: float
    reactive_power: float
    apparent_power: float
    power_factor: float
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: float
    humidity: float
    speed: float
    deg: float
    hour: int
    day_of_week: int
    month: int

class ForecastRequest(BaseModel):
    current: float
    voltage: float
    reactive_power: float
    apparent_power: float
    power_factor: float
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: float
    humidity: float
    speed: float
    deg: float
    hour: int
    day_of_week: int
    month: int
    periods: int  # Number of periods to forecast

@app.post("/predict")
def predict(request: RegressionRequest):
    input_data = np.array([getattr(request, col) for col in regression_columns]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = regression_model.predict(input_data_scaled)
    return {"predicted_active_power": prediction[0]}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    input_data = {
        'ds': datetime.now(),
        'current': request.current,
        'voltage': request.voltage,
        'reactive_power': request.reactive_power,
        'apparent_power': request.apparent_power,
        'power_factor': request.power_factor,
        'temp': request.temp,
        'feels_like': request.feels_like,
        'temp_min': request.temp_min,
        'temp_max': request.temp_max,
        'pressure': request.pressure,
        'humidity': request.humidity,
        'speed': request.speed,
        'deg': request.deg,
        'hour': request.hour,
        'day_of_week': request.day_of_week,
        'month': request.month
    }
    input_df = pd.DataFrame([input_data])
    
    future = prophet_model.make_future_dataframe(periods=request.periods, freq='H')
    for col in forecast_columns:
        if col != 'ds':
            future[col] = input_df[col].values[0]
    
    forecast = prophet_model.predict(future)
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(request.periods).to_dict(orient='records')
    return {"forecast": forecast_result}
