from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Define the columns to scale
columns_to_scale = ['current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor',
                    'temp', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 
                    'speed', 'deg', 'hour', 'day_of_week', 'month']

app = FastAPI()

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a Pydantic model for the request body
class PredictionRequest(BaseModel):
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

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = np.array([getattr(request, col) for col in columns_to_scale]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return {"predicted_active_power": prediction[0]}
