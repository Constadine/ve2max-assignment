# FastAPI Prediction Service

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)

## Description

This project is a FastAPI-based web service for predicting active power consumption based on various input parameters. It uses a pre-trained machine learning model and serves predictions via a RESTful API.

## Features

- **FastAPI**: High-performance, easy-to-use web framework for building APIs.
- **Docker**: Containerization for easy deployment and scaling.
- **Machine Learning**: Predictive model for active power consumption.
- **Scalability**: Easily deployable to various cloud platforms or on-premise environments.

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (if using containerization)

### Clone the Repository

```sh
git clone https://github.com/yourusername/fastapi-
```
### Install Dependencies

```sh
pip install -r requirements.txt
```

### Running the Application Locally

```sh
uvicorn main:app --reload
```
### Using Docker

### Build and Run
1. Build the Docker Image:
```sh
docker build -t fastapi-predictor .
```
2. Run the Docker Container:
```sh
docker run -d -p 80:80 fastapi-predictor
```

## Usage
### Testing the API

You can test the API endpoints using tools like curl, Postman, or any HTTP client.
Example Request

Using curl:
```sh
curl -X POST "http://127.0.0.1/predict" -H "Content-Type: application/json" -d @request.json
```

Where request.json contains:

```json
{
    "current": 2.53,
    "voltage": 122.2,
    "reactive_power": 159.09,
    "apparent_power": 309.17,
    "power_factor": 0.8575,
    "temp": 24.19,
    "feels_like": 23.68,
    "temp_min": 23.44,
    "temp_max": 27.5,
    "pressure": 1013,
    "humidity": 39,
    "speed": 0.0,
    "deg": 0,
    "hour": 14,
    "day_of_week": 5,
    "month": 11
}
```
## API Endpoints
/predict (POST)
- Description: Predict the active power consumption based on input parameters.
- Request Body: JSON object containing the input parameters.
- Response: JSON object with the predicted active power.

### Example Response
```json
{
    "predicted_active_power": 265.1
}
```
