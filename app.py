from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI(title="House Price Prediction API")

# Define input data model
class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

# Load the pre-trained model
# Use absolute path to ensure model.pkl is found
model_path = os.path.join(os.path.dirname(__file__), '../model.pkl')
model = joblib.load(model_path)

@app.get("/predict")
async def predict_price(features: HouseFeatures):
    try:
        # Convert input to DataFrame
        data = {
            'area': [features.area],
            'bedrooms': [features.bedrooms],
            'bathrooms': [features.bathrooms],
            'stories': [features.stories],
            'mainroad': [1 if features.mainroad.lower() == 'yes' else 0],
            'guestroom': [1 if features.guestroom.lower() == 'yes' else 0],
            'basement': [1 if features.basement.lower() == 'yes' else 0],
            'hotwaterheating': [1 if features.hotwaterheating.lower() == 'yes' else 0],
            'airconditioning': [1 if features.airconditioning.lower() == 'yes' else 0],
            'parking': [features.parking],
            'prefarea': [1 if features.prefarea.lower() == 'yes' else 0],
            'furnishingstatus_semi-furnished': [1 if features.furnishingstatus.lower() == 'semi-furnished' else 0],
            'furnishingstatus_unfurnished': [1 if features.furnishingstatus.lower() == 'unfurnished' else 0]
        }
        df = pd.DataFrame(data)

        # Ensure columns match training data
        expected_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                           'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                           'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
        df = df[expected_columns]

        # Predict
        prediction = model.predict(df)[0]
        
        return {"predicted_price": round(float(prediction))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API. Use POST /predict to get predictions."}
