
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
app = FastAPI()
model = joblib.load("wine_model.pkl")  # Load saved model

# 7. Define request body schema
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return{"message":"Welcome to the Iris Classifier API"}
# 8. Define endpoint
@app.post("/predict")
def predict(data: WineInput):
    input_array = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid,
                             data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
                             data.total_sulfur_dioxide, data.density, data.pH,
                             data.sulphates, data.alcohol]])
    prediction = model.predict(input_array)[0]
    return {"quality": "good" if prediction == 1 else "not good"}