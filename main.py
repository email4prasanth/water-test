import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI

# Import class Water from data_model.py
from data_model import Water


# Create an Instance/Method of Fast API
app = FastAPI(
    name = "Water Potability Prediciton",
    discription = "Potability Prediction of Water"
)

# Load pretrained model using pickle
# model = pickle.load(open(r"D:\PersonalRepo\MLOPS\Code\model.pkl","rb"))
with open("model.pkl","rb") as f:
    model = pickle.load(f)

# Using decorator '@' to functions or methods
@app.get('/')
def index():
    return "Welcome to MLOPS"

# Using decorator '@' to fucntion post method
@app.post('/predict')
def model_predict(water:Water):
    sample = pd.DataFrame(
        {
            'ph' : [water.ph],
            'Hardness' : [water.Hardness],
            'Solids' : [water.Solids],
            'Chloramines' : [water.Chloramines],
            'Sulfate' : [water.Sulfate],
            'Conductivity' : [water.Conductivity],
            'Organic_carbon' : [water.Organic_carbon],
            'Trihalomethanes' : [water.Trihalomethanes],
            'Turbidity' : [water.Turbidity]           
    })
    
    predicted_value = model.predict(sample)
    if predicted_value == 0:
            return "Water is not consumable"
    else:
            return "Water is Consumable"

