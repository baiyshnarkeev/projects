from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
with open("rf_titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class PassengerData(BaseModel):
    Pclass: int
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

@app.get("/")
def read_root():
    return "Hello, World!"

@app.post("/predict")
def predict_survival(data: PassengerData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make predictions
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    
    return {"prediction": result}