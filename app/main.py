from fastapi import FastAPI
from typing import List
import mlflow
import numpy as np
mlflow.set_tracking_uri(uri="http://flow:8080")

MLFLOW_PORT = 8080
app = FastAPI()
model_name = 'tracking-quickstart'
model_version = 'latest'
# Sepal Length, Sepal Width, Petal Length and Petal Width.

# model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
app.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

def load_model(version: str | int):
    app.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

@app.post("/predict")
async def create_prediction(inputs: List[float]):
    return app.model.predict(np.array(inputs).reshape((-1, 4))).tolist()

from pydantic import BaseModel

class VersionInput(BaseModel):
    version: str | int
    
@app.post('/update-model')
async def update_model(inputs: VersionInput):
    load_model(inputs.version)
    return True
