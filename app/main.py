from fastapi import FastAPI
from typing import List
import mlflow
import numpy as np
import os

app = FastAPI()
model_name = 'tracking-quickstart'
model_version = 'latest'

mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(uri=f"{mlflow_uri}")

app.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

def load_model(version: str | int):
    app.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

@app.get("/health")
async def check_health():
    return 'OK'

# Sepal Length, Sepal Width, Petal Length and Petal Width.
@app.post("/predict")
async def create_prediction(inputs: List[float]):
    return app.model.predict(np.array(inputs).reshape((-1, 4))).tolist()

from pydantic import BaseModel

class VersionInput(BaseModel):
    version: str | int
    
@app.post('/update-model')
async def update_model(inputs: VersionInput):
    try:
        load_model(inputs.version)
    except:
        return False
    return True
