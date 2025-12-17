import os
import random
from typing import List

import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model_name = "tracking-quickstart"
model_version = "latest"

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(uri=f"{mlflow_uri}")

# Load initial model
initial_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

app.current_model = initial_model
app.next_model = initial_model

CANARY_PROBABILITY = 0.8

@app.get("/health")
async def check_health():
    return "OK"


# Sepal Length, Sepal Width, Petal Length and Petal Width.
@app.post("/predict")
async def create_prediction(inputs: List[float]):
    if random.random() < CANARY_PROBABILITY:
        model = app.current_model
    else:
        model = app.next_model
        
    return model.predict(np.array(inputs).reshape((-1, 4))).tolist()


class VersionInput(BaseModel):
    version: str | int


@app.post("/update-model")
async def update_model(inputs: VersionInput):
    try:
        app.next_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{inputs.version}")
    except:
        return False
    return True

@app.post("/accept-next-model")
async def accept_next_model():
    app.current_model = app.next_model
    return True
