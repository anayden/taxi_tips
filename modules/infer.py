from fastapi import FastAPI
from .model import model
import numpy as np

app = FastAPI()


@app.post("/predict")
def predict():
    return model.predict(np.array([])).tolist()[0]


@app.get("/health")
def health_check():
    return "OK"
