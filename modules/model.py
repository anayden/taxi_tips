import pickle
import numpy as np
import os

model_tag = os.getenv("MODEL_TAG")

model_filename = f"./results/predict-model-{model_tag}.pkl"
scaler_filename = f"./results/predict-scaler-{model_tag}.pkl"


class Model:
    def __init__(self):
        with open(model_filename, 'rb') as file:
            self._model = pickle.load(file)
        with open(scaler_filename, 'rb') as file:
            self._scaler = pickle.load(file)

    def predict(self, input: np.array) -> np.array:
        input = np.expand_dims(input, axis=0)
        scaled_input = self._scaler.transform(input)
        return np.array(self._model.predict(scaled_input))


model = Model()
