import pickle
import numpy as np


filename = "results/taxi-model.pkl"


class Model:
    def __init__(self):
        with open(filename, 'rb') as file:
            self._model = pickle.load(file)

    def predict(self, input: np.array) -> np.array:
        x_test = np.array([[0.70241141, -0.44722295,  0.27440181, -0.07418688,  0.32060054,
                            1.07563487, -0.57038556,  0.20736666, -0.47792267,  0.08970088,
                            -0.19118768,  0.05976435,  0.21332606,  0.28652577]])
        return np.array(self._model.predict(x_test))

model = Model()
