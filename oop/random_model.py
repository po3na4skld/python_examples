import numpy as np


class RandomModel:
    def predict(self, inputs):
        return np.random.randint(0, 9)
