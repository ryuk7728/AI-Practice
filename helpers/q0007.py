"""Helper objects for Question 0007."""

import numpy as np


def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    differences = y_true - y_pred
    return np.mean(differences * differences)


class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0

    def predict(self, X):
        return self.w * X + self.b
