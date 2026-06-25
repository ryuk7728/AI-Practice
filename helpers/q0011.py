"""Helper objects for Question 0011."""

import torch
from torch import nn


class FixedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, -1.0],
                    ]
                )
            )
            self.linear.bias.copy_(torch.tensor([0.0, 0.0, 0.5]))

    def forward(self, X):
        return self.linear(X)


def make_classification_batch():
    X_test = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 4.0],
            [-2.0, -2.0],
            [2.0, 0.0],
        ]
    )
    y_test = torch.tensor([0, 1, 2, 1])
    return X_test, y_test
