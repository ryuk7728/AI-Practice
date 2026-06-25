"""Helper objects for Question 0014."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def make_tiny_loader(batch_size=4):
    X = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, 0.1, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5, 0.1],
            [1.8, 0.0, 0.1, 0.0],
            [0.0, 1.8, 0.0, 0.1],
            [0.1, 0.0, 1.8, 0.0],
            [1.2, 0.2, 0.0, 0.0],
            [0.0, 1.2, 0.2, 0.0],
            [0.0, 0.0, 1.2, 0.2],
        ]
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
