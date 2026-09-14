"""Helper objects for Question 0022."""

import torch
from torch import nn

from helpers.q0020 import Head


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        head_size,
        context_size,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(embedding_size, head_size, context_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(num_heads * head_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined))


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
