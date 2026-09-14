"""Helper objects for Question 0020."""

import torch
from torch import nn


class Head(nn.Module):
    def __init__(
        self,
        embedding_size,
        head_size,
        context_size,
        dropout=0.0,
    ):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size, dtype=torch.bool)),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sequence_length = x.shape[1]
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        scores = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        allowed = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(weights) @ value
