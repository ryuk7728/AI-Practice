"""Helper objects for Question 0023."""

from torch import nn

from helpers.q0022 import FeedForward, MultiHeadAttention


class Block(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        context_size,
        dropout=0.0,
    ):
        super().__init__()
        head_size = embedding_size // num_heads
        self.attention = MultiHeadAttention(
            embedding_size,
            num_heads,
            head_size,
            context_size,
            dropout,
        )
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.first_norm = nn.LayerNorm(embedding_size)
        self.second_norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.attention(self.first_norm(x))
        x = x + self.feed_forward(self.second_norm(x))
        return x
