import math

import torch
from torch import nn

from solutions.q0024 import generate_tokens


class RecordingLanguageModel(nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = []

    def forward(self, token_ids):
        self.calls.append(token_ids.detach().clone())
        scores = torch.zeros(
            *token_ids.shape,
            self.vocab_size,
            device=token_ids.device,
        )
        scores[..., -1] = 100.0
        return scores, None


class PositionSensitiveLanguageModel(nn.Module):
    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape
        scores = torch.full(
            (batch_size, sequence_length, 4),
            -100.0,
            device=token_ids.device,
        )
        scores[..., 0] = 100.0
        scores[:, -1, 0] = -100.0
        scores[:, -1, 3] = 100.0
        return scores, None


class ConstantDistributionLanguageModel(nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer(
            "scores",
            torch.tensor([math.log(probability) for probability in probabilities]),
        )

    def forward(self, token_ids):
        scores = self.scores.expand(*token_ids.shape, -1)
        return scores, None


def test_generate_tokens_appends_requested_number_for_each_batch_example():
    model = RecordingLanguageModel()
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    generated = generate_tokens(model, token_ids, max_new_tokens=4, context_size=8)

    assert generated.shape == (2, 7)
    torch.testing.assert_close(generated[:, :3], token_ids)
    assert generated.dtype == token_ids.dtype
    assert generated.device == token_ids.device


def test_generate_tokens_limits_each_model_context_to_recent_tokens():
    model = RecordingLanguageModel()
    token_ids = torch.tensor([[0, 1, 2, 3, 4, 0]], dtype=torch.long)

    generated = generate_tokens(model, token_ids, max_new_tokens=5, context_size=4)

    assert len(model.calls) == 5
    original_length = token_ids.shape[1]
    for step, model_input in enumerate(model.calls):
        available_prefix = generated[:, : original_length + step]
        expected_context = available_prefix[:, -4:]
        torch.testing.assert_close(model_input, expected_context)


def test_generate_tokens_uses_only_final_position_scores():
    model = PositionSensitiveLanguageModel()
    token_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)

    generated = generate_tokens(model, token_ids, max_new_tokens=3, context_size=6)

    torch.testing.assert_close(generated[:, -3:], torch.full((2, 3), 3))


def test_generate_tokens_samples_from_probability_distribution():
    probabilities = [0.65, 0.25, 0.10]
    model = ConstantDistributionLanguageModel(probabilities)
    token_ids = torch.zeros((4000, 1), dtype=torch.long)
    torch.manual_seed(24)

    generated = generate_tokens(model, token_ids, max_new_tokens=1, context_size=4)
    sampled = generated[:, -1]
    frequencies = torch.bincount(sampled, minlength=3).float() / len(sampled)

    assert torch.unique(sampled).numel() == 3
    torch.testing.assert_close(
        frequencies,
        torch.tensor(probabilities),
        rtol=0.08,
        atol=0.02,
    )


def test_generate_tokens_handles_zero_new_tokens_without_model_call():
    model = RecordingLanguageModel()
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    generated = generate_tokens(model, token_ids, max_new_tokens=0, context_size=2)

    torch.testing.assert_close(generated, token_ids)
    assert model.calls == []
