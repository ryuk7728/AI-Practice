from itertools import permutations

import torch
from torch import nn

from solutions.q0019 import Head


def projection_layers(model, embedding_size, head_size):
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear)
        and module.in_features == embedding_size
        and module.out_features == head_size
    ]


def test_head_is_torch_module_with_three_bias_free_projections():
    model = Head(embedding_size=6, head_size=3, context_size=8)
    projections = projection_layers(model, embedding_size=6, head_size=3)

    assert isinstance(model, nn.Module)
    assert len(projections) == 3
    assert all(layer.bias is None for layer in projections)


def test_head_returns_expected_shape_for_multiple_sequence_lengths():
    model = Head(embedding_size=8, head_size=4, context_size=10)
    model.eval()

    for batch_size, sequence_length in [(1, 1), (2, 5), (4, 10)]:
        output = model(torch.randn(batch_size, sequence_length, 8))

        assert output.shape == (batch_size, sequence_length, 4)


def test_head_matches_scaled_causal_attention_behavior():
    embedding_size = 6
    head_size = 3
    model = Head(
        embedding_size=embedding_size,
        head_size=head_size,
        context_size=8,
        dropout=0.0,
    )
    model.eval()
    projected = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: projected.append(output)
        )
        for layer in projection_layers(model, embedding_size, head_size)
    ]
    x = torch.randn(2, 5, embedding_size)

    try:
        actual = model(x)
    finally:
        for handle in handles:
            handle.remove()

    assert len(projected) == 3
    causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    possible_outputs = []

    for query, key, value in permutations(projected):
        scores = query @ key.transpose(-2, -1) * head_size**-0.5
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        possible_outputs.append(weights @ value)

    assert any(
        torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
        for expected in possible_outputs
    )


def test_head_does_not_use_future_positions():
    model = Head(embedding_size=7, head_size=4, context_size=8, dropout=0.0)
    model.eval()
    x = torch.randn(2, 8, 7)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20

    original_prefix = model(x)[:, :4]
    changed_prefix = model(changed)[:, :4]

    torch.testing.assert_close(original_prefix, changed_prefix)


def test_head_handles_zero_attention_scores_without_nan():
    model = Head(embedding_size=6, head_size=3, context_size=5, dropout=0.0)
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    output = model(torch.zeros(2, 5, 6))

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros(2, 5, 3))


def test_head_processes_batch_examples_independently():
    model = Head(embedding_size=5, head_size=3, context_size=6, dropout=0.0)
    model.eval()
    first = torch.randn(1, 6, 5)
    second = torch.randn(1, 6, 5)

    separate = model(first)
    batched = model(torch.cat((first, second), dim=0))[:1]

    torch.testing.assert_close(separate, batched)


def test_head_applies_attention_dropout_only_during_training():
    model = Head(embedding_size=6, head_size=3, context_size=5, dropout=1.0)
    x = torch.randn(2, 5, 6)

    model.train()
    training_output = model(x)
    model.eval()
    evaluation_output = model(x)

    torch.testing.assert_close(training_output, torch.zeros_like(training_output))
    assert torch.count_nonzero(evaluation_output) > 0


def test_head_propagates_gradients():
    model = Head(embedding_size=6, head_size=3, context_size=5, dropout=0.0)
    x = torch.randn(2, 5, 6, requires_grad=True)

    output = model(x)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
