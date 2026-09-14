import torch
from torch import nn

from solutions.q0021 import FeedForward


def linear_layers(model):
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def find_linear(model, in_features, out_features):
    matches = [
        layer
        for layer in linear_layers(model)
        if layer.in_features == in_features and layer.out_features == out_features
    ]
    assert len(matches) == 1
    return matches[0]


def test_feed_forward_is_torch_module_with_expected_dimensions():
    model = FeedForward(embedding_size=8, dropout=0.2)

    assert isinstance(model, nn.Module)
    assert len(linear_layers(model)) == 2
    find_linear(model, in_features=8, out_features=32)
    find_linear(model, in_features=32, out_features=8)


def test_feed_forward_returns_same_shape_for_sequence_inputs():
    model = FeedForward(embedding_size=6, dropout=0.0)
    model.eval()

    for shape in [(1, 1, 6), (2, 5, 6), (4, 9, 6)]:
        output = model(torch.randn(*shape))

        assert output.shape == shape


def test_feed_forward_applies_relu_between_linear_transformations():
    embedding_size = 5
    model = FeedForward(embedding_size=embedding_size, dropout=0.0)
    model.eval()
    expansion = find_linear(model, embedding_size, 4 * embedding_size)
    contraction = find_linear(model, 4 * embedding_size, embedding_size)
    observed = {}
    handles = [
        expansion.register_forward_hook(
            lambda _module, _inputs, output: observed.update(expanded=output)
        ),
        contraction.register_forward_pre_hook(
            lambda _module, inputs: observed.update(contracted_input=inputs[0])
        ),
        contraction.register_forward_hook(
            lambda _module, _inputs, output: observed.update(projected=output)
        ),
    ]

    try:
        output = model(torch.randn(2, 4, embedding_size))
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(observed["contracted_input"], torch.relu(observed["expanded"]))
    torch.testing.assert_close(output, observed["projected"])


def test_feed_forward_processes_positions_independently():
    model = FeedForward(embedding_size=7, dropout=0.0)
    model.eval()
    x = torch.randn(2, 6, 7)
    changed = x.clone()
    changed[0, 3] = torch.randn(7) * 20

    original = model(x)
    modified = model(changed)
    unchanged_positions = torch.ones((2, 6), dtype=torch.bool)
    unchanged_positions[0, 3] = False

    torch.testing.assert_close(
        original[unchanged_positions], modified[unchanged_positions]
    )


def test_feed_forward_applies_final_dropout_only_during_training():
    model = FeedForward(embedding_size=6, dropout=1.0)
    x = torch.randn(2, 5, 6)

    model.train()
    training_output = model(x)
    model.eval()
    evaluation_output = model(x)

    torch.testing.assert_close(training_output, torch.zeros_like(training_output))
    assert torch.count_nonzero(evaluation_output) > 0


def test_feed_forward_propagates_gradients():
    model = FeedForward(embedding_size=6, dropout=0.0)
    x = torch.randn(2, 5, 6, requires_grad=True)

    output = model(x)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
