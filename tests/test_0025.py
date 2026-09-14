import torch
from torch import nn

from solutions.q0025 import initialize_gpt_weights


class InitializationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.layers = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
        )
        self.normalization = nn.LayerNorm(128)
        self.extra = nn.Parameter(torch.full((128,), 3.0))


def assert_approximately_normal(values, expected_std):
    assert abs(values.mean().item()) < expected_std * 0.08
    assert abs(values.std().item() - expected_std) < expected_std * 0.08


def test_initialize_gpt_weights_returns_same_model():
    model = InitializationModel()

    returned = initialize_gpt_weights(model)

    assert returned is model


def test_initialize_gpt_weights_initializes_nested_linear_weights():
    model = InitializationModel()
    torch.manual_seed(25)

    initialize_gpt_weights(model, std=0.02)
    linear_layers = [
        module for module in model.modules() if isinstance(module, nn.Linear)
    ]

    assert len(linear_layers) == 2
    for layer in linear_layers:
        assert_approximately_normal(layer.weight.detach(), expected_std=0.02)


def test_initialize_gpt_weights_initializes_embedding_weights():
    model = InitializationModel()
    torch.manual_seed(250)

    initialize_gpt_weights(model, std=0.02)

    assert_approximately_normal(model.embedding.weight.detach(), expected_std=0.02)


def test_initialize_gpt_weights_uses_supplied_standard_deviation():
    model = InitializationModel()
    torch.manual_seed(2500)

    initialize_gpt_weights(model, std=0.08)

    assert_approximately_normal(model.embedding.weight.detach(), expected_std=0.08)


def test_initialize_gpt_weights_zeros_linear_biases_and_supports_missing_bias():
    model = InitializationModel()
    with torch.no_grad():
        model.layers[0].bias.fill_(4.0)

    initialize_gpt_weights(model)

    torch.testing.assert_close(
        model.layers[0].bias,
        torch.zeros_like(model.layers[0].bias),
    )
    assert model.layers[2].bias is None


def test_initialize_gpt_weights_preserves_other_parameters():
    model = InitializationModel()
    norm_weight = model.normalization.weight.detach().clone()
    norm_bias = model.normalization.bias.detach().clone()
    extra = model.extra.detach().clone()

    initialize_gpt_weights(model)

    torch.testing.assert_close(model.normalization.weight, norm_weight)
    torch.testing.assert_close(model.normalization.bias, norm_bias)
    torch.testing.assert_close(model.extra, extra)
