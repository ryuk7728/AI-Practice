import torch
from torch import nn

from solutions.q0013 import PranayNet


def test_pranay_net_is_torch_module():
    model = PranayNet()

    assert isinstance(model, nn.Module)


def test_pranay_net_has_expected_layers():
    model = PranayNet()

    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc2, nn.Linear)
    assert model.fc1.in_features == 784
    assert model.fc1.out_features == 10
    assert model.fc2.in_features == 10
    assert model.fc2.out_features == 10


def test_pranay_net_forward_returns_expected_shape():
    model = PranayNet()
    X = torch.randn(5, 784)

    output = model(X)

    assert output.shape == (5, 10)


def test_pranay_net_returns_raw_scores_not_probabilities():
    model = PranayNet()
    X = torch.randn(6, 784)

    output = model(X)

    assert not torch.allclose(output.sum(dim=1), torch.ones(6))


def test_pranay_net_applies_relu_between_layers():
    model = PranayNet()
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.copy_(torch.tensor([-1.0, 2.0] + [0.0] * 8))
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
        model.fc2.weight[0, 0] = 1.0
        model.fc2.weight[0, 1] = 1.0

    output = model(torch.zeros(1, 784))

    torch.testing.assert_close(output[0, 0], torch.tensor(2.0))
