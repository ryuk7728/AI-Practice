import torch

from solutions.q0010 import compute_gradient


def test_compute_gradient_returns_tensor():
    gradient = compute_gradient()

    assert isinstance(gradient, torch.Tensor)


def test_compute_gradient_has_expected_shape():
    gradient = compute_gradient()

    assert gradient.shape == (2,)


def test_compute_gradient_matches_expected_values():
    gradient = compute_gradient()

    torch.testing.assert_close(gradient, torch.tensor([8.0, 16.0]))


def test_compute_gradient_result_does_not_require_grad():
    gradient = compute_gradient()

    assert gradient.requires_grad is False
