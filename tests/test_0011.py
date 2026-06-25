import torch

from helpers.q0011 import FixedClassifier, make_classification_batch
from solutions.q0011 import compute_accuracy


def test_compute_accuracy_returns_tensor_scalar():
    model = FixedClassifier()
    X_test, y_test = make_classification_batch()

    accuracy = compute_accuracy(model, X_test, y_test)

    assert isinstance(accuracy, torch.Tensor)
    assert accuracy.ndim == 0


def test_compute_accuracy_matches_expected_value():
    model = FixedClassifier()
    X_test, y_test = make_classification_batch()

    accuracy = compute_accuracy(model, X_test, y_test)

    torch.testing.assert_close(accuracy, torch.tensor(0.75))


def test_compute_accuracy_sets_model_to_eval_mode():
    model = FixedClassifier()
    model.train()
    X_test, y_test = make_classification_batch()

    compute_accuracy(model, X_test, y_test)

    assert model.training is False


def test_compute_accuracy_does_not_accumulate_gradients():
    model = FixedClassifier()
    X_test, y_test = make_classification_batch()

    compute_accuracy(model, X_test, y_test)

    assert all(parameter.grad is None for parameter in model.parameters())
