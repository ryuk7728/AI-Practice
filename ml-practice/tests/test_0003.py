import numpy as np

from solutions.q0003 import cross_entropy


def test_cross_entropy_known_single_example():
    y_true = np.array([[1, 0, 0]])
    y_pred = np.array([[0.8, 0.1, 0.1]])

    np.testing.assert_allclose(cross_entropy(y_true, y_pred), -np.log(0.8))


def test_cross_entropy_known_batch_example():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])

    expected = (-np.log(0.7) - np.log(0.6)) / 2

    np.testing.assert_allclose(cross_entropy(y_true, y_pred), expected)


def test_cross_entropy_handles_zero_probabilities():
    y_true = np.array([[1, 0, 0]])
    y_pred = np.array([[0.0, 0.5, 0.5]])

    result = cross_entropy(y_true, y_pred)

    assert np.isfinite(result)
    assert result > 0


def test_cross_entropy_matches_reference_on_random_distributions():
    rng = np.random.default_rng(123)
    logits = rng.random(size=(12, 5))
    y_pred = logits / logits.sum(axis=1, keepdims=True)
    class_ids = rng.integers(0, 5, size=12)
    y_true = np.eye(5)[class_ids]
    true_class_probabilities = y_pred[np.arange(12), class_ids]
    expected = -np.mean(np.log(true_class_probabilities))

    np.testing.assert_allclose(cross_entropy(y_true, y_pred), expected)
