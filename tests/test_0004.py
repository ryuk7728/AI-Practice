import numpy as np

from solutions.q0004 import mse_loss


def test_mse_loss_is_zero_for_identical_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert mse_loss(y_true, y_pred) == 0


def test_mse_loss_known_vector_example():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    np.testing.assert_allclose(mse_loss(y_true, y_pred), 1 / 3)


def test_mse_loss_accepts_python_lists():
    y_true = [2, 4, 6]
    y_pred = [1, 5, 8]

    np.testing.assert_allclose(mse_loss(y_true, y_pred), 2.0)


def test_mse_loss_handles_2d_inputs():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 3.0], [5.0, 4.0]])

    np.testing.assert_allclose(mse_loss(y_true, y_pred), 1.25)


def test_mse_loss_matches_expected_random_batch():
    rng = np.random.default_rng(123)
    y_true = rng.normal(size=(20, 3))
    y_pred = rng.normal(size=(20, 3))
    differences = y_true - y_pred
    expected = np.sum(differences * differences) / differences.size

    np.testing.assert_allclose(mse_loss(y_true, y_pred), expected)
