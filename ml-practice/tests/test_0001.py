import numpy as np

from solutions.q0001 import relu


def test_relu_turns_negative_values_to_zero():
    x = np.array([-5, -1, -0.1])
    expected = np.array([0, 0, 0])

    np.testing.assert_array_equal(relu(x), expected)


def test_relu_keeps_positive_values_unchanged():
    x = np.array([0.5, 1.0, 7.25])

    np.testing.assert_array_equal(relu(x), x)


def test_relu_handles_zero():
    x = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])

    np.testing.assert_array_equal(relu(x), expected)


def test_relu_preserves_shape():
    x = np.array([[-2, -1, 0], [1, 2, 3]])

    assert relu(x).shape == x.shape


def test_relu_matches_reference_on_random_inputs():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(20, 10))
    expected = np.maximum(x, 0)

    np.testing.assert_array_equal(relu(x), expected)
