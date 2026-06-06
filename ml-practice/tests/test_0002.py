import numpy as np

from solutions.q0002 import softmax


def test_softmax_outputs_sum_to_one():
    result = softmax(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(np.sum(result), 1.0)


def test_softmax_outputs_are_non_negative():
    result = softmax(np.array([-2.0, 0.0, 2.0]))

    assert np.all(result >= 0)


def test_softmax_matches_reference_on_random_vectors():
    rng = np.random.default_rng(123)

    for _ in range(10):
        x = rng.normal(size=8)
        result = softmax(x)
        np.testing.assert_allclose(np.sum(result), 1.0)
        assert np.all(result >= 0)
        assert np.argmax(result) == np.argmax(x)


def test_softmax_preserves_1d_shape():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    assert softmax(x).shape == x.shape


def test_softmax_handles_large_values_without_overflow():
    x = np.array([1000.0, 1001.0, 1002.0])
    expected = np.array([0.09003057317038045, 0.24472847105479764, 0.6652409557748218])

    result = softmax(x)

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_softmax_computes_row_wise_for_2d_inputs():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    expected = np.array(
        [
            [0.09003057317038045, 0.24472847105479764, 0.6652409557748218],
            [0.6652409557748218, 0.24472847105479764, 0.09003057317038045],
        ]
    )

    result = softmax(x)

    assert result.shape == x.shape
    np.testing.assert_allclose(np.sum(result, axis=1), np.ones(2))
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_softmax_handles_large_2d_values_without_overflow():
    x = np.array([[1000.0, 1001.0, 1002.0], [2002.0, 2001.0, 2000.0]])

    result = softmax(x)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    np.testing.assert_allclose(np.sum(result, axis=1), np.ones(2))
    assert np.argmax(result[0]) == 2
    assert np.argmax(result[1]) == 0
