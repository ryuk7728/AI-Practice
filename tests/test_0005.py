import numpy as np

from solutions.q0005 import LinearRegression


def test_linear_regression_initializes_parameters_to_zero():
    model = LinearRegression()

    assert model.w == 0
    assert model.b == 0


def test_linear_regression_predicts_scalar_input():
    model = LinearRegression()
    model.w = 2
    model.b = 1

    assert model.predict(3) == 7


def test_linear_regression_predicts_list_input():
    model = LinearRegression()
    model.w = 2
    model.b = 1

    np.testing.assert_array_equal(model.predict([1, 2, 3]), np.array([3, 5, 7]))


def test_linear_regression_predicts_numpy_array_input():
    model = LinearRegression()
    model.w = -0.5
    model.b = 3
    X = np.array([0.0, 2.0, 4.0])
    expected = np.array([3.0, 2.0, 1.0])

    np.testing.assert_allclose(model.predict(X), expected)


def test_linear_regression_predict_preserves_array_shape():
    model = LinearRegression()
    model.w = 4
    model.b = -2
    X = np.array([[1, 2], [3, 4]])

    assert model.predict(X).shape == X.shape


def test_linear_regression_uses_current_parameters():
    model = LinearRegression()
    X = np.array([1, 2, 3])

    model.w = 1
    model.b = 0
    first = model.predict(X)

    model.w = 3
    model.b = -1
    second = model.predict(X)

    np.testing.assert_array_equal(first, np.array([1, 2, 3]))
    np.testing.assert_array_equal(second, np.array([2, 5, 8]))
