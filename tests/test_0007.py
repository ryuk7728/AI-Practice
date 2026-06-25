import numpy as np

from helpers.q0007 import LinearRegression, mse_loss
from solutions.q0007 import train


def make_dataset():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = 2.5 * X - 0.75
    return X[:4], X[4:], y[:4], y[4:]


def reference_train(
    model, X_train, y_train, X_test, y_test, learning_rate=0.003, epochs=1000
):
    losses = []
    n = len(X_train)

    for _ in range(epochs):
        y_pred_test = model.predict(X_test)
        losses.append(mse_loss(y_test, y_pred_test))

        y_pred_train = model.predict(X_train)
        dW = (-2 / n) * np.sum(X_train * (y_train - y_pred_train))
        dB = (-2 / n) * np.sum(y_train - y_pred_train)

        model.w = model.w - learning_rate * dW
        model.b = model.b - learning_rate * dB

    return losses


def test_train_returns_one_loss_per_epoch():
    X_train, X_test, y_train, y_test = make_dataset()
    model = LinearRegression()

    losses = train(model, X_train, y_train, X_test, y_test, epochs=25)

    assert len(losses) == 25


def test_train_returns_numeric_losses():
    X_train, X_test, y_train, y_test = make_dataset()
    model = LinearRegression()

    losses = train(model, X_train, y_train, X_test, y_test, epochs=5)

    assert all(np.isscalar(loss) for loss in losses)
    assert all(np.isfinite(loss) for loss in losses)


def test_train_updates_model_parameters():
    X_train, X_test, y_train, y_test = make_dataset()
    model = LinearRegression()

    train(model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=20)

    assert model.w != 0
    assert model.b != 0


def test_train_matches_reference_implementation():
    X_train, X_test, y_train, y_test = make_dataset()
    actual_model = LinearRegression()
    expected_model = LinearRegression()

    actual_losses = train(
        actual_model,
        X_train,
        y_train,
        X_test,
        y_test,
        learning_rate=0.01,
        epochs=40,
    )
    expected_losses = reference_train(
        expected_model,
        X_train,
        y_train,
        X_test,
        y_test,
        learning_rate=0.01,
        epochs=40,
    )

    np.testing.assert_allclose(actual_losses, expected_losses)
    np.testing.assert_allclose(actual_model.w, expected_model.w)
    np.testing.assert_allclose(actual_model.b, expected_model.b)


def test_train_reduces_training_loss():
    X_train, X_test, y_train, y_test = make_dataset()
    model = LinearRegression()

    initial_train_loss = mse_loss(y_train, model.predict(X_train))
    train(model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=80)
    final_train_loss = mse_loss(y_train, model.predict(X_train))

    assert final_train_loss < initial_train_loss


def test_train_handles_single_epoch():
    X_train, X_test, y_train, y_test = make_dataset()
    model = LinearRegression()
    expected_model = LinearRegression()

    losses = train(model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1)
    expected_losses = reference_train(
        expected_model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1
    )

    assert len(losses) == 1
    np.testing.assert_allclose(losses, expected_losses)
    np.testing.assert_allclose(model.w, expected_model.w)
    np.testing.assert_allclose(model.b, expected_model.b)
