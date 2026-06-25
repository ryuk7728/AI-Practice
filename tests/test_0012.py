import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from solutions.q0012 import prepare_mnist_loaders


def make_mnist_like_data():
    X = np.arange(20 * 784, dtype=np.float32).reshape(20, 784) % 256
    y = np.arange(20) % 10
    return X, y


def expected_split(X, y):
    X_scaled = X / 255
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def test_prepare_mnist_loaders_returns_expected_objects():
    X, y = make_mnist_like_data()
    outputs = prepare_mnist_loaders(X, y)

    assert len(outputs) == 6
    assert all(isinstance(output, torch.Tensor) for output in outputs[:4])
    assert isinstance(outputs[4], DataLoader)
    assert isinstance(outputs[5], DataLoader)


def test_prepare_mnist_loaders_matches_expected_split():
    X, y = make_mnist_like_data()
    actual = prepare_mnist_loaders(X, y, batch_size=4)
    expected = expected_split(X, y)

    for actual_tensor, expected_array in zip(actual[:4], expected):
        torch.testing.assert_close(actual_tensor.cpu(), torch.from_numpy(expected_array))


def test_prepare_mnist_loaders_uses_expected_dtypes():
    X, y = make_mnist_like_data()
    X_train, X_test, y_train, y_test, _, _ = prepare_mnist_loaders(X, y)

    assert X_train.dtype == torch.float32
    assert X_test.dtype == torch.float32
    assert y_train.dtype == torch.long
    assert y_test.dtype == torch.long


def test_prepare_mnist_loaders_normalizes_features():
    X, y = make_mnist_like_data()
    X_train, X_test, _, _, _, _ = prepare_mnist_loaders(X, y)

    assert torch.all(X_train >= 0)
    assert torch.all(X_train <= 1)
    assert torch.all(X_test >= 0)
    assert torch.all(X_test <= 1)


def test_prepare_mnist_loaders_creates_tensor_datasets():
    X, y = make_mnist_like_data()
    X_train, X_test, y_train, y_test, train_loader, test_loader = prepare_mnist_loaders(
        X, y, batch_size=4
    )

    assert isinstance(train_loader.dataset, TensorDataset)
    assert isinstance(test_loader.dataset, TensorDataset)
    torch.testing.assert_close(train_loader.dataset.tensors[0], X_train)
    torch.testing.assert_close(train_loader.dataset.tensors[1], y_train)
    torch.testing.assert_close(test_loader.dataset.tensors[0], X_test)
    torch.testing.assert_close(test_loader.dataset.tensors[1], y_test)


def test_prepare_mnist_loaders_configures_batching_and_shuffle():
    X, y = make_mnist_like_data()
    _, _, _, _, train_loader, test_loader = prepare_mnist_loaders(X, y, batch_size=4)

    assert train_loader.batch_size == 4
    assert test_loader.batch_size == 4
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
