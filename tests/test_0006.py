from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from solutions.q0006 import load_data


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Housing.csv"


def expected_outputs(path):
    df = pd.read_csv(path)

    X = np.array(df["area"]).reshape(-1, 1)
    y = np.array(df["price"]).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    return X_train, X_test, y_train, y_test


def test_load_data_returns_four_arrays():
    outputs = load_data(DATA_PATH)

    assert len(outputs) == 4
    assert all(isinstance(output, np.ndarray) for output in outputs)


def test_load_data_returns_expected_shapes():
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    assert X_train.shape == (8, 1)
    assert X_test.shape == (2, 1)
    assert y_train.shape == (8, 1)
    assert y_test.shape == (2, 1)


def test_load_data_matches_expected_preprocessing():
    actual = load_data(DATA_PATH)
    expected = expected_outputs(DATA_PATH)

    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_allclose(actual_array, expected_array)


def test_load_data_standardizes_training_features():
    X_train, _, _, _ = load_data(DATA_PATH)

    np.testing.assert_allclose(np.mean(X_train), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(X_train), 1.0, atol=1e-12)


def test_load_data_log_transforms_targets():
    _, _, y_train, y_test = load_data(DATA_PATH)
    _, _, expected_y_train, expected_y_test = expected_outputs(DATA_PATH)

    np.testing.assert_allclose(y_train, expected_y_train)
    np.testing.assert_allclose(y_test, expected_y_test)


def test_load_data_accepts_string_path():
    from_string = load_data(str(DATA_PATH))
    from_path = load_data(DATA_PATH)

    for string_array, path_array in zip(from_string, from_path):
        np.testing.assert_allclose(string_array, path_array)
