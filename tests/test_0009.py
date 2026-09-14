import numpy as np
import torch

from solutions.q0009 import tensor_basics


EXPECTED_KEYS = {
    "direct_creation",
    "array_conversion",
    "sum_last_axis",
    "random",
    "dimensions",
    "element_type",
    "dimension_count",
    "device_before_transfer",
    "transferred",
    "array_output",
    "float_tensor",
}


def test_tensor_basics_returns_expected_keys():
    results = tensor_basics(device="cpu")

    assert set(results) == EXPECTED_KEYS


def test_tensor_basics_creates_expected_tensors():
    results = tensor_basics(device="cpu")
    expected = torch.tensor([[1, 2, 3], [4, 5, 6]])

    torch.testing.assert_close(
        results["direct_creation"], expected.to(results["direct_creation"].dtype)
    )
    torch.testing.assert_close(results["array_conversion"], expected)


def test_tensor_basics_sums_over_last_axis_with_dimensions():
    results = tensor_basics(device="cpu")
    expected = torch.tensor([[6], [15]])

    torch.testing.assert_close(results["sum_last_axis"], expected)


def test_tensor_basics_random_tensor_has_expected_properties():
    results = tensor_basics(device="cpu")
    random_values = results["random"]

    assert random_values.shape == (3, 2)
    assert torch.all(random_values >= 0)
    assert torch.all(random_values < 1)


def test_tensor_basics_reports_metadata():
    results = tensor_basics(device="cpu")

    assert results["dimensions"] == torch.Size([2, 3])
    assert results["element_type"] == results["array_conversion"].dtype
    assert results["dimension_count"] == 2
    assert results["device_before_transfer"] == torch.device("cpu")


def test_tensor_basics_transfers_to_requested_device():
    results = tensor_basics(device="cpu")

    assert results["transferred"].device == torch.device("cpu")


def test_tensor_basics_converts_tensor_to_numpy():
    results = tensor_basics(device="cpu")

    assert isinstance(results["array_output"], np.ndarray)
    np.testing.assert_array_equal(
        results["array_output"], np.array([[1, 2, 3], [4, 5, 6]])
    )


def test_tensor_basics_converts_tensor_to_float():
    results = tensor_basics(device="cpu")

    assert results["float_tensor"].dtype == torch.float32
    torch.testing.assert_close(
        results["float_tensor"], torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    )
