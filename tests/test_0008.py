import torch

from solutions.q0008 import get_device_info


def test_get_device_info_returns_expected_keys():
    info = get_device_info()

    assert set(info) == {"cuda_available", "device", "device_name"}


def test_get_device_info_reports_cuda_availability():
    info = get_device_info()

    assert info["cuda_available"] == torch.cuda.is_available()


def test_get_device_info_selects_available_device():
    info = get_device_info()
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert info["device"] == expected


def test_get_device_info_reports_device_name():
    info = get_device_info()
    expected = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    assert info["device_name"] == expected
