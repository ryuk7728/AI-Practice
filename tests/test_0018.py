import torch

from solutions.q0018 import sample_text_batch


def assert_rows_are_valid_windows(inputs, targets, source):
    for input_row, target_row in zip(inputs.cpu(), targets.cpu()):
        torch.testing.assert_close(input_row[1:], target_row[:-1])
        matches = (source == input_row[0]).nonzero(as_tuple=True)[0]
        assert any(
            torch.equal(input_row, source[start : start + len(input_row)])
            and torch.equal(target_row, source[start + 1 : start + len(input_row) + 1])
            for start in matches.tolist()
        )


def test_sample_text_batch_returns_expected_shapes_and_dtype():
    train_data = torch.arange(40, dtype=torch.long)
    val_data = torch.arange(100, 130, dtype=torch.long)

    inputs, targets = sample_text_batch(
        train_data, val_data, "train", batch_size=6, block_size=5, device="cpu"
    )

    assert inputs.shape == (6, 5)
    assert targets.shape == (6, 5)
    assert inputs.dtype == train_data.dtype
    assert targets.dtype == train_data.dtype


def test_sample_text_batch_uses_training_data():
    train_data = torch.arange(50, dtype=torch.long)
    val_data = torch.arange(100, 150, dtype=torch.long)

    inputs, targets = sample_text_batch(
        train_data, val_data, "train", batch_size=8, block_size=6, device="cpu"
    )

    assert_rows_are_valid_windows(inputs, targets, train_data)


def test_sample_text_batch_uses_validation_data():
    train_data = torch.arange(50, dtype=torch.long)
    val_data = torch.arange(100, 150, dtype=torch.long)

    inputs, targets = sample_text_batch(
        train_data, val_data, "val", batch_size=8, block_size=6, device="cpu"
    )

    assert_rows_are_valid_windows(inputs, targets, val_data)


def test_sample_text_batch_pairs_each_input_with_shifted_targets():
    train_data = torch.arange(80, dtype=torch.long)
    val_data = torch.arange(100, 180, dtype=torch.long)
    torch.manual_seed(18)

    inputs, targets = sample_text_batch(
        train_data, val_data, "train", batch_size=12, block_size=9, device="cpu"
    )

    torch.testing.assert_close(targets, inputs + 1)
    assert_rows_are_valid_windows(inputs, targets, train_data)


def test_sample_text_batch_handles_only_available_window():
    train_data = torch.tensor([7, 3, 9, 2, 8])
    val_data = torch.tensor([6, 1, 4, 0, 5])

    inputs, targets = sample_text_batch(
        train_data, val_data, "train", batch_size=3, block_size=4, device="cpu"
    )

    expected_inputs = train_data[:4].repeat(3, 1)
    expected_targets = train_data[1:].repeat(3, 1)
    torch.testing.assert_close(inputs, expected_inputs)
    torch.testing.assert_close(targets, expected_targets)


def test_sample_text_batch_samples_multiple_starting_positions():
    train_data = torch.arange(200, dtype=torch.long)
    val_data = torch.arange(300, 500, dtype=torch.long)
    torch.manual_seed(1818)

    inputs, _ = sample_text_batch(
        train_data, val_data, "train", batch_size=32, block_size=8, device="cpu"
    )

    assert torch.unique(inputs[:, 0]).numel() > 1


def test_sample_text_batch_places_outputs_on_requested_device():
    train_data = torch.arange(30, dtype=torch.long)
    val_data = torch.arange(30, 60, dtype=torch.long)

    inputs, targets = sample_text_batch(
        train_data, val_data, "val", batch_size=4, block_size=5, device="cpu"
    )

    assert inputs.device == torch.device("cpu")
    assert targets.device == torch.device("cpu")


def test_sample_text_batch_selects_an_available_device_by_default():
    train_data = torch.arange(30, dtype=torch.long)
    val_data = torch.arange(30, 60, dtype=torch.long)
    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    inputs, targets = sample_text_batch(
        train_data, val_data, "train", batch_size=4, block_size=5
    )

    assert inputs.device.type == expected_device_type
    assert targets.device.type == expected_device_type
