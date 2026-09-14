import random

import torch

from solutions.q0017 import prepare_text_data


def expected_encoding(text):
    ordered_characters = sorted(set(text))
    character_ids = {
        character: index for index, character in enumerate(ordered_characters)
    }
    return torch.tensor([character_ids[character] for character in text], dtype=torch.long)


def test_prepare_text_data_returns_two_long_tensors():
    training, validation = prepare_text_data("machine learning", train_fraction=0.7)

    assert isinstance(training, torch.Tensor)
    assert isinstance(validation, torch.Tensor)
    assert training.dtype == torch.long
    assert validation.dtype == torch.long
    assert training.ndim == 1
    assert validation.ndim == 1


def test_prepare_text_data_uses_deterministic_character_ids():
    training, validation = prepare_text_data("banana", train_fraction=0.5)

    torch.testing.assert_close(training, torch.tensor([1, 0, 2]))
    torch.testing.assert_close(validation, torch.tensor([0, 2, 0]))


def test_prepare_text_data_preserves_whitespace_and_punctuation():
    text = "Hi, Hi!"
    training, validation = prepare_text_data(text, train_fraction=0.75)
    expected = expected_encoding(text)
    split_index = int(0.75 * len(text))

    torch.testing.assert_close(training, expected[:split_index])
    torch.testing.assert_close(validation, expected[split_index:])


def test_prepare_text_data_uses_training_fraction_for_split_position():
    text = "abcdefghij"

    for fraction in [0.1, 0.55, 0.9]:
        training, validation = prepare_text_data(text, train_fraction=fraction)
        split_index = int(fraction * len(text))

        assert len(training) == split_index
        assert len(validation) == len(text) - split_index


def test_prepare_text_data_handles_one_distinct_character():
    training, validation = prepare_text_data("zzzzz", train_fraction=0.6)

    torch.testing.assert_close(training, torch.tensor([0, 0, 0]))
    torch.testing.assert_close(validation, torch.tensor([0, 0]))


def test_prepare_text_data_matches_randomized_inputs():
    generator = random.Random(17)
    alphabet = "abcXYZ !?"

    for _ in range(10):
        text = "".join(generator.choice(alphabet) for _ in range(40))
        fraction = generator.uniform(0.1, 0.9)
        training, validation = prepare_text_data(text, train_fraction=fraction)
        expected = expected_encoding(text)
        split_index = int(fraction * len(text))

        torch.testing.assert_close(training, expected[:split_index])
        torch.testing.assert_close(validation, expected[split_index:])
