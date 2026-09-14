import torch
from torch import nn

from helpers.q0014 import TinyClassifier, make_tiny_loader
from solutions.q0014 import train_classifier


def reference_train_classifier(train_loader, device="cpu", epochs=2, learning_rate=0.01):
    model = TinyClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss / len(train_loader))

    return model, losses


def test_train_classifier_returns_model_and_losses():
    train_loader = make_tiny_loader()

    model, losses = train_classifier(train_loader, device="cpu", epochs=3, learning_rate=0.01)

    assert isinstance(model, TinyClassifier)
    assert len(losses) == 3


def test_train_classifier_returns_numeric_epoch_losses():
    train_loader = make_tiny_loader()

    _, losses = train_classifier(train_loader, device="cpu", epochs=4, learning_rate=0.01)

    assert all(isinstance(loss, float) for loss in losses)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_train_classifier_moves_model_to_requested_device():
    train_loader = make_tiny_loader()

    model, _ = train_classifier(train_loader, device="cpu", epochs=1, learning_rate=0.01)

    assert next(model.parameters()).device == torch.device("cpu")


def test_train_classifier_updates_parameters():
    torch.manual_seed(123)
    reference_model = TinyClassifier()
    initial_parameters = [parameter.detach().clone() for parameter in reference_model.parameters()]

    torch.manual_seed(123)
    train_loader = make_tiny_loader()
    model, _ = train_classifier(train_loader, device="cpu", epochs=2, learning_rate=0.01)

    assert any(
        not torch.allclose(before, after.detach().cpu())
        for before, after in zip(initial_parameters, model.parameters())
    )


def test_train_classifier_sets_model_to_train_mode():
    train_loader = make_tiny_loader()

    model, _ = train_classifier(train_loader, device="cpu", epochs=1, learning_rate=0.01)

    assert model.training is True


def test_train_classifier_uses_cross_entropy_compatible_outputs():
    train_loader = make_tiny_loader()
    model, _ = train_classifier(train_loader, device="cpu", epochs=1, learning_rate=0.01)
    criterion = nn.CrossEntropyLoss()
    X_batch, y_batch = next(iter(train_loader))

    logits = model(X_batch)
    loss = criterion(logits, y_batch)

    assert logits.shape[1] == 3
    assert torch.isfinite(loss)


def test_train_classifier_clears_gradients_for_each_batch():
    train_loader = make_tiny_loader()

    torch.manual_seed(123)
    actual_model, actual_losses = train_classifier(
        train_loader, device="cpu", epochs=2, learning_rate=0.01
    )

    torch.manual_seed(123)
    expected_model, expected_losses = reference_train_classifier(
        train_loader, device="cpu", epochs=2, learning_rate=0.01
    )

    torch.testing.assert_close(torch.tensor(actual_losses), torch.tensor(expected_losses))
    for actual_parameter, expected_parameter in zip(
        actual_model.parameters(), expected_model.parameters()
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter)
