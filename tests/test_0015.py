import torch
from torch import nn

from solutions.q0015 import PranayNet


def modules_of_type(model, module_type):
    return [module for module in model.modules() if isinstance(module, module_type)]


def test_pranay_net_is_torch_module():
    model = PranayNet()

    assert isinstance(model, nn.Module)


def test_pranay_net_has_expected_convolution_channels():
    model = PranayNet()
    expected_channels = [
        ("conv1", 3, 16),
        ("conv2", 16, 32),
        ("conv3", 32, 64),
        ("conv4", 64, 128),
        ("conv5", 128, 128),
    ]

    for name, in_channels, out_channels in expected_channels:
        layer = getattr(model, name)
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == in_channels
        assert layer.out_channels == out_channels


def test_pranay_net_has_matching_batch_norm_layers():
    model = PranayNet()
    expected_features = [
        ("bn1", 16),
        ("bn2", 32),
        ("bn3", 64),
        ("bn4", 128),
        ("bn5", 128),
    ]

    for name, num_features in expected_features:
        layer = getattr(model, name)
        assert isinstance(layer, nn.BatchNorm2d)
        assert layer.num_features == num_features


def test_pranay_net_has_pool_flatten_dropout_and_classifier_layers():
    model = PranayNet()
    pools = modules_of_type(model, nn.MaxPool2d)
    flatten_layers = modules_of_type(model, nn.Flatten)
    dropout_layers = modules_of_type(model, nn.Dropout)
    linear_layers = modules_of_type(model, nn.Linear)

    assert pools
    assert flatten_layers
    assert dropout_layers
    assert all(layer.p == 0.5 for layer in dropout_layers)
    assert len(linear_layers) == 3

    expected_dimensions = [
        (128 * 4 * 4, 128),
        (128, 64),
        (64, 2),
    ]
    actual_dimensions = [
        (layer.in_features, layer.out_features) for layer in linear_layers
    ]

    assert sorted(actual_dimensions) == sorted(expected_dimensions)


def test_pranay_net_forward_returns_expected_shape():
    model = PranayNet()
    model.eval()
    X = torch.randn(4, 3, 128, 128)

    with torch.no_grad():
        output = model(X)

    assert output.shape == (4, 2)


def test_pranay_net_returns_raw_logits_not_probabilities():
    model = PranayNet()
    model.eval()
    X = torch.randn(3, 3, 128, 128)

    with torch.no_grad():
        output = model(X)

    assert not torch.allclose(output.sum(dim=1), torch.ones(3), atol=1e-5)


def test_pranay_net_convolution_stage_order():
    model = PranayNet()
    model.eval()
    events = []
    handles = []

    for name in ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4", "conv5", "bn5"]:
        module = getattr(model, name)
        handles.append(module.register_forward_hook(lambda _module, _inputs, _output, n=name: events.append(n)))

    for module in modules_of_type(model, nn.ReLU):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output: events.append("relu")
            )
        )

    for module in modules_of_type(model, nn.MaxPool2d):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output: events.append("pool")
            )
        )

    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 128, 128))
    finally:
        for handle in handles:
            handle.remove()

    assert events[:20] == [
        "conv1",
        "bn1",
        "relu",
        "pool",
        "conv2",
        "bn2",
        "relu",
        "pool",
        "conv3",
        "bn3",
        "relu",
        "pool",
        "conv4",
        "bn4",
        "relu",
        "pool",
        "conv5",
        "bn5",
        "relu",
        "pool",
    ]


def test_pranay_net_classifier_head_order():
    model = PranayNet()
    model.eval()
    events = []
    handles = []

    observed_types = [
        (nn.Flatten, "flatten"),
        (nn.ReLU, "relu"),
        (nn.Dropout, "dropout"),
    ]
    for module_type, event_name in observed_types:
        for module in modules_of_type(model, module_type):
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=event_name: events.append(name)
                )
            )

    linear_event_names = {
        (128 * 4 * 4, 128): "linear1",
        (128, 64): "linear2",
        (64, 2): "linear3",
    }
    for module in modules_of_type(model, nn.Linear):
        event_name = linear_event_names[(module.in_features, module.out_features)]
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=event_name: events.append(name)
            )
        )

    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 128, 128))
    finally:
        for handle in handles:
            handle.remove()

    classifier_events = events[events.index("flatten") :]

    assert classifier_events == [
        "flatten",
        "linear1",
        "relu",
        "dropout",
        "linear2",
        "relu",
        "dropout",
        "linear3",
    ]


def test_pranay_net_supports_training_mode_with_dropout():
    model = PranayNet()
    model.train()
    X = torch.randn(2, 3, 128, 128)

    output = model(X)

    assert output.shape == (2, 2)
