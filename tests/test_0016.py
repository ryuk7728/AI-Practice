import torch
from torch import nn

from solutions.q0016 import ResidualBlock, ResNet


def modules_of_type(model, module_type):
    return [module for module in model.modules() if isinstance(module, module_type)]


def find_linear(model, in_features, out_features):
    matches = [
        layer
        for layer in modules_of_type(model, nn.Linear)
        if layer.in_features == in_features and layer.out_features == out_features
    ]
    assert len(matches) == 1
    return matches[0]


def find_stem_convolution(model):
    matches = [
        layer
        for layer in modules_of_type(model, nn.Conv2d)
        if layer.in_channels == 1 and layer.out_channels == 4
    ]
    assert len(matches) == 1
    return matches[0]


def test_residual_block_is_torch_module():
    block = ResidualBlock()

    assert isinstance(block, nn.Module)


def test_residual_block_has_expected_layers():
    block = ResidualBlock()

    assert isinstance(block.conv1, nn.Conv2d)
    assert isinstance(block.conv2, nn.Conv2d)
    assert block.conv1.in_channels == 4
    assert block.conv1.out_channels == 4
    assert block.conv2.in_channels == 4
    assert block.conv2.out_channels == 4


def test_residual_block_preserves_shape():
    block = ResidualBlock()
    X = torch.randn(3, 4, 4, 4)

    output = block(X)

    assert output.shape == X.shape


def test_residual_block_uses_skip_connection_before_final_relu():
    block = ResidualBlock()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()

    X = torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]]] * 4])
    output = block(X)

    torch.testing.assert_close(output, torch.relu(X))


def test_residual_block_applies_activations_in_expected_flow():
    block = ResidualBlock()
    observed = {}
    handles = [
        block.conv1.register_forward_hook(
            lambda _module, _inputs, output: observed.update(conv1_output=output)
        ),
        block.conv2.register_forward_pre_hook(
            lambda _module, inputs: observed.update(conv2_input=inputs[0])
        ),
        block.conv2.register_forward_hook(
            lambda _module, _inputs, output: observed.update(conv2_output=output)
        ),
    ]
    X = torch.randn(2, 4, 4, 4)

    try:
        output = block(X)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(observed["conv2_input"], torch.relu(observed["conv1_output"]))
    torch.testing.assert_close(output, torch.relu(observed["conv2_output"] + X))


def test_resnet_is_torch_module():
    model = ResNet(n_layers=3)

    assert isinstance(model, nn.Module)


def test_resnet_has_expected_layer_stack():
    model = ResNet(n_layers=4)
    stem = find_stem_convolution(model)
    residual_blocks = modules_of_type(model, ResidualBlock)

    assert stem.in_channels == 1
    assert stem.out_channels == 4
    assert len(residual_blocks) == 3


def test_resnet_has_expected_classifier_layers():
    model = ResNet(n_layers=3)
    linear_layers = modules_of_type(model, nn.Linear)

    assert len(linear_layers) == 2
    find_linear(model, 4 * 4 * 4, 10)
    find_linear(model, 10, 2)


def test_resnet_forward_returns_expected_shape():
    model = ResNet(n_layers=3)
    X = torch.randn(5, 1, 4, 4)

    output = model(X)

    assert output.shape == (5, 2)


def test_resnet_returns_raw_logits_not_probabilities():
    model = ResNet(n_layers=3)
    X = torch.randn(4, 1, 4, 4)

    output = model(X)

    assert not torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_resnet_convolution_stack_uses_single_flow():
    model = ResNet(n_layers=4)
    stem = find_stem_convolution(model)
    residual_blocks = modules_of_type(model, ResidualBlock)
    observed = {"block_inputs": [], "block_outputs": []}
    handles = [
        stem.register_forward_hook(
            lambda _module, _inputs, output: observed.update(stem_output=output)
        )
    ]

    for block in residual_blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, inputs: observed["block_inputs"].append(inputs[0])
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, output: observed["block_outputs"].append(output)
            )
        )

    try:
        model(torch.randn(1, 1, 4, 4))
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed["block_inputs"]) == 3
    assert len(observed["block_outputs"]) == 3
    torch.testing.assert_close(
        observed["block_inputs"][0], torch.relu(observed["stem_output"])
    )
    for previous_output, next_input in zip(
        observed["block_outputs"], observed["block_inputs"][1:]
    ):
        torch.testing.assert_close(next_input, previous_output)


def test_resnet_classifier_uses_flatten_hidden_relu_and_raw_logits():
    model = ResNet(n_layers=3)
    hidden = find_linear(model, 4 * 4 * 4, 10)
    output_layer = find_linear(model, 10, 2)
    observed = {}
    handles = [
        hidden.register_forward_pre_hook(
            lambda _module, inputs: observed.update(hidden_input=inputs[0])
        ),
        hidden.register_forward_hook(
            lambda _module, _inputs, output: observed.update(hidden_output=output)
        ),
        output_layer.register_forward_pre_hook(
            lambda _module, inputs: observed.update(output_input=inputs[0])
        ),
        output_layer.register_forward_hook(
            lambda _module, _inputs, output: observed.update(raw_logits=output)
        ),
    ]

    try:
        output = model(torch.randn(2, 1, 4, 4))
    finally:
        for handle in handles:
            handle.remove()

    assert observed["hidden_input"].shape == (2, 4 * 4 * 4)
    torch.testing.assert_close(
        observed["output_input"], torch.relu(observed["hidden_output"])
    )
    torch.testing.assert_close(output, observed["raw_logits"])


def test_resnet_respects_requested_number_of_layers():
    for n_layers in [1, 2, 5]:
        model = ResNet(n_layers=n_layers)

        assert len(modules_of_type(model, ResidualBlock)) == n_layers - 1
