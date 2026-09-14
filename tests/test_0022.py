import torch
from torch import nn

from helpers.q0020 import Head
from helpers.q0022 import FeedForward, MultiHeadAttention
from solutions.q0022 import Block


def modules_of_type(model, module_type):
    return [module for module in model.modules() if isinstance(module, module_type)]


def test_block_is_torch_module_with_expected_components():
    model = Block(
        embedding_size=12,
        num_heads=3,
        context_size=10,
        dropout=0.2,
    )
    attention_modules = modules_of_type(model, MultiHeadAttention)
    feed_forward_modules = modules_of_type(model, FeedForward)
    normalizations = modules_of_type(model, nn.LayerNorm)
    heads = modules_of_type(model, Head)

    assert isinstance(model, nn.Module)
    assert len(attention_modules) == 1
    assert len(feed_forward_modules) == 1
    assert len(normalizations) == 2
    assert all(layer.normalized_shape == (12,) for layer in normalizations)
    assert len(heads) == 3
    for head in heads:
        projections = [
            module
            for module in head.modules()
            if isinstance(module, nn.Linear)
            and module.in_features == 12
            and module.out_features == 4
        ]
        assert len(projections) == 3


def test_block_returns_same_shape_for_supported_sequence_lengths():
    model = Block(
        embedding_size=8,
        num_heads=2,
        context_size=12,
        dropout=0.0,
    )
    model.eval()

    for shape in [(1, 1, 8), (2, 6, 8), (4, 12, 8)]:
        output = model(torch.randn(*shape))

        assert output.shape == shape


def test_block_uses_pre_norm_and_two_residual_connections():
    embedding_size = 8
    model = Block(
        embedding_size=embedding_size,
        num_heads=2,
        context_size=8,
        dropout=0.0,
    )
    model.eval()
    attention = modules_of_type(model, MultiHeadAttention)[0]
    feed_forward = modules_of_type(model, FeedForward)[0]
    normalizations = modules_of_type(model, nn.LayerNorm)
    observed = {
        "norm_inputs": [],
        "norm_outputs": [],
    }
    handles = []

    def record_normalization(_module, inputs, output):
        observed["norm_inputs"].append(inputs[0])
        observed["norm_outputs"].append(output)

    for normalization in normalizations:
        handles.append(
            normalization.register_forward_hook(record_normalization)
        )

    handles.extend(
        [
            attention.register_forward_pre_hook(
                lambda _module, inputs: observed.update(attention_input=inputs[0])
            ),
            attention.register_forward_hook(
                lambda _module, _inputs, output: observed.update(
                    attention_output=output
                )
            ),
            feed_forward.register_forward_pre_hook(
                lambda _module, inputs: observed.update(feed_forward_input=inputs[0])
            ),
            feed_forward.register_forward_hook(
                lambda _module, _inputs, output: observed.update(
                    feed_forward_output=output
                )
            ),
        ]
    )
    x = torch.randn(2, 5, embedding_size)

    try:
        output = model(x)
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed["norm_inputs"]) == 2
    assert len(observed["norm_outputs"]) == 2
    torch.testing.assert_close(observed["norm_inputs"][0], x)
    torch.testing.assert_close(
        observed["attention_input"], observed["norm_outputs"][0]
    )

    first_residual = x + observed["attention_output"]
    torch.testing.assert_close(observed["norm_inputs"][1], first_residual)
    torch.testing.assert_close(
        observed["feed_forward_input"], observed["norm_outputs"][1]
    )
    torch.testing.assert_close(
        output, first_residual + observed["feed_forward_output"]
    )


def test_block_preserves_causal_behavior():
    model = Block(
        embedding_size=8,
        num_heads=2,
        context_size=8,
        dropout=0.0,
    )
    model.eval()
    x = torch.randn(2, 8, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20

    original_prefix = model(x)[:, :4]
    changed_prefix = model(changed)[:, :4]

    torch.testing.assert_close(original_prefix, changed_prefix)


def test_block_processes_batch_examples_independently():
    model = Block(
        embedding_size=8,
        num_heads=2,
        context_size=6,
        dropout=0.0,
    )
    model.eval()
    first = torch.randn(1, 6, 8)
    second = torch.randn(1, 6, 8)

    separate = model(first)
    batched = model(torch.cat((first, second), dim=0))[:1]

    torch.testing.assert_close(separate, batched)


def test_block_passes_dropout_to_both_components():
    model = Block(
        embedding_size=8,
        num_heads=2,
        context_size=6,
        dropout=1.0,
    )
    x = torch.randn(2, 6, 8)

    model.train()
    training_output = model(x)
    model.eval()
    evaluation_output = model(x)

    torch.testing.assert_close(training_output, x)
    assert not torch.allclose(evaluation_output, x)


def test_block_propagates_gradients():
    model = Block(
        embedding_size=8,
        num_heads=2,
        context_size=6,
        dropout=0.0,
    )
    x = torch.randn(2, 6, 8, requires_grad=True)

    output = model(x)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
