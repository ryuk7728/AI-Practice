import torch
from torch import nn

from helpers.q0020 import Head
from solutions.q0020 import MultiHeadAttention


def modules_of_type(model, module_type):
    return [module for module in model.modules() if isinstance(module, module_type)]


def find_output_projection(model, combined_size, embedding_size):
    matches = [
        module
        for module in modules_of_type(model, nn.Linear)
        if module.in_features == combined_size
        and module.out_features == embedding_size
    ]
    assert len(matches) == 1
    return matches[0]


def test_multi_head_attention_is_torch_module_with_independent_heads():
    model = MultiHeadAttention(
        embedding_size=8,
        num_heads=3,
        head_size=4,
        context_size=10,
        dropout=0.2,
    )
    heads = modules_of_type(model, Head)

    assert isinstance(model, nn.Module)
    assert len(heads) == 3
    assert len({id(head) for head in heads}) == 3


def test_multi_head_attention_has_expected_output_projection():
    model = MultiHeadAttention(
        embedding_size=8,
        num_heads=3,
        head_size=4,
        context_size=10,
        dropout=0.2,
    )
    projection = find_output_projection(model, combined_size=12, embedding_size=8)

    assert isinstance(projection, nn.Linear)


def test_multi_head_attention_returns_expected_shapes():
    model = MultiHeadAttention(
        embedding_size=9,
        num_heads=2,
        head_size=3,
        context_size=12,
        dropout=0.0,
    )
    model.eval()

    for batch_size, sequence_length in [(1, 1), (2, 6), (4, 12)]:
        output = model(torch.randn(batch_size, sequence_length, 9))

        assert output.shape == (batch_size, sequence_length, 9)


def test_multi_head_attention_runs_heads_in_parallel_and_combines_features():
    embedding_size = 7
    num_heads = 3
    head_size = 2
    model = MultiHeadAttention(
        embedding_size=embedding_size,
        num_heads=num_heads,
        head_size=head_size,
        context_size=8,
        dropout=0.0,
    )
    model.eval()
    heads = modules_of_type(model, Head)
    projection = find_output_projection(
        model,
        combined_size=num_heads * head_size,
        embedding_size=embedding_size,
    )
    observed = {"head_inputs": [], "head_outputs": []}
    handles = []

    for head in heads:
        handles.append(
            head.register_forward_pre_hook(
                lambda _module, inputs: observed["head_inputs"].append(inputs[0])
            )
        )
        handles.append(
            head.register_forward_hook(
                lambda _module, _inputs, output: observed["head_outputs"].append(output)
            )
        )

    handles.extend(
        [
            projection.register_forward_pre_hook(
                lambda _module, inputs: observed.update(projection_input=inputs[0])
            ),
            projection.register_forward_hook(
                lambda _module, _inputs, output: observed.update(
                    projection_output=output
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

    assert len(observed["head_inputs"]) == num_heads
    assert len(observed["head_outputs"]) == num_heads
    for head_input in observed["head_inputs"]:
        torch.testing.assert_close(head_input, x)
    torch.testing.assert_close(
        observed["projection_input"],
        torch.cat(observed["head_outputs"], dim=-1),
    )
    torch.testing.assert_close(output, observed["projection_output"])


def test_multi_head_attention_preserves_causal_behavior():
    model = MultiHeadAttention(
        embedding_size=8,
        num_heads=2,
        head_size=3,
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


def test_multi_head_attention_applies_output_dropout_during_training():
    model = MultiHeadAttention(
        embedding_size=8,
        num_heads=2,
        head_size=3,
        context_size=6,
        dropout=1.0,
    )
    x = torch.randn(2, 6, 8)

    model.train()
    training_output = model(x)
    model.eval()
    evaluation_output = model(x)

    torch.testing.assert_close(training_output, torch.zeros_like(training_output))
    assert torch.count_nonzero(evaluation_output) > 0


def test_multi_head_attention_propagates_gradients():
    model = MultiHeadAttention(
        embedding_size=8,
        num_heads=2,
        head_size=3,
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
