import torch
from torch import nn
from torch.nn import functional as F

from helpers.q0020 import Head
from helpers.q0023 import Block
from solutions.q0023 import GPTLanguageModel


def modules_of_type(model, module_type):
    return [module for module in model.modules() if isinstance(module, module_type)]


def find_embedding(model, num_embeddings, embedding_size):
    matches = [
        module
        for module in modules_of_type(model, nn.Embedding)
        if module.num_embeddings == num_embeddings
        and module.embedding_dim == embedding_size
    ]
    assert len(matches) == 1
    return matches[0]


def find_output_layer(model, embedding_size, vocab_size):
    matches = [
        module
        for module in modules_of_type(model, nn.Linear)
        if module.in_features == embedding_size and module.out_features == vocab_size
    ]
    assert len(matches) == 1
    return matches[0]


def final_normalizations(model, blocks):
    block_module_ids = {
        id(module)
        for block in blocks
        for module in block.modules()
    }
    return [
        module
        for module in modules_of_type(model, nn.LayerNorm)
        if id(module) not in block_module_ids
    ]


def test_gpt_model_has_expected_components_and_head_sizes():
    model = GPTLanguageModel(
        vocab_size=41,
        embedding_size=12,
        context_size=16,
        num_heads=3,
        num_layers=4,
        dropout=0.2,
    )
    blocks = modules_of_type(model, Block)

    assert isinstance(model, nn.Module)
    find_embedding(model, num_embeddings=41, embedding_size=12)
    find_embedding(model, num_embeddings=16, embedding_size=12)
    find_output_layer(model, embedding_size=12, vocab_size=41)
    assert len(blocks) == 4
    assert len(final_normalizations(model, blocks)) == 1

    for block in blocks:
        heads = modules_of_type(block, Head)
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


def test_gpt_model_returns_sequence_scores_without_targets():
    model = GPTLanguageModel(
        vocab_size=31,
        embedding_size=8,
        context_size=10,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    token_ids = torch.randint(0, 31, (3, 7))

    scores, loss = model(token_ids)

    assert scores.shape == (3, 7, 31)
    assert loss is None


def test_gpt_model_combines_embeddings_and_uses_continuous_block_flow():
    vocab_size = 37
    embedding_size = 8
    context_size = 9
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        context_size=context_size,
        num_heads=2,
        num_layers=3,
        dropout=0.0,
    )
    model.eval()
    token_embedding = find_embedding(model, vocab_size, embedding_size)
    position_embedding = find_embedding(model, context_size, embedding_size)
    blocks = modules_of_type(model, Block)
    final_norm = final_normalizations(model, blocks)[0]
    output_layer = find_output_layer(model, embedding_size, vocab_size)
    observed = {
        "block_inputs": [],
        "block_outputs": [],
    }
    handles = [
        token_embedding.register_forward_hook(
            lambda _module, _inputs, output: observed.update(token_embedding=output)
        ),
        position_embedding.register_forward_hook(
            lambda _module, _inputs, output: observed.update(position_embedding=output)
        ),
        final_norm.register_forward_pre_hook(
            lambda _module, inputs: observed.update(final_norm_input=inputs[0])
        ),
        final_norm.register_forward_hook(
            lambda _module, _inputs, output: observed.update(final_norm_output=output)
        ),
        output_layer.register_forward_pre_hook(
            lambda _module, inputs: observed.update(output_input=inputs[0])
        ),
        output_layer.register_forward_hook(
            lambda _module, _inputs, output: observed.update(raw_scores=output)
        ),
    ]

    for block in blocks:
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

    token_ids = torch.randint(0, vocab_size, (2, 6))

    try:
        scores, _ = model(token_ids)
    finally:
        for handle in handles:
            handle.remove()

    combined = observed["token_embedding"] + observed["position_embedding"]
    torch.testing.assert_close(observed["block_inputs"][0], combined)
    for previous_output, next_input in zip(
        observed["block_outputs"], observed["block_inputs"][1:]
    ):
        torch.testing.assert_close(next_input, previous_output)
    torch.testing.assert_close(
        observed["final_norm_input"], observed["block_outputs"][-1]
    )
    torch.testing.assert_close(
        observed["output_input"], observed["final_norm_output"]
    )
    torch.testing.assert_close(scores, observed["raw_scores"])


def test_gpt_model_returns_flat_scores_and_cross_entropy_with_targets():
    vocab_size = 29
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=8,
        context_size=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    token_ids = torch.randint(0, vocab_size, (3, 6))
    targets = torch.randint(0, vocab_size, (3, 6))

    scores, loss = model(token_ids, targets)
    expected_loss = F.cross_entropy(scores, targets.reshape(-1))

    assert scores.shape == (18, vocab_size)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected_loss)


def test_gpt_model_preserves_causal_behavior():
    model = GPTLanguageModel(
        vocab_size=43,
        embedding_size=8,
        context_size=8,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    token_ids = torch.randint(0, 43, (2, 8))
    changed = token_ids.clone()
    changed[:, 4:] = torch.randint(0, 43, (2, 4))

    original_prefix = model(token_ids)[0][:, :4]
    changed_prefix = model(changed)[0][:, :4]

    torch.testing.assert_close(original_prefix, changed_prefix)


def test_gpt_model_processes_batch_examples_independently():
    model = GPTLanguageModel(
        vocab_size=35,
        embedding_size=8,
        context_size=7,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    first = torch.randint(0, 35, (1, 7))
    second = torch.randint(0, 35, (1, 7))

    separate = model(first)[0]
    batched = model(torch.cat((first, second), dim=0))[0][:1]

    torch.testing.assert_close(separate, batched)


def test_gpt_model_passes_dropout_to_transformer_blocks():
    model = GPTLanguageModel(
        vocab_size=33,
        embedding_size=8,
        context_size=6,
        num_heads=2,
        num_layers=2,
        dropout=1.0,
    )
    model.train()
    blocks = modules_of_type(model, Block)
    observed = {"inputs": [], "outputs": []}
    handles = []

    for block in blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, inputs: observed["inputs"].append(inputs[0])
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, output: observed["outputs"].append(output)
            )
        )

    try:
        model(torch.randint(0, 33, (2, 6)))
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed["inputs"]) == 2
    assert len(observed["outputs"]) == 2
    for block_input, block_output in zip(observed["inputs"], observed["outputs"]):
        torch.testing.assert_close(block_output, block_input)


def test_gpt_model_propagates_gradients_from_language_model_loss():
    model = GPTLanguageModel(
        vocab_size=27,
        embedding_size=8,
        context_size=6,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    token_ids = torch.randint(0, 27, (2, 6))
    targets = torch.randint(0, 27, (2, 6))

    _, loss = model(token_ids, targets)
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
