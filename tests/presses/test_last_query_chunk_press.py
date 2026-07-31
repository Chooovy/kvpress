# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

import kvpress.presses.last_query_chunk_press as last_query_chunk_press_module
from kvpress import BSAPress, MeanPoolingPress


class DummyAttention(nn.Module):
    def __init__(self, num_query_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_query_heads,
            num_key_value_heads=num_kv_heads,
        )
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        hidden_size = num_query_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(hidden_size))


def make_synthetic_inputs(
    seq_len: int,
    batch_size: int = 1,
    num_query_heads: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 2,
):
    module = DummyAttention(num_query_heads, num_kv_heads, head_dim)
    hidden_states = torch.ones(batch_size, seq_len, num_query_heads * head_dim)
    keys = torch.zeros(batch_size, num_kv_heads, seq_len, head_dim)
    token_ids = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
    values = token_ids.expand(batch_size, num_kv_heads, seq_len, head_dim).clone()
    position_embeddings = (
        torch.ones(batch_size, seq_len, head_dim),
        torch.zeros(batch_size, seq_len, head_dim),
    )
    kwargs = {"position_embeddings": position_embeddings}
    return module, hidden_states, keys, values, kwargs


@pytest.mark.parametrize(
    "use_prerope_query, use_prerope_keys",
    (
        (False, False),
        (True, True),
        (False, True),
        (True, False),
    ),
)
def test_mean_pooling_routes_expected_scoring_query_and_keys(
    monkeypatch,
    use_prerope_query,
    use_prerope_keys,
):
    press = MeanPoolingPress(
        chunk_size=2,
        protected_window_size=0,
        use_prerope_query=use_prerope_query,
        use_prerope_keys=use_prerope_keys,
    )
    module = DummyAttention(num_query_heads=1, num_kv_heads=1, head_dim=2)
    hidden_states = torch.zeros(1, 4, 2)
    cached_postrope_keys = torch.full((1, 1, 4, 2), 11.0)
    prerope_query = torch.tensor([[[[1.0, 2.0]]]])
    prerope_keys = torch.full_like(cached_postrope_keys, 22.0)
    calls = {"query": 0, "keys": 0}

    def fake_get_prerope_query_states(attention_module, selected_hidden_states):
        assert attention_module is module
        assert torch.equal(selected_hidden_states, hidden_states[:, -1:])
        calls["query"] += 1
        return prerope_query

    def fake_get_prerope_key_states(attention_module, selected_hidden_states):
        assert attention_module is module
        assert torch.equal(selected_hidden_states, hidden_states)
        calls["keys"] += 1
        return prerope_keys

    monkeypatch.setattr(
        last_query_chunk_press_module,
        "get_prerope_query_states",
        fake_get_prerope_query_states,
    )
    monkeypatch.setattr(
        last_query_chunk_press_module,
        "get_prerope_key_states",
        fake_get_prerope_key_states,
    )
    position_embeddings = (
        torch.zeros(1, 4, 2),
        torch.ones(1, 4, 2),
    )

    scoring_query, scoring_keys = press._get_scoring_query_and_keys(
        module,
        hidden_states,
        cached_postrope_keys,
        position_embeddings,
    )

    expected_query = prerope_query.squeeze(2) if use_prerope_query else torch.tensor([[[-2.0, 1.0]]])
    assert torch.equal(scoring_query, expected_query)
    assert scoring_keys is (prerope_keys if use_prerope_keys else cached_postrope_keys)
    assert calls == {"query": 1, "keys": int(use_prerope_keys)}


def test_default_mean_pooling_does_not_project_prerope_keys(monkeypatch):
    press = MeanPoolingPress(compression_ratio=0.5, chunk_size=4, protected_window_size=4)
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(seq_len=16)

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        pytest.fail("Default post/post scoring must not project pre-RoPE keys")

    monkeypatch.setattr(last_query_chunk_press_module, "get_prerope_key_states", fail_if_called)

    compressed_keys, compressed_values = press.compress(
        module,
        hidden_states,
        keys,
        values,
        None,
        kwargs,
    )

    assert compressed_keys.shape[2] == 8
    assert compressed_values.shape[2] == 8


def test_prerope_keys_score_remote_and_local_but_gather_original_cache(monkeypatch):
    press = MeanPoolingPress(
        compression_ratio=0.5,
        chunk_size=4,
        protected_window_size=4,
        use_prerope_keys=True,
    )
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(seq_len=16)
    cached_postrope_keys = torch.arange(keys.numel(), dtype=keys.dtype).reshape_as(keys) + 100.0
    keys.copy_(cached_postrope_keys)
    prerope_keys = torch.zeros_like(keys)
    prerope_keys[:, :, 0:4] = 4.0
    prerope_keys[:, :, 4:8] = 2.0
    prerope_keys[:, :, 8:12] = 1.0
    prerope_keys[:, :, 12:16] = 9.0
    captured = {}

    monkeypatch.setattr(
        last_query_chunk_press_module,
        "get_prerope_key_states",
        lambda attention_module, selected_hidden_states: prerope_keys,
    )

    original_compute_remote = press._compute_remote_log_mass_proxy

    def capture_remote(query_states, remote_key_chunks, scale):
        captured["remote_keys"] = remote_key_chunks.detach().clone()
        return original_compute_remote(query_states, remote_key_chunks, scale)

    monkeypatch.setattr(press, "_compute_remote_log_mass_proxy", capture_remote)
    original_einsum = torch.einsum

    def capture_local(equation, *operands):
        if equation == "bhgd,bhld->bhgl":
            captured["local_keys"] = operands[1].detach().clone()
        return original_einsum(equation, *operands)

    monkeypatch.setattr(last_query_chunk_press_module.torch, "einsum", capture_local)

    compressed_keys, compressed_values = press.compress(
        module,
        hidden_states,
        keys,
        values,
        None,
        kwargs,
    )

    expected_remote = prerope_keys[:, :, :12].reshape(1, 1, 3, 4, 2)
    expected_indices = [0, 1, 2, 3, 12, 13, 14, 15]
    assert torch.equal(captured["remote_keys"], expected_remote)
    assert torch.equal(captured["local_keys"], prerope_keys[:, :, 12:])
    assert torch.equal(compressed_keys, cached_postrope_keys[:, :, expected_indices])
    assert torch.equal(compressed_values, values[:, :, expected_indices])
    assert not torch.equal(compressed_keys, prerope_keys[:, :, expected_indices])


def test_bsa_log_mass_is_fp32_and_scale_is_inside_logsumexp():
    press = BSAPress(chunk_size=2, protected_window_size=1)
    query_states = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    remote_chunks = torch.tensor([[[[[2.0], [2.0]], [[3.0], [0.0]]]]], dtype=torch.float64)
    scale = 0.1

    actual = press._compute_remote_log_mass_proxy(query_states, remote_chunks, scale)
    raw_logits = torch.einsum("bhgd,bhnsd->bhgns", query_states.float(), remote_chunks.float())
    expected = torch.logsumexp(raw_logits * scale, dim=-1)
    incorrectly_scaled_after_reduction = torch.logsumexp(raw_logits, dim=-1) * scale
    full_attention_chunk_mass = (
        torch.softmax((raw_logits * scale).flatten(-2), dim=-1).reshape_as(raw_logits).sum(dim=-1)
    )

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)
    assert actual.argmax(dim=-1).item() == 0
    assert torch.equal(actual.argmax(dim=-1), full_attention_chunk_mass.argmax(dim=-1))
    assert incorrectly_scaled_after_reduction.argmax(dim=-1).item() == 1


def test_mean_pooling_proxy_is_q_mean_k_plus_log_chunk_size():
    press = MeanPoolingPress(chunk_size=4, protected_window_size=1)
    query_states = torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float64)
    remote_chunks = torch.arange(16, dtype=torch.float64).reshape(1, 1, 2, 4, 2)
    scale = 0.25

    actual = press._compute_remote_log_mass_proxy(query_states, remote_chunks, scale)
    mean_keys = remote_chunks.float().mean(dim=-2)
    mean_logits = torch.einsum("bhgd,bhnd->bhgn", query_states.float(), mean_keys) * scale
    token_logits = torch.einsum("bhgd,bhnsd->bhgns", query_states.float(), remote_chunks.float()) * scale

    assert actual.dtype == torch.float32
    assert torch.allclose(mean_logits, token_logits.mean(dim=-1))
    assert torch.allclose(actual, mean_logits + math.log(press.chunk_size))


def test_local_inclusive_normalized_max_is_batched_and_gqa_grouped():
    normal = torch.tensor([[10.0, 0.0], [0.0, 1.0]])
    flipped = normal.flip(-1)
    remote_log_mass_proxy = torch.stack(
        (
            torch.stack((normal, flipped)),
            torch.stack((flipped, normal)),
        )
    )
    local_log_mass = torch.tensor([20.0, -10.0]).view(1, 1, 2).expand(2, 2, 2)

    actual = BSAPress._normalize_and_reduce_gqa(remote_log_mass_proxy, local_log_mass)
    remote_only = torch.log_softmax(remote_log_mass_proxy, dim=-1).max(dim=2).values

    assert torch.equal(actual.argmax(dim=-1), torch.tensor([[1, 0], [0, 1]]))
    assert torch.equal(remote_only.argmax(dim=-1), torch.tensor([[0, 1], [1, 0]]))


def test_mean_pooling_log_chunk_size_changes_cross_head_winner():
    mean_logits = torch.tensor([[[[-5.0, -2.0], [-2.0, -3.0]]]])
    local_log_mass = torch.tensor([[[-2.0, -3.0]]])

    with_log_chunk_size = MeanPoolingPress._normalize_and_reduce_gqa(
        mean_logits + math.log(4),
        local_log_mass,
    )
    without_log_chunk_size = MeanPoolingPress._normalize_and_reduce_gqa(
        mean_logits,
        local_log_mass,
    )

    assert with_log_chunk_size.argmax(dim=-1).item() == 1
    assert without_log_chunk_size.argmax(dim=-1).item() == 0


def test_padded_attention_mask_fails_closed_when_no_remote_chunk_is_kept():
    press = BSAPress(compression_ratio=0.75, chunk_size=4, protected_window_size=4)
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(seq_len=16)
    kwargs["attention_mask"] = torch.tensor([[0] + [1] * 15])

    with pytest.raises(ValueError, match="unpadded, non-packed"):
        press.compress(module, hidden_states, keys, values, None, kwargs)


@pytest.mark.parametrize(
    "seq_len, compression_ratio, expected_kept, expected_protected, expected_remote_chunks",
    (
        (16, 0.0, 16, 4, 3),
        (16, 0.5, 8, 4, 1),
        (16, 0.75, 4, 4, 0),
        (17, 0.5, 5, 5, 0),
    ),
)
def test_budget_rounding_and_runtime_stats(
    seq_len,
    compression_ratio,
    expected_kept,
    expected_protected,
    expected_remote_chunks,
):
    press = BSAPress(
        compression_ratio=compression_ratio,
        chunk_size=4,
        protected_window_size=4,
    )
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(seq_len)

    compressed_keys, compressed_values = press.compress(
        module,
        hidden_states,
        keys,
        values,
        None,
        kwargs,
    )

    assert compressed_keys.shape[2] == expected_kept
    assert compressed_values.shape[2] == expected_kept
    assert press.last_input_tokens == seq_len
    assert press.last_kept_tokens == expected_kept
    assert press.last_protected_tokens == expected_protected
    assert press.last_kept_remote_chunks == expected_remote_chunks
    assert isinstance(press.last_input_tokens, int)
    assert isinstance(press.last_kept_tokens, int)
    assert isinstance(press.last_actual_compression_ratio, float)
    assert press.last_actual_compression_ratio == pytest.approx(1.0 - expected_kept / seq_len)

    requested_kept_budget = int(seq_len * (1.0 - compression_ratio))
    if compression_ratio > 0:
        assert expected_kept <= requested_kept_budget
        assert 0 <= press.last_actual_compression_ratio - compression_ratio < press.chunk_size / seq_len


@pytest.mark.parametrize(
    "seq_len, chunk_size, protected_window_size, compression_ratio, expected_message",
    (
        (3, 4, 4, 0.5, "No complete remote chunk"),
        (16, 4, 8, 0.75, "Requested compression is infeasible"),
    ),
)
def test_infeasible_budget_fails_closed(
    seq_len,
    chunk_size,
    protected_window_size,
    compression_ratio,
    expected_message,
):
    press = BSAPress(
        compression_ratio=compression_ratio,
        chunk_size=chunk_size,
        protected_window_size=protected_window_size,
    )
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(seq_len)

    with pytest.raises(ValueError, match=expected_message) as exc_info:
        press.compress(module, hidden_states, keys, values, None, kwargs)

    message = str(exc_info.value)
    assert f"input_tokens={seq_len}" in message
    assert f"chunk_size={chunk_size}" in message
    assert f"protected_window_size={protected_window_size}" in message
    assert "actual_protected_tokens=" in message
    assert f"requested_compression_ratio={compression_ratio:.6f}" in message
    assert "maximum_feasible_compression_ratio=" in message
    assert press.last_input_tokens is None


def test_whole_chunk_gather_is_batched_headwise_and_chronological():
    batch_size, num_heads, seq_len, head_dim = 2, 2, 21, 2
    press = MeanPoolingPress(
        compression_ratio=0.38,
        chunk_size=4,
        protected_window_size=4,
    )
    module, hidden_states, keys, values, kwargs = make_synthetic_inputs(
        seq_len,
        batch_size=batch_size,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
    )
    chunk_logits = torch.tensor(
        [
            [[0.0, 3.0, 1.0, 4.0], [4.0, 0.0, 3.0, 1.0]],
            [[0.0, 4.0, 3.0, 1.0], [3.0, 4.0, 0.0, 1.0]],
        ]
    )
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for chunk_idx in range(4):
                start = chunk_idx * press.chunk_size
                end = start + press.chunk_size
                keys[batch_idx, head_idx, start:end] = chunk_logits[batch_idx, head_idx, chunk_idx]

    compressed_keys, compressed_values = press.compress(
        module,
        hidden_states,
        keys,
        values,
        None,
        kwargs,
    )

    expected_chunks = (
        ((1, 3), (0, 2)),
        ((1, 2), (0, 1)),
    )
    protected_indices = list(range(16, 21))
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            remote_indices = [
                token_idx
                for chunk_idx in expected_chunks[batch_idx][head_idx]
                for token_idx in range(chunk_idx * 4, chunk_idx * 4 + 4)
            ]
            expected_indices = remote_indices + protected_indices
            actual_indices = compressed_values[batch_idx, head_idx, :, 0].to(torch.long).tolist()
            assert actual_indices == expected_indices
            assert torch.equal(
                compressed_keys[batch_idx, head_idx],
                keys[batch_idx, head_idx, expected_indices],
            )

    assert press.last_kept_tokens == 13
    assert press.last_protected_tokens == 5
    assert press.last_kept_remote_chunks == 2
    assert "last_input_tokens" not in repr(press)
    assert "last_input_tokens" not in inspect.signature(MeanPoolingPress).parameters
    assert not any(
        isinstance(value, torch.Tensor)
        for value in (
            press.last_input_tokens,
            press.last_kept_tokens,
            press.last_actual_compression_ratio,
            press.last_protected_tokens,
            press.last_kept_remote_chunks,
        )
    )


def make_tiny_gqa_model(model_type: str, attn_implementation: str | None = None):
    config_kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
    }
    if model_type == "llama":
        config = LlamaConfig(**config_kwargs)
        model_cls = LlamaForCausalLM
    elif model_type == "qwen3":
        config = Qwen3Config(**config_kwargs, head_dim=8)
        model_cls = Qwen3ForCausalLM
    else:
        raise ValueError(f"Unsupported tiny model type: {model_type}")
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    return model_cls(config).eval()


@pytest.mark.parametrize("press_cls", (BSAPress, MeanPoolingPress))
@pytest.mark.parametrize("model_type", ("llama", "qwen3"))
def test_tiny_gqa_model_end_to_end_and_runtime_stats(press_cls, model_type):
    torch.manual_seed(0)
    model = make_tiny_gqa_model(model_type)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 24))

    baseline_cache = DynamicCache()
    with torch.no_grad():
        model(input_ids, past_key_values=baseline_cache, use_cache=True)

    press = press_cls(compression_ratio=0.5, chunk_size=4, protected_window_size=4)
    compressed_cache = DynamicCache()
    with torch.no_grad(), press(model):
        model(input_ids, past_key_values=compressed_cache, use_cache=True)

    compressed_prefixes = [(layer.keys.clone(), layer.values.clone()) for layer in compressed_cache.layers]
    for baseline_layer, compressed_layer in zip(baseline_cache.layers, compressed_cache.layers):
        assert compressed_layer.keys.shape[2] == 12
        assert compressed_layer.keys.numel() == baseline_layer.keys.numel() // 2
        assert torch.allclose(compressed_layer.keys[:, :, -4:], baseline_layer.keys[:, :, -4:])
        assert torch.allclose(compressed_layer.values[:, :, -4:], baseline_layer.values[:, :, -4:])

        for batch_idx in range(input_ids.shape[0]):
            for head_idx in range(model.config.num_key_value_heads):
                selected_chunks = []
                for compressed_chunk_idx in range(2):
                    start = compressed_chunk_idx * 4
                    compressed_key_chunk = compressed_layer.keys[batch_idx, head_idx, start : start + 4]
                    matches = [
                        original_chunk_idx
                        for original_chunk_idx in range(5)
                        if torch.allclose(
                            compressed_key_chunk,
                            baseline_layer.keys[
                                batch_idx,
                                head_idx,
                                original_chunk_idx * 4 : original_chunk_idx * 4 + 4,
                            ],
                        )
                    ]
                    assert len(matches) == 1
                    selected_chunks.append(matches[0])
                    original_start = matches[0] * 4
                    assert torch.allclose(
                        compressed_layer.values[batch_idx, head_idx, start : start + 4],
                        baseline_layer.values[batch_idx, head_idx, original_start : original_start + 4],
                    )
                assert selected_chunks == sorted(selected_chunks)

    assert press.last_input_tokens == 24
    assert press.last_kept_tokens == 12
    assert press.last_actual_compression_ratio == pytest.approx(0.5)
    assert press.last_protected_tokens == 4
    assert press.last_kept_remote_chunks == 2

    question_ids = torch.randint(0, model.config.vocab_size, (2, 3))
    position_ids = torch.arange(24, 27).view(1, -1).expand(2, -1)
    with torch.no_grad():
        continued_output = model(
            question_ids,
            past_key_values=compressed_cache,
            position_ids=position_ids,
            use_cache=True,
        )

    assert torch.isfinite(continued_output.logits).all()
    for (prefix_keys, prefix_values), continued_layer in zip(compressed_prefixes, compressed_cache.layers):
        assert continued_layer.keys.shape[2] == 15
        assert torch.equal(continued_layer.keys[:, :, :12], prefix_keys)
        assert torch.equal(continued_layer.values[:, :, :12], prefix_values)


@pytest.mark.parametrize(
    "use_prerope",
    (False, True),
    ids=("post_q_post_k", "pre_q_pre_k"),
)
def test_mean_pooling_supports_zero_protected_window(use_prerope):
    torch.manual_seed(0)
    model = make_tiny_gqa_model("qwen3")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 24))
    press = MeanPoolingPress(
        compression_ratio=0.5,
        chunk_size=4,
        protected_window_size=0,
        use_prerope_query=use_prerope,
        use_prerope_keys=use_prerope,
    )
    cache = DynamicCache()

    with torch.no_grad(), press(model):
        model(input_ids, past_key_values=cache, use_cache=True)

    assert all(layer.keys.shape[2] == 12 for layer in cache.layers)
    assert all(layer.values.shape[2] == 12 for layer in cache.layers)
    assert press.last_input_tokens == 24
    assert press.last_kept_tokens == 12
    assert press.last_protected_tokens == 0
    assert press.last_kept_remote_chunks == 3
    assert press.last_actual_compression_ratio == pytest.approx(0.5)


@pytest.mark.parametrize("model_type", ("llama", "qwen3"))
def test_postrope_last_query_reproduces_eager_attention_last_row(model_type):
    torch.manual_seed(0)
    model = make_tiny_gqa_model(model_type, attn_implementation="eager")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    attention_module = model.model.layers[0].self_attn
    captured = {}

    def capture_attention_inputs(module, args, kwargs):
        del module, args
        captured["hidden_states"] = kwargs["hidden_states"].detach()
        captured["position_embeddings"] = tuple(tensor.detach() for tensor in kwargs["position_embeddings"])

    hook = attention_module.register_forward_pre_hook(capture_attention_inputs, with_kwargs=True)
    cache = DynamicCache()
    try:
        with torch.no_grad():
            output = model(
                input_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )
    finally:
        hook.remove()

    press = BSAPress(chunk_size=4, protected_window_size=4)
    query_states = press._get_postrope_last_query(
        attention_module,
        captured["hidden_states"],
        captured["position_embeddings"],
    )
    keys = cache.layers[0].keys
    group_size = model.config.num_attention_heads // model.config.num_key_value_heads
    repeated_keys = keys.repeat_interleave(group_size, dim=1)
    logits = (
        torch.matmul(query_states.float().unsqueeze(2), repeated_keys.float().transpose(2, 3)).squeeze(2)
        * attention_module.scaling
    )
    expected_attention = torch.softmax(logits, dim=-1)
    observed_attention = output.attentions[0][:, :, -1, :].float()

    assert torch.allclose(expected_attention, observed_attention, atol=1e-5, rtol=1e-5)
