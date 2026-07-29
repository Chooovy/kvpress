# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from kvpress import KVzipPress
from tests.fixtures import unit_test_model  # noqa: F401


def make_fake_model(num_layers: int):
    layers = []
    for layer_idx in range(num_layers):
        self_attn = SimpleNamespace(
            layer_idx=layer_idx,
            config=SimpleNamespace(_attn_implementation="sdpa"),
            masked_key_indices=None,
        )
        layers.append(SimpleNamespace(self_attn=self_attn))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def masked_coordinates(layer) -> set[tuple[int, int, int]]:
    masked_key_indices = layer.self_attn.masked_key_indices
    if masked_key_indices is None:
        return set()
    return set(zip(*(indices.tolist() for indices in masked_key_indices)))


def make_ranked_chunk_scores(
    num_layers: int = 2,
    num_kv_heads: int = 2,
    context_length: int = 10,
    chunk_size: int = 4,
) -> torch.Tensor:
    scores = torch.full((num_layers, 1, num_kv_heads, context_length), -100.0)
    rank = 0
    for layer_idx in range(num_layers):
        for head_idx in range(num_kv_heads):
            for chunk_idx in range(context_length // chunk_size):
                start = chunk_idx * chunk_size
                scores[layer_idx, 0, head_idx, start : start + chunk_size] = rank
                rank += 1
    return scores


def assert_common_runtime_stats_are_python_scalars(press: KVzipPress):
    assert isinstance(press.last_total_kv_slots, int)
    assert isinstance(press.last_masked_kv_slots, int)
    assert isinstance(press.last_actual_masked_slot_ratio, float)
    assert isinstance(press.last_masked_slots_per_layer, tuple)
    assert all(isinstance(value, int) for value in press.last_masked_slots_per_layer)


def test_token_selection_matches_original_global_bottom_k():
    scores = torch.tensor(
        [
            [[[0.0, 8.0, 9.0, 10.0, 11.0], [1.0, 12.0, 13.0, 14.0, 15.0]]],
            [[[2.0, 3.0, 16.0, 17.0, 18.0], [4.0, 5.0, 6.0, 19.0, 20.0]]],
        ]
    )
    press = KVzipPress(
        compression_ratio=0.3,
        layerwise=False,
        selection_granularity="token",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    press.compress_post(model)

    num_pruned = int(scores.numel() * press.compression_ratio)
    slots_per_layer = scores.shape[1] * scores.shape[2] * scores.shape[3]
    expected_by_layer = [set() for _ in model.model.layers]
    for flat_index in torch.topk(-scores.reshape(-1), num_pruned).indices.tolist():
        layer_idx, layer_offset = divmod(flat_index, slots_per_layer)
        head_idx, seq_idx = divmod(layer_offset, scores.shape[-1])
        expected_by_layer[layer_idx].add((0, head_idx, seq_idx))

    assert [masked_coordinates(layer) for layer in model.model.layers] == expected_by_layer
    assert press.last_total_kv_slots == scores.numel()
    assert press.last_masked_kv_slots == num_pruned
    assert press.last_actual_masked_slot_ratio == pytest.approx(0.3)
    assert press.last_masked_chunks is None
    assert press.last_masked_slots_per_layer == (2, 4)
    assert_common_runtime_stats_are_python_scalars(press)


def test_token_selection_matches_original_layerwise_bottom_k():
    scores = torch.tensor(
        [
            [[[0.0, 8.0, 9.0, 10.0, 11.0], [1.0, 12.0, 13.0, 14.0, 15.0]]],
            [[[2.0, 3.0, 16.0, 17.0, 18.0], [4.0, 5.0, 6.0, 19.0, 20.0]]],
        ]
    )
    press = KVzipPress(
        compression_ratio=0.3,
        layerwise=True,
        selection_granularity="token",
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    press.compress_post(model)

    num_pruned_per_layer = int(scores[0].numel() * press.compression_ratio)
    expected_by_layer = []
    for layer_scores in scores:
        coordinates = set()
        for flat_index in torch.topk(-layer_scores.reshape(-1), num_pruned_per_layer).indices.tolist():
            head_idx, seq_idx = divmod(flat_index, scores.shape[-1])
            coordinates.add((0, head_idx, seq_idx))
        expected_by_layer.append(coordinates)

    assert [masked_coordinates(layer) for layer in model.model.layers] == expected_by_layer
    assert press.last_masked_slots_per_layer == (3, 3)


def test_chunk_scores_are_fp32_sums_and_exclude_partial_tail():
    press = KVzipPress(selection_granularity="chunk", selection_chunk_size=4)
    token_scores = torch.tensor(
        [
            [
                [
                    [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, -100.0, -200.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25, -300.0, -400.0],
                ]
            ]
        ],
        dtype=torch.float16,
    )

    chunk_scores, complete_end = press._aggregate_chunk_scores(token_scores)
    expected = token_scores[..., :8].float().reshape(1, 1, 2, 2, 4).sum(dim=-1)

    assert complete_end == 8
    assert chunk_scores.dtype == torch.float32
    assert torch.equal(chunk_scores, expected)


def test_chunk_selection_is_per_head_and_never_masks_partial_tail():
    scores = torch.tensor(
        [
            [
                [
                    [0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, -100.0, -100.0],
                    [10.0, 10.0, 10.0, 10.0, 0.2, 0.2, 0.2, 0.2, -100.0, -100.0],
                ]
            ]
        ]
    )
    press = KVzipPress(
        compression_ratio=0.4,
        layerwise=False,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=1)

    press.compress_post(model)

    assert masked_coordinates(model.model.layers[0]) == {
        *((0, 0, seq_idx) for seq_idx in range(4)),
        *((0, 1, seq_idx) for seq_idx in range(4, 8)),
    }
    assert all(seq_idx < 8 for _, _, seq_idx in masked_coordinates(model.model.layers[0]))
    assert press.last_total_kv_slots == 20
    assert press.last_masked_kv_slots == 8
    assert press.last_actual_masked_slot_ratio == pytest.approx(0.4)
    assert press.last_masked_chunks == 2
    assert press.last_masked_slots_per_layer == (8,)


def test_global_chunk_budget_preserves_adaptive_layer_allocation_and_rounding_bound():
    scores = make_ranked_chunk_scores()
    press = KVzipPress(
        compression_ratio=0.5,
        layerwise=False,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    press.compress_post(model)

    requested_slots = int(scores.numel() * press.compression_ratio)
    assert press.last_masked_slots_per_layer == (16, 4)
    assert press.last_masked_kv_slots == 20
    assert press.last_masked_chunks == 5
    assert 0 <= requested_slots - press.last_masked_kv_slots < press.selection_chunk_size
    assert_common_runtime_stats_are_python_scalars(press)


def test_layerwise_chunk_budget_rounds_once_per_layer():
    scores = make_ranked_chunk_scores()
    press = KVzipPress(
        compression_ratio=0.5,
        layerwise=True,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    press.compress_post(model)

    requested_slots_per_layer = int(scores[0].numel() * press.compression_ratio)
    assert press.last_masked_slots_per_layer == (8, 8)
    assert press.last_masked_kv_slots == 16
    assert press.last_masked_chunks == 4
    for masked_slots in press.last_masked_slots_per_layer:
        assert 0 <= requested_slots_per_layer - masked_slots < press.selection_chunk_size
    assert (
        0
        <= requested_slots_per_layer * scores.shape[0] - press.last_masked_kv_slots
        < scores.shape[0] * press.selection_chunk_size
    )


def test_zero_chunk_budget_is_a_valid_noop_and_clears_stale_mask():
    scores = torch.ones(1, 1, 1, 3)
    press = KVzipPress(
        compression_ratio=0.5,
        layerwise=False,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=1)
    model.model.layers[0].self_attn.masked_key_indices = (
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
    )

    press.compress_post(model)

    assert model.model.layers[0].self_attn.masked_key_indices is None
    assert press.last_total_kv_slots == 3
    assert press.last_masked_kv_slots == 0
    assert press.last_actual_masked_slot_ratio == 0.0
    assert press.last_masked_chunks == 0
    assert press.last_masked_slots_per_layer == (0,)


def test_infeasible_chunk_budget_fails_when_requested_chunks_exceed_available_chunks():
    scores = torch.ones(2, 1, 2, 3)
    press = KVzipPress(
        compression_ratio=0.9,
        layerwise=False,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    with pytest.raises(
        ValueError,
        match=r"requested_chunks=2, available_chunks=0",
    ):
        press.compress_post(model)


def test_infeasible_layerwise_chunk_budget_fails_per_layer():
    scores = torch.ones(2, 1, 2, 3)
    press = KVzipPress(
        compression_ratio=0.9,
        layerwise=True,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)

    with pytest.raises(
        ValueError,
        match=r"requested_chunks_per_layer=1, available_chunks_per_layer=0",
    ):
        press.compress_post(model)


@pytest.mark.parametrize("invalid_chunk_size", (0, -1, True, 1.5))
def test_runtime_chunk_size_validation_cannot_be_bypassed_by_mutation(invalid_chunk_size):
    press = KVzipPress(
        compression_ratio=0.5,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.selection_chunk_size = invalid_chunk_size
    press.score_val = torch.ones(1, 1, 1, 8)

    with pytest.raises(ValueError, match="selection_chunk_size"):
        press.compress_post(make_fake_model(num_layers=1))


def test_zero_budget_layer_explicitly_clears_stale_mask():
    scores = torch.tensor(
        [
            [[[-10.0, -10.0, -10.0, -10.0]]],
            [[[10.0, 10.0, 10.0, 10.0]]],
        ]
    )
    press = KVzipPress(
        compression_ratio=0.5,
        layerwise=False,
        selection_granularity="chunk",
        selection_chunk_size=4,
    )
    press.score_val = scores
    model = make_fake_model(num_layers=2)
    stale_mask = (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = stale_mask

    press.compress_post(model)

    assert masked_coordinates(model.model.layers[0]) == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
    }
    assert model.model.layers[1].self_attn.masked_key_indices is None
    assert press.last_masked_slots_per_layer == (4, 0)


def test_internal_reset_preserves_last_runtime_stats_until_explicit_stats_reset():
    press = KVzipPress()
    press.last_total_kv_slots = 80
    press.last_masked_kv_slots = 32
    press.last_actual_masked_slot_ratio = 0.4
    press.last_masked_chunks = 8
    press.last_masked_slots_per_layer = (16, 16)
    press.score_val = torch.ones(2, 1, 1, 4)
    press.context_length = 4
    press._context_ids = torch.ones(1, 4, dtype=torch.long)

    press._reset_internal_parameters()

    assert press.score_val is None
    assert press.context_length == 0
    assert press._context_ids is None
    assert press.last_total_kv_slots == 80
    assert press.last_masked_kv_slots == 32
    assert press.last_actual_masked_slot_ratio == 0.4
    assert press.last_masked_chunks == 8
    assert press.last_masked_slots_per_layer == (16, 16)

    press.reset_runtime_stats()

    assert press.last_total_kv_slots is None
    assert press.last_masked_kv_slots is None
    assert press.last_actual_masked_slot_ratio is None
    assert press.last_masked_chunks is None
    assert press.last_masked_slots_per_layer is None


@pytest.mark.parametrize(
    "selection_granularity, expected_masked_chunks",
    (
        ("token", None),
        ("chunk", 4),
    ),
)
def test_real_context_manager_keeps_cache_length_and_fake_mask_decode_works(
    unit_test_model,  # noqa: F811
    selection_granularity,
    expected_masked_chunks,
):
    context_length = 128
    press = KVzipPress(
        compression_ratio=0.5,
        layerwise=False,
        selection_granularity=selection_granularity,
        selection_chunk_size=64,
    )
    cache = DynamicCache()
    input_ids = torch.randint(0, 1024, (1, context_length), device=unit_test_model.device)

    with press(unit_test_model):
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == context_length
    assert press.score_val is None
    assert press.last_total_kv_slots == (
        unit_test_model.config.num_hidden_layers * unit_test_model.config.num_key_value_heads * context_length
    )
    assert press.last_masked_kv_slots == press.last_total_kv_slots // 2
    assert press.last_actual_masked_slot_ratio == pytest.approx(0.5)
    assert press.last_masked_chunks == expected_masked_chunks

    next_token = torch.randint(0, 1024, (1, 1), device=unit_test_model.device)
    unit_test_model(
        next_token,
        past_key_values=cache,
        position_ids=torch.tensor([[context_length]], device=unit_test_model.device),
    )
    assert cache.get_seq_length() == context_length + 1
