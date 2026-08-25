# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import inspect

import pytest
import torch
from torch import nn
from transformers import DynamicCache

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    ChunkPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DMSPress,
    FacilityLocationPress,
    FastKVzipPress,
    KeyRerotationPress,
    KnormPress,
    KVComposePress,
    KVzipPress,
    MergingPress,
    ObservedAttentionPress,
    ScorerPress,
    SnapKVPress,
    ThinKPress,
)
from tests.default_presses import default_presses
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def init_press_from_model(press, model):
    """
    Call ``post_init_from_model`` for presses that need it, on a model shared across kwargs.

    ``GQAIndexerPress`` attaches an indexer module to each attention layer and refuses to
    reuse one whose geometry differs from what it was configured for. Since this loop drives
    many kwargs sets through a single session-scoped model, later variants (e.g. ``rope_dim=0,
    head_dim=32``) would otherwise inherit the first variant's indexer and never exercise the
    geometry they name, so ask for a fresh one.
    """
    if not hasattr(press, "post_init_from_model"):
        return
    if "force_reinit" in inspect.signature(press.post_init_from_model).parameters:
        press.post_init_from_model(model, force_reinit=True)
    else:
        press.post_init_from_model(model)


def test_composed_press(unit_test_model):  # noqa: F811
    press1 = KnormPress(compression_ratio=0.5)
    press2 = ThinKPress(key_channel_compression_ratio=0.5, window_size=2)
    composed_press = ComposedPress([press1, press2])
    with composed_press(unit_test_model):
        input_ids = unit_test_model.dummy_inputs["input_ids"].to(unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_chunk_press(unit_test_model):  # noqa: F811
    press = KnormPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


def test_chunk_press_slices_position_embeddings_per_chunk(unit_test_model):  # noqa: F811
    """
    Each chunk must be scored with its own RoPE tables, not the whole sequence's.

    ChunkPress slices hidden_states/keys/values per chunk but used to forward kwargs
    untouched, so a press that rotates queries saw positions the chunk does not contain.
    That is silent for SnapKV -- ``cos[:, -window_size:]`` is the end of the sequence rather
    than of the chunk, so it picks different keys with no error -- and fatal for a press that
    rotates the full chunk, which is how the mismatch was noticed.
    """
    seen = []
    # Read from __dict__, not the class, so the saved value is the staticmethod descriptor
    # itself -- restoring a plain function would leave later tests calling it as a bound
    # method, silently shifting every argument by one.
    original = SnapKVPress.__dict__["compute_window_attention"]
    unwrapped = SnapKVPress.compute_window_attention

    def spy(module, hidden_states, keys, window_size, position_embeddings):
        seen.append((hidden_states.shape[1], position_embeddings[0].shape[-2]))
        return unwrapped(module, hidden_states, keys, window_size, position_embeddings)

    SnapKVPress.compute_window_attention = staticmethod(spy)
    try:
        press = ChunkPress(press=SnapKVPress(compression_ratio=0.5, window_size=4), chunk_length=24)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 72), device=unit_test_model.device)
            unit_test_model(input_ids, past_key_values=DynamicCache())
    finally:
        SnapKVPress.compute_window_attention = original

    assert seen
    for hidden_len, rope_len in seen:
        assert rope_len == hidden_len, f"rope table spans {rope_len} positions for {hidden_len} tokens"


def test_chunkkv_press(unit_test_model):  # noqa: F811
    press = SnapKVPress(compression_ratio=0.5)
    for chunk_length in [2, 4, 8, 128]:
        composed_press = ChunkKVPress(press=press, chunk_length=chunk_length)
        with composed_press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 256), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache).past_key_values
            assert cache.get_seq_length() == 128


@pytest.mark.parametrize("press_dict", default_presses)
@pytest.mark.parametrize(
    "wrapper_press",
    [
        None,
        ComposedPress,
        KeyRerotationPress,
        AdaKVPress,
        ChunkPress,
        CriticalKVPress,
        CriticalAdaKVPress,
        DMSPress,
        MergingPress,
    ],
)
def test_presses_run(unit_test_model, press_dict, wrapper_press):  # noqa: F811
    cls = press_dict["cls"]
    for kwargs in press_dict["kwargs"]:
        press = cls(**kwargs)
        if wrapper_press is not None:
            init_press_from_model(press, unit_test_model)
            if issubclass(wrapper_press, ComposedPress):
                if isinstance(press, (KVzipPress, FastKVzipPress, KVComposePress)):
                    # KVzipPress, FastKVzipPress and KVComposePress are currently not compatible with ComposedPress
                    return
                press = ComposedPress(presses=[press])
            elif not isinstance(press, ScorerPress):  # remaining wrapper presses only support ScorerPress
                return
            elif issubclass(
                wrapper_press,
                (KeyRerotationPress, AdaKVPress, CriticalKVPress, CriticalAdaKVPress, MergingPress),
            ):
                press = wrapper_press(press=press)
            elif issubclass(wrapper_press, ChunkPress):
                press = ChunkPress(press=press, chunk_length=24)
            elif issubclass(wrapper_press, DMSPress):
                press = DMSPress(press=press, threshold=-0.5, sliding_window_size=32)

        # TODO: Handle post_init_from_model differently
        init_press_from_model(press, unit_test_model)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
            unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
        # Check that the press has a compression_ratio attribute
        assert hasattr(press, "compression_ratio")


def test_presses_run_observed_attention(unit_test_model_output_attention):  # noqa: F811
    for cls in [ObservedAttentionPress, FacilityLocationPress]:
        for compresion_ratio in [0.2, 0.8]:
            press = cls(compression_ratio=compresion_ratio)
            with press(unit_test_model_output_attention):
                input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"].to(
                    unit_test_model_output_attention.device
                )
                unit_test_model_output_attention(input_ids, past_key_values=DynamicCache()).past_key_values


@dataclass
class StoreKnormPress(ScorerPress):
    def __post_init__(self):
        self.scores = []

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        scores = -keys.norm(dim=-1)
        self.scores.append(scores)
        return scores


@torch.no_grad()
def test_presses_keep_highest_score(unit_test_model):  # noqa: F811
    """
    Test that kept keys are those with the highest score
    """
    for compresion_ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
        press = StoreKnormPress(compression_ratio=compresion_ratio)
        with press(unit_test_model):
            input_ids = torch.randint(0, 3_000, (5, 256), device=unit_test_model.device)
            past_key_values = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

        keys = [layer.keys for layer in past_key_values.layers]
        for scores, key in zip(press.scores, keys):
            max_scores = -key.norm(dim=-1)
            for batch_idx in range(scores.shape[0]):
                for head_idx in range(scores.shape[1]):
                    assert torch.allclose(
                        scores[batch_idx, head_idx].sort().values[-max_scores.shape[-1] :],
                        max_scores[batch_idx, head_idx].sort().values,
                    )
