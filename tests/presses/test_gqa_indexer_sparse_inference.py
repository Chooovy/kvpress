# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for :class:`~kvpress.presses.gqa_indexer.sparse_inference.SparseAttentionContext`.

CPU-only: :func:`sparse_gqa_attention` falls back to its pure-torch reference when CUDA/Triton
are absent, so the whole selection + attention path runs here without a GPU. The kernel itself is
tested separately in ``test_gqa_indexer_sparse_attention.py``; these tests check the *wiring* --
that the indexer selects, that the indexer key-cache stays in lockstep with the model cache across
decode steps, and that when the selection is a no-op the model reproduces ordinary dense attention.
"""

from __future__ import annotations

import pytest
import torch

from kvpress import GQAIndexerPress, SparseAttentionContext

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def _tiny_model() -> Qwen3ForCausalLM:
    """A minimal Qwen3 in fp32 on CPU, with a random (default-init) indexer attached."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        attn_implementation="sdpa",
    )
    return Qwen3ForCausalLM(config).to(torch.float32).eval()


def _press(model) -> GQAIndexerPress:
    # gate_scale=True mirrors how the eval loads an e2e checkpoint; it is unused by selection.
    press = GQAIndexerPress(compression_ratio=0.0, gate_scale=True)
    press.post_init_from_model(model)
    return press


@torch.no_grad()
def test_full_topk_reduces_to_dense_attention():
    """topk >= k_len selects every causal key, so sparse attention == dense attention."""
    model = _tiny_model()
    press = _press(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 24))

    dense = model(input_ids=ids).logits

    with SparseAttentionContext(model, press, topk=1024, force_sink=0, force_local=0):
        sparse = model(input_ids=ids).logits

    # Selection is a no-op here, so the only differences are fp32 reduction-order noise between
    # sdpa and the gather reference -- tight, not loose.
    assert torch.allclose(dense, sparse, atol=1e-4, rtol=1e-4), (
        f"max abs diff {(dense - sparse).abs().max().item():.2e}"
    )


@torch.no_grad()
def test_decode_keeps_indexer_cache_in_lockstep():
    """Prefill then two decode steps must not trip the cache-length assertion, and must select."""
    model = _tiny_model()
    press = _press(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 20))

    with SparseAttentionContext(model, press, topk=8, force_sink=2, force_local=2):
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        assert cache.get_seq_length() == 20
        next_id = out.logits[:, -1:].argmax(-1)
        for step in range(2):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1:].argmax(-1)
        assert cache.get_seq_length() == 22
        assert torch.isfinite(out.logits).all()


@torch.no_grad()
def test_small_topk_runs_and_differs_from_dense():
    """A genuinely sparse budget produces finite output that differs from full attention."""
    model = _tiny_model()
    press = _press(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 32))

    dense = model(input_ids=ids).logits
    with SparseAttentionContext(model, press, topk=4, force_sink=1, force_local=1):
        sparse = model(input_ids=ids).logits

    assert torch.isfinite(sparse).all()
    assert not torch.allclose(dense, sparse, atol=1e-3), "topk=4 should not match full attention"


@torch.no_grad()
def test_attn_implementation_restored_on_exit():
    """The registry swap and config pointer are undone, even though the block ran."""
    model = _tiny_model()
    press = _press(model)
    before = model.config._attn_implementation
    with SparseAttentionContext(model, press, topk=8):
        assert model.config._attn_implementation == "kvpress_gqa_indexer_sparse"
        model(input_ids=torch.randint(0, model.config.vocab_size, (1, 8)))
    assert model.config._attn_implementation == before
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert "kvpress_gqa_indexer_sparse" not in type(ALL_ATTENTION_FUNCTIONS)._global_mapping
