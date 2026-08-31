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
def test_dma_prefill_and_decode_use_value_states():
    """DMA scores newly projected values across context, question prefill, and decode."""
    model = _tiny_model()
    press = GQAIndexerPress(compression_ratio=0.0, scorer="dma")
    press.post_init_from_model(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 20))

    with SparseAttentionContext(
        model, press, topk=8, force_sink=2, force_local=2, query_independent=False
    ) as sparse:
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        assert all(k_idx.shape == (1, 20, 2) for k_idx in sparse._k_idx.values())

        question_ids = torch.randint(0, model.config.vocab_size, (1, 3))
        out = model(input_ids=question_ids, past_key_values=cache, use_cache=True)
        assert all(k_idx.shape == (1, 23, 2) for k_idx in sparse._k_idx.values())

        next_id = out.logits[:, -1:].argmax(-1)
        for _ in range(2):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1:].argmax(-1)
        assert all(k_idx.shape == (1, 25, 2) for k_idx in sparse._k_idx.values())

    assert cache.get_seq_length() == 25
    assert torch.isfinite(out.logits).all()


@torch.no_grad()
def test_prefix_scorer_decodes_with_a_cached_prefix():
    """The prefix arm decodes, and its cache stays in lockstep with the model's KV.

    The prefix readout is a function of the key's whole prefix, so ``score_keys`` refuses a
    suffix outright: without the indexer caching its own K/V, a decode step could not be scored
    at all. This is the test that the context enables that cache, advances it by exactly one key
    per step, and tears it down on exit -- a leak would silently score the next generation
    against the previous one's prefix.
    """
    model = _tiny_model()
    press = GQAIndexerPress(
        compression_ratio=0.0,
        scorer="prefix",
        scalar_mid_dim=16,
        prefix_head_dim=8,
        prefix_value_dim=8,
        prefix_zero_init=False,
    )
    press.post_init_from_model(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 20))

    with SparseAttentionContext(
        model, press, topk=8, force_sink=2, force_local=2, query_independent=False
    ):
        indexer = press.get_indexer(model.model.layers[0].self_attn)
        assert indexer.cached_length == 0

        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        assert indexer.cached_length == 20

        next_id = out.logits[:, -1:].argmax(-1)
        for expected in (21, 22):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            assert indexer.cached_length == expected == cache.get_seq_length()
            next_id = out.logits[:, -1:].argmax(-1)
        assert torch.isfinite(out.logits).all()

    assert press.get_indexer(model.model.layers[0].self_attn).cached_length == 0


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


@torch.no_grad()
def test_precision_reaches_the_kernel():
    """
    ``precision`` must arrive at the kernel call, and default to ``tf32``.

    Not a style preference: the parameter simply was not plumbed, so the eval silently ran the
    kernel at its ``"ieee"`` default. That forgoes tensor cores, which turned ``BLOCK_G``'s
    Triton-3.3 padding into a measured 7x per prefill (67.0 s against 9.4 s at ``L=8192,
    topk=2048`` on an H20). A default asserted here is a default that cannot quietly regress.
    """
    import kvpress.presses.gqa_indexer.sparse_inference as si

    model = _tiny_model()
    press = _press(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 8))

    seen: list[str | None] = []
    original = si.sparse_gqa_attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("precision"))
        return original(*args, **kwargs)

    si.sparse_gqa_attention = spy
    try:
        with SparseAttentionContext(model, press, topk=8):
            model(input_ids=ids)
        assert seen and set(seen) == {"tf32"}, f"expected tf32 everywhere, got {set(seen)}"

        seen.clear()
        with SparseAttentionContext(model, press, topk=8, precision="ieee"):
            model(input_ids=ids)
        assert seen and set(seen) == {"ieee"}, f"expected ieee everywhere, got {set(seen)}"
    finally:
        si.sparse_gqa_attention = original


def test_precision_is_validated():
    """A typo must fail on construction, not silently fall through to the kernel default."""
    model = _tiny_model()
    press = _press(model)
    with pytest.raises(ValueError, match="precision must be"):
        SparseAttentionContext(model, press, topk=8, precision="fp32")
