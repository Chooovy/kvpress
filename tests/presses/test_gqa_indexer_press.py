# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GQA lightning indexer press, loss and training helpers."""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    IndexerNorm,
    IndexerTrainConfig,
    aggregate_chunk_scores,
    build_indexer_mask,
    build_dense_indexer_target,
    build_sparse_indexer_target,
    compute_indexer_loss,
    freeze_all_but_indexer,
    get_attention_modules,
    get_input_layernorms,
    indexer_layer_loss,
    indexer_state_dict,
    load_indexer_state_dict,
    masked_log_softmax,
    normalize_indexer_target,
    reduce_queries,
    slice_rope_tables,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG, apply_rotary, rotate_half
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def make_indexer(n_heads=2, head_dim=8, hidden_size=16, **kw):
    config = GQAIndexerConfig(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=head_dim,
        **kw,
    )
    return GQAIndexer(config)


def hf_rope_tables(seq_len, width, base=10000.0):
    """Build cos/sin exactly the way HuggingFace does: cat([freqs, freqs], -1)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, width, 2, dtype=torch.float) / width))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0), inv_freq


# ----------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rope_dim", [2, 4, 6, 8])
def test_rope_narrowing_matches_ground_truth(rope_dim):
    """
    Narrowed tables must rotate pair (j, j + r/2) by the angle of frequency j.

    This is the property a contiguous prefix slice silently violates: it drives the two
    halves of the pair with different frequencies.
    """
    seq_len, width = 6, 8
    cos_full, sin_full, inv_freq = hf_rope_tables(seq_len, width)
    cos, sin = slice_rope_tables(cos_full, sin_full, rope_dim)
    assert cos.shape[-1] == rope_dim

    x = torch.randn(1, 1, seq_len, rope_dim)
    got = apply_rotary(x, cos, sin)[0, 0]

    half = rope_dim // 2
    expected = torch.empty_like(x[0, 0])
    for t in range(seq_len):
        for j in range(half):
            ang = t * inv_freq[j]
            c, s = math.cos(ang), math.sin(ang)
            a, b = x[0, 0, t, j], x[0, 0, t, j + half]
            expected[t, j] = a * c - b * s
            expected[t, j + half] = b * c + a * s
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_rope_is_norm_preserving_and_relative():
    """RoPE must preserve norms and make dot products depend only on relative position."""
    cos_full, sin_full, _ = hf_rope_tables(8, 8)
    cos, sin = slice_rope_tables(cos_full, sin_full, 4)

    x = torch.randn(1, 1, 8, 4)
    rotated = apply_rotary(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)

    q = torch.randn(1, 1, 1, 4).expand(1, 1, 8, 4).contiguous()
    k = torch.randn(1, 1, 1, 4).expand(1, 1, 8, 4).contiguous()
    rq, rk = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    # same offset (m - n = 3) at two absolute positions -> identical similarity
    d1 = (rq[0, 0, 4] * rk[0, 0, 1]).sum()
    d2 = (rq[0, 0, 7] * rk[0, 0, 4]).sum()
    torch.testing.assert_close(d1, d2, rtol=1e-5, atol=1e-5)


def test_rope_passes_through_unrotated_tail():
    """Channels beyond rope_dim are NoPE and must be untouched."""
    cos_full, sin_full, _ = hf_rope_tables(4, 8)
    cos, sin = slice_rope_tables(cos_full, sin_full, 4)
    x = torch.randn(1, 1, 4, 8)
    out = apply_rotary(x, cos, sin)
    torch.testing.assert_close(out[..., 4:], x[..., 4:])
    assert not torch.allclose(out[..., :4], x[..., :4])


def test_rotate_half_matches_hf_convention():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def test_mask_is_causal_during_prefill():
    mask = build_indexer_mask(5, 5, torch.device("cpu"))[0, 0]
    allowed = mask == 0
    torch.testing.assert_close(allowed, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert allowed[0].sum() == 1  # first query sees only itself
    assert allowed[4].sum() == 5


def test_mask_handles_decode_and_chunked_prefill():
    """A query appended after existing keys may attend the whole cache."""
    assert (build_indexer_mask(1, 6, torch.device("cpu"))[0, 0, 0] == 0).sum() == 6
    # queries 4 and 5 of a 6-long cache
    mask = build_indexer_mask(2, 6, torch.device("cpu"))[0, 0]
    assert (mask[0] == 0).sum() == 5
    assert (mask[1] == 0).sum() == 6


def test_mask_applies_padding_from_2d_keep_mask():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    mask = build_indexer_mask(5, 5, torch.device("cpu"), attention_mask=attention_mask)
    assert (mask[1, 0, :, :2] == MASK_NEG).all()  # pad keys blocked for every query
    torch.testing.assert_close(mask[0, 0] == 0, torch.tril(torch.ones(5, 5, dtype=torch.bool)))


def test_mask_accepts_4d_additive_mask():
    additive = torch.zeros(1, 1, 4, 4)
    additive[..., 0] = torch.finfo(torch.float32).min  # block key 0
    mask = build_indexer_mask(4, 4, torch.device("cpu"), attention_mask=additive)
    assert (mask[0, 0, :, 0] == MASK_NEG).all()
    assert mask[0, 0, 1, 1] == 0


# ----------------------------------------------------------------------
# Indexer forward
# ----------------------------------------------------------------------
def test_indexer_emits_one_score_per_head():
    indexer = make_indexer(n_heads=3)
    out = indexer(torch.randn(2, 7, 16))
    assert out.shape == (2, 3, 7, 7)
    assert out.dtype == torch.float32  # fp32 accumulation


def test_indexer_is_mqa_on_the_key_side():
    """
    w_q emits n_heads heads; w_k emits exactly ONE.

    The single key head is what keeps the indexer's own cache at head_dim per token
    instead of n_heads * head_dim (8x smaller for Llama-3.1-8B).
    """
    indexer = make_indexer(n_heads=8, head_dim=128, hidden_size=4096)
    assert indexer.w_q.weight.shape == (8 * 128, 4096)
    assert indexer.w_k.weight.shape == (128, 4096)  # MQA: one key head
    assert indexer.w_q.bias is None and indexer.w_k.bias is None  # bias-free like DSA/M3


def test_indexer_has_no_activation_or_pooling_parameters():
    """
    The DSA components that only exist to collapse heads must be gone.

    A per-head scalar weight cannot reorder a row (it is constant along the key axis), and
    an activation cannot change a per-head top-k. Keeping either would be dead weight.
    """
    indexer = make_indexer()
    assert not hasattr(indexer, "weights_proj")
    assert not hasattr(indexer, "group_reduce")
    param_names = {n for n, _ in indexer.named_parameters()}
    assert param_names == {
        "w_q.weight",
        "w_k.weight",
        "q_norm.weight",
        "q_norm.bias",
        "k_norm.weight",
        "k_norm.bias",
    }


def test_scores_are_a_plain_dot_product():
    """No activation: the score must be exactly q . k, so it can take either sign."""
    indexer = make_indexer(n_heads=2, head_dim=8, hidden_size=16)
    hidden = torch.randn(1, 6, 16)
    out = indexer(hidden)

    q = indexer.project_q(hidden)
    k = indexer.project_k(hidden)
    expected = torch.einsum("bhqd,bkd->bhqk", q.float(), k.float())
    torch.testing.assert_close(out, expected)
    # both signs present -> nothing is clamping the range
    assert (out > 0).any() and (out < 0).any()


def test_per_head_scores_are_distinct():
    """The whole point: heads must not collapse to one shared score."""
    indexer = make_indexer(n_heads=4)
    out = indexer(torch.randn(1, 6, 16))
    for i in range(1, 4):
        assert not torch.allclose(out[0, 0], out[0, i]), f"head {i} duplicates head 0"


def test_indexer_respects_mask():
    indexer = make_indexer()
    hidden = torch.randn(1, 5, 16)
    mask = build_indexer_mask(5, 5, hidden.device)
    out = indexer(hidden, mask=mask)
    # Index the upper triangle explicitly: out.triu(1) would zero-fill the lower triangle
    # and those zeros would then fail the comparison.
    rows, cols = torch.triu_indices(5, 5, offset=1)
    assert (out[0, 0][rows, cols] < MASK_NEG / 2).all()
    # and the allowed (lower-triangular) entries are untouched
    rows, cols = torch.tril_indices(5, 5)
    assert (out[0, 0][rows, cols] > MASK_NEG / 2).all()


def test_indexer_accepts_separate_key_hidden_states():
    """Decode-time scoring: 1 query against a longer key history."""
    indexer = make_indexer()
    out = indexer(torch.randn(1, 1, 16), key_hidden_states=torch.randn(1, 9, 16))
    assert out.shape == (1, 2, 1, 9)


def test_config_rejects_invalid_settings():
    with pytest.raises(ValueError, match="rope_dim must be even"):
        GQAIndexerConfig(hidden_size=16, n_heads=2, head_dim=8, rope_dim=3)
    with pytest.raises(ValueError, match="cannot exceed head_dim"):
        GQAIndexerConfig(hidden_size=16, n_heads=2, head_dim=8, rope_dim=16)
    with pytest.raises(ValueError, match="n_heads must be positive"):
        GQAIndexerConfig(hidden_size=16, n_heads=0, head_dim=8)


# ----------------------------------------------------------------------
# Why activation / per-head weights were dropped.
# These pin the arguments so nobody re-adds them without a new reason.
# ----------------------------------------------------------------------
def test_topk_is_invariant_to_strictly_increasing_activation():
    """softplus / exp cannot change a per-head selection, so they buy nothing."""
    scores = torch.randn(64)
    k = 20
    base = set(scores.topk(k).indices.tolist())
    for name, fn in [("softplus", F.softplus), ("exp", torch.exp), ("affine", lambda x: 2 * x + 1)]:
        assert set(fn(scores).topk(k).indices.tolist()) == base, f"{name} changed the selection"


def test_relu_would_tie_negative_scores_and_randomise_selection():
    """
    ReLU is NOT strictly increasing: it flattens every negative score to 0.

    At moderate compression the keep boundary falls inside that negative region, so ReLU
    would decide part of the selection by arbitrary tie-break. This is the concrete reason
    the indexer has no activation.
    """
    torch.manual_seed(0)
    scores = torch.randn(64) - 1.0  # mostly negative
    k = 40  # boundary lands among the negatives
    assert (scores < 0).sum() > k // 2, "fixture should be mostly negative"

    raw = set(scores.topk(k).indices.tolist())
    relued = set(F.relu(scores).topk(k).indices.tolist())
    assert F.relu(scores).eq(0).sum() > 1, "relu should produce ties"
    assert raw != relued, "relu must be shown to perturb the selection"


def test_per_head_scalar_weight_cannot_reorder_a_row():
    """
    A weights_proj-style scalar is constant along the key axis: a no-op when positive and a
    ranking reversal when negative. Either way it adds nothing to per-head selection.
    """
    scores = torch.randn(64)
    k = 20
    base = set(scores.topk(k).indices.tolist())
    assert set((3.7 * scores).topk(k).indices.tolist()) == base
    assert set((-3.7 * scores).topk(k).indices.tolist()) != base


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["mean", "max", "last", "recency"])
def test_reduce_queries_shapes(mode):
    scores = torch.randn(2, 3, 6, 7)
    out = reduce_queries(scores, mode, last_n_query=4 if mode == "last" else None)
    assert out.shape == (2, 3, 7)


def test_reduce_queries_semantics():
    scores = torch.tensor([[[[1.0, 5.0], [3.0, 1.0]]]])  # (1,1,2,2)
    torch.testing.assert_close(reduce_queries(scores, "mean"), torch.tensor([[[2.0, 3.0]]]))
    torch.testing.assert_close(reduce_queries(scores, "max"), torch.tensor([[[3.0, 5.0]]]))
    # last-1 keeps only the final query row
    torch.testing.assert_close(
        reduce_queries(scores, "last", last_n_query=1), torch.tensor([[[3.0, 1.0]]])
    )


def test_recency_weighting_favours_recent_queries():
    scores = torch.zeros(1, 1, 4, 1)
    scores[0, 0, -1, 0] = 1.0  # only newest query votes
    recent = reduce_queries(scores, "recency", recency_half_life=1.0)
    mean = reduce_queries(scores, "mean")
    assert recent.item() > mean.item()


@pytest.mark.parametrize("mode,expected", [("mean", [1.5, 5.5]), ("max", [3.0, 7.0])])
def test_aggregate_chunk_scores(mode, expected):
    token_scores = torch.arange(10, dtype=torch.float).view(1, 1, 10)
    chunks, complete_end = aggregate_chunk_scores(token_scores, 4, mode)
    assert complete_end == 8  # ragged tail excluded
    torch.testing.assert_close(chunks, torch.tensor(expected).view(1, 1, 2))


def test_aggregate_chunk_scores_handles_short_input():
    chunks, end = aggregate_chunk_scores(torch.randn(1, 1, 3), 8, "mean")
    assert end == 0 and chunks.shape[-1] == 0


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def test_kl_is_zero_when_student_matches_teacher():
    logits = torch.randn(1, 2, 3, 5)
    valid = torch.ones_like(logits, dtype=torch.bool)
    log_probs = masked_log_softmax(logits, valid)
    target = log_probs.exp()
    from kvpress.presses.gqa_indexer.loss import indexer_kl_per_row

    assert indexer_kl_per_row(target, log_probs, valid).abs().max() < 1e-6


def test_kl_is_nonnegative_and_masked_entries_are_ignored():
    from kvpress.presses.gqa_indexer.loss import indexer_kl_per_row

    logits = torch.randn(1, 2, 3, 6)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[..., 4:] = False
    log_probs = masked_log_softmax(logits, valid)
    target = normalize_indexer_target(torch.rand(1, 2, 3, 6).masked_fill(~valid, 0.0))
    kl = indexer_kl_per_row(target, log_probs, valid)
    assert (kl >= -1e-6).all() and torch.isfinite(kl).all()


def test_masked_log_softmax_survives_fully_masked_row():
    """A row with no valid key must stay finite so the caller can drop it."""
    logits = torch.randn(1, 1, 2, 4)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[0, 0, 0] = False
    out = masked_log_softmax(logits, valid)
    assert torch.isfinite(out).all()
    # valid row still normalizes correctly over its valid entries
    torch.testing.assert_close(out[0, 0, 1].exp().sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_dense_target_groups_heads_by_kv_group():
    """
    Each KV group's target must come from its own attention heads.

    Averaging across all heads (the DSA behaviour) would give every indexer head an
    identical target and waste the per-head capacity.
    """
    attn = torch.rand(1, 8, 3, 5)
    attn = attn / attn.sum(-1, keepdim=True)
    valid = torch.ones(1, 4, 3, 5, dtype=torch.bool)
    target = build_dense_indexer_target(attn, valid, n_kv_heads=4)

    assert target.shape == (1, 4, 3, 5)
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 4, 3))
    # group 0 is built from heads 0-1 only
    expected0 = normalize_indexer_target(attn[:, :2].mean(1))
    torch.testing.assert_close(target[:, 0], expected0)
    assert not torch.allclose(target[:, 0], target[:, 1])


def test_dense_target_zeroes_masked_keys():
    attn = torch.rand(1, 4, 2, 6)
    valid = torch.ones(1, 2, 2, 6, dtype=torch.bool)
    valid[..., 3:] = False
    target = build_dense_indexer_target(attn, valid, n_kv_heads=2)
    assert (target[..., 3:] == 0).all()
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 2, 2))


def test_sparse_target_aligns_with_topk_and_ignores_padding_slots():
    attn = torch.rand(1, 4, 2, 6)
    attn = attn / attn.sum(-1, keepdim=True)
    topk = torch.tensor([[[[0, 2, -1], [1, 3, 4]]] * 2])  # (1,2,2,3)
    target = build_sparse_indexer_target(attn, topk, n_kv_heads=2)
    assert target.shape == (1, 2, 2, 3)
    assert target[0, 0, 0, 2] == 0  # the -1 slot carries no mass
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 2, 2))


# ----------------------------------------------------------------------
# Layer loss / training
# ----------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["dense", "sparse"])
def test_indexer_layer_loss_is_finite_and_differentiable(stage):
    n_kv_heads, n_heads, q_len, k_len = 2, 4, 5, 5
    indexer = make_indexer(n_heads=n_kv_heads)
    hidden = torch.randn(1, q_len, 16)
    mask = build_indexer_mask(q_len, k_len, hidden.device)
    logits = indexer(hidden, mask=mask)

    attn = torch.rand(1, n_heads, q_len, k_len).tril()
    attn = attn / attn.sum(-1, keepdim=True)

    config = IndexerTrainConfig(stage=stage, topk=3)
    loss = indexer_layer_loss(logits, attn, config)
    assert torch.isfinite(loss) and loss.item() >= 0

    loss.backward()
    grads = [p.grad for p in indexer.parameters() if p.grad is not None]
    assert grads, "loss produced no indexer gradients"
    assert any(g.abs().sum() > 0 for g in grads)


def test_layer_loss_does_not_backprop_into_teacher():
    """The frozen model's attention must receive no gradient."""
    indexer = make_indexer()
    hidden = torch.randn(1, 4, 16)
    logits = indexer(hidden, mask=build_indexer_mask(4, 4, hidden.device))
    attn = torch.rand(1, 4, 4, 4, requires_grad=True).tril()
    attn = attn / attn.sum(-1, keepdim=True)
    indexer_layer_loss(logits, attn, IndexerTrainConfig()).backward()
    assert attn.grad is None


def test_layer_loss_excludes_ignored_label_rows():
    """Rows whose query position is masked out by labels must not affect the loss."""
    indexer = make_indexer()
    hidden = torch.randn(1, 4, 16)
    logits = indexer(hidden, mask=build_indexer_mask(4, 4, hidden.device))
    attn = torch.rand(1, 4, 4, 4).tril()
    attn = attn / attn.sum(-1, keepdim=True)

    labels_all = torch.zeros(1, 4, dtype=torch.long)
    labels_some = torch.tensor([[-100, -100, 0, 0]])
    cfg = IndexerTrainConfig()
    assert not torch.allclose(
        indexer_layer_loss(logits, attn, cfg, labels=labels_all),
        indexer_layer_loss(logits, attn, cfg, labels=labels_some),
    )


def test_layer_loss_rejects_shape_mismatch():
    indexer = make_indexer()
    logits = indexer(torch.randn(1, 4, 16))
    with pytest.raises(ValueError, match="shape mismatch"):
        indexer_layer_loss(logits, torch.rand(1, 4, 3, 3), IndexerTrainConfig())


# ----------------------------------------------------------------------
# Press integration
# ----------------------------------------------------------------------
def test_press_scores_are_per_kv_head(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    module = get_attention_modules(unit_test_model)[0]

    n_kv = unit_test_model.config.num_key_value_heads
    head_dim = module.head_dim
    hidden = torch.randn(1, 12, unit_test_model.config.hidden_size, device=unit_test_model.device)
    keys = torch.randn(1, n_kv, 12, head_dim, device=unit_test_model.device)

    scores = press.score(module, hidden, keys, keys, None, {})
    assert scores.shape == (1, n_kv, 12)


def test_press_protects_sink_and_local(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5, n_sink=2, n_local=3)
    press.post_init_from_model(unit_test_model)
    module = get_attention_modules(unit_test_model)[0]

    n_kv = unit_test_model.config.num_key_value_heads
    hidden = torch.randn(1, 16, unit_test_model.config.hidden_size, device=unit_test_model.device)
    keys = torch.randn(1, n_kv, 16, module.head_dim, device=unit_test_model.device)

    scores = press.score(module, hidden, keys, keys, None, {})
    # Protection uses a finite sentinel, not +inf, so scores stay usable in arithmetic.
    assert torch.isfinite(scores).all()
    protected = torch.cat([scores[..., :2], scores[..., -3:]], dim=-1)
    scored = scores[..., 2:-3]
    assert protected.min() > scored.max(), "protected tokens must outrank every scored token"


def test_press_protected_tokens_survive_compression(unit_test_model):  # noqa: F811
    """Sink and local tokens must still be in the cache after eviction."""
    press = GQAIndexerPress(compression_ratio=0.5, n_sink=2, n_local=2)
    press.post_init_from_model(unit_test_model)
    module = get_attention_modules(unit_test_model)[0]

    n_kv, seq_len = unit_test_model.config.num_key_value_heads, 16
    hidden = torch.randn(1, seq_len, unit_test_model.config.hidden_size, device=unit_test_model.device)
    keys = torch.randn(1, n_kv, seq_len, module.head_dim, device=unit_test_model.device)

    scores = press.score(module, hidden, keys, keys, None, {})
    n_kept = int(seq_len * (1 - press.compression_ratio))
    kept = scores.topk(n_kept, dim=-1).indices[0, 0].tolist()
    for idx in (0, 1, seq_len - 2, seq_len - 1):
        assert idx in kept, f"protected index {idx} was evicted"


def test_press_chunk_mode_gives_chunk_uniform_scores(unit_test_model):  # noqa: F811
    """Chunk selection is expressed as a score transform, so chunk members must tie."""
    press = GQAIndexerPress(compression_ratio=0.5, chunk_size=4, n_sink=0, n_local=0)
    press.post_init_from_model(unit_test_model)
    module = get_attention_modules(unit_test_model)[0]

    n_kv = unit_test_model.config.num_key_value_heads
    hidden = torch.randn(1, 16, unit_test_model.config.hidden_size, device=unit_test_model.device)
    keys = torch.randn(1, n_kv, 16, module.head_dim, device=unit_test_model.device)

    scores = press.score(module, hidden, keys, keys, None, {})
    for start in range(0, 16, 4):
        chunk = scores[0, 0, start : start + 4]
        assert torch.allclose(chunk, chunk[0].expand(4)), f"chunk at {start} is not uniform"


@pytest.mark.parametrize("compression_ratio", [0.25, 0.5])
def test_press_end_to_end_compresses_cache(unit_test_model, compression_ratio):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=compression_ratio)
    seq_len = 64
    input_ids = torch.randint(0, 1024, (1, seq_len), device=unit_test_model.device)

    with press(unit_test_model):
        cache = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values

    # ScorerPress.compress uses int(k_len * (1 - compression_ratio)) with no floor.
    expected = int(seq_len * (1 - compression_ratio))
    assert cache.layers[0].keys.shape[2] == expected


def test_press_mean_head_matches_head_uniform_behaviour(unit_test_model):  # noqa: F811
    """mean_head=True is the ablation back to a single shared selection."""
    press = GQAIndexerPress(compression_ratio=0.5, mean_head=True)
    input_ids = torch.randint(0, 1024, (1, 32), device=unit_test_model.device)
    with press(unit_test_model):
        cache = unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values
    assert cache.layers[0].keys.shape[2] == 16


def test_press_geometry_defaults_to_one_head_per_kv_head(unit_test_model):  # noqa: F811
    """
    The default must be n_heads == num_key_value_heads, NOT num_attention_heads.

    Mirroring the full query-head count would cost ~4x the parameters and FLOPs while
    still producing only num_key_value_heads usable scores.
    """
    press = GQAIndexerPress()
    press.post_init_from_model(unit_test_model)
    module = get_attention_modules(unit_test_model)[0]
    indexer = press.get_indexer(module)

    config = unit_test_model.config
    assert indexer.n_heads == config.num_key_value_heads
    assert indexer.head_dim == module.head_dim
    # w_k stays single-head regardless of n_heads
    assert indexer.w_k.weight.shape[0] == indexer.head_dim


def test_post_init_is_idempotent(unit_test_model):  # noqa: F811
    press = GQAIndexerPress()
    press.post_init_from_model(unit_test_model)
    first = press.get_indexer(get_attention_modules(unit_test_model)[0])
    press.post_init_from_model(unit_test_model)
    assert press.get_indexer(get_attention_modules(unit_test_model)[0]) is first


def test_press_errors_without_post_init():
    press = GQAIndexerPress()
    with pytest.raises(RuntimeError, match="post_init_from_model"):
        press.get_indexer(torch.nn.Linear(4, 4))


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------
def test_freeze_all_but_indexer(unit_test_model):  # noqa: F811
    press = GQAIndexerPress()
    press.post_init_from_model(unit_test_model)
    trainable = freeze_all_but_indexer(unit_test_model)

    assert trainable
    for name, param in unit_test_model.named_parameters():
        assert param.requires_grad == (".indexer." in name), name


def test_indexer_state_dict_roundtrip(unit_test_model):  # noqa: F811
    press = GQAIndexerPress()
    press.post_init_from_model(unit_test_model)

    sd = indexer_state_dict(unit_test_model)
    assert sd and all(".indexer." in k for k in sd)
    # only the indexer travels, not the backbone
    assert len(sd) < len(unit_test_model.state_dict())

    module = get_attention_modules(unit_test_model)[0]
    with torch.no_grad():
        press.get_indexer(module).w_k.weight.add_(1.0)
    load_indexer_state_dict(unit_test_model, sd)
    torch.testing.assert_close(
        press.get_indexer(module).w_k.weight.cpu(),
        sd[[k for k in sd if k.endswith("layers.0.self_attn.indexer.w_k.weight")][0]],
    )


def test_load_indexer_state_dict_rejects_unrelated_dict(unit_test_model):  # noqa: F811
    with pytest.raises(ValueError, match="no 'indexer' keys"):
        load_indexer_state_dict(unit_test_model, {"model.embed_tokens.weight": torch.zeros(1)})


def test_compute_indexer_loss_over_all_layers(unit_test_model_output_attention):  # noqa: F811
    model = unit_test_model_output_attention
    press = GQAIndexerPress()
    press.post_init_from_model(model)
    freeze_all_but_indexer(model)

    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)
    out = model(input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)

    loss, per_layer = compute_indexer_loss(
        press,
        get_attention_modules(model),
        out.hidden_states,
        out.attentions,
        IndexerTrainConfig(stage="dense"),
        model=model,
    )
    assert torch.isfinite(loss) and len(per_layer) == len(out.attentions)

    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_compute_indexer_loss_uses_layer_input_hidden_states(unit_test_model_output_attention):  # noqa: F811
    """
    Layer i must be scored from layernorm(hidden_states[i]), not hidden_states[i+1].

    Two distinct mistakes are ruled out here. Using i+1 would leak the layer's own output,
    which the real forward pass never has. And using hidden_states[i] *raw* would skip the
    block's input_layernorm -- but kvpress hooks self_attn, which sits after it, so the press
    scores the normalized tensor at inference.
    """
    model = unit_test_model_output_attention
    press = GQAIndexerPress()
    press.post_init_from_model(model)

    input_ids = torch.randint(0, 1024, (1, 12), device=model.device)
    out = model(input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)
    modules = get_attention_modules(model)
    norms = get_input_layernorms(model)

    seen = []
    original = press.indexer_logits

    def spy(module, hidden_states, kwargs, k_len=None):
        seen.append(hidden_states)
        return original(module, hidden_states, kwargs, k_len=k_len)

    press.indexer_logits = spy
    compute_indexer_loss(
        press, modules, out.hidden_states, out.attentions, IndexerTrainConfig(), model=model
    )
    press.indexer_logits = original

    assert len(seen) == len(out.attentions)
    for layer_idx, hs in enumerate(seen):
        torch.testing.assert_close(hs, norms[layer_idx](out.hidden_states[layer_idx]))
        assert hs is not out.hidden_states[layer_idx + 1]


def test_get_rope_tables_raises_when_rope_is_wanted_but_absent(unit_test_model):  # noqa: F811
    """
    A RoPE-aware indexer with no position_embeddings must fail, not silently score NoPE.

    At inference the press is hooked onto self_attn and so always receives them; an absent
    table means a caller is about to build a different student than the one that runs. The old
    behaviour returned (None, None), which trained the indexer with no positional signal at all
    while every shape and loss value stayed plausible.
    """
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    indexer = press.get_indexer(get_attention_modules(unit_test_model)[0])
    assert indexer.rope_dim > 0, "this fixture must want RoPE for the test to mean anything"

    with pytest.raises(ValueError, match="no position_embeddings"):
        press.get_rope_tables(indexer, {})


def test_get_rope_tables_allows_deliberate_nope(unit_test_model):  # noqa: F811
    """rope_dim=0 is an explicit opt-in, so it must stay silent."""
    press = GQAIndexerPress(compression_ratio=0.5, rope_dim=0)
    press.post_init_from_model(unit_test_model)
    indexer = press.get_indexer(get_attention_modules(unit_test_model)[0])
    assert indexer.rope_dim == 0
    assert press.get_rope_tables(indexer, {}) == (None, None)


def test_compute_indexer_loss_applies_rope(unit_test_model_output_attention):  # noqa: F811
    """
    The dense path must give the indexer positions, or it trains a NoPE student.

    Checked by comparing against an explicit rope_dim=0 press: if the default path were also
    running without positions the two losses would coincide.
    """
    model = unit_test_model_output_attention
    input_ids = torch.randint(0, 1024, (1, 12), device=model.device)
    out = model(input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)
    modules = get_attention_modules(model)

    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    indexer = press.get_indexer(modules[0])
    assert indexer.rope_dim > 0, "this fixture must want RoPE for the test to mean anything"

    rope_loss, _ = compute_indexer_loss(
        press, modules, out.hidden_states, out.attentions,
        IndexerTrainConfig(stage="dense"), model=model,
    )

    # Same weights, RoPE switched off in place: the only difference is the rotation.
    original = {m: press.get_indexer(m).rope_dim for m in modules}
    try:
        for module in modules:
            press.get_indexer(module).rope_dim = 0
        nope_loss, _ = compute_indexer_loss(
            press, modules, out.hidden_states, out.attentions,
            IndexerTrainConfig(stage="dense"), model=model,
        )
    finally:
        for module, rope_dim in original.items():
            press.get_indexer(module).rope_dim = rope_dim

    assert torch.isfinite(rope_loss) and torch.isfinite(nope_loss)
    assert not torch.allclose(rope_loss, nope_loss), (
        "identical to the NoPE student, so the default path is not applying RoPE"
    )


def test_compute_indexer_loss_rejects_missing_attentions(unit_test_model):  # noqa: F811
    """output_attentions=True is required; None must say so rather than raise a TypeError."""
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    input_ids = torch.randint(0, 1024, (1, 8), device=unit_test_model.device)
    out = unit_test_model(input_ids, output_hidden_states=True, use_cache=False)
    assert getattr(out, "attentions", None) is None

    with pytest.raises(ValueError, match="output_attentions=True"):
        compute_indexer_loss(
            press, get_attention_modules(unit_test_model), out.hidden_states, None,
            IndexerTrainConfig(stage="dense"), model=unit_test_model,
        )


# ----------------------------------------------------------------------
# fp32 norm statistics
# ----------------------------------------------------------------------
def test_indexer_norm_matches_layer_norm_in_fp32():
    """
    In fp32 there is nothing to upcast, so IndexerNorm must equal nn.LayerNorm exactly.

    This is what makes the swap safe: the fp32 path is unchanged, and only reduced-precision
    inputs take a different (more accurate) route.
    """
    torch.manual_seed(0)
    dim = 16
    ours = IndexerNorm(dim)
    theirs = torch.nn.LayerNorm(dim)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(dim))
        ours.bias.copy_(torch.randn(dim))
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)

    x = torch.randn(3, 5, dim)
    torch.testing.assert_close(ours(x), theirs(x))


def test_indexer_norm_reduces_in_fp32_for_bf16_input():
    """
    bf16 in, bf16 out, but the statistics computed in fp32.

    nn.LayerNorm on a bf16 module reduces in bf16, whose 8 significant bits give ~7e-2 median
    relative error over head_dim channels. That error lands on q/k and is then amplified by the
    head_dim-long dot product, so the score would inherit it before its own fp32 GEMM begins.
    """
    torch.manual_seed(0)
    dim = 128
    ours = IndexerNorm(dim).to(torch.bfloat16)
    theirs = torch.nn.LayerNorm(dim).to(torch.bfloat16)
    with torch.no_grad():
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)

    x = (torch.randn(4, 32, dim) * 10).to(torch.bfloat16)
    got = ours(x)
    assert got.dtype == torch.bfloat16, "the output must stay in the module's dtype"

    reference = torch.nn.functional.layer_norm(
        x.float(), (dim,), ours.weight.float(), ours.bias.float(), ours.eps
    )
    ours_err = (got.float() - reference).abs().max()
    theirs_err = (theirs(x).float() - reference).abs().max()
    assert ours_err <= theirs_err, (
        f"fp32 reduction should not be worse: {ours_err} vs {theirs_err}"
    )


def test_indexer_norm_does_not_narrow_float64():
    """An fp64 caller asked for fp64; .float() would silently narrow it."""
    dim = 8
    norm = IndexerNorm(dim).to(torch.float64)
    x = torch.randn(2, 3, dim, dtype=torch.float64)
    out = norm(x)
    assert out.dtype == torch.float64

    reference = torch.nn.functional.layer_norm(
        x, (dim,), norm.weight, norm.bias, norm.eps
    )
    torch.testing.assert_close(out, reference)


def test_indexer_norm_is_checkpoint_compatible_with_layer_norm():
    """
    Parameter names and shapes must match nn.LayerNorm, or existing checkpoints break.

    The swap is meant to be a numerics change only; a state_dict saved before it must load
    after it with no renaming.
    """
    dim = 12
    legacy = torch.nn.LayerNorm(dim)
    with torch.no_grad():
        legacy.weight.copy_(torch.randn(dim))
        legacy.bias.copy_(torch.randn(dim))

    ours = IndexerNorm(dim)
    # strict=True is the assertion: any renamed or reshaped parameter raises here.
    result = ours.load_state_dict(legacy.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(ours.weight, legacy.weight)
    torch.testing.assert_close(ours.bias, legacy.bias)


def test_indexer_norm_gradients_flow():
    """The upcast must not detach: both parameters have to receive gradients."""
    dim = 8
    norm = IndexerNorm(dim).to(torch.bfloat16)
    x = torch.randn(2, 4, dim, dtype=torch.bfloat16, requires_grad=True)
    norm(x).float().pow(2).sum().backward()
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0
    assert norm.bias.grad is not None
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_indexer_uses_the_fp32_norm():
    """The indexer must pick up IndexerNorm, not nn.LayerNorm."""
    indexer = make_indexer()
    assert isinstance(indexer.q_norm, IndexerNorm)
    assert isinstance(indexer.k_norm, IndexerNorm)
