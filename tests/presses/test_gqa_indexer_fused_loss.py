# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tiled (O(L)-memory) indexer distillation loss."""

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    build_indexer_mask,
    fused_indexer_ce_rows,
    fused_indexer_loss,
    make_recompute_teacher,
    normalize_captured_lse,
    teacher_lse_from_qk,
    teacher_probs_from_lse,
)
from kvpress.presses.gqa_indexer.teacher_lse import assert_lse_mask_compatible


def make_case(bsz=2, h=3, group_size=2, q_len=9, k_len=9, dim=4, d_tea=5, seed=0, causal=True):
    """Build a consistent (student, teacher, mask, lse) fixture."""
    torch.manual_seed(seed)
    n_heads = h * group_size
    q_idx = torch.randn(bsz, h, q_len, dim, dtype=torch.float64, requires_grad=True)
    k_idx = torch.randn(bsz, k_len, dim, dtype=torch.float64, requires_grad=True)
    q_tea = torch.randn(bsz, n_heads, q_len, d_tea, dtype=torch.float64)
    k_tea = torch.randn(bsz, n_heads, k_len, d_tea, dtype=torch.float64)
    scaling = d_tea**-0.5

    mask = build_indexer_mask(q_len, k_len, q_idx.device, dtype=torch.float64) if causal else None
    teacher_alpha = make_recompute_teacher(q_tea, k_tea, scaling, group_size)
    # lse must be computed under the SAME mask the loss uses.
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=4).to(torch.float64)
    return q_idx, k_idx, teacher_alpha, lse, mask, group_size, n_heads


def dense_reference(q_idx, k_idx, teacher_alpha, lse, mask, group_size, k_len):
    """Materialize everything and compute the same objective the naive way."""
    logits = torch.einsum("bhqd,bkd->bhqk", q_idx, k_idx)
    alpha = teacher_alpha(0, k_len)
    if mask is not None:
        logits = logits + mask
        alpha = alpha + mask
    p_bar = teacher_probs_from_lse(alpha, lse, group_size).to(logits.dtype)
    lse_student = torch.logsumexp(logits, dim=-1)
    return lse_student - (p_bar * logits).sum(-1), p_bar, torch.exp(logits - lse_student.unsqueeze(-1))


# ----------------------------------------------------------------------
# Teacher reconstruction from lse
# ----------------------------------------------------------------------
def test_teacher_probs_from_lse_are_normalized():
    """exp(alpha - lse) must sum to 1 once the mask is folded in before the lse."""
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    alpha = teacher_alpha(0, k_idx.shape[1]) + mask
    p_bar = teacher_probs_from_lse(alpha, lse, group_size)
    torch.testing.assert_close(p_bar.sum(-1), torch.ones_like(p_bar.sum(-1)))


def test_teacher_probs_match_explicit_softmax():
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, n_heads = make_case()
    alpha = teacher_alpha(0, k_idx.shape[1]) + mask
    expected = torch.softmax(alpha, dim=-1)
    expected = expected.view(
        alpha.shape[0], n_heads // group_size, group_size, alpha.shape[2], alpha.shape[3]
    ).mean(2)
    torch.testing.assert_close(teacher_probs_from_lse(alpha, lse, group_size), expected)


def test_teacher_lse_from_qk_is_tile_invariant():
    torch.manual_seed(0)
    q_tea = torch.randn(1, 4, 8, 6, dtype=torch.float64)
    k_tea = torch.randn(1, 4, 8, 6, dtype=torch.float64)
    ref = teacher_lse_from_qk(q_tea, k_tea, 6**-0.5, key_tile=8)
    for tile in (1, 3, 5, 32):
        got = teacher_lse_from_qk(q_tea, k_tea, 6**-0.5, key_tile=tile)
        torch.testing.assert_close(got, ref)


def test_teacher_lse_matches_logsumexp():
    torch.manual_seed(0)
    q_tea = torch.randn(1, 2, 6, 5, dtype=torch.float64)
    k_tea = torch.randn(1, 2, 6, 5, dtype=torch.float64)
    scaling = 5**-0.5
    got = teacher_lse_from_qk(q_tea, k_tea, scaling, key_tile=4)
    expected = torch.logsumexp(torch.einsum("bhqd,bhkd->bhqk", q_tea, k_tea) * scaling, dim=-1)
    torch.testing.assert_close(got, expected)


def test_recompute_teacher_expands_kv_heads():
    """A GQA teacher may hand us KV heads; they must be repeated like attention does."""
    torch.manual_seed(0)
    q_tea = torch.randn(1, 8, 5, 4, dtype=torch.float64)
    k_kv = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    alpha = make_recompute_teacher(q_tea, k_kv, 1.0, 4)(0, 5)
    assert alpha.shape == (1, 8, 5, 5)
    # heads 0..3 share KV head 0
    expected0 = torch.einsum("bqd,bkd->bqk", q_tea[:, 0], k_kv[:, 0])
    torch.testing.assert_close(alpha[:, 0], expected0)
    expected4 = torch.einsum("bqd,bkd->bqk", q_tea[:, 4], k_kv[:, 1])
    torch.testing.assert_close(alpha[:, 4], expected4)


# ----------------------------------------------------------------------
# Forward equivalence
# ----------------------------------------------------------------------
@pytest.mark.parametrize("key_tile", [1, 2, 4, 9, 64])
def test_fused_loss_matches_dense_reference(key_tile):
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    rows = fused_indexer_ce_rows(
        q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=key_tile
    )
    expected, _, _ = dense_reference(
        q_idx, k_idx, teacher_alpha, lse, mask, group_size, k_idx.shape[1]
    )
    torch.testing.assert_close(rows, expected)


def test_fused_loss_is_tile_invariant():
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    ref = fused_indexer_ce_rows(
        q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=9
    )
    for tile in (1, 2, 3, 5, 128):
        got = fused_indexer_ce_rows(
            q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=tile
        )
        torch.testing.assert_close(got, ref)


def test_fused_loss_without_mask():
    q_idx, k_idx, teacher_alpha, lse, _, group_size, _ = make_case(causal=False)
    rows = fused_indexer_ce_rows(
        q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=None, key_tile=4
    )
    expected, _, _ = dense_reference(
        q_idx, k_idx, teacher_alpha, lse, None, group_size, k_idx.shape[1]
    )
    torch.testing.assert_close(rows, expected)


def test_cross_entropy_exceeds_kl_by_teacher_entropy():
    """
    The fused objective is CE, not KL. It differs from the true KL by exactly H(pbar),
    which is constant in the student -- same gradients, shifted curve.
    """
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    rows = fused_indexer_ce_rows(
        q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=4
    )
    _, p_bar, p_hat = dense_reference(
        q_idx, k_idx, teacher_alpha, lse, mask, group_size, k_idx.shape[1]
    )
    safe = p_bar.clamp_min(1e-300)
    kl = (p_bar * (safe.log() - p_hat.clamp_min(1e-300).log())).sum(-1)
    entropy = -(p_bar * safe.log()).sum(-1)
    torch.testing.assert_close(rows, kl + entropy)
    assert (rows >= kl - 1e-9).all()


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------
@pytest.mark.parametrize("key_tile", [2, 4, 9])
def test_gradients_match_autograd_through_the_dense_path(key_tile):
    """
    The analytic dQ/dK must equal what autograd derives from the dense formulation.

    This is the load-bearing test: forward accumulates dQ under a unit upstream gradient
    and backward runs a transposed pass for dK, so an error in either would show up only
    here.
    """
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    grad_out = torch.rand(q_idx.shape[0], q_idx.shape[1], q_idx.shape[2], dtype=torch.float64)

    rows = fused_indexer_ce_rows(
        q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=key_tile
    )
    (rows * grad_out).sum().backward()
    fused_dq, fused_dk = q_idx.grad.clone(), k_idx.grad.clone()

    q2 = q_idx.detach().clone().requires_grad_(True)
    k2 = k_idx.detach().clone().requires_grad_(True)
    dense_rows, _, _ = dense_reference(q2, k2, teacher_alpha, lse, mask, group_size, k_idx.shape[1])
    (dense_rows * grad_out).sum().backward()

    torch.testing.assert_close(fused_dq, q2.grad)
    torch.testing.assert_close(fused_dk, k2.grad)


def test_gradients_are_tile_invariant():
    grads = []
    for tile in (1, 3, 9):
        q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
        rows = fused_indexer_ce_rows(
            q_idx, k_idx, teacher_alpha, lse, group_size=group_size, mask=mask, key_tile=tile
        )
        rows.sum().backward()
        grads.append((q_idx.grad.clone(), k_idx.grad.clone()))
    for dq, dk in grads[1:]:
        torch.testing.assert_close(dq, grads[0][0])
        torch.testing.assert_close(dk, grads[0][1])


def test_teacher_receives_no_gradient():
    """The teacher is a frozen reference; nothing may flow back into it."""
    torch.manual_seed(0)
    q_tea = torch.randn(1, 4, 6, 5, dtype=torch.float64, requires_grad=True)
    k_tea = torch.randn(1, 4, 6, 5, dtype=torch.float64, requires_grad=True)
    q_idx = torch.randn(1, 2, 6, 4, dtype=torch.float64, requires_grad=True)
    k_idx = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)

    mask = build_indexer_mask(6, 6, q_idx.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea.detach(), k_tea.detach(), 5**-0.5, mask=mask)
    alpha_fn = make_recompute_teacher(q_tea.detach(), k_tea.detach(), 5**-0.5, 2)

    fused_indexer_ce_rows(
        q_idx, k_idx, alpha_fn, lse, group_size=2, mask=mask, key_tile=3
    ).sum().backward()
    assert q_tea.grad is None and k_tea.grad is None
    assert q_idx.grad is not None and k_idx.grad is not None


# ----------------------------------------------------------------------
# Module-level entry point
# ----------------------------------------------------------------------
def test_fused_indexer_loss_reaches_indexer_parameters():
    torch.manual_seed(0)
    indexer = GQAIndexer(GQAIndexerConfig(hidden_size=16, n_heads=2, head_dim=8)).double()
    hidden = torch.randn(1, 7, 16, dtype=torch.float64)
    q_tea = torch.randn(1, 4, 7, 8, dtype=torch.float64)
    k_tea = torch.randn(1, 4, 7, 8, dtype=torch.float64)

    mask = build_indexer_mask(7, 7, hidden.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea, k_tea, 8**-0.5, mask=mask)
    loss = fused_indexer_loss(
        indexer,
        hidden,
        make_recompute_teacher(q_tea, k_tea, 8**-0.5, 2),
        lse,
        group_size=2,
        mask=mask,
        key_tile=3,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in indexer.named_parameters():
        assert param.grad is not None, name
    assert any(p.grad.abs().sum() > 0 for p in indexer.parameters())


def test_fused_indexer_loss_honours_row_valid():
    torch.manual_seed(0)
    indexer = GQAIndexer(GQAIndexerConfig(hidden_size=16, n_heads=2, head_dim=8)).double()
    hidden = torch.randn(1, 6, 16, dtype=torch.float64)
    q_tea = torch.randn(1, 2, 6, 8, dtype=torch.float64)
    k_tea = torch.randn(1, 2, 6, 8, dtype=torch.float64)
    mask = build_indexer_mask(6, 6, hidden.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea, k_tea, 8**-0.5, mask=mask)
    alpha_fn = make_recompute_teacher(q_tea, k_tea, 8**-0.5, 1)

    kwargs = dict(group_size=1, mask=mask, key_tile=3)
    all_rows = fused_indexer_loss(indexer, hidden, alpha_fn, lse, **kwargs)
    subset = torch.zeros(1, 1, 6, dtype=torch.bool)
    subset[..., 3:] = True
    partial = fused_indexer_loss(indexer, hidden, alpha_fn, lse, row_valid=subset, **kwargs)
    assert not torch.allclose(all_rows, partial)


def test_fused_ce_rows_validates_shapes():
    q_idx, k_idx, teacher_alpha, lse, mask, group_size, _ = make_case()
    with pytest.raises(ValueError, match="q_idx must be"):
        fused_indexer_ce_rows(q_idx[0], k_idx, teacher_alpha, lse, group_size=group_size)
    with pytest.raises(ValueError, match="k_idx must be"):
        fused_indexer_ce_rows(q_idx, k_idx[0], teacher_alpha, lse, group_size=group_size)
    with pytest.raises(ValueError, match="key_tile must be positive"):
        fused_indexer_ce_rows(q_idx, k_idx, teacher_alpha, lse, group_size=group_size, key_tile=0)


def test_teacher_probs_rejects_indivisible_group_size():
    alpha = torch.randn(1, 5, 3, 3)
    lse = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match="not divisible"):
        teacher_probs_from_lse(alpha, lse, group_size=2)


# ----------------------------------------------------------------------
# Dtype handling
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_dtype_follows_input(dtype):
    """
    Accumulation upcasts low precision but must never DOWNCAST float64.

    An unconditional ``.float()`` silently truncated float64 inputs, which both discarded
    precision the caller asked for and broke gradient comparisons against a float64
    reference.
    """
    torch.manual_seed(0)
    q_idx = torch.randn(1, 2, 6, 4, dtype=dtype, requires_grad=True)
    k_idx = torch.randn(1, 6, 4, dtype=dtype, requires_grad=True)
    q_tea = torch.randn(1, 4, 6, 5, dtype=dtype)
    k_tea = torch.randn(1, 4, 6, 5, dtype=dtype)

    mask = build_indexer_mask(6, 6, q_idx.device, dtype=dtype)
    lse = teacher_lse_from_qk(q_tea, k_tea, 5**-0.5, mask=mask)
    assert lse.dtype == dtype

    rows = fused_indexer_ce_rows(
        q_idx, k_idx, make_recompute_teacher(q_tea, k_tea, 5**-0.5, 2), lse,
        group_size=2, mask=mask, key_tile=3,
    )
    assert rows.dtype == dtype
    rows.sum().backward()
    assert q_idx.grad.dtype == dtype and k_idx.grad.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_accumulate_in_fp32(dtype):
    torch.manual_seed(0)
    q_idx = torch.randn(1, 2, 6, 4, dtype=dtype)
    k_idx = torch.randn(1, 6, 4, dtype=dtype)
    q_tea = torch.randn(1, 2, 6, 5, dtype=dtype)
    k_tea = torch.randn(1, 2, 6, 5, dtype=dtype)

    mask = build_indexer_mask(6, 6, q_idx.device, dtype=torch.float32)
    lse = teacher_lse_from_qk(q_tea, k_tea, 5**-0.5, mask=mask)
    assert lse.dtype == torch.float32  # upcast, not left in half precision

    rows = fused_indexer_ce_rows(
        q_idx, k_idx, make_recompute_teacher(q_tea, k_tea, 5**-0.5, 1), lse,
        group_size=1, mask=mask, key_tile=3,
    )
    assert rows.dtype == torch.float32
    assert torch.isfinite(rows).all()


# ----------------------------------------------------------------------
# lse / mask compatibility guard
# ----------------------------------------------------------------------
def test_assert_lse_mask_compatible_accepts_causal_only():
    assert_lse_mask_compatible(None, "test")
    assert_lse_mask_compatible(torch.ones(2, 8, dtype=torch.long), "test")
    causal = build_indexer_mask(6, 6, torch.device("cpu"))
    assert_lse_mask_compatible(causal, "test")


def test_assert_lse_mask_compatible_rejects_padding():
    """
    flash-attn's lse only covers causal masking, so padding would leave the teacher rows
    un-normalized. That must fail loudly rather than train slightly wrong.
    """
    padded = torch.ones(2, 8, dtype=torch.long)
    padded[1, :3] = 0
    with pytest.raises(ValueError, match="un-normalized"):
        assert_lse_mask_compatible(padded, "flash-attn")

    causal = build_indexer_mask(6, 6, torch.device("cpu"))
    causal[..., 0] = -1e4  # extra key masking on top of causal
    with pytest.raises(ValueError, match="un-normalized"):
        assert_lse_mask_compatible(causal, "flash-attn")


def test_unnormalized_teacher_is_what_the_guard_prevents():
    """Pin the failure mode: masking after the lse loses probability mass."""
    torch.manual_seed(0)
    q_tea = torch.randn(1, 2, 6, 5, dtype=torch.float64)
    k_tea = torch.randn(1, 2, 6, 5, dtype=torch.float64)
    scaling = 5**-0.5
    mask = build_indexer_mask(6, 6, q_tea.device, dtype=torch.float64)

    good = teacher_probs_from_lse(
        make_recompute_teacher(q_tea, k_tea, scaling, 1)(0, 6) + mask,
        teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask),
        1,
    )
    torch.testing.assert_close(good.sum(-1), torch.ones_like(good.sum(-1)))

    # lse WITHOUT the mask, probabilities zeroed afterwards -> rows no longer sum to 1
    bad = teacher_probs_from_lse(
        make_recompute_teacher(q_tea, k_tea, scaling, 1)(0, 6),
        teacher_lse_from_qk(q_tea, k_tea, scaling),
        1,
    ) * (mask > -1e3)
    assert not torch.allclose(bad.sum(-1), torch.ones_like(bad.sum(-1)))


# ----------------------------------------------------------------------
# flash-attention contract
# ----------------------------------------------------------------------
def test_causal_mask_matches_flash_attn_bottom_right_alignment():
    """
    build_indexer_mask must use flash-attn's bottom-right causal alignment.

    A captured `lse` is only consistent with the loss if both use the same causal
    convention. Cases quoted verbatim from the flash_attn_func docstring: for
    seqlen_q=2/seqlen_k=5 the keep mask is [[1,1,1,1,0],[1,1,1,1,1]], and for
    seqlen_q=5/seqlen_k=2 it is [[0,0],[0,0],[0,0],[1,0],[1,1]]. Top-left alignment
    would differ, and nothing else in the pipeline would notice.
    """
    device = torch.device("cpu")

    wide = build_indexer_mask(2, 5, device) == 0
    expected_wide = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(wide[0, 0], expected_wide)

    tall = build_indexer_mask(5, 2, device) == 0
    expected_tall = torch.tensor(
        [[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]], dtype=torch.bool
    )
    torch.testing.assert_close(tall[0, 0], expected_tall)


def test_gqa_head_mapping_matches_flash_attn():
    """
    Query head i must read KV head i // group_size.

    flash_attn_func: "if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will
    attention to head 0 of K, V, and head 3, 4, 5 of Q will attention to head 1".
    """
    torch.manual_seed(0)
    q_tea = torch.randn(1, 6, 4, 3, dtype=torch.float64)
    k_kv = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    alpha = make_recompute_teacher(q_tea, k_kv, 1.0, 3)(0, 4)

    for q_head, kv_head in enumerate([0, 0, 0, 1, 1, 1]):
        expected = torch.einsum("bqd,bkd->bqk", q_tea[:, q_head], k_kv[:, kv_head])
        torch.testing.assert_close(alpha[:, q_head], expected)


# ----------------------------------------------------------------------
# lse layout normalization
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,bsz,n_heads,q_len",
    [((2, 4, 8), 2, 4, 8), ((2, 8, 4), 2, 4, 8), ((4, 8), 1, 4, 8), ((8, 4), 1, 4, 8)],
)
def test_normalize_captured_lse_layouts(shape, bsz, n_heads, q_len):
    out = normalize_captured_lse(torch.zeros(shape), bsz, n_heads, q_len)
    assert out.shape == (bsz, n_heads, q_len)


def test_normalize_captured_lse_transposes_correctly():
    lse = torch.arange(2 * 3 * 5, dtype=torch.float).view(2, 5, 3)  # (B, Sq, H)
    out = normalize_captured_lse(lse, bsz=2, n_heads=3, q_len=5)
    torch.testing.assert_close(out, lse.transpose(1, 2))


def test_normalize_captured_lse_rejects_unknown_layout():
    with pytest.raises(ValueError, match="cannot interpret"):
        normalize_captured_lse(torch.zeros(2, 7, 9), bsz=2, n_heads=4, q_len=8)
