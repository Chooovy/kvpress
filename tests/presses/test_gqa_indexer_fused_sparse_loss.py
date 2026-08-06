# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for stage-2 sparse indexer distillation.

Two independent things are pinned here: that the support selection matches a dense
``topk`` (and honours the forced slots), and that the KL over that support matches a
materialize-everything reference and autograd. Everything runs in float64 so gradient
comparisons are limited by the analytic derivation, not by fp32 rounding.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    build_dense_indexer_target,
    build_indexer_mask,
    build_sparse_indexer_target,
    expand_to_heads,
    forced_support_positions,
    fused_sparse_indexer_kl_rows,
    fused_sparse_indexer_loss,
    gather_support_keys,
    make_sparse_recompute_teacher,
    masked_log_softmax,
    resolve_topk,
    sort_support,
    streaming_topk_support,
    support_teacher_lse,
    teacher_lse_from_qk,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG


def make_case(bsz=2, h=3, group_size=2, q_len=9, k_len=9, dim=4, d_tea=5, topk=4, seed=0):
    """Student q/k, teacher q/k, mask and lse, all float64."""
    torch.manual_seed(seed)
    n_heads = h * group_size
    q_idx = torch.randn(bsz, h, q_len, dim, dtype=torch.float64, requires_grad=True)
    k_idx = torch.randn(bsz, k_len, dim, dtype=torch.float64, requires_grad=True)
    q_tea = torch.randn(bsz, n_heads, q_len, d_tea, dtype=torch.float64)
    k_tea = torch.randn(bsz, h, k_len, d_tea, dtype=torch.float64)
    scaling = d_tea**-0.5
    mask = build_indexer_mask(q_len, k_len, q_idx.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=4)
    return dict(
        q_idx=q_idx,
        k_idx=k_idx,
        q_tea=q_tea,
        k_tea=k_tea,
        scaling=scaling,
        mask=mask,
        lse=lse,
        group_size=group_size,
        h=h,
        n_heads=n_heads,
        topk=topk,
        k_len=k_len,
        q_len=q_len,
    )


def dense_sparse_reference(case, support, valid):
    """
    The naive stage-2 objective: build the dense teacher, gather, renormalize, KL.

    Deliberately routed through ``build_sparse_indexer_target`` and ``masked_log_softmax``
    -- the dense stage-2 path in ``train.py`` -- so this asserts the fused loss agrees with
    code that already existed rather than with a second copy of its own derivation.
    """
    logits = torch.einsum("bhqd,bkd->bhqk", case["q_idx"], case["k_idx"])
    k_rep = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    alpha = torch.einsum("bhqd,bhkd->bhqk", case["q_tea"], k_rep) * case["scaling"] + case["mask"]
    attn = torch.softmax(alpha, dim=-1)

    target = build_sparse_indexer_target(
        attn, support, n_kv_heads=case["h"], head_reduce="mean"
    )
    student = logits.gather(-1, support.clamp_min(0).long())
    log_q = masked_log_softmax(student, valid)
    rows = torch.where(
        valid, target * (torch.log(target.clamp_min(1e-10)) - log_q), torch.zeros_like(target)
    ).sum(-1)
    return rows, target, torch.exp(log_q) * valid


def build_teacher(case, support, valid, mode="global", topk_tile=None):
    return make_sparse_recompute_teacher(
        case["q_tea"],
        case["k_tea"],
        case["scaling"],
        case["group_size"],
        teacher_lse=case["lse"] if mode == "global" else None,
        teacher_mode=mode,
        support=support,
        valid=valid,
        topk_tile=topk_tile,
    )


def pick_support(case, **kw):
    return streaming_topk_support(
        case["q_idx"].detach(), case["k_idx"].detach(), case["topk"], mask=case["mask"], **kw
    )


# ----------------------------------------------------------------------
# Support selection
# ----------------------------------------------------------------------
@pytest.mark.parametrize("key_tile", [1, 2, 3, 9, 64])
def test_streaming_topk_matches_dense_topk(key_tile):
    """
    The streamed tournament merge must select the same keys as a dense topk.

    Compared as sorted *values*, not indices: among exactly-equal logits the tie-break may
    legitimately differ between a tiled merge and one global sort, and that difference is
    invisible to the objective.
    """
    case = make_case()
    support, valid = pick_support(case, key_tile=key_tile, query_tile=4)

    logits = torch.einsum("bhqd,bkd->bhqk", case["q_idx"], case["k_idx"]).detach()
    logits = logits.masked_fill(case["mask"] <= MASK_NEG / 2, -float("inf"))
    dense_v, _ = logits.topk(case["topk"], dim=-1)

    got = logits.gather(-1, support.clamp_min(0).long())
    got = torch.where(valid, got, torch.full_like(got, -float("inf")))
    torch.testing.assert_close(got.sort(-1).values, dense_v.sort(-1).values)


def test_support_is_sorted_and_free_of_duplicates():
    case = make_case(q_len=12, k_len=12, topk=5)
    support, valid = pick_support(case, key_tile=3, query_tile=5)

    filled = torch.where(valid, support, torch.full_like(support, case["k_len"]))
    assert (filled.diff(dim=-1) > 0).logical_or(filled[..., 1:] == case["k_len"]).all(), (
        "support must be strictly ascending, so no key is selected twice"
    )
    # empty slots may only trail
    assert (~valid).logical_or(valid.cummin(-1).values).all()


def test_support_never_selects_a_masked_key():
    case = make_case(q_len=8, k_len=8, topk=4)
    support, valid = pick_support(case, key_tile=3, query_tile=3)
    keep = case["mask"] > MASK_NEG / 2  # (1, 1, Sq, Sk)
    keep = keep.expand(support.shape[0], support.shape[1], *keep.shape[-2:])
    picked_allowed = keep.gather(-1, support.clamp_min(0).long())
    assert (picked_allowed | ~valid).all(), "a causally-masked key entered the support"


def test_support_fills_every_available_slot():
    """A row must keep min(topk, #visible keys) slots -- no silent under-filling."""
    case = make_case(q_len=6, k_len=6, topk=4)
    support, valid = pick_support(case, key_tile=2, query_tile=2)
    visible = (case["mask"] > MASK_NEG / 2).sum(-1)  # (1, 1, Sq)
    expected = visible.clamp(max=case["topk"]).expand(support.shape[0], support.shape[1], -1)
    torch.testing.assert_close(valid.sum(-1), expected)


@pytest.mark.parametrize("force_sink,force_local", [(2, 0), (0, 3), (2, 2), (1, 1)])
def test_forced_positions_are_always_selected(force_sink, force_local):
    """Forced slots must survive even when their logits are the worst in the row."""
    case = make_case(q_len=10, k_len=10, topk=6)
    support, valid = pick_support(
        case, force_sink=force_sink, force_local=force_local, key_tile=3, query_tile=4
    )
    q_len, k_len = case["q_len"], case["k_len"]
    offset = k_len - q_len

    for t in range(q_len):
        limit = t + offset
        want = {s for s in range(force_sink) if s <= limit}
        want |= {s for s in range(max(0, limit - force_local + 1), limit + 1)}
        got = set(support[0, 0, t][valid[0, 0, t]].tolist())
        assert want <= got, f"row {t}: forced {sorted(want - got)} missing from {sorted(got)}"

    filled = torch.where(valid, support, torch.full_like(support, k_len))
    assert (filled.diff(dim=-1) > 0).logical_or(filled[..., 1:] == k_len).all(), (
        "forcing must not duplicate a key the topk would also have picked"
    )


def test_forced_positions_beyond_topk_raise():
    """
    Silently truncating the forced set would be worse than failing.

    A row that drops its local block trains a different objective than configured, with no
    signal anywhere.
    """
    case = make_case(topk=3)
    with pytest.raises(ValueError, match="exceeds topk"):
        pick_support(case, force_sink=2, force_local=2)


def test_forced_support_positions_marks_unreachable_slots():
    q_index = torch.arange(4)
    forced = forced_support_positions(q_index, force_sink=3, force_local=2, query_offset=0, k_len=8)
    # row 0 sees only key 0: the other two sinks and the extra local slot are unusable
    assert forced[0].tolist() == [0, -1, -1, -1, -1]
    # row 3 sees keys 0..3: all sinks fit, and local {2, 3} does not collide with sink [0,3)
    assert forced[3].tolist() == [0, 1, 2, -1, 3]


def test_forced_local_deduplicates_against_the_sink():
    """The sink block is exactly [0, force_sink), so overlap is detectable exactly."""
    forced = forced_support_positions(
        torch.tensor([1]), force_sink=4, force_local=3, query_offset=0, k_len=8
    )
    sink, local = forced[0, :4].tolist(), forced[0, 4:].tolist()
    assert sink == [0, 1, -1, -1]  # row 1 sees keys 0..1
    assert local == [-1, -1, -1], "keys 0 and 1 are already in the sink block"


def test_resolve_topk_and_sort_support():
    assert resolve_topk(100, None, 0.25) == 25
    assert resolve_topk(100, 8, 0.25) == 8
    assert resolve_topk(4, 99, 0.5) == 4, "topk must be clamped to the key axis"
    assert resolve_topk(100, None, 0.001) == 1, "never degenerate to an empty support"

    support, valid = sort_support(torch.tensor([[[[5, -1, 2, -1, 9]]]]), k_len=10)
    assert support[0, 0, 0].tolist() == [2, 5, 9, -1, -1]
    assert valid[0, 0, 0].tolist() == [True, True, True, False, False]


def test_streaming_topk_without_a_mask_uses_causal_arithmetic():
    """mask=None must give the same result as an explicit causal mask, at no memory cost."""
    case = make_case(q_len=7, k_len=7, topk=3)
    with_mask, v1 = pick_support(case, key_tile=3, query_tile=3)
    without, v2 = streaming_topk_support(
        case["q_idx"].detach(), case["k_idx"].detach(), case["topk"], key_tile=3, query_tile=3
    )
    torch.testing.assert_close(with_mask, without)
    torch.testing.assert_close(v1, v2)


def test_streaming_topk_handles_bottom_right_alignment():
    """Sq != Sk must align to the bottom-right corner, as flash-attention does."""
    case = make_case(q_len=4, k_len=9, topk=3)
    support, valid = pick_support(case, key_tile=4, query_tile=2)
    offset = 9 - 4
    for t in range(4):
        picked = support[0, 0, t][valid[0, 0, t]]
        assert (picked <= t + offset).all(), f"row {t} selected a future key"


def test_gather_support_keys_matches_indexing():
    case = make_case()
    support, _ = pick_support(case, key_tile=4)
    gathered = gather_support_keys(case["k_idx"].detach(), support)
    for b in (0, 1):
        for hh in range(case["h"]):
            for t in (0, case["q_len"] - 1):
                for j in range(support.shape[-1]):
                    idx = support[b, hh, t, j].clamp_min(0)
                    torch.testing.assert_close(gathered[b, hh, t, j], case["k_idx"][b, idx].detach())


def test_gather_support_keys_rejects_head_mismatch():
    case = make_case()
    support, _ = pick_support(case, key_tile=4)
    bad = torch.randn(2, case["h"] + 1, case["k_len"], 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="heads"):
        gather_support_keys(bad, support)


# ----------------------------------------------------------------------
# The loss against a dense reference
# ----------------------------------------------------------------------
@pytest.mark.parametrize("topk_tile", [1, 2, 3, 4, 64])
def test_sparse_loss_matches_dense_reference(topk_tile):
    case = make_case()
    support, valid = pick_support(case, key_tile=4)
    rows = fused_sparse_indexer_kl_rows(
        case["q_idx"],
        case["k_idx"],
        support,
        valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"],
        query_tile=3,
        topk_tile=topk_tile,
    )
    expected, _, _ = dense_sparse_reference(case, support, valid)
    torch.testing.assert_close(rows, expected)


@pytest.mark.parametrize("query_tile,topk_tile", [(1, 1), (2, 3), (3, 2), (5, 4), (64, 64)])
def test_sparse_loss_is_tile_invariant(query_tile, topk_tile):
    case = make_case()
    support, valid = pick_support(case, key_tile=4)
    teacher = build_teacher(case, support, valid)
    base = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid, teacher,
        group_size=case["group_size"], query_tile=64, topk_tile=64,
    )
    got = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid, teacher,
        group_size=case["group_size"], query_tile=query_tile, topk_tile=topk_tile,
    )
    torch.testing.assert_close(got, base)


def test_sparse_kl_is_non_negative():
    """
    Stage 2 reports full KL, so it must be >= 0 -- the check stage 1's CE cannot make.

    This is the assertion that would catch a mis-derived normalizer: a missing ``log Z``
    shows up immediately as a negative KL.
    """
    case = make_case(q_len=12, k_len=12, topk=5)
    support, valid = pick_support(case, key_tile=4)
    rows = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=4, topk_tile=2,
    )
    assert (rows[valid.any(-1)] >= -1e-9).all(), f"negative KL: min {rows.min().item()}"


def test_full_support_reproduces_the_dense_stage_one_objective():
    """
    With topk == k_len the sparse path must reduce to stage 1's KL exactly.

    A strong end-to-end check: the two implementations share no code, so agreeing here
    means both the teacher reconstruction and the normalizer are right.
    """
    case = make_case(q_len=7, k_len=7, topk=7)
    support, valid = pick_support(case, key_tile=7)
    rows = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=3, topk_tile=3,
    )

    logits = torch.einsum("bhqd,bkd->bhqk", case["q_idx"], case["k_idx"]) + case["mask"]
    k_rep = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    alpha = torch.einsum("bhqd,bhkd->bhqk", case["q_tea"], k_rep) * case["scaling"] + case["mask"]
    keep = (case["mask"] > MASK_NEG / 2).expand_as(logits)
    target = build_dense_indexer_target(
        torch.softmax(alpha, -1), keep, n_kv_heads=case["h"], head_reduce="mean"
    )
    log_q = masked_log_softmax(logits, keep)
    expected = torch.where(
        keep, target * (torch.log(target.clamp_min(1e-10)) - log_q), torch.zeros_like(target)
    ).sum(-1)
    torch.testing.assert_close(rows, expected)


def test_recall_reports_the_captured_teacher_mass():
    case = make_case(q_len=10, k_len=10, topk=3)
    support, valid = pick_support(case, key_tile=4)
    stats: dict = {}
    fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=4, topk_tile=2, stats=stats,
    )
    _, target, _ = dense_sparse_reference(case, support, valid)

    k_rep = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    alpha = torch.einsum("bhqd,bhkd->bhqk", case["q_tea"], k_rep) * case["scaling"] + case["mask"]
    attn = torch.softmax(alpha, -1)
    grouped = attn.view(*attn.shape[:1], case["h"], case["group_size"], *attn.shape[2:]).mean(2)
    expected = torch.where(
        valid, grouped.gather(-1, support.clamp_min(0).long()), torch.zeros_like(target)
    ).sum(-1)
    torch.testing.assert_close(stats["recall"], expected)
    assert (stats["recall"] <= 1.0 + 1e-9).all(), "recall is a probability mass"


def test_recall_is_one_when_the_support_is_everything():
    case = make_case(q_len=6, k_len=6, topk=6)
    support, valid = pick_support(case, key_tile=6)
    stats: dict = {}
    fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=6, topk_tile=6, stats=stats,
    )
    torch.testing.assert_close(stats["recall"], torch.ones_like(stats["recall"]))


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------
@pytest.mark.parametrize("query_tile,topk_tile", [(1, 1), (2, 3), (3, 2), (64, 64)])
def test_gradients_match_autograd_through_the_dense_path(query_tile, topk_tile):
    """
    The hand-derived dQ/dK must equal what autograd produces from the dense formulation.

    ``dK`` is the interesting half: the forward pass accumulates only ``dQ``, and ``dK`` is
    scattered in a second transposed pass with ``index_add``.
    """
    case = make_case()
    support, valid = pick_support(case, key_tile=4)
    rows = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=query_tile, topk_tile=topk_tile,
    )
    torch.manual_seed(1)
    upstream = torch.randn_like(rows)
    (rows * upstream).sum().backward()
    got_q, got_k = case["q_idx"].grad.clone(), case["k_idx"].grad.clone()

    q2 = case["q_idx"].detach().clone().requires_grad_(True)
    k2 = case["k_idx"].detach().clone().requires_grad_(True)
    ref_case = dict(case, q_idx=q2, k_idx=k2)
    ref_rows, _, _ = dense_sparse_reference(ref_case, support, valid)
    (ref_rows * upstream).sum().backward()

    torch.testing.assert_close(got_q, q2.grad)
    torch.testing.assert_close(got_k, k2.grad)


def test_gradcheck_on_a_small_case():
    """
    Independent confirmation via finite differences, with the support held fixed.

    The support has to be frozen outside the checked function: top-k is piecewise constant,
    so perturbing an input across a selection boundary makes the numeric derivative
    meaningless while the analytic one stays correct.
    """
    case = make_case(bsz=1, h=2, group_size=2, q_len=5, k_len=5, dim=3, d_tea=3, topk=3, seed=4)
    support, valid = pick_support(case, key_tile=5)
    teacher = build_teacher(case, support, valid)

    def fn(q, k):
        return fused_sparse_indexer_kl_rows(
            q, k, support, valid, teacher,
            group_size=case["group_size"], query_tile=2, topk_tile=2,
        )

    assert torch.autograd.gradcheck(
        fn, (case["q_idx"], case["k_idx"]), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_dk_accumulates_across_query_tiles():
    """
    dK sums over queries, so every query tile must ADD into it.

    A missing accumulation leaves only the last tile's contribution while the loss stays
    perfectly correct -- silent, and exactly the bug this asserts against.
    """
    case = make_case(q_len=8, k_len=8, topk=4)
    support, valid = pick_support(case, key_tile=4)
    teacher = build_teacher(case, support, valid)

    grads = []
    for query_tile in (8, 4, 2, 1):
        q = case["q_idx"].detach().clone().requires_grad_(True)
        k = case["k_idx"].detach().clone().requires_grad_(True)
        fused_sparse_indexer_kl_rows(
            q, k, support, valid, teacher,
            group_size=case["group_size"], query_tile=query_tile, topk_tile=2,
        ).sum().backward()
        grads.append(k.grad.clone())
    for other in grads[1:]:
        torch.testing.assert_close(other, grads[0])
    assert grads[0].abs().sum() > 0


def test_teacher_receives_no_gradient():
    case = make_case()
    case["q_tea"].requires_grad_(True)
    case["k_tea"].requires_grad_(True)
    support, valid = pick_support(case, key_tile=4)
    fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid),
        group_size=case["group_size"], query_tile=3, topk_tile=2,
    ).sum().backward()
    assert case["q_tea"].grad is None
    assert case["k_tea"].grad is None


# ----------------------------------------------------------------------
# teacher_mode
# ----------------------------------------------------------------------
def test_support_mode_teacher_rows_sum_to_one():
    """Support-mode normalizes over the support, so Z is identically 1."""
    case = make_case(q_len=8, k_len=8, topk=3)
    support, valid = pick_support(case, key_tile=4)
    stats: dict = {}
    fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="support", topk_tile=2),
        group_size=case["group_size"], query_tile=3, topk_tile=2, stats=stats,
    )
    live = valid.any(-1)
    torch.testing.assert_close(stats["recall"][live], torch.ones_like(stats["recall"][live]))


def test_support_mode_matches_an_explicit_sparse_softmax():
    case = make_case(q_len=7, k_len=7, topk=3)
    support, valid = pick_support(case, key_tile=4)
    rows = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="support"),
        group_size=case["group_size"], query_tile=3, topk_tile=2,
    )

    # reference: per-head softmax over the support, then group mean
    k_rep = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    dense_alpha = torch.einsum("bhqd,bhkd->bhqk", case["q_tea"], k_rep) * case["scaling"]
    sup_h = support.repeat_interleave(case["group_size"], dim=1)
    val_h = valid.repeat_interleave(case["group_size"], dim=1)
    alpha = dense_alpha.gather(-1, sup_h.clamp_min(0).long())
    probs = torch.softmax(alpha.masked_fill(~val_h, -float("inf")), dim=-1)
    probs = probs.masked_fill(~val_h, 0.0)
    target = probs.view(
        probs.shape[0], case["h"], case["group_size"], *probs.shape[2:]
    ).mean(2)

    logits = torch.einsum("bhqd,bkd->bhqk", case["q_idx"], case["k_idx"])
    log_q = masked_log_softmax(logits.gather(-1, support.clamp_min(0).long()), valid)
    expected = torch.where(
        valid, target * (torch.log(target.clamp_min(1e-10)) - log_q), torch.zeros_like(target)
    ).sum(-1)
    torch.testing.assert_close(rows, expected)


def test_the_two_teacher_modes_are_different_objectives():
    """
    Not two ways to compute one thing.

    They coincide only when every head in a group captures the same support mass, which is
    not the case in general -- so picking a mode is a modelling decision, and the docs say so.
    """
    case = make_case(q_len=10, k_len=10, topk=3)
    support, valid = pick_support(case, key_tile=4)
    kw = dict(group_size=case["group_size"], query_tile=4, topk_tile=2)
    a = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="global"), **kw
    )
    b = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="support"), **kw
    )
    assert not torch.allclose(a, b, atol=1e-6)


def test_modes_agree_when_the_support_is_the_whole_axis():
    """With topk == k_len both normalizers span the same keys, so they must coincide."""
    case = make_case(q_len=6, k_len=6, topk=6)
    support, valid = pick_support(case, key_tile=6)
    kw = dict(group_size=case["group_size"], query_tile=3, topk_tile=3)
    a = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="global"), **kw
    )
    b = fused_sparse_indexer_kl_rows(
        case["q_idx"], case["k_idx"], support, valid,
        build_teacher(case, support, valid, mode="support"), **kw
    )
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("topk_tile", [1, 2, 3, 64])
def test_support_teacher_lse_is_tile_invariant(topk_tile):
    torch.manual_seed(0)
    alpha = torch.randn(2, 4, 5, 6, dtype=torch.float64)
    valid = torch.rand(2, 4, 5, 6) > 0.3
    valid[..., 0] = True
    got = support_teacher_lse(alpha, valid, topk_tile=topk_tile)
    expected = torch.logsumexp(alpha.masked_fill(~valid, -float("inf")), dim=-1)
    torch.testing.assert_close(got, expected)


# ----------------------------------------------------------------------
# Head expansion
# ----------------------------------------------------------------------
def test_expand_to_heads_matches_the_group_mean_layout():
    """
    ``expand`` cannot turn h into H -- both are non-singleton, so it raises.

    The expansion also has to produce the exact layout ``view(B, h, g, ...)`` assumes,
    or the group mean would average the wrong heads together: a silent correctness bug
    that leaves every shape intact.
    """
    bsz, kv_heads, group_size, dq, tk = 2, 3, 4, 5, 6
    n_heads = kv_heads * group_size
    valid = torch.rand(bsz, kv_heads, dq, tk) > 0.4

    with pytest.raises(RuntimeError):
        valid.expand(bsz, n_heads, dq, tk)

    got = expand_to_heads(valid, n_heads, group_size)
    assert got.shape == (bsz, n_heads, dq, tk)
    torch.testing.assert_close(got, valid.repeat_interleave(group_size, dim=1))
    # round-tripping through the group axis must recover the original
    back = got.view(bsz, kv_heads, group_size, dq, tk)
    for i in range(group_size):
        torch.testing.assert_close(back[:, :, i], valid)


def test_expand_to_heads_is_a_noop_when_already_expanded():
    x = torch.randn(2, 8, 3, 4)
    assert expand_to_heads(x, 8, 4) is x


def test_expand_to_heads_rejects_an_impossible_expansion():
    with pytest.raises(ValueError, match="cannot expand"):
        expand_to_heads(torch.randn(2, 3, 4, 5), n_heads=8, group_size=2)


def test_support_teacher_lse_accepts_a_per_kv_head_valid():
    """The caller holds one support per KV group, so valid arrives with h heads, not H."""
    torch.manual_seed(0)
    bsz, kv_heads, group_size, dq, tk = 1, 2, 3, 4, 5
    n_heads = kv_heads * group_size
    alpha = torch.randn(bsz, n_heads, dq, tk, dtype=torch.float64)
    valid = torch.rand(bsz, kv_heads, dq, tk) > 0.3
    valid[..., 0] = True

    got = support_teacher_lse(alpha, valid, group_size=group_size, topk_tile=2)
    expanded = valid.repeat_interleave(group_size, dim=1)
    expected = torch.logsumexp(alpha.masked_fill(~expanded, -float("inf")), dim=-1)
    torch.testing.assert_close(got, expected)


def test_support_teacher_lse_keeps_empty_rows_finite():
    """An all-invalid row would give -inf, poisoning every later exp."""
    alpha = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    valid = torch.ones(1, 2, 3, 4, dtype=torch.bool)
    valid[0, 0, 1] = False
    for tile in (1, 2, 4):
        lse = support_teacher_lse(alpha, valid, topk_tile=tile)
        assert torch.isfinite(lse).all()
        assert lse[0, 0, 1] == 0.0


def test_global_mode_requires_a_teacher_lse():
    case = make_case()
    support, valid = pick_support(case, key_tile=4)
    with pytest.raises(ValueError, match="needs teacher_lse"):
        make_sparse_recompute_teacher(
            case["q_tea"], case["k_tea"], case["scaling"], case["group_size"],
            teacher_mode="global", support=support, valid=valid,
        )


def test_support_mode_requires_the_full_support():
    """A row's normalizer spans all its slots, so it cannot come from one tile."""
    case = make_case()
    with pytest.raises(ValueError, match="full support"):
        make_sparse_recompute_teacher(
            case["q_tea"], case["k_tea"], case["scaling"], case["group_size"],
            teacher_mode="support",
        )


def test_unknown_teacher_mode_raises():
    case = make_case()
    with pytest.raises(ValueError, match="teacher_mode"):
        make_sparse_recompute_teacher(
            case["q_tea"], case["k_tea"], case["scaling"], case["group_size"],
            teacher_mode="mixture",
        )


# ----------------------------------------------------------------------
# The module-level entry point
# ----------------------------------------------------------------------
def build_indexer(hidden=16, n_heads=3, head_dim=4, dtype=torch.float64):
    indexer = GQAIndexer(
        GQAIndexerConfig(hidden_size=hidden, n_heads=n_heads, head_dim=head_dim, rope_dim=0)
    )
    return indexer.to(dtype)


def test_fused_sparse_indexer_loss_reaches_indexer_parameters():
    torch.manual_seed(0)
    bsz, q_len, hidden, h, group_size, d_tea = 2, 8, 16, 3, 2, 5
    indexer = build_indexer(hidden, h)
    hidden_states = torch.randn(bsz, q_len, hidden, dtype=torch.float64)
    q_tea = torch.randn(bsz, h * group_size, q_len, d_tea, dtype=torch.float64)
    k_tea = torch.randn(bsz, h, q_len, d_tea, dtype=torch.float64)
    scaling = d_tea**-0.5
    mask = build_indexer_mask(q_len, q_len, hidden_states.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=4)

    with torch.no_grad():
        support, valid = streaming_topk_support(
            indexer.project_q(hidden_states), indexer.project_k(hidden_states), 4,
            mask=mask, key_tile=4,
        )
    teacher = make_sparse_recompute_teacher(
        q_tea, k_tea, scaling, group_size, teacher_lse=lse,
        teacher_mode="global", support=support, valid=valid,
    )
    loss = fused_sparse_indexer_loss(
        indexer, hidden_states, support, valid, teacher,
        group_size=group_size, query_tile=3, topk_tile=2, loss_coeff=2.0,
    )
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    for name in ("w_q", "w_k"):
        grad = getattr(indexer, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} received no gradient"


def test_fused_sparse_indexer_loss_honours_row_valid():
    torch.manual_seed(0)
    bsz, q_len, hidden, h, group_size, d_tea = 1, 6, 16, 2, 2, 4
    indexer = build_indexer(hidden, h)
    hidden_states = torch.randn(bsz, q_len, hidden, dtype=torch.float64)
    q_tea = torch.randn(bsz, h * group_size, q_len, d_tea, dtype=torch.float64)
    k_tea = torch.randn(bsz, h, q_len, d_tea, dtype=torch.float64)
    scaling = d_tea**-0.5
    mask = build_indexer_mask(q_len, q_len, hidden_states.device, dtype=torch.float64)
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=3)
    with torch.no_grad():
        support, valid = streaming_topk_support(
            indexer.project_q(hidden_states), indexer.project_k(hidden_states), 3,
            mask=mask, key_tile=3,
        )
    teacher = make_sparse_recompute_teacher(
        q_tea, k_tea, scaling, group_size, teacher_lse=lse,
        teacher_mode="global", support=support, valid=valid,
    )
    kw = dict(group_size=group_size, query_tile=3, topk_tile=2)
    everything = fused_sparse_indexer_loss(
        indexer, hidden_states, support, valid, teacher,
        row_valid=torch.ones(bsz, h, q_len, dtype=torch.bool), **kw
    )
    subset_rows = torch.zeros(bsz, h, q_len, dtype=torch.bool)
    subset_rows[..., :2] = True
    subset = fused_sparse_indexer_loss(
        indexer, hidden_states, support, valid, teacher, row_valid=subset_rows, **kw
    )
    assert not torch.allclose(everything, subset)
    assert torch.isfinite(subset)


def test_sparse_rows_validates_shapes():
    case = make_case()
    support, valid = pick_support(case, key_tile=4)
    teacher = build_teacher(case, support, valid)
    kw = dict(group_size=case["group_size"])

    with pytest.raises(ValueError, match=r"q_idx must be"):
        fused_sparse_indexer_kl_rows(
            case["q_idx"][0], case["k_idx"], support, valid, teacher, **kw
        )
    with pytest.raises(ValueError, match=r"k_idx must be"):
        fused_sparse_indexer_kl_rows(
            case["q_idx"], case["k_idx"][0], support, valid, teacher, **kw
        )
    with pytest.raises(ValueError, match="support"):
        fused_sparse_indexer_kl_rows(
            case["q_idx"], case["k_idx"], support, valid[..., :-1], teacher, **kw
        )
    with pytest.raises(ValueError, match="does not match q_idx"):
        fused_sparse_indexer_kl_rows(
            case["q_idx"], case["k_idx"], support[:, :-1], valid[:, :-1], teacher, **kw
        )
    with pytest.raises(ValueError, match="must be positive"):
        fused_sparse_indexer_kl_rows(
            case["q_idx"], case["k_idx"], support, valid, teacher, query_tile=0, **kw
        )


def test_streaming_topk_validates_shapes():
    case = make_case()
    with pytest.raises(ValueError, match=r"q_idx must be"):
        streaming_topk_support(case["q_idx"][0], case["k_idx"], 3)
    with pytest.raises(ValueError, match=r"k_idx must be"):
        streaming_topk_support(case["q_idx"], case["k_idx"][0], 3)
    with pytest.raises(ValueError, match="must be positive"):
        streaming_topk_support(case["q_idx"], case["k_idx"], 3, key_tile=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_dtype_follows_input(dtype):
    torch.manual_seed(0)
    bsz, h, group_size, q_len, dim, d_tea = 1, 2, 2, 6, 3, 4
    q_idx = torch.randn(bsz, h, q_len, dim, dtype=dtype, requires_grad=True)
    k_idx = torch.randn(bsz, q_len, dim, dtype=dtype, requires_grad=True)
    q_tea = torch.randn(bsz, h * group_size, q_len, d_tea, dtype=dtype)
    k_tea = torch.randn(bsz, h, q_len, d_tea, dtype=dtype)
    mask = build_indexer_mask(q_len, q_len, q_idx.device, dtype=dtype)
    lse = teacher_lse_from_qk(q_tea, k_tea, d_tea**-0.5, mask=mask, key_tile=3)
    with torch.no_grad():
        support, valid = streaming_topk_support(q_idx, k_idx, 3, mask=mask, key_tile=3)
    teacher = make_sparse_recompute_teacher(
        q_tea, k_tea, d_tea**-0.5, group_size, teacher_lse=lse,
        teacher_mode="global", support=support, valid=valid,
    )
    rows = fused_sparse_indexer_kl_rows(
        q_idx, k_idx, support, valid, teacher, group_size=group_size, query_tile=3, topk_tile=2
    )
    assert rows.dtype == dtype, "fp64 must not be silently downcast to fp32"
    rows.sum().backward()
    assert q_idx.grad.dtype == dtype
    assert k_idx.grad.dtype == dtype


# ----------------------------------------------------------------------
# Storage width
# ----------------------------------------------------------------------
def test_support_is_stored_as_int32():
    """
    The support is the largest retained tensor in stage 2, so int64 doubles the dominant term.

    int32 addresses 2.1e9 keys, far past any real sequence, and gather/index_add cast to int64
    per tile -- an O(query_tile * topk_tile) transient rather than an O(L * topk) resident.
    """
    case = make_case(q_len=8, k_len=8, topk=3)
    support, valid = pick_support(case, key_tile=4)
    assert support.dtype == torch.int32
    assert valid.dtype == torch.bool

    # The values must survive the narrowing, sentinel included.
    logits = torch.einsum("bhqd,bkd->bhqk", case["q_idx"], case["k_idx"]).detach()
    logits = logits.masked_fill(case["mask"] <= MASK_NEG / 2, -float("inf"))
    gathered = logits.gather(-1, support.clamp_min(0).long())
    assert torch.isfinite(gathered[valid]).all(), "a narrowed index pointed at a masked key"
    assert (support[~valid] == -1).all(), "the -1 sentinel must survive the cast"


def test_sort_support_rejects_sequences_beyond_int32():
    """Silently wrapping would corrupt the support with no error anywhere downstream."""
    with pytest.raises(ValueError, match="exceeds int32"):
        sort_support(torch.zeros(1, 1, 1, 1, dtype=torch.long), k_len=2**31)


def test_gather_support_keys_accepts_both_index_widths():
    """Callers holding int64 support (e.g. a hand-built one) must keep working."""
    case = make_case(q_len=6, k_len=6, topk=3)
    support, _ = pick_support(case, key_tile=3)
    keys = case["k_idx"].detach()
    from_int32 = gather_support_keys(keys, support)
    from_int64 = gather_support_keys(keys, support.long())
    torch.testing.assert_close(from_int32, from_int64)


def test_teacher_gathers_once_per_kv_head():
    """
    A KV group shares one support, so the teacher must gather at h heads, not H.

    Gathering at H would build a group_size-times larger tile for identical arithmetic -- and
    the gathered tile is stage 2's dominant transient. Verified by value: the result must equal
    an explicit repeat_interleave reference.
    """
    case = make_case(q_len=7, k_len=7, topk=3)
    support, valid = pick_support(case, key_tile=4)
    teacher = build_teacher(case, support, valid)
    alpha, _ = teacher(0, case["q_len"], support, valid)

    # Reference: expand the keys to H heads first, then gather per attention head.
    k_rep = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    sup_h = support.repeat_interleave(case["group_size"], dim=1)
    expected = torch.einsum(
        "bhqd,bhqtd->bhqt", case["q_tea"], gather_support_keys(k_rep, sup_h)
    ) * case["scaling"]
    torch.testing.assert_close(alpha, expected)
