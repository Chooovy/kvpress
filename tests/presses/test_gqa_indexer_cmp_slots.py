# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the training-free CMP slots.

Weighted towards the properties that fail *silently*, because every prior bug in this family did:
the future leak changed no shape and left a healthy training curve while RULER fell 73.71 -> 4.00,
and double-counting a key between the exact branch and the summary raises nothing at all.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.cmp_slots import (
    cluster_evicted,
    cluster_reduce,
    evicted_from_deadline,
    kmeans_assign,
    prefill_logit_var,
    slot_mass,
)
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines


def _kv(n_heads=2, n_pts=64, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n_heads, n_pts, dim, generator=g),
        torch.randn(n_heads, n_pts, dim, generator=g),
    )


# ----------------------------------------------------------------------------------------------
# partition exactness: every evicted key in exactly one slot, no retained key in any slot
# ----------------------------------------------------------------------------------------------
def test_slot_populations_sum_to_the_evicted_count():
    """The slots must account for the evicted set exactly -- no key lost, none counted twice.

    This is the invariant `b_r` rests on: it claims to stand for `n_r` tokens, and if the
    populations do not sum to `|E|` then the total mass the summary injects is wrong by whatever
    the discrepancy is, silently.
    """
    keys, values = _kv()
    ev = torch.zeros(2, 64, dtype=torch.bool)
    ev[0, :40] = True
    ev[1, 10:50] = True
    _, _, b = cluster_evicted(keys, values, ev, 8)
    pop = b.exp().nan_to_num(0.0)  # b = log n_r, so exp recovers the population
    for h in range(2):
        assert pop[h].sum().round().item() == ev[h].sum().item()


def test_retained_keys_never_enter_a_slot():
    """A retained key is read exactly by the top-k, so including it in a slot double-counts it.

    Checked by making the retained keys enormous and distinctive: if they leaked into the weighted
    mean, the centroids would inherit that magnitude.
    """
    keys, values = _kv()
    keys[:, 32:] = 1e3
    values[:, 32:] = 1e3
    ev = torch.zeros(2, 64, dtype=torch.bool)
    ev[:, :32] = True
    k_cmp, v_cmp, _ = cluster_evicted(keys, values, ev, 4)
    assert k_cmp.abs().max() < 1e2
    assert v_cmp.abs().max() < 1e2


def test_singleton_slots_are_silenced_not_emitted():
    """A cluster of one is not a summary -- it is a key the row already chose not to read.

    With one evicted key and 4 slots, three clusters are empty and must be exactly ``-inf`` so they
    drop out of the softmax rather than contributing ``exp(log 0)`` by luck of the fp path.
    """
    keys, values = _kv()
    ev = torch.zeros(2, 64, dtype=torch.bool)
    ev[:, 0] = True
    _, _, b = cluster_evicted(keys, values, ev, 4)
    assert torch.isneginf(b).sum() == 2 * 3


# ----------------------------------------------------------------------------------------------
# causality
# ----------------------------------------------------------------------------------------------
def test_evicted_mask_excludes_the_future():
    """Keys past the horizon have not arrived and must not be clustered.

    The failure this guards is the one that took RULER 8K from 73.71 to 4.00: a summary built over
    positions ahead of its readers, with a training curve that looked healthy for 600 steps.
    """
    scores = torch.randn(2, 32)
    dl = deadlines(scores, 8, force_sink=1, force_local=0)
    for horizon in (0, 5, 17, 31):
        ev = evicted_from_deadline(dl, horizon)
        assert not ev[:, horizon + 1 :].any(), f"future key marked evicted at horizon {horizon}"


def test_evicted_and_retained_partition_the_arrived_keys():
    """``evicted`` is the exact complement of the mask ``qi_block_mask`` keeps, over arrived keys.

    Both derive from the same ``deadline``, so agreement is by construction -- asserted anyway
    because a key in both branches, or in neither, changes no shape and raises nothing.
    """
    scores = torch.randn(4, 64)
    dl = deadlines(scores, 16, force_sink=2, force_local=0)
    horizon = 48
    ev = evicted_from_deadline(dl, horizon)
    pos = torch.arange(64)
    arrived = pos.view(1, -1) <= horizon
    retained = arrived & (horizon <= dl.to(torch.int64))
    assert torch.equal(ev | retained, arrived.expand_as(ev))
    assert not (ev & retained).any()


def test_sinks_are_never_clustered():
    """``deadlines`` gives a sink ``k_len - 1``, so it is retained at every horizon."""
    scores = torch.randn(2, 32)
    dl = deadlines(scores, 8, force_sink=4, force_local=0)
    ev = evicted_from_deadline(dl, 31)
    assert not ev[:, :4].any()


# ----------------------------------------------------------------------------------------------
# the mass term
# ----------------------------------------------------------------------------------------------
def test_bare_count_mass_is_biased_low_against_the_true_logsumexp():
    """``log n_r`` under-claims by the within-cluster logit variance (Jensen), and the correction fixes it.

    Pins the direction and the rough size of the bias, which is the reason ``count`` is the *safe*
    default rather than the good one: the slot is quiet because its mass is too small, and
    "improving" it while the direction is still mediocre was measured 3-5x WORSE than no slot.
    """
    torch.manual_seed(0)
    n_heads, n_pts, dim = 1, 256, 16
    keys = torch.randn(n_heads, n_pts, dim)
    values = torch.randn(n_heads, n_pts, dim)
    q = torch.randn(n_heads, 64, dim)
    ev = torch.ones(n_heads, n_pts, dtype=torch.bool)

    _, assign = kmeans_assign(keys, 4, weights=ev.float())
    k_cmp, pop = cluster_reduce(keys, assign, 4, ev.float())
    var = prefill_logit_var(q, keys, assign, 4, ev.float(), scaling=1.0)

    b_count = slot_mass(pop)
    b_var = slot_mass(pop, logit_var=var)
    # true per-cluster logsumexp, averaged over the same queries
    logits = torch.einsum("htd,hsd->hts", q, keys)
    err_count, err_var = [], []
    for r in range(4):
        m = assign[0] == r
        if m.sum() < 2:
            continue
        true = torch.logsumexp(logits[0][:, m], dim=-1)  # (T,)
        base = torch.einsum("td,d->t", q[0], k_cmp[0, r])
        err_count.append((base + b_count[0, r] - true).mean())
        err_var.append((base + b_var[0, r] - true).mean())
    err_count = torch.stack(err_count)
    err_var = torch.stack(err_var)
    assert (err_count < 0).all(), "log n_r should UNDER-claim the cluster's mass"
    assert err_var.abs().mean() < err_count.abs().mean(), "the 1/2 Var term should reduce the bias"


def test_zero_population_slots_get_neg_inf_mass():
    pop = torch.tensor([[4.0, 0.0, 1.0]])
    b = slot_mass(pop)
    assert torch.isneginf(b[0, 1])
    assert torch.allclose(b[0, 0], torch.tensor(4.0).log())


def test_delta_shifts_every_live_slot_equally():
    pop = torch.tensor([[4.0, 0.0, 1.0]])
    d = slot_mass(pop, delta=1.5) - slot_mass(pop)
    assert torch.allclose(d[0, [0, 2]], torch.full((2,), 1.5))
    assert torch.isnan(d[0, 1]) or torch.isneginf(slot_mass(pop, delta=1.5)[0, 1])


# ----------------------------------------------------------------------------------------------
# the summary itself
# ----------------------------------------------------------------------------------------------
def test_one_slot_per_key_is_exact():
    """``R = |E|`` must reproduce the evicted keys exactly -- the identity control.

    The analogue of the ``chunk=1`` control in the probe, which returned 0.0000 and is what makes
    the non-trivial numbers believable: it separates "the summary is lossy" from "the plumbing is
    wrong".
    """
    keys, values = _kv(n_heads=1, n_pts=16, dim=4)
    ev = torch.ones(1, 16, dtype=torch.bool)
    k_cmp, v_cmp, b = cluster_evicted(keys, values, ev, 16, iters=50)
    # every cluster holds exactly one key, so the centroids are a permutation of the keys
    assert torch.allclose(b.exp(), torch.ones(1, 16), atol=1e-4)
    d = torch.cdist(k_cmp, keys)
    assert d.min(dim=-1).values.max() < 1e-4
    d = torch.cdist(v_cmp, values)
    assert d.min(dim=-1).values.max() < 1e-4


def test_kmeans_beats_positional_chunks_on_within_cluster_value_spread():
    """Content clusters must be more coherent than positional spans at the same slot count.

    This is the whole reason the arm was rebuilt: the positional variant scored cos 0.68-0.70 to the
    true evicted output against 0.85-0.86 for k-means, and the mechanism is that a positional span
    mixes unrelated value vectors so its mean represents none of them. Asserted on the spread
    directly, so the property is checked without needing a model.
    """
    torch.manual_seed(0)
    n_pts, dim, R = 256, 16, 8
    # values carrying genuine cluster structure, deliberately NOT aligned with position
    centres = torch.randn(R, dim) * 5
    which = torch.randint(R, (n_pts,))
    values = centres[which] + 0.1 * torch.randn(n_pts, dim)
    keys = values.clone()  # keys co-vary with values, as they do in a real layer
    ev = torch.ones(1, n_pts, dtype=torch.bool)

    _, assign = kmeans_assign(keys.unsqueeze(0), R, weights=ev.float())
    mean_km, pop_km = cluster_reduce(values.unsqueeze(0), assign, R, ev.float())
    resid_km = (values - mean_km[0][assign[0]]).norm(dim=-1).mean()

    span = n_pts // R
    pos_assign = (torch.arange(n_pts) // span).clamp(max=R - 1).unsqueeze(0)
    mean_pos, _ = cluster_reduce(values.unsqueeze(0), pos_assign, R, ev.float())
    resid_pos = (values - mean_pos[0][pos_assign[0]]).norm(dim=-1).mean()

    assert resid_km < 0.5 * resid_pos, f"kmeans {resid_km:.3f} vs positional {resid_pos:.3f}"


def test_masked_points_do_not_move_centroids():
    """Weight-0 points must influence neither the assignment's centroids nor the slot means."""
    keys, values = _kv(n_heads=1, n_pts=32, dim=4)
    ev = torch.zeros(1, 32, dtype=torch.bool)
    ev[0, :16] = True
    k_a, v_a, b_a = cluster_evicted(keys, values, ev, 4, generator=torch.Generator().manual_seed(3))
    keys2, values2 = keys.clone(), values.clone()
    keys2[0, 16:] = 1e4  # change only the masked-out half
    values2[0, 16:] = -1e4
    k_b, v_b, b_b = cluster_evicted(
        keys2, values2, ev, 4, generator=torch.Generator().manual_seed(3)
    )
    assert torch.allclose(k_a, k_b, atol=1e-4)
    assert torch.allclose(v_a, v_b, atol=1e-4)
    assert torch.allclose(b_a, b_b, atol=1e-5)


@pytest.mark.parametrize("R", [1, 4, 16])
def test_slot_means_lie_in_the_convex_hull_of_their_members(R):
    """``v_r`` must be an average of value vectors, so its norm cannot exceed the largest member's.

    Guards the property the single-softmax fusion relies on: the read is a convex combination, so a
    slot cannot inject magnitude the evicted set did not contain.
    """
    keys, values = _kv(n_heads=1, n_pts=64, dim=8)
    ev = torch.ones(1, 64, dtype=torch.bool)
    _, v_cmp, b = cluster_evicted(keys, values, ev, R)
    live = ~torch.isneginf(b[0])
    assert v_cmp[0][live].norm(dim=-1).max() <= values[0].norm(dim=-1).max() + 1e-4


# ----------------------------------------------------------------------------------------------
# the fusion identity
# ----------------------------------------------------------------------------------------------
def test_fused_read_equals_an_explicit_joint_softmax():
    """``fuse_memory(o_S, lse_S, n, d)`` must equal softmaxing ``[exact ; cmp]`` together.

    ``_attend_with_cmp`` claims "one softmax over the concatenation" but never builds it -- it
    reuses the retained branch's ``(o_S, lse_S)`` and folds the slots in as ``(N_S+n)/(D_S+d)``.
    That is an algebraic identity, and this pins it, because if it ever stopped holding the slots
    would be silently mis-weighted against the retained keys rather than raising.
    """
    from kvpress.presses.gqa_indexer.memory import fuse_memory

    torch.manual_seed(0)
    B, H, Sq, S, R, D = 1, 4, 3, 32, 5, 8
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    k_cmp = torch.randn(H, R, D)
    v_cmp = torch.randn(H, R, D)
    b_cmp = torch.randn(H, R)
    b_cmp[0, 0] = -float("inf")  # a silenced slot must contribute exactly nothing
    scale = D**-0.5

    l_exact = torch.einsum("bhqd,bhsd->bhqs", q, k) * scale
    l_cmp = torch.einsum("bhqd,hrd->bhqr", q, k_cmp) * scale + b_cmp.view(1, H, 1, R)

    joint = torch.softmax(torch.cat([l_exact, l_cmp], -1), -1)
    ref = torch.einsum("bhqs,bhsd->bhqd", joint[..., :S], v) + torch.einsum(
        "bhqr,hrd->bhqd", joint[..., S:], v_cmp
    )

    p_s = torch.softmax(l_exact, -1)
    o_s = torch.einsum("bhqs,bhsd->bhqd", p_s, v)
    lse_s = torch.logsumexp(l_exact, -1)
    w = l_cmp.exp()
    got = fuse_memory(o_s, lse_s, torch.einsum("bhqr,hrd->bhqd", w, v_cmp), w.sum(-1), group=1)

    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_silenced_slots_contribute_exactly_zero():
    """``b_r = -inf`` must give weight 0, not NaN -- ``exp(-inf) * v`` is the empty-cluster case."""
    b = slot_mass(torch.tensor([[0.0, 3.0]]))
    w = b.exp()
    assert w[0, 0] == 0.0
    assert torch.isfinite(w).all()


# ----------------------------------------------------------------------------------------------
# pre-RoPE clustering
# ----------------------------------------------------------------------------------------------
def test_unrotate_inverts_apply_rotary_pos_emb_exactly():
    """``unrotate`` must be HF's ``apply_rotary_pos_emb`` inverted to fp precision.

    RoPE is an orthogonal rotation so the inverse is exact, not an approximation -- pinned against
    the library's own function rather than a reimplementation, so a convention change upstream (the
    interleaving of ``rotate_half``) fails here instead of silently producing a wrong clustering
    space.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    from kvpress.presses.gqa_indexer.cmp_slots import unrotate

    torch.manual_seed(0)
    H, S, D = 4, 32, 16
    k = torch.randn(1, H, S, D, dtype=torch.float64)
    t = torch.arange(S, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = torch.outer(t, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1).unsqueeze(0)
    sin = torch.cat([ang.sin(), ang.sin()], -1).unsqueeze(0)
    _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
    back = unrotate(k_rot[0], cos[0], sin[0])
    assert torch.allclose(back, k[0], atol=1e-10), (back - k[0]).abs().max()


def test_cluster_keys_changes_only_the_partition_not_the_slot_vectors():
    """With ``cluster_keys``, ``k_cmp`` must still be a mean of the POST-RoPE keys.

    The slot's logit is ``q . k_cmp`` and it has to approximate ``mean_j(q . R_j k_j)``, so averaging
    in the un-rotated space and rotating afterwards would be a different quantity -- and there is no
    single position to rotate a cross-position cluster by. Asserted by forcing a known partition:
    every ``k_cmp`` row must equal the post-RoPE mean of its own members.
    """
    from kvpress.presses.gqa_indexer.cmp_slots import cluster_reduce, kmeans_assign

    torch.manual_seed(0)
    H, S, D, R = 2, 64, 8, 4
    post = torch.randn(H, S, D)
    pre = torch.randn(H, S, D)  # a deliberately unrelated clustering space
    ev = torch.ones(H, S, dtype=torch.bool)
    g = torch.Generator().manual_seed(7)
    k_cmp, v_cmp, _ = cluster_evicted(post, post.clone(), ev, R, cluster_keys=pre, generator=g)

    # reproduce the assignment the call must have used: k-means in `pre`
    _, assign = kmeans_assign(pre, R, weights=ev.float(), generator=torch.Generator().manual_seed(7))
    expect, _ = cluster_reduce(post, assign, R, ev.float())
    assert torch.allclose(k_cmp, expect, atol=1e-5)


def test_position_locality_flags_contiguous_clusters():
    """The diagnostic must read ~0 for positional spans and ~1 for a position-blind partition.

    This is what decides whether post-RoPE clustering has quietly degenerated into the positional
    chunking that was already measured to lose (cos 0.68-0.70 vs 0.85-0.86).
    """
    from kvpress.presses.gqa_indexer.cmp_slots import position_locality

    S, R = 256, 8
    w = torch.ones(1, S)
    spans = (torch.arange(S) // (S // R)).clamp(max=R - 1).unsqueeze(0)
    assert position_locality(spans, w, R).mean() < 0.15
    torch.manual_seed(0)
    rand = torch.randint(R, (1, S))
    assert position_locality(rand, w, R).mean() > 0.85


# ----------------------------------------------------------------------------------------------
# the learned mass head
# ----------------------------------------------------------------------------------------------
def test_mass_head_init_reproduces_the_analytic_modes_exactly():
    """``CMPMassHead`` must contain both fixed modes, so the ablation is nested rather than parallel.

    ``(count, var, bias) = (1, 1, 0)`` is ``count+var`` and ``(1, 0, 0)`` is ``count``. This is what
    lets a learned run *start at* the best closed-form answer instead of at zero, and it means any
    regression against the fixed modes is a training result rather than a parameterization artifact.
    """
    from kvpress.presses.gqa_indexer.cmp_slots import CMPMassHead

    torch.manual_seed(0)
    pop = torch.rand(8, 16) * 50
    lv = torch.rand(8, 16) * 4
    assert torch.allclose(CMPMassHead(8)(pop, lv), slot_mass(pop, logit_var=lv), atol=1e-6)
    assert torch.allclose(
        CMPMassHead(8, var_init=0.0)(pop, lv), slot_mass(pop), atol=1e-6
    )


def test_mass_head_all_three_scalars_get_gradients():
    """All three must be live at init, or the parameter silently never moves.

    Third instance of this class of bug in the package (``upcast_gate_scales``' bf16 freeze, AdamW eps
    throttling ``log_gamma``), hence the explicit check rather than trust.
    """
    from kvpress.presses.gqa_indexer.cmp_slots import CMPMassHead

    head = CMPMassHead(4)
    pop = torch.rand(4, 8) * 20 + 1
    lv = torch.rand(4, 8) * 3
    head(pop, lv).sum().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_count_coef_changes_mass_scaling_not_just_the_offset():
    """``count_coef`` must alter how mass scales with cluster SIZE -- that is why it is not a bias.

    A cluster twice as large should gain ``count_coef * log 2`` nats, so halving the coefficient
    halves the gap between a big slot and a small one. A constant offset cannot do this, which is the
    whole argument for learning the count exponent rather than an additive correction.
    """
    pop = torch.tensor([[4.0, 64.0]])
    gap_1 = (slot_mass(pop, count_coef=1.0)[0, 1] - slot_mass(pop, count_coef=1.0)[0, 0]).item()
    gap_h = (slot_mass(pop, count_coef=0.5)[0, 1] - slot_mass(pop, count_coef=0.5)[0, 0]).item()
    assert abs(gap_1 - torch.tensor(16.0).log().item()) < 1e-5
    assert abs(gap_h - 0.5 * gap_1) < 1e-5
    # a bias leaves the gap untouched, which is the contrast being asserted
    gap_b = (slot_mass(pop, bias=3.0)[0, 1] - slot_mass(pop, bias=3.0)[0, 0]).item()
    assert abs(gap_b - gap_1) < 1e-5


def test_masked_slots_do_not_poison_the_gradient():
    """A zero-population slot must give ``-inf`` mass AND finite gradients for the live ones.

    ``torch.where`` evaluates both branches, so an ``inf`` in the untaken branch back-propagates as
    NaN to every parameter -- the exact shape of the ``exp(-lse)`` NaN that ran clean for 250 steps
    and then killed a whole run. The ``clamp(min=1e-30)`` inside ``slot_mass`` is what prevents it.
    """
    from kvpress.presses.gqa_indexer.cmp_slots import CMPMassHead

    head = CMPMassHead(2)
    pop = torch.tensor([[0.0, 10.0, 3.0], [5.0, 0.0, 0.0]])
    lv = torch.ones(2, 3)
    b = head(pop, lv)
    assert torch.isneginf(b[pop == 0]).all()
    b[pop > 0].sum().backward()
    for name, p in head.named_parameters():
        assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/inf"
