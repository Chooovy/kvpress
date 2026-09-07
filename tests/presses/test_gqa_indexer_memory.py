# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the linear-memory eviction compensation.

The properties here are chosen for a specific reason: **every one of these failures is silent**.
A double-counted key, a lost key, a ``gamma`` that absorbed ``1/|E|``, a decay applied to the
numerator only -- none of them raise, none make the loss diverge. They make the metric quietly
worse, which is indistinguishable from "the idea did not work". So they are asserted rather than
inspected.
"""

from __future__ import annotations

import math

import pytest
import torch

from kvpress.presses.gqa_indexer.memory import (
    MemoryConfig,
    MemoryKernel,
    fuse_memory,
    memory_mass_share,
)
from kvpress.presses.gqa_indexer.memory_schedule import (
    block_horizons,
    block_memory_states,
    entry_block,
    evicted_counts,
    ingestion_horizon,
)
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines

HKV, D, R = 4, 32, 8


def make_kernel(**kwargs) -> MemoryKernel:
    torch.manual_seed(0)
    cfg = MemoryConfig(n_kv_heads=HKV, head_dim=D, rank=kwargs.pop("rank", R), mid_dim=64, **kwargs)
    return MemoryKernel(cfg)


def make_kv(k_len, bsz=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(bsz, HKV, k_len, D, generator=g)
    v = torch.randn(bsz, HKV, k_len, D, generator=g)
    return k, v


# ----------------------------------------------------------------------
# Partition: no key counted twice, none lost
# ----------------------------------------------------------------------
@pytest.mark.parametrize("k_len,topk,n_sink,n_local", [(512, 128, 4, 0), (512, 128, 4, 32), (384, 64, 0, 0)])
def test_partition_is_exact(k_len, topk, n_sink, n_local):
    """
    Per (head, query block): retained + ingested == causally visible, with no overlap.

    The invariant the whole design rests on. A key counted in both branches is weighted twice; a
    key in neither loses its mass outright. Neither shows up as anything but a slightly worse
    number, so it is checked entry by entry against the *same* ``deadline`` the attention mask is
    built from.
    """
    torch.manual_seed(0)
    scores = torch.randn(HKV, k_len)
    dl = deadlines(scores, topk, force_sink=n_sink, force_local=n_local)
    block = 128
    q_len = k_len
    horizons = block_horizons(q_len, k_len, block=block, n_local=n_local)
    enter = ingestion_horizon(dl)
    n_blocks = (q_len + block - 1) // block
    entry = entry_block(enter, horizons)
    counts = evicted_counts(entry, n_blocks)

    key_idx = torch.arange(k_len)
    for b in range(n_blocks):
        start = b * block
        limit = min(start + (k_len - q_len), k_len - 1)
        horizon = limit - n_local
        for h in range(HKV):
            # Exactly the mask_mod of qi_block_mask, evaluated at the block's first row.
            sink = key_idx < n_sink
            local = (key_idx > limit - n_local) & (key_idx >= n_sink)
            alive = horizon <= dl[h].to(torch.int64)
            chosen = (key_idx >= n_sink) & (key_idx <= horizon) & alive
            retained = (key_idx <= limit) & (sink | local | chosen)

            ingested = entry[h] <= b

            assert not bool((retained & ingested).any()), (
                f"head {h} block {b}: {int((retained & ingested).sum())} keys are in BOTH the "
                "attention mask and the memory -- their mass is counted twice"
            )
            visible = key_idx <= limit
            assert bool((retained | ingested).eq(visible).all()), (
                f"head {h} block {b}: {int((visible & ~(retained | ingested)).sum())} visible keys "
                "are in NEITHER branch -- their mass vanishes"
            )
            assert int(counts[h, b]) == int(ingested.sum()), (
                f"head {h} block {b}: |E| reported {int(counts[h, b])} but the partition has "
                f"{int(ingested.sum())}"
            )


def test_sinks_are_never_ingested():
    """A sink's deadline is ``k_len-1``, so its ingestion horizon is unreachable.

    Checked separately from the partition test because it is the mechanism that makes sink/local
    protection need no special case anywhere in the memory path -- if it broke, the partition test
    would still pass while the sink's mass moved into the low-rank state.
    """
    k_len, n_sink = 256, 8
    torch.manual_seed(0)
    dl = deadlines(torch.randn(HKV, k_len), 64, force_sink=n_sink)
    enter = ingestion_horizon(dl)
    assert bool((enter[:, :n_sink] > k_len - 1).all())

    kernel = make_kernel()
    w = kernel.ingest_weights(enter, k_len)
    assert torch.equal(w[:, :n_sink], torch.zeros_like(w[:, :n_sink]))


# ----------------------------------------------------------------------
# The off state really is off
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_evicted", [128.0, 4096.0, 32768.0, 131072.0])
def test_gamma_off_recovers_eviction_baseline(n_evicted):
    """
    A calibrated ``log_gamma`` makes the memory a no-op, **at every ``|E|``**.

    This is what lets a run start at the eviction baseline it is trying to beat, so the first steps
    cannot be a regression that later has to be recovered from.

    The threshold is computed rather than hardcoded, and that is the substance of the test. What
    reaches the softmax is ``d = gamma |E| <phi_hat, z_hat>``, and the simplex inner product is
    empirically ``~1/R`` -- both ``phi_hat`` and ``z_hat`` are near-uniform over ``R`` coordinates at
    initialization (measured 0.1239 at R=8 against 1/8 = 0.125, and 0.0627 at R=16 against 0.0625).
    So the off-state condition is analytic:

        ``d ~ gamma |E| / R``,  which must be << 1

    i.e. ``log_gamma`` has to be set against ``|E|`` **and** ``R``, not chosen once. That dependence
    is exactly why two earlier values were wrong: ``-10`` under the un-normalized mass reached
    ``d = 2.0`` at ``|E|=128K`` while still being called "off", and ``-18`` was off everywhere but 21
    log units away from useful. :data:`~.memory.DEFAULT_LOG_GAMMA` records the calibration for the
    real geometry; this test checks the *relation* holds for whatever geometry it is given.
    """
    from kvpress.presses.gqa_indexer.memory import DEFAULT_LOG_GAMMA

    torch.manual_seed(0)
    kernel = make_kernel()
    # Two rescalings of the shipped default, both from the relation above, so this test exercises the
    # calibration rather than a magic number:
    #
    #  * ``R/16`` -- this fixture's rank against the real geometry's, since ``d ~ 1/R``.
    #  * ``D_S`` -- the retained branch's normalizer. The shipped value is calibrated against the
    #    real model, where topk=2048 keys of genuine attention logits give ``D_S ~ 3.6e4`` (implied
    #    by the live run: mass_share 1.59e-10 at ``gamma|E|/R = 5.7e-6``). This fixture's synthetic
    #    ``lse ~ N(2,1)`` gives ``D_S ~ e^2 = 7.4``, five thousand times smaller -- so the same
    #    ``gamma`` is a no-op on the model and dominant here. Without this term the test would be
    #    asserting that the default is wrong for a regime it was never set for.
    real_d_s, test_d_s = 3.6e4, math.exp(2.0)
    kernel.log_gamma.data.fill_(
        DEFAULT_LOG_GAMMA + math.log(R / 16) + math.log(test_d_s / real_d_s)
    )
    k, v = make_kv(256)
    q = torch.randn(1, HKV, 64, D)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 256), 64, force_sink=4))
    w = kernel.ingest_weights(enter, 256)
    state = kernel.ingest(k, v, w)
    n, d = kernel.read(q, state, torch.full((64,), n_evicted))

    o_s = torch.randn(1, HKV, 64, D)
    lse_s = torch.full((1, HKV, 64), 2.0)
    fused = fuse_memory(o_s, lse_s, n, d)

    # The predicted mass, from the analytic relation -- checked against the actual one, so a change
    # that broke the |E| or R dependence would show up here rather than as a mysterious tolerance.
    predicted = math.exp(float(kernel.log_gamma.mean())) * n_evicted / R
    assert float(d.mean()) < 10 * predicted, (
        f"d = {float(d.mean()):.3e} exceeds 10x the predicted gamma|E|/R = {predicted:.3e}"
    )
    # And the memory is genuinely off: its share of the softmax mass stays negligible.
    share = float(memory_mass_share(lse_s, d).max())
    assert share < 1e-3, f"memory holds {share:.3e} of the mass at |E|={n_evicted:g}, not off"
    # The output deviation stays small relative to the values themselves.
    rel = float((fused - o_s).abs().max() / o_s.abs().max())
    assert rel < 1e-2, f"the default log_gamma perturbs the output by {rel:.3e} at |E|={n_evicted:g}"


# ----------------------------------------------------------------------
# Gradients are alive at the initialization point
# ----------------------------------------------------------------------
def test_gradients_nonzero_at_init():
    """
    ``beta_h`` and ``psi``'s last layer both receive nonzero gradient at the start.

    The three dead points in the module docstring are all of the form "the gradient is exactly
    zero, so training never leaves initialization" -- and a zero gradient produces a flat curve
    that looks like a hard problem rather than a bug. ``psi_out`` is the one that the naive
    zero-init would kill (``abs``'s subgradient at 0 is 0), which is why the off state lives in
    ``gamma`` instead.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    k, v = make_kv(256)
    q = torch.randn(1, HKV, 64, D)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 256), 64, force_sink=4))
    w = kernel.ingest_weights(enter, 256)
    state = kernel.ingest(k, v, w)
    n, d = kernel.read(q, state, torch.full((64,), 128.0))
    o_s = torch.randn(1, HKV, 64, D)
    lse_s = torch.randn(1, HKV, 64)
    fuse_memory(o_s, lse_s, n, d).square().sum().backward()

    for name in ("log_gamma", "psi_out", "phi_out", "log_tau"):
        param = getattr(kernel, name)
        assert param.grad is not None, f"{name} received no gradient at all"
        assert float(param.grad.abs().max()) > 0, (
            f"{name}'s gradient is identically zero at initialization, so it can never move"
        )
    for name in ("psi_in", "psi_mid", "phi_in"):
        grad = getattr(kernel, name).weight.grad
        assert grad is not None and float(grad.abs().max()) > 0, (
            f"{name} received no usable gradient at initialization"
        )


def test_ingest_scale_gradient_nonzero_when_off():
    """``a`` is a live parameter at ``a = 0``: ``dw/da = w s~ != 0``, so it self-starts.

    The point of defaulting the ``g(s~)`` feature off rather than removing it. Its own gradient
    does not depend on it already being on.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    assert float(kernel.ingest_scale) == 0.0
    k_len = 256
    enter = ingestion_horizon(deadlines(torch.randn(HKV, k_len), 64, force_sink=4))
    scores = torch.randn(HKV, k_len)
    # ingest_weights short-circuits the score path at exactly a == 0 (it is a no-op there and the
    # standardisation is not free), so the gradient is taken from the general form.
    kernel.ingest_scale.data.fill_(1e-6)
    w = kernel.ingest_weights(enter, k_len, scores=scores)
    w.sum().backward()
    assert kernel.ingest_scale.grad is not None
    assert float(kernel.ingest_scale.grad.abs()) > 0


# ----------------------------------------------------------------------
# The two normalization invariants
# ----------------------------------------------------------------------
def test_read_is_invariant_to_evicted_count_scaling():
    """
    Same ``o_E`` at different ``|E|`` gives the same ``n/d``, and mass proportional to ``|E|``.

    The explicit ``|E|`` factor exists so ``gamma`` cannot learn to be ``1/|E|`` at the training
    length and then be wrong at every other length. Two checks in one: the *direction* ``n/d`` must
    not move with ``|E|`` at all, and the *mass* ``d`` must move exactly linearly with it.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    kernel.log_gamma.data.fill_(0.0)  # gamma = 1, so the effect is visible
    k, v = make_kv(256)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 256), 64, force_sink=4))
    state = kernel.ingest(k, v, kernel.ingest_weights(enter, 256))
    q = torch.randn(1, HKV, 16, D)

    n1, d1 = kernel.read(q, state, torch.full((16,), 100.0))
    n2, d2 = kernel.read(q, state, torch.full((16,), 5000.0))

    ratio1 = n1 / d1.unsqueeze(-1).clamp(min=1e-20)
    ratio2 = n2 / d2.unsqueeze(-1).clamp(min=1e-20)
    assert torch.allclose(ratio1, ratio2, atol=1e-5), "n/d must not depend on |E|"
    assert torch.allclose(d2, d1 * 50.0, rtol=1e-4), "d must be linear in |E|"


def test_decay_scales_H_and_z_together():
    """
    Changing ``lambda`` leaves ``H/z`` fixed on a single-key evicted set.

    Invariant 1: ``gamma``, ``lambda`` and ``w_j`` must all hit ``H`` and ``z`` identically. If a
    decay reached the numerator alone, ``H/z`` would stop being a weighted mean of value vectors and
    ``d`` would stop meaning softmax mass -- and the model would still run. On one ingested key,
    ``H/z`` should be that key's value vector exactly, whatever the decay is.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    k_len = 64
    k, v = make_kv(k_len)
    # Exactly one ingested key: deadline = -1 for key 5 (evicted immediately), never for the rest.
    dl = torch.full((HKV, k_len), k_len - 1, dtype=torch.int32)
    dl[:, 5] = -1
    enter = ingestion_horizon(dl)

    ratios = []
    for tau in (10.0, 1e3, 1e6):
        kernel.log_tau.data.fill_(math.log(tau))
        H, z, W = kernel.ingest(k, v, kernel.ingest_weights(enter, k_len))
        # H is (B, Hkv, R, D), z is (B, Hkv, R): H/z per rank row is the value vector.
        ratios.append(H[0, :, 0] / z[0, :, 0].unsqueeze(-1).clamp(min=1e-20))
    for later in ratios[1:]:
        assert torch.allclose(ratios[0], later, atol=1e-4), "H/z moved when only lambda changed"
    # And it is that key's value vector, RAW. H is built from unnormalized v so that
    # `n/d = sum_j a_j v_j` lands in the same space as the target `oE*`; see MemoryKernel.ingest.
    expect = v[0, :, 5]
    assert torch.allclose(ratios[0], expect, atol=1e-3)


def test_global_decay_does_not_change_direction():
    """A uniform rescaling of all ``w_j`` leaves ``n/d`` untouched -- so it is ``gamma``'s job.

    Written down as a test because it is the reason ``ingest_bias`` is not redundant with
    ``log_gamma`` in a confusing way: ``b`` genuinely cannot express anything ``gamma`` cannot, and
    a future change that made it appear to would be a bug.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    k, v = make_kv(128)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 128), 32, force_sink=4))
    q = torch.randn(1, HKV, 8, D)
    counts = torch.full((8,), 50.0)

    outs = []
    for bias in (0.0, 3.0, -2.0):
        kernel.ingest_bias.data.fill_(bias)
        state = kernel.ingest(k, v, kernel.ingest_weights(enter, 128))
        n, d = kernel.read(q, state, counts)
        outs.append(n / d.unsqueeze(-1).clamp(min=1e-20))
    for later in outs[1:]:
        assert torch.allclose(outs[0], later, atol=1e-4)


# ----------------------------------------------------------------------
# Fusion arithmetic
# ----------------------------------------------------------------------
def test_fusion_matches_fp64_dense_reference():
    """
    ``fuse_memory`` reproduces ``(N_S + n)/(D_S + d)`` computed longhand in fp64.

    The fusion is written in the ``exp(-lse)`` form for stability, which is an identity rather than
    an approximation -- this is what pins that claim to a number.
    """
    torch.manual_seed(0)
    B, H, Sq, Sk = 1, 8, 16, 64
    group = H // HKV
    q = torch.randn(B, H, Sq, D, dtype=torch.float64)
    k = torch.randn(B, H, Sk, D, dtype=torch.float64)
    v = torch.randn(B, H, Sk, D, dtype=torch.float64)
    scale = D**-0.5
    keep = torch.rand(B, H, Sq, Sk) > 0.4

    logits = (q @ k.transpose(-1, -2)) * scale
    masked = logits.masked_fill(~keep, float("-inf"))
    lse_s = torch.logsumexp(masked, -1)
    o_s = torch.softmax(masked, -1) @ v

    n = torch.rand(B, HKV, Sq, D, dtype=torch.float64) * 3.0
    d = torch.rand(B, HKV, Sq, dtype=torch.float64) * 5.0

    # longhand: N_S = exp(lse_S) * o_S, then (N_S + n)/(D_S + d)
    N_S = torch.exp(lse_s).unsqueeze(-1) * o_s
    D_S = torch.exp(lse_s)
    n_rep = n.repeat_interleave(group, 1)
    d_rep = d.repeat_interleave(group, 1)
    expect = (N_S + n_rep) / (D_S + d_rep).unsqueeze(-1)

    got = fuse_memory(o_s, lse_s, n, d, group=group)
    assert torch.allclose(got, expect, atol=1e-10), f"max err {(got - expect).abs().max():.3e}"


def test_fusion_is_identity_when_memory_is_zero():
    """``n = d = 0`` returns ``o_S`` bit-identically -- no ``0/0``, no drift."""
    torch.manual_seed(0)
    o_s = torch.randn(1, 8, 16, D)
    lse_s = torch.randn(1, 8, 16)
    n = torch.zeros(1, HKV, 16, D)
    d = torch.zeros(1, HKV, 16)
    assert torch.equal(fuse_memory(o_s, lse_s, n, d, group=2), o_s)


def test_fusion_survives_large_lse():
    """A large positive ``lse`` must underflow the memory term, not overflow the fusion.

    ``exp(+lse)`` would be ``inf`` here; the implementation uses ``exp(-lse)`` precisely so the
    dominated case degrades to "no memory contribution" instead of to NaN.
    """
    o_s = torch.randn(1, 4, 8, D)
    lse_s = torch.full((1, 4, 8), 200.0)
    n = torch.randn(1, 4, 8, D)
    d = torch.rand(1, 4, 8)
    out = fuse_memory(o_s, lse_s, n, d)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, o_s, atol=1e-6)


@pytest.mark.parametrize("lse_val", [-20.0, -90.0, -400.0, -1e4])
@pytest.mark.parametrize("empty_memory", [True, False])
def test_fusion_survives_extreme_lse(lse_val, empty_memory):
    """
    A very *negative* ``lse`` must stay finite, including where the memory is empty.

    This is the bug that killed the first real 8K run, and it is worth the parametrization. A query
    row early in the sequence retains only a sink key or two, so its branch is a single logit and its
    ``lse`` is far below zero -- ``exp(-lse)`` then overflows fp32 past -88.7. Where ``n > 0`` that
    gives ``inf``; where ``n == 0`` it gives ``inf * 0 = NaN``, and ``n`` is *identically* zero for
    every row of query block 0 because nothing has been evicted yet. So the rows the memory does not
    touch at all were the ones that produced NaN.

    Observed: clean through step 250, all-NaN from step 260, with every parameter still finite and in
    range. Nothing in the loss curve or the checkpoint pointed at the fusion.
    """
    torch.manual_seed(0)
    o_s = torch.randn(1, 4, 8, D)
    lse_s = torch.full((1, 4, 8), lse_val)
    n = torch.zeros(1, 4, 8, D) if empty_memory else torch.randn(1, 4, 8, D)
    d = torch.zeros(1, 4, 8) if empty_memory else torch.rand(1, 4, 8)
    out = fuse_memory(o_s, lse_s, n, d)
    assert torch.isfinite(out).all(), (
        f"fusion produced non-finite output at lse={lse_val:g}, empty_memory={empty_memory}"
    )
    if empty_memory:
        # An empty memory must be exactly a no-op regardless of how extreme the lse is.
        assert torch.equal(out, o_s)
    assert torch.isfinite(memory_mass_share(lse_s, d)).all()


# ----------------------------------------------------------------------
# Streaming schedule
# ----------------------------------------------------------------------
def test_block_states_are_prefix_sums():
    """Block ``b``'s state equals a direct sum over the keys ingested at or before ``b``.

    The prefix-sum construction is what makes the streaming schedule ``O(L)`` instead of
    ``O(L * n_blocks)``; this checks it against the definition it is standing in for.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    k_len = q_len = 512
    block = 128
    k, v = make_kv(k_len)
    dl = deadlines(torch.randn(HKV, k_len), 128, force_sink=4)
    H, z, W, counts = block_memory_states(kernel, k, v, dl, q_len=q_len, block=block)

    enter = ingestion_horizon(dl)
    entry = entry_block(enter, block_horizons(q_len, k_len, block=block))
    n_blocks = q_len // block
    w = kernel.ingest_weights(enter, k_len)
    psi = kernel.psi(k).float() * w.unsqueeze(0).unsqueeze(-1)
    v_n = v.float()  # H is built from RAW values; see MemoryKernel.ingest

    for b in range(n_blocks):
        sel = (entry <= b).float()  # (Hkv, Sk)
        H_ref = torch.einsum("bhsr,bhsd,hs->bhrd", psi, v_n, sel)
        z_ref = torch.einsum("bhsr,hs->bhr", psi, sel)
        assert torch.allclose(H[:, :, b], H_ref, atol=1e-4)
        assert torch.allclose(z[:, :, b], z_ref, atol=1e-4)
        assert torch.allclose(W[:, :, b], (w * sel).sum(-1).unsqueeze(0), atol=1e-4)


def test_evicted_count_grows_with_block():
    """``|E_b|`` is non-decreasing -- the irreversibility the scalar scorer guarantees.

    If a key could re-enter the top-k, a single accumulated state could not represent the evicted
    set at all (it would have to *subtract*), so this is the property the whole design depends on
    rather than a convenience. ``scalar_indexer`` measures 0 returns over 1500 steps with the
    position tilt on.
    """
    torch.manual_seed(0)
    k_len = 1024
    dl = deadlines(torch.randn(HKV, k_len), 256, force_sink=4)
    entry = entry_block(ingestion_horizon(dl), block_horizons(k_len, k_len, block=128))
    counts = evicted_counts(entry, k_len // 128)
    assert bool((counts[:, 1:] >= counts[:, :-1]).all())


def test_rank_zero_state_is_the_value_mean():
    """The rank-0 ablation reads back the weighted mean of the evicted values.

    Its purpose is to be the cheap comparison that isolates whether rank buys anything -- measured
    78% relative residual against the exact evicted output, so it is expected to lose. That only
    holds if it is genuinely the mean, hence the check.
    """
    torch.manual_seed(0)
    kernel = make_kernel(rank=0)
    assert kernel.rank_zero
    k_len = 128
    k, v = make_kv(k_len)
    dl = torch.full((HKV, k_len), k_len - 1, dtype=torch.int32)
    dl[:, :10] = -1  # first ten keys evicted immediately
    enter = ingestion_horizon(dl)
    w = kernel.ingest_weights(enter, k_len)
    H, z, W = kernel.ingest(k, v, w)
    got = H[0, :, 0] / z[0, :, 0].unsqueeze(-1)
    v_n = v
    expect = torch.einsum("hs,hsd->hd", w[:, :10], v_n[0, :, :10]) / w[:, :10].sum(-1, keepdim=True)
    assert torch.allclose(got, expect, atol=1e-4)


def test_scalars_are_fp32_and_upcast_is_idempotent():
    """
    The scalars stay fp32 through a bf16 cast of the parent module.

    Not a precision nicety. ``upcast_gate_scales``' docstring records ``gate_scale`` frozen at its
    bf16 initialization for 30 steps across all 36 layers while the loss fell 4.52 -> 2.42, because
    bf16's spacing near the init value exceeds a warmup-sized step. Same parameters, same failure,
    so the same fix -- and it has to happen before the optimizer captures the tensors.
    """
    kernel = make_kernel()
    for name in ("log_gamma", "log_tau", "ingest_scale", "ingest_bias"):
        assert getattr(kernel, name).dtype == torch.float32
    assert kernel.upcast_scalars() == 0  # already fp32

    kernel.to(torch.bfloat16)
    assert kernel.log_gamma.dtype == torch.bfloat16
    assert kernel.upcast_scalars() == 4
    for name in ("log_gamma", "log_tau", "ingest_scale", "ingest_bias"):
        param = getattr(kernel, name)
        assert param.dtype == torch.float32
        assert param.is_leaf and param.requires_grad or name == "log_tau"
    assert kernel.upcast_scalars() == 0


def test_parameter_groups_partition_all_parameters():
    """Every parameter lands in exactly one of the two optimizer groups.

    They are trained at different learning rates, so a parameter in neither group silently never
    trains and one in both gets stepped twice.
    """
    kernel = make_kernel()
    scalars = kernel.scalar_parameters()
    kernels = kernel.kernel_parameters()
    ids = {id(p) for p in scalars} | {id(p) for p in kernels}
    assert len(scalars) + len(kernels) == len(list(kernel.parameters()))
    assert ids == {id(p) for p in kernel.parameters()}


def test_psi_and_phi_are_non_negative():
    """``abs`` keeps both kernels non-negative, which is what keeps ``d >= 0``.

    A negative ``d`` can drive the fused denominator through zero -- not a worse model, an
    undefined one.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    k, v = make_kv(64)
    q = torch.randn(1, HKV, 16, D) * 5.0
    assert bool((kernel.psi(k) >= 0).all())
    assert bool((kernel.phi(q) >= 0).all())


# ----------------------------------------------------------------------
# Press wiring
# ----------------------------------------------------------------------
def test_press_rejects_memory_with_query_dependent_scorer():
    """``memory=True`` with a pairwise scorer is refused at construction.

    Structural rather than a missing feature: one accumulated state can only stand for the evicted
    set if that set is a function of the query's *position*. A pairwise scorer evicts a different set
    per query, so the module would run happily and summarize the wrong keys -- there is no shape or
    loss that reveals it, hence the up-front error.
    """
    from kvpress.presses.gqa_indexer.press import GQAIndexerPress

    for scorer in ("pairwise", "prefix", "dma"):
        with pytest.raises(ValueError, match="requires scorer='scalar'"):
            GQAIndexerPress(compression_ratio=0.5, scorer=scorer, memory=True)
    # The supported combination constructs.
    GQAIndexerPress(compression_ratio=0.5, scorer="scalar", memory=True)


def test_get_memory_raises_without_memory():
    """Reading the memory off a press built without one raises instead of returning ``None``.

    A ``None`` would let a caller skip the memory term and report plain-eviction numbers under the
    memory arm's name.
    """
    from kvpress.presses.gqa_indexer.press import GQAIndexerPress

    press = GQAIndexerPress(compression_ratio=0.5, scorer="scalar")
    module = torch.nn.Linear(2, 2)  # any module; it simply has no memory attribute
    with pytest.raises(RuntimeError, match="memory=True"):
        press.get_memory(module)


def test_trainer_parameter_groups_carry_their_own_eps():
    """The scalar group ships an ``eps`` below the gradient's own scale.

    ``dL/dlog_gamma`` is proportional to ``gamma`` (~2e-9 at init), which is *under* AdamW's default
    ``eps=1e-8`` -- so the denominator becomes eps rather than the gradient scale and the step
    degrades from ``lr`` to ``lr * g / eps``. Measured: 5x slower than the learning rate implies.
    The eps therefore has to travel with the group rather than be left to the caller's defaults.
    """
    from kvpress.presses.gqa_indexer.memory import DEFAULT_SCALAR_EPS

    kernel = make_kernel()
    # parameter_groups walks the model, so exercise the group construction directly against the
    # kernel's own accessors -- the walk itself is covered by the smoke path.
    scalars = kernel.scalar_parameters()
    groups = [
        {"params": kernel.kernel_parameters(), "lr": 1e-3},
        {"params": scalars, "lr": 0.05, "eps": DEFAULT_SCALAR_EPS},
    ]
    opt = torch.optim.AdamW(groups)
    assert opt.param_groups[1]["eps"] == DEFAULT_SCALAR_EPS
    assert opt.param_groups[1]["eps"] < 1e-8, "the default eps is what throttles the bootstrap"


def test_gamma_bootstraps_off_the_off_state():
    """
    ``log_gamma`` escapes its initialization within ~100 steps at the default scalar LR.

    The counterpart to :func:`test_gamma_off_recovers_eviction_baseline`: that one checks the memory
    starts off, this one checks it can *turn on*. Both are needed -- an init small enough to be a
    genuine no-op is also small enough to raise "can it ever climb back", and the answer depends on
    the optimizer settings rather than on the initial value, since AdamW's step is ~``lr`` almost
    independently of the gradient's size.
    """
    from kvpress.presses.gqa_indexer.memory import DEFAULT_SCALAR_EPS, DEFAULT_SCALAR_LR

    torch.manual_seed(0)
    kernel = make_kernel()
    opt = torch.optim.AdamW(
        [
            {"params": kernel.kernel_parameters(), "lr": 1e-3},
            {"params": kernel.scalar_parameters(), "lr": DEFAULT_SCALAR_LR, "eps": DEFAULT_SCALAR_EPS},
        ]
    )
    k, v = make_kv(256)
    q = torch.randn(1, HKV, 32, D)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 256), 64, force_sink=4))
    o_s = torch.randn(1, HKV, 32, D)
    lse_s = torch.randn(1, HKV, 32) + 2.0
    target = torch.randn(1, HKV, 32, D)  # something the memory must contribute to reach
    counts = torch.full((32,), 192.0)

    start = float(kernel.gamma.mean())
    for _ in range(100):
        state = kernel.ingest(k, v, kernel.ingest_weights(enter, 256))
        n, d = kernel.read(q, state, counts)
        loss = (fuse_memory(o_s, lse_s, n, d) - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    end = float(kernel.gamma.mean())
    # Compounding, not linear: dL/dlog_gamma grows with gamma, so each decade arrives faster than
    # the last. Two orders of magnitude in 100 steps is the observed behaviour on the real model
    # (1.5e-8 -> 2.4e-5 over 240 steps there, against a much harder objective).
    assert end > start * 100, f"gamma only moved {start:.3e} -> {end:.3e} in 100 steps"


def test_mass_is_invariant_to_kernel_magnitude():
    """
    Scaling ``phi`` and ``psi`` up leaves ``d`` unchanged and ``n/d`` unchanged.

    The property that keeps ``gamma`` meaningful across layers. ``n/d`` was always
    scale-invariant (both terms carry the factor), but ``d`` -- the softmax *mass* -- was not until
    the L1 normalization: with ``/W`` it was proportional to the kernels' own learned magnitude, and
    the magnitudes diverge across depth. Measured on Qwen3-8B at one shared ``gamma``: ``d`` = 0.21
    at layer 0 against **6.3e3** at layer 35, i.e. late layers had collapsed into pure linear
    attention while early ones were still switched off. That is not something one scalar per head can
    absorb, and it is what drove the run to NaN once ``gamma`` grew.
    """
    torch.manual_seed(0)
    kernel = make_kernel()
    kernel.log_gamma.data.fill_(0.0)
    k, v = make_kv(256)
    q = torch.randn(1, HKV, 16, D)
    enter = ingestion_horizon(deadlines(torch.randn(HKV, 256), 64, force_sink=4))
    counts = torch.full((16,), 192.0)

    outs = []
    for scale in (1.0, 10.0, 100.0):
        with torch.no_grad():
            # Scale both readouts, which scales phi and psi by the same factor each.
            base = make_kernel()
            base.log_gamma.data.fill_(0.0)
            base.phi_out.mul_(scale)
            base.psi_out.mul_(scale)
            state = base.ingest(k, v, base.ingest_weights(enter, 256))
            outs.append(base.read(q, state, counts))

    d0, n0 = outs[0][1], outs[0][0]
    for n_i, d_i in outs[1:]:
        assert torch.allclose(d_i, d0, rtol=1e-3), (
            f"d moved with the kernel magnitude: {float(d0.mean()):.4e} -> {float(d_i.mean()):.4e}"
        )
        ratio0 = n0 / d0.unsqueeze(-1).clamp(min=1e-20)
        ratio_i = n_i / d_i.unsqueeze(-1).clamp(min=1e-20)
        assert torch.allclose(ratio0, ratio_i, atol=1e-3), "n/d moved with the kernel magnitude"


def test_streaming_and_oneshot_agree_at_matched_horizon():
    """
    The streaming schedule's last block equals one-shot compression **at the same horizon**.

    The two schedules are the train-time and deploy-time views, so they must describe the same model
    where they overlap -- otherwise the memory is trained against a partition that never occurs at
    inference, and both paths still run.

    "At the same horizon" is the whole content of the test. The streaming state for a block is taken
    at the block's **first** row (:func:`block_horizons`, deliberately: it is the smallest evicted set
    in the block, so no key is credited to the memory while some row still attends to it). One-shot
    compresses at the sequence end. So the two agree at the horizon the *block* uses, not at
    ``k_len - 1`` -- comparing them there is off by up to one block's worth of keys, which is what a
    first version of this test got wrong.
    """
    from kvpress.presses.gqa_indexer.memory_schedule import block_horizons

    torch.manual_seed(0)
    kernel = make_kernel()
    k_len = q_len = 512
    block = 128
    k, v = make_kv(k_len)
    dl = deadlines(torch.randn(HKV, k_len), 128, force_sink=4)
    H, z, W, counts = block_memory_states(kernel, k, v, dl, q_len=q_len, block=block)

    # The final block's horizon, which is what its state was accumulated at.
    horizon = int(block_horizons(q_len, k_len, block=block)[-1])
    enter = ingestion_horizon(dl)
    # One-shot at that horizon: ingest exactly the keys the block considers evicted. entry_block
    # encodes both conditions (evicted AND arrived), so reuse it rather than restating them.
    ingested = entry_block(enter, block_horizons(q_len, k_len, block=block)) <= (H.shape[2] - 1)
    weights = kernel.ingest_weights(enter, k_len) * ingested.to(torch.float32)
    state = kernel.ingest(k, v, weights)

    assert torch.allclose(state[0], H[:, :, -1], atol=1e-3), "H diverges at the final block"
    assert torch.allclose(state[1], z[:, :, -1], atol=1e-3), "z diverges at the final block"
    assert torch.equal(ingested.sum(-1), counts[:, -1]), "|E| diverges at the final block"
    assert horizon == q_len - block + (k_len - q_len)


# ----------------------------------------------------------------------
# Causality
# ----------------------------------------------------------------------
def test_memory_state_never_contains_a_rows_future():
    """
    No query row's memory state includes a key at a position it cannot see.

    **The bug this exists for.** The inference path originally read one state -- built over the whole
    evicted set -- from every query row, on the reasoning that the press compresses once and decode
    only appends. That is true for decode, where the single query sits after everything in the cache.
    Over a *prefill* it is a future leak: at ``L=8192`` query row 0 read a state accumulated from
    **6208** keys, every one of them ahead of it. Training was unaffected (the streaming schedule is
    causal by construction), so the loss curve looked healthy the whole way while RULER 8K went from
    73.71 to **4.00**.

    Checked as a property of the schedule rather than of the model, so it holds at any geometry: for
    each block, every ingested key must lie at or before the block's first row.
    """
    torch.manual_seed(0)
    k_len = q_len = 512
    block = 128
    dl = deadlines(torch.randn(HKV, k_len), 128, force_sink=4)
    horizons = block_horizons(q_len, k_len, block=block)
    entry = entry_block(ingestion_horizon(dl), horizons)
    n_blocks = q_len // block

    key_idx = torch.arange(k_len)
    for b in range(n_blocks):
        first_row = b * block + (k_len - q_len)
        ingested = entry <= b
        for h in range(HKV):
            future = ingested[h] & (key_idx > first_row)
            assert not bool(future.any()), (
                f"head {h} block {b}: {int(future.sum())} ingested keys are at positions > "
                f"{first_row}, the block's first query row -- the state leaks that row's future"
            )


def test_oneshot_schedule_refuses_a_full_prefill():
    """
    ``schedule="oneshot"`` raises rather than silently leaking on a full-sequence forward.

    The one-state construction is valid only where the query rows all sit after the keys the state
    holds -- decode, or a suffix of a longer cache. Over ``q_len == k_len`` it is the future leak
    above, and the failure is invisible: shapes match, no NaN, the loss curve is fine. So it is
    refused at the call rather than left to be discovered in an eval score.
    """
    from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer
    from kvpress.presses.gqa_indexer.press import GQAIndexerPress

    press = GQAIndexerPress(compression_ratio=0.5, scorer="scalar", memory=True)
    trainer = MemoryTrainer(press=press, schedule="oneshot")
    kernel = make_kernel()
    k_len = q_len = 256
    k, v = make_kv(k_len)
    q = torch.randn(1, HKV * 2, q_len, D)  # group = 2
    dl = deadlines(torch.randn(HKV, k_len), 64, force_sink=4)
    with pytest.raises(ValueError, match="own future"):
        trainer.memory_terms(
            kernel, q, k, v, dl,
            scores=torch.randn(HKV, k_len), q_len=q_len, group=2,
        )
