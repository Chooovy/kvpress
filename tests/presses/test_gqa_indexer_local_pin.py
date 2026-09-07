# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The ``local`` pin: an always-attended causal window held out of the gate's normalizer.

SP-KV's ``1_win(t, s) = 1[0 <= t - s < w]``. Keys inside the window are read at gate ``0`` *and*
excluded from the budget, so the router never spends mass on a neighbour and ranks only what lies
beyond the window -- the division of labour the eviction path already assumes at inference through
``force_local``.

Two properties are load-bearing:

1. **``local`` at width 1 IS ``self``.** The old mode is the ``n_local=1`` special case, so it must
    stay bit-identical -- otherwise this refactor silently retrained every existing
    ``pin_mode="self"`` configuration.
2. **All three implementations agree.** The fused Triton kernel, the concat-on-SDPA fold and the
    explicit reference each rebuild the window from their own ``(q_pos, k_pos)`` arithmetic. A
    disagreement between them is the kind of bug that trains fine and evaluates wrong.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.gate_pin import local_width, pinned_mask, pins_local, pins_sink
from kvpress.presses.gqa_indexer.gated_attention import gated_attention_full, gated_attention_reference
from kvpress.presses.gqa_indexer.triton_gated_attention import gated_kernels_available

CPU = torch.device("cpu")


def _tensors(q_len=96, k_len=96, dim=16, di=4, dtype=torch.float64, device="cpu", seed=0):
    """A small (q, k, v, q_idx, k_idx) set. fp64 by default, so the reference is the arbiter."""
    gen = torch.Generator().manual_seed(seed)
    bsz, n_heads, n_kv = 1, 4, 2
    made = (
        torch.randn(bsz, n_heads, q_len, dim, generator=gen, dtype=torch.float64),
        torch.randn(bsz, n_kv, k_len, dim, generator=gen, dtype=torch.float64),
        torch.randn(bsz, n_kv, k_len, dim, generator=gen, dtype=torch.float64),
        torch.randn(bsz, n_kv, q_len, di, generator=gen, dtype=torch.float64),
        torch.randn(bsz, k_len, di, generator=gen, dtype=torch.float64),
    )
    return tuple(t.to(device=device, dtype=dtype) for t in made)


def test_local_width_one_reproduces_the_self_mask():
    """``local`` at width 1 and ``self`` must select the same pairs -- the refactor's premise."""
    for q_len, k_len, offset in [(8, 8, 0), (4, 12, 8), (16, 16, 0)]:
        as_self = pinned_mask("self", q_len, k_len, CPU, query_offset=offset)
        as_local = pinned_mask("local", q_len, k_len, CPU, query_offset=offset, n_local=1)
        assert torch.equal(as_self, as_local), (q_len, k_len, offset)


def test_local_window_is_causal_and_exactly_w_wide():
    """``0 <= t - s < w``: never a future key, never more than ``w`` keys per row."""
    q_len = k_len = 64
    width = 8
    pinned = pinned_mask("local", q_len, k_len, CPU, n_local=width)
    q_pos = torch.arange(q_len).unsqueeze(-1)
    k_pos = torch.arange(k_len).unsqueeze(0)
    age = q_pos - k_pos

    assert not (pinned & (k_pos > q_pos)).any(), "the window reached a future key"
    torch.testing.assert_close(pinned, (age >= 0) & (age < width))
    assert pinned[width:].sum(-1).unique().tolist() == [width]


def test_self_ignores_n_local():
    """``self`` names an exact geometry, so a stray width must not silently widen it."""
    assert local_width("self", n_local=128) == 1
    assert local_width("self+sink", n_local=128) == 1
    assert pinned_mask("self", 16, 16, CPU, n_local=128).sum() == 16


def test_local_sink_is_the_union_of_both_pins():
    """``local+sink`` pins the leading keys AND the window, not one or the other."""
    q_len = k_len = 32
    n_sink, width = 4, 8
    both = pinned_mask("local+sink", q_len, k_len, CPU, n_sink=n_sink, n_local=width)
    only_local = pinned_mask("local", q_len, k_len, CPU, n_local=width)
    only_sink = pinned_mask("sink", q_len, k_len, CPU, n_sink=n_sink)

    assert torch.equal(both, only_local | only_sink)
    assert both[:, :n_sink].all()


def test_mode_predicates():
    assert pins_local("local") and pins_local("local+sink")
    assert pins_local("self"), "self is the width-1 window, so it takes the same path"
    assert not pins_local("sink") and not pins_local("none")
    assert pins_sink("local+sink") and not pins_sink("local")
    assert local_width("sink") == 0 and local_width("none") == 0
    with pytest.raises(ValueError, match="n_local >= 1"):
        local_width("local", n_local=0)


def test_window_shrinks_the_history_the_budget_is_spread_over():
    """
    The point of the pin, as a test: a wider window leaves fewer gated keys.

    This is what makes the router rank only distant keys. Were the window merely "always kept"
    without also leaving the normalizer, these counts would not move.
    """
    q_len = k_len = 128
    causal = torch.arange(k_len).unsqueeze(0) <= torch.arange(q_len).unsqueeze(-1)
    counts = []
    for width in (1, 8, 128):
        pinned = pinned_mask("local", q_len, k_len, CPU, n_local=width)
        counts.append(int((causal & ~pinned).sum()))

    assert counts[0] > counts[1] > counts[2], counts
    assert counts[2] == 0, "a window as wide as the sequence leaves no history to gate"


@pytest.mark.parametrize("pin_mode", ["local", "local+sink"])
@pytest.mark.parametrize("n_local", [1, 16])
def test_reference_matches_the_sdpa_path(pin_mode, n_local):
    """The two torch paths must agree; fp64 keeps this about geometry rather than rounding."""
    q, k, v, q_idx, k_idx = _tensors()
    kw = dict(gate_scale=0.5, gate_budget=1.0, pin_mode=pin_mode, n_sink=4, n_local=n_local)

    want = gated_attention_reference(q, k, v, q_idx, k_idx, **kw)
    got = gated_attention_full(q, k, v, q_idx, k_idx, **kw)
    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)


def test_local_width_one_matches_self_end_to_end():
    """
    The regression that matters most: an existing ``self`` run must be unchanged.

    Compared through the whole attention rather than only the mask, so a divergence in the
    normalizer or the dispatch is caught too.
    """
    q, k, v, q_idx, k_idx = _tensors()
    common = dict(gate_scale=0.5, gate_budget=1.0, n_sink=4)

    as_self = gated_attention_reference(q, k, v, q_idx, k_idx, pin_mode="self+sink", **common)
    as_local = gated_attention_reference(q, k, v, q_idx, k_idx, pin_mode="local+sink", n_local=1, **common)
    torch.testing.assert_close(as_local, as_self, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs CUDA")
@pytest.mark.parametrize("n_local", [1, 8, 128])
def test_triton_kernel_matches_the_reference(n_local):
    """The kernel rebuilds the window from its own index arithmetic -- check that it agrees."""
    q, k, v, q_idx, k_idx = _tensors(q_len=256, k_len=256, dim=32, di=8, dtype=torch.float32, device="cuda")
    if not gated_kernels_available(q, k, v, q_idx, k_idx):
        pytest.skip("gated Triton kernels unavailable")
    kw = dict(gate_scale=0.5, gate_budget=1.0, pin_mode="local+sink", n_sink=4, n_local=n_local)

    want = gated_attention_reference(q.double(), k.double(), v.double(), q_idx.double(), k_idx.double(), **kw)
    got = gated_attention_full(q, k, v, q_idx, k_idx, **kw)
    torch.testing.assert_close(got, want.float(), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs CUDA")
def test_triton_backward_matches_the_reference():
    """
    The gate's gradient is the whole objective, so the backward is checked too.

    The window enters the backward through the same ``pinned`` predicate as the forward, so a
    pinned pair must contribute no gradient to the score.
    """
    q, k, v, q_idx, k_idx = _tensors(q_len=128, k_len=128, dim=32, di=8, dtype=torch.float32, device="cuda")
    if not gated_kernels_available(q, k, v, q_idx, k_idx):
        pytest.skip("gated Triton kernels unavailable")
    kw = dict(gate_scale=0.5, gate_budget=1.0, pin_mode="local+sink", n_sink=4, n_local=16)

    grads = []
    for fn in (gated_attention_reference, gated_attention_full):
        qi = q_idx.clone().requires_grad_(True)
        ki = k_idx.clone().requires_grad_(True)
        fn(q, k, v, qi, ki, **kw).sum().backward()
        grads.append((qi.grad, ki.grad))

    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=2e-4, atol=2e-4)
