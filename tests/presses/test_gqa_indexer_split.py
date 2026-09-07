# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The C1/C2 split-context objective.

The load-bearing property is the **gradient structure**: with the loss on C2 only, every unit of
router gradient must land on C1 keys and none on pinned (sink or C2) keys. That is what makes the
objective train future utility rather than local retention, and it is the first thing that would
break if the pin were implemented as "drop the gate term" instead of "pin to 0".
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.delta_loss import IGNORE_INDEX
from kvpress.presses.gqa_indexer.gate_pin import history_mask, pinned_mask
from kvpress.presses.gqa_indexer.gated_attention import (
    gated_attention,
    gated_attention_full,
    gated_attention_reference,
)
from kvpress.presses.gqa_indexer.split_loss import (
    resolve_split,
    split_labels,
)


def tensors(B=1, Hq=2, Hkv=1, S=16, D=8, Di=2, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, Hq, S, D, dtype=torch.double),
        torch.randn(B, Hkv, S, D, dtype=torch.double),
        torch.randn(B, Hkv, S, D, dtype=torch.double),
        torch.randn(B, Hkv, S, Di, dtype=torch.double),
        torch.randn(B, S, Di, dtype=torch.double),
    )


# ------------------------------------------------------------------ geometry
def test_tail_pin_exempts_the_suffix_and_confines_history_to_the_prefix():
    S, split, ns = 12, 8, 2
    p = pinned_mask("sink", S, S, torch.device("cpu"), n_sink=ns, pin_from=split)
    assert p[10, :ns].all() and p[10, split:].all()
    assert not p[10, ns:split].any()
    # The normalizer's history is the gated, visible region: the prefix past the sink.
    h = history_mask(p, None, S, S, torch.device("cpu"))
    assert h[10].tolist() == [False, False] + [True] * 6 + [False] * 4
    # causality still applies inside the prefix
    assert h[3].tolist() == [False, False, True, True] + [False] * 8


def test_tail_pin_alone_needs_no_pin_mode():
    S = 8
    assert pinned_mask("none", S, S, torch.device("cpu")) is None
    p = pinned_mask("none", S, S, torch.device("cpu"), pin_from=4)
    assert p is not None and p[:, 4:].all() and not p[:, :4].any()


def test_out_of_range_split_is_inert():
    """A split at or past k_len pins nothing -- so a sequence shorter than the split is a no-op."""
    S = 8
    assert pinned_mask("none", S, S, torch.device("cpu"), pin_from=S) is None
    assert pinned_mask("none", S, S, torch.device("cpu"), pin_from=99) is None


# ------------------------------------------------------------------ the gradient property
def test_c2_loss_sends_all_router_gradient_to_c1():
    """The objective's whole point: C1 keys get gradient, pinned keys get exactly zero.

    If ``pin_from`` were implemented by dropping the gate term on cross-segment pairs instead of
    pinning C2's own keys, C1's gradient would be identically zero here and the arm would be
    training nothing.
    """
    q, k, v, qi, ki = tensors()
    ki = ki.clone().requires_grad_(True)
    split, n_sink = 8, 2
    out = gated_attention_reference(
        q, k, v, qi, ki,
        gate_scale=torch.tensor(1.0, dtype=torch.double),
        pin_mode="sink", n_sink=n_sink, pin_from=split,
    )
    out[:, :, split:, :].sum().backward()  # loss on C2 rows only
    g = ki.grad.abs().sum(-1)[0]
    assert (g[:n_sink] == 0).all(), "sink keys are pinned and must receive no gate gradient"
    assert (g[split:] == 0).all(), "C2 keys are pinned and must receive no gate gradient"
    assert (g[n_sink:split] > 0).all(), "every C1 key must receive gradient from the C2 loss"


def test_c2_queries_still_gate_c1_keys():
    """Removing the split must change C2's output -- otherwise the gate is not acting on C1."""
    q, k, v, qi, ki = tensors()
    gs = torch.tensor(1.0, dtype=torch.double)
    a = gated_attention_reference(q, k, v, qi, ki, gate_scale=gs, pin_mode="sink", n_sink=2)
    b = gated_attention_reference(
        q, k, v, qi, ki, gate_scale=gs, pin_mode="sink", n_sink=2, pin_from=8
    )
    assert (a - b).abs().max() > 1e-6


# ------------------------------------------------------------------ path agreement
@pytest.mark.parametrize("split", [None, 4, 8, 12])
def test_full_path_matches_reference_under_a_tail_pin(split):
    q, k, v, qi, ki = tensors()
    gs = torch.tensor(1.0, dtype=torch.double)
    kw = dict(gate_scale=gs, pin_mode="sink", n_sink=2, pin_from=split)
    ref = gated_attention_reference(q, k, v, qi, ki, **kw)
    got = gated_attention_full(q, k, v, qi, ki, **kw)
    assert torch.allclose(got, ref, atol=1e-12), (got - ref).abs().max()


def test_sparse_scope_rejects_a_split():
    q, k, v, qi, ki = tensors()
    idx = torch.zeros(1, 1, 16, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="pin_from is a full-scope quantity"):
        gated_attention(
            q, k, v, qi, ki, scope="sparse", indices=idx, pin_mode="none", pin_from=8
        )


# ------------------------------------------------------------------ label masking
def test_split_labels_masks_c1_and_the_gap():
    labels = torch.arange(20).unsqueeze(0)
    m = split_labels(labels, 8)
    assert (m[0, :8] == IGNORE_INDEX).all()
    assert (m[0, 8:] == labels[0, 8:]).all()
    g = split_labels(labels, 8, gap=4)
    assert (g[0, :12] == IGNORE_INDEX).all()
    assert (g[0, 12:] == labels[0, 12:]).all()


def test_split_labels_does_not_mutate_its_input():
    labels = torch.arange(10).unsqueeze(0).clone()
    before = labels.clone()
    split_labels(labels, 4)
    assert torch.equal(labels, before)


def test_split_labels_rejects_out_of_range():
    labels = torch.arange(10).unsqueeze(0)
    with pytest.raises(ValueError, match="out of range"):
        split_labels(labels, 11)


# ------------------------------------------------------------------ split resolution
@pytest.mark.parametrize("seq_len,frac,expect", [(16384, 0.5, 8192), (8192, 0.25, 2048)])
def test_resolve_split_scales_with_length(seq_len, frac, expect):
    assert resolve_split(seq_len, frac) == expect


def test_resolve_split_keeps_both_sides_usable():
    """A tiny fraction on a modest sequence is clamped, not silently accepted."""
    assert resolve_split(256, 0.01, min_side=64) == 64
    assert resolve_split(256, 0.99, min_side=64) == 192


def test_resolve_split_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="split_frac"):
        resolve_split(1024, 0.0)
    with pytest.raises(ValueError, match="split_frac"):
        resolve_split(1024, 1.0)
    with pytest.raises(ValueError, match="too short"):
        resolve_split(64, 0.5, min_side=64)
