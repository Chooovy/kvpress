# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TrimKV's per-key lifetime on the scalar indexer.

The property everything else rests on is the **fold**: the decayed gate
``s_j + log_beta_j * (i - j) / ref`` is bilinear, so it equals an ordinary ``qi . ki`` at
``Di = 2 * n_heads``. If that identity holds, the fused kernel, the gate's ``lse`` normalizer and
the whole backward pass are correct by construction -- none of them know decay exists. Most of the
tests here are therefore about the fold and about the offsets that feed it.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.gated_attention import (
    gated_attention,
    gated_attention_reference,
)
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig
from kvpress.presses.gqa_indexer.train import infer_scalar_decay


def build(n_heads=4, hidden=32, mid_dim=8, decay=True, **kw):
    cfg = ScalarIndexerConfig(
        hidden_size=hidden,
        n_heads=n_heads,
        mid_dim=mid_dim,
        decay=decay,
        gate_scale=True,
        pos_slope=0.0,
        **kw,
    )
    torch.manual_seed(0)
    return ScalarIndexer(cfg).double()


@pytest.mark.parametrize("query_offset", [0, 7, 4096])
@pytest.mark.parametrize("mid_dim", [0, 8])
def test_fold_matches_explicit_pairwise_score(query_offset, mid_dim):
    """``project_q . project_k`` reproduces the explicit per-pair decayed score.

    This is the identity the whole port depends on. It is checked against ``forward``, which
    computes ``s_j + log_beta_j * age`` directly with no folding.

    Compared in fp32: ``forward``/``score_keys`` deliberately return fp32 regardless of module
    dtype (a bf16 score resolves too few distinct values for top-k), while ``project_k`` casts to
    the model's dtype for the gate einsum. So the two sides differ in dtype by design, and the
    tolerance is fp32's, not fp64's.
    """
    m = build(mid_dim=mid_dim)
    h = torch.randn(1, 24, 32, dtype=torch.double)
    qi = m.project_q(h, n_kv_heads=m.n_heads, query_offset=query_offset)
    ki = m.project_k(h)
    fold = torch.einsum("bhqd,bkd->bhqk", qi, ki).float()
    ref = m.forward(h, query_offset=query_offset).float()
    assert torch.allclose(fold, ref, atol=1e-5), (fold - ref).abs().max()


def test_indexer_width_doubles_only_with_decay():
    h = torch.randn(1, 8, 32, dtype=torch.double)
    assert build(decay=True).project_k(h).shape[-1] == 8  # 2 * n_heads
    assert build(decay=False).project_k(h).shape[-1] == 4  # n_heads


def test_decay_off_is_the_original_arm():
    """With decay off, q_idx is still the constant one-hot selector and k_idx the raw score.

    Guards the A/B: the decay arm is only interpretable if the flag being off reproduces the
    checkpointed baseline exactly, not approximately.
    """
    m = build(decay=False)
    h = torch.randn(1, 8, 32, dtype=torch.double)
    qi = m.project_q(h, n_kv_heads=m.n_heads)
    assert torch.equal(qi[0, :, 0, :], torch.eye(4, dtype=torch.double))
    assert torch.allclose(m.project_k(h).double(), m.score_keys(h).transpose(1, 2).double())
    # every query row identical -- the query-independence the deadline path assumes
    assert torch.allclose(qi[0, :, 0, :], qi[0, :, -1, :])


def test_key_offset_makes_a_suffix_score_like_a_slice():
    """Scoring a suffix at its true offset equals scoring the whole sequence and slicing.

    This is the decode invariant. The fold puts ``-log_beta_j * j / ref`` on the KEY side, so a
    wrong ``key_offset`` silently shifts every cached key's age.
    """
    m = build()
    h = torch.randn(1, 32, 32, dtype=torch.double)
    whole = m.project_k(h)
    suffix = m.project_k(h[:, 16:, :], key_offset=16)
    assert torch.allclose(whole[:, 16:, :], suffix)


def test_query_offset_shifts_ages_by_a_constant():
    """A query at absolute position ``i`` sees age ``i - j`` regardless of how rows are split."""
    m = build()
    h = torch.randn(1, 16, 32, dtype=torch.double)
    full = m.forward(h, query_offset=0)
    # row 8 of the full pass == row 0 of a pass whose first query sits at position 8
    shifted = m.forward(h[:, :1, :], key_hidden_states=h, query_offset=8)
    assert torch.allclose(full[:, :, 8:9, :], shifted)


def test_log_beta_is_non_positive_and_starts_at_init():
    """``log_beta <= 0`` always: a positive value would make keys grow with age.

    ``decay_init=0`` is the inert ablation and must construct rather than raise -- it is what an
    A/B against "decay present but off" needs.
    """
    for init in (-1.0, -0.1, 0.0):
        m = build(decay_init=init)
        h = torch.randn(1, 16, 32, dtype=torch.double)
        _, log_beta = m._score_and_decay(h)
        assert (log_beta <= 1e-12).all()
        # zero-init weights mean every key starts at exactly decay_init
        assert torch.allclose(log_beta, torch.full_like(log_beta, init), atol=1e-6)


def test_gated_attention_matches_reference_with_decay():
    """The real gate path agrees with the explicit reference once decay is folded in.

    ``gated_attention`` dispatches to the fused kernel or the concat identity; neither knows about
    decay. That they still match the reference is what says the fold is complete.
    """
    m = build(n_heads=2)
    torch.manual_seed(1)
    h = torch.randn(1, 20, 32, dtype=torch.double)
    q = torch.randn(1, 4, 20, 16, dtype=torch.double)
    k = torch.randn(1, 2, 20, 16, dtype=torch.double)
    v = torch.randn(1, 2, 20, 16, dtype=torch.double)
    qi, ki = m.project_q(h, n_kv_heads=2), m.project_k(h)
    gs = m.require_gate_scale().double()
    ref = gated_attention_reference(q, k, v, qi, ki, gate_scale=gs, pin_mode="sink", n_sink=2)
    got = gated_attention(q, k, v, qi, ki, scope="full", gate_scale=gs, pin_mode="sink", n_sink=2)
    assert torch.allclose(got, ref, atol=1e-12), (got - ref).abs().max()


def test_gradient_reaches_the_lifetime_head():
    """The LM loss must be able to train ``log_beta``, not just the magnitude.

    If this fails the arm is silently the plain scalar indexer with extra parameters.
    """
    m = build(n_heads=2)
    torch.manual_seed(1)
    h = torch.randn(1, 20, 32, dtype=torch.double)
    q = torch.randn(1, 4, 20, 16, dtype=torch.double)
    k = torch.randn(1, 2, 20, 16, dtype=torch.double)
    v = torch.randn(1, 2, 20, 16, dtype=torch.double)
    qi, ki = m.project_q(h, n_kv_heads=2), m.project_k(h)
    gated_attention(
        q, k, v, qi, ki, scope="full",
        gate_scale=m.require_gate_scale().double(), pin_mode="sink", n_sink=2,
    ).sum().backward()
    assert m.w_decay.weight.grad is not None and m.w_decay.weight.grad.norm() > 0
    assert m.w_decay.bias.grad is not None and m.w_decay.bias.grad.norm() > 0


def test_score_at_reduces_to_score_keys_at_age_zero():
    m = build()
    h = torch.randn(1, 16, 32, dtype=torch.double)
    assert torch.allclose(m.score_at(h, 0.0), m.score_keys(h))


def test_score_at_is_monotone_in_query_position():
    """Later queries see older keys, so every key's score can only fall (log_beta <= 0)."""
    m = build()
    h = torch.randn(1, 32, 32, dtype=torch.double)
    early, late = m.score_at(h, 8.0), m.score_at(h, 4096.0)
    assert (late <= early + 1e-12).all()


def test_decay_ref_scales_the_age_term():
    """Halving ``decay_ref`` doubles the age term -- the knob does what it says."""
    h = torch.randn(1, 32, 32, dtype=torch.double)
    a = build(decay_ref=16384.0)
    b = build(decay_ref=8192.0)
    ta = a.score_at(h, 1000.0) - a.score_keys(h)
    tb = b.score_at(h, 1000.0) - b.score_keys(h)
    assert torch.allclose(2 * ta, tb, atol=1e-9)


def test_config_rejects_positive_init_and_nonpositive_ref():
    with pytest.raises(ValueError, match="decay_init"):
        ScalarIndexerConfig(hidden_size=32, n_heads=4, decay=True, decay_init=0.5)
    with pytest.raises(ValueError, match="decay_ref"):
        ScalarIndexerConfig(hidden_size=32, n_heads=4, decay=True, decay_ref=0.0)


def test_checkpoint_detection_prefers_weights_over_config():
    """``decay`` is read from the presence of ``w_decay``; a contradicting config is an error."""
    with_decay = {f"m.0.indexer.{k}": v for k, v in build().state_dict().items()}
    without = {f"m.0.indexer.{k}": v for k, v in build(decay=False).state_dict().items()}

    got = infer_scalar_decay(with_decay, {"scalar_decay": True, "scalar_decay_ref": 8192.0})
    assert got == {"scalar_decay": True, "scalar_decay_ref": 8192.0}
    # a checkpoint predating decay records nothing and has no head -> press default (off)
    assert infer_scalar_decay(without, {}) == {}
    with pytest.raises(ValueError, match="w_decay"):
        infer_scalar_decay(without, {"scalar_decay": True})
    with pytest.raises(ValueError, match="w_decay"):
        infer_scalar_decay(with_decay, {"scalar_decay": False})
