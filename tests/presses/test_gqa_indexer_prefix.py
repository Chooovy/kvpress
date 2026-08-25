# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the prefix-attention indexer -- the "read the key's whole prefix, stay
query-independent" arm.

Four properties carry the design, and each has a test that fails without it:

* **it is a strict superset of the scalar arm.** With the prefix branch zero-initialized the
  score is bit-identical to :class:`~.scalar_indexer.ScalarIndexer`'s, which is what makes the
  A/B single-variable -- "reads the prefix" is the only thing that changed.
* **the zero init is an escapable saddle, not a dead start.** ``w_a`` receives gradient at step
  one even though the branch's own projections do not, so the branch is live from step two.
* **the score stays query-independent, hence irreversible.** ``s_j`` depends only on
  ``prefix <= j``, so a key that leaves the top-k never returns -- the property that makes
  eviction safe, and the whole reason this arm exists rather than the pairwise one.
* **padding does not leak across the prefix attention.** A padded key must not influence any
  other key's score, which is stronger than the scalar arm needs (there, padding only had to
  be excluded from the final ranking).
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.prefix_indexer import (
    PrefixIndexer,
    PrefixIndexerConfig,
    score_variance_profile,
)
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig


def _prefix(**kw):
    kw.setdefault("hidden_size", 64)
    kw.setdefault("n_heads", 4)  # per-head, the granularity GQA evicts at
    kw.setdefault("mid_dim", 32)
    kw.setdefault("head_dim", 16)
    kw.setdefault("value_dim", 16)
    return PrefixIndexer(PrefixIndexerConfig(**kw)).double()


@pytest.mark.parametrize("mid_dim", [0, 32])
@pytest.mark.parametrize("n_heads", [1, 4])
def test_shape_and_fp32_score(mid_dim, n_heads):
    """Scores are ``(B, n_heads, Sk)`` and always fp32, matching the scalar arm's contract.

    fp32 is load-bearing for the same reason as there: a bf16 score resolves only ~12% distinct
    values at ``L=8030``, so top-k would decide large blocks of ties by index order.
    """
    m = PrefixIndexer(
        PrefixIndexerConfig(
            hidden_size=64, n_heads=n_heads, mid_dim=mid_dim, head_dim=16, value_dim=16
        )
    ).to(torch.bfloat16)
    s = m.score_keys(torch.randn(2, 12, 64, dtype=torch.bfloat16))
    assert s.shape == (2, n_heads, 12)
    assert s.dtype == torch.float32


@pytest.mark.parametrize("mid_dim", [0, 32])
def test_zero_init_is_bit_identical_to_scalar(mid_dim):
    """``w_a = 0`` reproduces :class:`ScalarIndexer` exactly, so the A/B is single-variable.

    Not "close to": bit-identical. If this drifts, every prefix-vs-scalar comparison is
    confounded by whatever else changed, and the confound would be invisible -- both arms would
    still train and both would still report a plausible loss.
    """
    p = _prefix(mid_dim=mid_dim)  # zero_init_prefix=True by default
    s = ScalarIndexer(
        ScalarIndexerConfig(hidden_size=64, n_heads=4, mid_dim=mid_dim)
    ).double()
    # The shared parameters are the same modules under the same names -- that is the point of
    # subclassing -- so the scalar arm loads straight out of the prefix arm's state dict.
    weights = p.state_dict()
    s.load_state_dict({k: weights[k] for k in s.state_dict()}, strict=True)

    h = torch.randn(2, 24, 64, dtype=torch.float64)
    assert torch.equal(p.score_keys(h), s.score_keys(h))


def test_nonzero_prefix_branch_actually_changes_the_score():
    """The converse of the test above: once ``w_a != 0`` the prefix branch does something.

    Without this, ``test_zero_init_is_bit_identical_to_scalar`` would also pass if the branch
    were wired up wrong and contributed nothing at any weight.
    """
    p = _prefix()
    h = torch.randn(2, 24, 64, dtype=torch.float64)
    before = p.score_keys(h)
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)
    assert not torch.allclose(before, p.score_keys(h))


def test_zero_init_prefix_branch_escapes_saddle():
    """Zero init is a one-step saddle, not a dead start.

    At ``w_a == 0`` the branch's projections get no gradient (``dL/da = W_a^T dL/dz = 0``), but
    ``w_a`` itself does (``dL/dW_a = dL/dz (x) norm(a)``, and ``norm(a) != 0``). So one step moves
    ``w_a`` off zero and the projections are live from step two. This is the check that
    distinguishes it from :attr:`~.indexer.GQAIndexer.gate_scale`'s genuinely-dead zero start,
    where the gradient is proportional to the parameter itself.
    """
    p = _prefix()
    h = torch.randn(2, 16, 64, dtype=torch.float64)

    p.score_keys(h).sum().backward()
    assert p.w_a.weight.grad.norm() > 0, "w_a must receive gradient at step one"
    for name in ("w_pq", "w_pk", "w_pv"):
        grad = getattr(p, name).weight.grad
        assert grad is None or grad.norm() == 0, f"{name} cannot have gradient while w_a == 0"

    torch.optim.SGD(p.parameters(), lr=0.1).step()
    p.zero_grad(set_to_none=True)
    p.score_keys(h).sum().backward()
    for name in ("w_pq", "w_pk", "w_pv"):
        assert getattr(p, name).weight.grad.norm() > 0, f"{name} must be live at step two"


def test_score_is_query_independent_and_causal():
    """``s_j`` depends only on ``prefix <= j``: perturbing key ``j+1..`` cannot move it.

    This is the property the whole arm rests on. A ``q . k`` inside the scorer invites the
    opposite reading, so it is asserted directly rather than argued: the query is projected from
    the *key token's own* hidden state, and the attention is causal.
    """
    p = _prefix()
    h = torch.randn(1, 32, 64, dtype=torch.float64)
    base = p.score_keys(h)

    cut = 20
    perturbed = h.clone()
    perturbed[:, cut:] = torch.randn_like(perturbed[:, cut:])
    after = p.score_keys(perturbed)

    assert torch.equal(base[..., :cut], after[..., :cut]), "future tokens changed a past score"
    assert not torch.allclose(base[..., cut:], after[..., cut:])


def test_recency_tilt_keeps_topk_irreversible():
    """A key that leaves the top-k never re-enters it.

    Inherited from the scalar arm, and re-tested here rather than assumed: irreversibility is
    what makes a dropped KV entry safe to *free*, and it is a joint property of the tilt and of
    the score being frozen at arrival. The prefix branch changes how the score is computed, so
    it could in principle break the second half.
    """
    p = _prefix(pos_slope=1e-6)
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)  # exercise the branch, not the scalar fallback
    s = p.score_keys(torch.randn(1, 400, 64, dtype=torch.float64))[0, 0]

    keep, ever_out, returns = 16, set(), 0
    for t in range(keep, s.shape[0]):
        top = set(torch.topk(s[: t + 1], keep).indices.tolist())
        returns += len(ever_out & top)
        ever_out |= set(range(t + 1)) - top
    assert returns == 0, f"{returns} keys re-entered the top-k"


def test_padding_does_not_leak_through_prefix_attention():
    """A padded key must not influence any *other* key's score.

    Stronger than the scalar arm needs: there a padded key only had to be excluded from the
    final ranking, because a score never read another token. Here the prefix attention would
    happily attend to garbage in the padded slots, so the mask has to reach inside it. Checked
    by changing the padded content and requiring every real score to be unmoved.
    """
    p = _prefix()
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)

    h = torch.randn(1, 20, 64, dtype=torch.float64)
    keep = torch.ones(1, 20, dtype=torch.bool)
    keep[:, 14:] = False

    first = p.score_keys(h, mask=keep)
    poisoned = h.clone()
    poisoned[:, 14:] = 1e3 * torch.randn_like(poisoned[:, 14:])
    second = p.score_keys(poisoned, mask=keep)

    assert torch.allclose(first[..., :14], second[..., :14], atol=1e-12)
    assert torch.equal(first[..., 14:], torch.full_like(first[..., 14:], MASK_NEG))


def test_unmasked_and_all_true_mask_agree():
    """An all-``True`` mask takes the flash path and must match the explicit-mask arithmetic.

    :meth:`score_keys` drops an all-real mask to stay on SDPA's ``is_causal`` fast path (``O(L)``
    memory instead of a quadratic bool mask). That optimization must not change the answer.
    """
    p = _prefix()
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)
    h = torch.randn(1, 18, 64, dtype=torch.float64)
    explicit = p.prefix_readout(p.in_norm(h), keep=torch.ones(1, 18, dtype=torch.bool))
    flash = p.prefix_readout(p.in_norm(h), keep=None)
    assert torch.allclose(explicit, flash, atol=1e-12)


def test_key_offset_is_rejected():
    """A non-zero ``key_offset`` raises rather than scoring a suffix against a truncated prefix.

    The scalar arm accepts an offset because its score reads one token. Here an offset means
    ``hidden_states`` is a suffix, so every key's prefix attention would silently see only part
    of the prefix -- and the score would then depend on how the prefill was chunked, which is
    exactly the class of silent framing bug this package has been bitten by before.
    """
    p = _prefix()
    h = torch.randn(1, 8, 64, dtype=torch.float64)
    with pytest.raises(ValueError, match="key_offset=0"):
        p.score_keys(h, key_offset=4)


def test_rope_is_rejected_and_rope_dim_is_zero():
    """The prefix attention is NoPE, and the module reports it as such.

    ``rope_dim == 0`` is what routes the press's RoPE plumbing and :mod:`~.cross_replay` down the
    scalar arm's path unchanged, so it is part of the interface rather than an implementation
    detail.
    """
    p = _prefix()
    assert p.rope_dim == 0 and p.config.rope_dim == 0
    cos = torch.ones(1, 8, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="rope"):
        p.forward(torch.randn(1, 8, 64, dtype=torch.float64), cos=cos, sin=cos)


def test_forward_is_a_broadcast_view_of_score_keys():
    """The pairwise-protocol view is a broadcast, not an ``O(Sq * Sk)`` recomputation.

    Inherited from the scalar arm, and worth pinning: if the prefix arm ever computed a genuine
    per-query score it would stop being query-independent, and the deadline path in
    :mod:`~.qi_flex_attention` would silently produce a different selection than the press.
    """
    p = _prefix()
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)
    h = torch.randn(1, 10, 64, dtype=torch.float64)
    pairs = p.forward(h)
    assert pairs.shape == (1, 4, 10, 10)
    keys = p.score_keys(h)
    for row in range(10):
        assert torch.equal(pairs[:, :, row, :], keys)


def test_score_variance_profile_flags_a_collapsing_score():
    """The diagnostic reports a monotone decay when there is one, and stays flat otherwise.

    ``softmax(...) V`` lies in the convex hull of ``{v_i}``, so a *pure* prefix readout loses
    across-key spread as position grows -- late keys become mutually indistinguishable while
    top-k still compares them against early ones. The ``W_in norm(h_j)`` residual is what
    mitigates it, so the healthy case here is the real module and the unhealthy case is a
    synthetic score that collapses by construction.
    """
    k_len = 512
    collapsing = torch.randn(1, 2, k_len) * torch.linspace(1.0, 0.02, k_len)
    _, var = score_variance_profile(collapsing, n_bins=4)
    assert var[0] > 4 * var[-1], "a collapsing score must show up as decaying variance"

    p = _prefix(hidden_size=128, head_dim=32, value_dim=32)
    with torch.no_grad():
        p.w_a.weight.normal_(0, 0.5)
    healthy = p.score_keys(torch.randn(1, k_len, 128, dtype=torch.float64))
    _, var = score_variance_profile(healthy, n_bins=4)
    assert var[-1] > 0.3 * var[0], f"residual path should hold variance up, got {var.tolist()}"


def test_variance_profile_ignores_masked_and_detrends_slope():
    """The diagnostic measures *content* spread: not the tilt, not the ``MASK_NEG`` sentinel."""
    k_len = 256
    flat = torch.zeros(1, 1, k_len)
    pos = torch.arange(k_len, dtype=torch.float32)
    _, var = score_variance_profile(flat + 1e-3 * pos, n_bins=4)
    assert torch.allclose(var, torch.zeros_like(var), atol=1e-10), "tilt was not detrended"

    with_pad = torch.randn(1, 1, k_len)
    with_pad[..., 200:] = MASK_NEG
    _, var = score_variance_profile(with_pad, n_bins=4)
    assert var[-1] < 10.0, "MASK_NEG leaked into the variance"


def test_prefix_config_rejects_bad_geometry():
    for kw in ({"head_dim": 0}, {"value_dim": -1}):
        with pytest.raises(ValueError):
            PrefixIndexerConfig(hidden_size=64, n_heads=4, **kw)
