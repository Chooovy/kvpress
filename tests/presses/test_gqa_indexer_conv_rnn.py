# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the two history-aggregating scorers: :mod:`~.conv_indexer` and :mod:`~.rnn_indexer`.

Mirrors ``test_gqa_indexer_prefix.py``'s structure, because these arms exist for the same reason
and have the same failure modes. The load-bearing assertions are:

* **zero-init nests the scalar arm bit-identically** -- without this the A/B is not single-variable
  and any RULER delta could be capacity rather than structure;
* **the branch is strictly causal** -- a scorer that peeks at ``h_j`` (conv tap 0) or at ``S_j``
  (unshifted state) is scoring with information the eviction decision does not have, which would
  make a *positive* result an artifact;
* **chunked prefill equals a one-pass score** -- the property every history arm can silently
  violate, and the one that decides whether the arm can be evaluated at all;
* **the scan matches an explicit sequential reference** -- the log-depth recurrence is the only
  non-obvious arithmetic in either module.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.conv_indexer import ConvIndexer, ConvIndexerConfig
from kvpress.presses.gqa_indexer.rnn_indexer import (
    RNNIndexer,
    RNNIndexerConfig,
    gated_scan,
)
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig
from tests.fixtures import unit_test_model  # noqa: F401

HIDDEN = 32
N_HEADS = 4


def _conv(**kw):
    """A float64 ConvIndexer, so the nesting assertions can be exact."""
    cfg = ConvIndexerConfig(
        hidden_size=HIDDEN,
        n_heads=N_HEADS,
        mid_dim=kw.pop("mid_dim", 16),
        conv_kernel=kw.pop("conv_kernel", 4),
        conv_dim=kw.pop("conv_dim", 8),
        **kw,
    )
    torch.manual_seed(0)
    return ConvIndexer(cfg).double().eval()


def _rnn(**kw):
    cfg = RNNIndexerConfig(
        hidden_size=HIDDEN,
        n_heads=N_HEADS,
        mid_dim=kw.pop("mid_dim", 16),
        state_dim=kw.pop("state_dim", 8),
        **kw,
    )
    torch.manual_seed(0)
    return RNNIndexer(cfg).double().eval()


def _scalar(mid_dim=16, **kw):
    cfg = ScalarIndexerConfig(hidden_size=HIDDEN, n_heads=N_HEADS, mid_dim=mid_dim, **kw)
    torch.manual_seed(0)
    return ScalarIndexer(cfg).double().eval()


def _bump(dtype=torch.float64):
    """A non-uniform per-channel perturbation.

    ``in_norm`` is a LayerNorm, so a uniform shift across channels is in its null space and would
    change nothing -- making any "this must move" assertion vacuously fail.
    """
    return torch.randn(HIDDEN, generator=torch.Generator().manual_seed(5), dtype=dtype)


def _h(bsz=2, seq=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(bsz, seq, HIDDEN, generator=g, dtype=torch.float64)


# ----------------------------------------------------------------------
# The scan
# ----------------------------------------------------------------------
@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 64])
def test_gated_scan_matches_sequential_reference(length):
    """Blelloch doubling == the explicit recurrence it replaces."""
    g = torch.Generator().manual_seed(length)
    a = torch.rand(2, length, 5, generator=g)
    b = torch.randn(2, length, 5, generator=g)

    got = gated_scan(a, b)

    want = torch.zeros_like(got)
    s = torch.zeros(2, 5)
    for t in range(length):
        s = a[:, t] * s + b[:, t]
        want[:, t] = s
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


def test_gated_scan_holds_and_forgets_at_the_limits():
    """a=1 accumulates, a=0 keeps only the current term -- the two ends of the gate's range."""
    b = torch.ones(1, 6, 1)
    held = gated_scan(torch.ones(1, 6, 1), b)
    assert torch.allclose(held.squeeze(), torch.arange(1, 7, dtype=torch.float32))
    forgot = gated_scan(torch.zeros(1, 6, 1), b)
    assert torch.allclose(forgot.squeeze(), torch.ones(6))


def test_gated_scan_rejects_bad_shapes():
    with pytest.raises(ValueError, match="same shape"):
        gated_scan(torch.ones(1, 4, 2), torch.ones(1, 5, 2))
    with pytest.raises(ValueError, match=r"\(B, L, D\)"):
        gated_scan(torch.ones(4, 2), torch.ones(4, 2))


# ----------------------------------------------------------------------
# Zero-init nesting: the property the A/B rests on
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mid_dim", [0, 16])
@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_zero_init_is_bit_identical_to_scalar(builder, mid_dim):
    """With w_a == 0 the history branch contributes exactly nothing."""
    mod = builder(mid_dim=mid_dim)
    ref = _scalar(mid_dim=mid_dim)
    # Share every parameter the two have in common; only the branch differs.
    ref.load_state_dict({k: v for k, v in mod.state_dict().items() if k in ref.state_dict()})

    h = _h()
    got = mod.score_keys(h)
    want = ref.score_keys(h)
    assert torch.equal(got, want), (got - want).abs().max()


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_nonzero_branch_actually_changes_the_score(builder):
    """Guards against a dead branch: the test above would also pass if it were unreachable."""
    mod = builder()
    h = _h()
    base = mod.score_keys(h)
    with torch.no_grad():
        mod.w_a.weight.normal_(0, 0.5)
    assert not torch.allclose(base, mod.score_keys(h))


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_zero_init_escapes_the_saddle(builder):
    """w_a gets gradient at w_a == 0, so the first step moves it off zero."""
    mod = builder().float()
    loss = mod.score_keys(_h().float()).square().mean()
    loss.backward()
    assert mod.w_a.weight.grad is not None
    assert mod.w_a.weight.grad.abs().max() > 0


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_every_parameter_receives_gradient(builder):
    """A parameter with no gradient path is a silently dead feature."""
    mod = builder().float()
    with torch.no_grad():  # leave the saddle so the branch's own params are reached
        mod.w_a.weight.normal_(0, 0.5)
    mod.score_keys(_h().float()).square().mean().backward()
    dead = [n for n, p in mod.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, dead


# ----------------------------------------------------------------------
# Causality / query-independence
# ----------------------------------------------------------------------
@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_score_is_causal_in_the_hidden_states(builder):
    """Perturbing h_t must not move any score at a position < t.

    The perturbation has to be non-uniform across channels: ``in_norm`` is a LayerNorm, so adding
    the same constant to every channel of a row lands in its null space and changes nothing
    anywhere -- which would make the vacuity guard below fail for a reason unrelated to causality.
    """
    mod = builder()
    with torch.no_grad():
        mod.w_a.weight.normal_(0, 0.5)
    h = _h(bsz=1, seq=12)
    base = mod.score_keys(h)

    t = 7
    h2 = h.clone()
    h2[0, t] += _bump(h.dtype)
    moved = mod.score_keys(h2)
    assert torch.equal(base[..., :t], moved[..., :t])
    # and it must actually affect something at or after t, else the test is vacuous
    assert not torch.equal(base[..., t:], moved[..., t:])


def test_conv_excludes_the_current_token():
    """exclude_self=True means h_j reaches the score only through W_in, not through the conv."""
    mod = _conv(exclude_self=True)
    with torch.no_grad():
        mod.w_a.weight.normal_(0, 0.5)
        # Kill the W_in path so the conv branch is the ONLY route from h to the score.
        mod.w_in.weight.zero_()
    h = _h(bsz=1, seq=10)
    a = mod.conv_readout(mod.in_norm(h))

    h2 = h.clone()
    h2[0, 4] += _bump(h.dtype)
    a2 = mod.conv_readout(mod.in_norm(h2))
    # Position 4's own readout must be untouched; position 5 onward must move.
    assert torch.equal(a[:, :5], a2[:, :5])
    assert not torch.equal(a[:, 5], a2[:, 5])


def test_conv_include_self_does_read_the_current_token():
    """The ablation flag genuinely flips the behaviour the test above pins down."""
    mod = _conv(exclude_self=False)
    h = _h(bsz=1, seq=10)
    a = mod.conv_readout(mod.in_norm(h))
    h2 = h.clone()
    h2[0, 4] += _bump(h.dtype)
    a2 = mod.conv_readout(mod.in_norm(h2))
    assert not torch.equal(a[:, 4], a2[:, 4])


def test_rnn_readout_is_the_shifted_state():
    """Row 0 sees no history, and row j sees S_{j-1} -- not S_j."""
    mod = _rnn()
    h = _h(bsz=1, seq=8)
    a = mod.state_readout(mod.in_norm(h))
    assert torch.equal(a[:, 0], torch.zeros_like(a[:, 0]))

    h2 = h.clone()
    h2[0, 3] += _bump(h.dtype)
    a2 = mod.state_readout(mod.in_norm(h2))
    assert torch.equal(a[:, :4], a2[:, :4])  # up to and including row 3
    assert not torch.equal(a[:, 4], a2[:, 4])  # row 4 reads S_3, which moved


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_forward_is_a_broadcast_view_of_score_keys(builder):
    """The pairwise protocol view must not be an independent computation."""
    mod = builder()
    h = _h()
    pairs = mod.forward(h)
    keys = mod.score_keys(h)
    assert pairs.shape == (h.shape[0], N_HEADS, h.shape[1], h.shape[1])
    assert torch.equal(pairs, keys.unsqueeze(2).expand_as(pairs))


# ----------------------------------------------------------------------
# Cache: chunked prefill and decode must equal a one-pass score
# ----------------------------------------------------------------------
@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_key_offset_is_rejected_without_a_cache(builder):
    mod = builder()
    with pytest.raises(ValueError, match="key_offset"):
        mod.score_keys(_h(), key_offset=4)


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_cache_rejects_a_diverged_offset(builder):
    mod = builder()
    mod.enable_cache()
    mod.score_keys(_h(seq=8))
    with pytest.raises(ValueError, match="diverged"):
        mod.score_keys(_h(seq=4), key_offset=99)


@pytest.mark.parametrize("chunks", [[8, 8], [4, 4, 4, 4], [12, 1, 1, 1, 1]])
@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_chunked_prefill_matches_one_pass(builder, chunks):
    """The property that decides whether the arm is evaluable at all."""
    mod = builder(conv_kernel=4) if builder is _conv else builder()
    with torch.no_grad():
        mod.w_a.weight.normal_(0, 0.5)
    h = _h(bsz=2, seq=sum(chunks))

    mod.disable_cache()
    want = mod.score_keys(h)

    mod.enable_cache()
    got, start = [], 0
    for n in chunks:
        got.append(mod.score_keys(h[:, start : start + n], key_offset=start))
        start += n
    got = torch.cat(got, dim=-1)

    assert torch.allclose(got, want, atol=1e-9), (got - want).abs().max()


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_cached_length_tracks_absolute_position(builder):
    mod = builder()
    mod.enable_cache()
    assert mod.cached_length == 0
    mod.score_keys(_h(seq=5))
    assert mod.cached_length == 5
    mod.score_keys(_h(seq=3), key_offset=5)
    assert mod.cached_length == 8
    mod.disable_cache()
    assert mod.cached_length == 0


def test_conv_cache_state_is_bounded_by_the_kernel():
    """The arm's O(1)-decode claim: the carry must not grow with the sequence."""
    mod = _conv(conv_kernel=4)
    mod.enable_cache()
    for i in range(6):
        mod.score_keys(_h(seq=8, seed=i), key_offset=8 * i)
        assert mod._cache_z.shape[-1] <= 4


def test_rnn_cache_state_is_one_vector():
    mod = _rnn(state_dim=8)
    mod.enable_cache()
    for i in range(4):
        mod.score_keys(_h(bsz=2, seq=8, seed=i), key_offset=8 * i)
        assert mod._cache_state.shape == (2, 8)


# ----------------------------------------------------------------------
# Decay / gate fold, inherited from the scalar arm
# ----------------------------------------------------------------------
@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_decay_widens_the_indexer_and_folds_exactly(builder):
    """The trunk override must leave the decay fold's identity intact.

    ``gate_key``/``gate_query`` are built in the module's dtype so the einsum is exact, while
    ``forward`` returns fp32 by contract (scores are always fp32 for top-k resolution) -- hence
    the explicit cast before comparing.
    """
    mod = builder(decay=True)
    assert mod.idx_dim == 2 * N_HEADS

    h = _h(bsz=1, seq=10)
    q_len = h.shape[1]
    ki = mod.gate_key(h, dtype=h.dtype)
    qi = mod.gate_query(q_len, 1, N_HEADS, device=h.device, dtype=h.dtype)
    folded = torch.einsum("bhqd,bkd->bhqk", qi, ki)
    explicit = mod.forward(h)
    assert torch.allclose(folded.float(), explicit, atol=1e-6), (
        folded.float() - explicit
    ).abs().max()


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_gradient_reaches_the_lifetime_head(builder):
    mod = builder(decay=True).float()
    mod.score_at(_h().float(), query_pos=32.0).square().mean().backward()
    assert mod.w_decay.weight.grad.abs().max() > 0


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_rope_is_rejected_and_rope_dim_is_zero(builder):
    mod = builder()
    assert mod.rope_dim == 0
    with pytest.raises(ValueError, match="RoPE|rotate"):
        mod.forward(_h(), cos=torch.ones(1), sin=torch.ones(1))


@pytest.mark.parametrize("builder", [_conv, _rnn])
def test_padding_mask_is_rejected(builder):
    """Documented scope limit: pack sequences rather than pad them."""
    mod = builder()
    h = _h(bsz=1, seq=8)
    keep = torch.ones(1, 8, dtype=torch.bool)
    keep[0, -2:] = False
    with pytest.raises(ValueError, match="padding mask"):
        mod.score_keys(h, mask=keep)
    # An all-true mask is the no-op case and must be accepted.
    mod.score_keys(h, mask=torch.ones(1, 8, dtype=torch.bool))


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------
def test_conv_config_rejects_bad_geometry():
    for kw in ({"conv_kernel": 0}, {"conv_dim": -1}):
        with pytest.raises(ValueError):
            ConvIndexerConfig(hidden_size=HIDDEN, n_heads=N_HEADS, **kw)


def test_rnn_config_rejects_bad_geometry():
    with pytest.raises(ValueError, match="state_dim"):
        RNNIndexerConfig(hidden_size=HIDDEN, n_heads=N_HEADS, state_dim=0)
    with pytest.raises(ValueError, match="gate_mode"):
        RNNIndexerConfig(hidden_size=HIDDEN, n_heads=N_HEADS, gate_mode="gru")
    with pytest.raises(ValueError, match="fixed_half_life"):
        RNNIndexerConfig(hidden_size=HIDDEN, n_heads=N_HEADS, fixed_half_life=0)


def test_rnn_fixed_gate_is_the_probed_ablation():
    """gate_mode='fixed' has one scalar retention and no per-token gate projection."""
    mod = _rnn(gate_mode="fixed")
    assert mod.w_g is None
    assert mod.logit_retain.numel() == 1
    names = dict(mod.named_parameters())
    assert "w_g.weight" not in names
    # It must still be a live, trainable parameter -- a frozen constant would be a different arm.
    mod = mod.float()
    with torch.no_grad():
        mod.w_a.weight.normal_(0, 0.5)
    mod.score_keys(_h().float()).square().mean().backward()
    assert mod.logit_retain.grad is not None and mod.logit_retain.grad.abs().item() > 0


# ----------------------------------------------------------------------
# Press integration, on the shared toy model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scorer", ["conv", "rnn"])
def test_press_attaches_and_scores(unit_test_model, scorer):  # noqa: F811
    """The new arms reach ``token_scores`` through the same press method as the scalar one."""
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer.conv_indexer import ConvIndexer
    from kvpress.presses.gqa_indexer.press import get_language_model
    from kvpress.presses.gqa_indexer.rnn_indexer import RNNIndexer

    press = GQAIndexerPress(compression_ratio=0.5, scorer=scorer, gate_scale=True)
    # force_reinit: the model fixture is shared, so another parametrisation may have left a
    # different scorer attached.
    press.post_init_from_model(unit_test_model, force_reinit=True)
    attn = get_language_model(unit_test_model).layers[0].self_attn
    indexer = press.get_indexer(attn)
    assert isinstance(indexer, ConvIndexer if scorer == "conv" else RNNIndexer)
    assert indexer.n_heads == unit_test_model.config.num_key_value_heads
    assert indexer.rope_dim == 0

    hidden = torch.randn(1, 12, unit_test_model.config.hidden_size, device=unit_test_model.device)
    assert press.token_scores(attn, hidden, {}, k_len=12).shape == (
        1,
        unit_test_model.config.num_key_value_heads,
        12,
    )


@pytest.mark.parametrize("scorer", ["conv", "rnn"])
def test_press_rejects_head_dim_with_a_useful_hint(scorer):
    """--head_dim has no meaning for these arms; each names its own width knob instead."""
    from kvpress import GQAIndexerPress

    hint = "conv_dim" if scorer == "conv" else "state_dim"
    press = GQAIndexerPress(compression_ratio=0.5, scorer=scorer, head_dim=64)
    with pytest.raises(ValueError, match=hint):
        press.build_indexer_config(_FakeModel(), _FakeModel())


class _FakeModel:
    """Minimal stand-in exposing the two attributes ``build_indexer_config`` reads."""

    class config:  # noqa: N801
        hidden_size = 64
        num_key_value_heads = 4
        num_attention_heads = 8


@pytest.mark.parametrize("scorer", ["conv", "rnn"])
def test_e2e_trainer_trains_the_history_router(unit_test_model, scorer):  # noqa: F811
    """The LM loss reaches every parameter of both arms, through the unmodified trainer seam.

    ``indexer_qk`` is the trainer's only seam onto the router, and both arms satisfy it by
    inheriting ``project_q``/``require_gate_scale`` and keeping ``project_k``'s full signature --
    so this also asserts that no trainer change was needed. A missing gradient here is what a
    narrowed ``project_k`` signature previously caused for the prefix arm.

    ``w_a`` is perturbed off zero first: at the zero-init saddle the branch's own projections
    correctly receive no gradient on the *first* step, which is a documented property rather than
    a defect, and would otherwise read as dead parameters.
    """
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer.e2e_trainer import E2EIndexerTrainer
    from kvpress.presses.gqa_indexer.press import get_language_model

    torch.manual_seed(0)
    extra = {"conv_dim": 32, "conv_kernel": 4} if scorer == "conv" else {"state_dim": 32}
    press = GQAIndexerPress(
        compression_ratio=0.5,
        scorer=scorer,
        scalar_mid_dim=32,
        gate_scale=True,
        **extra,
    )
    press.post_init_from_model(unit_test_model, force_reinit=True)
    layers = get_language_model(unit_test_model).layers
    with torch.no_grad():
        for layer in layers:
            press.get_indexer(layer.self_attn).w_a.weight.normal_(0, 0.5)

    trainer = E2EIndexerTrainer(press=press, stage="dense", keep_ratio=0.5, pin_mode="sink")
    # The model fixture is session-scoped, so restore requires_grad afterwards.
    was_trainable = {n: p.requires_grad for n, p in unit_test_model.named_parameters()}
    trainer.freeze_backbone(unit_test_model)

    ids = torch.randint(
        0, unit_test_model.config.vocab_size, (1, 32), device=unit_test_model.device
    )
    with trainer.hooks(unit_test_model):
        loss = unit_test_model(ids, labels=ids, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss)

    dead = [
        f"layer{i}.{name}"
        for i, layer in enumerate(layers)
        for name, p in press.get_indexer(layer.self_attn).named_parameters()
        if p.grad is None or not p.grad.any()
    ]
    assert not dead, f"LM loss did not reach: {dead}"
    trainable = [n for n, p in unit_test_model.named_parameters() if p.requires_grad]
    assert trainable and all(press.scorer_attr in n for n in trainable)

    for n, p in unit_test_model.named_parameters():
        p.requires_grad_(was_trainable.get(n, True))
