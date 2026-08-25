# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the query-independent scalar indexer (the SparseK/DMA baseline arm).

Three properties carry the design and each has a test that fails without it:

* the score plugs into the existing gate as its ``Di = 1`` case, so no new kernel is needed;
* the recency tilt keeps top-k **irreversible**, which is what makes eviction safe;
* pinning still closes the flat-gate no-op hole for a query-independent score.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig
from tests.fixtures import unit_test_model  # noqa: F401


def _mod(**kw):
    kw.setdefault("hidden_size", 64)
    kw.setdefault("n_heads", 4)  # per-head, the default granularity
    return ScalarIndexer(ScalarIndexerConfig(**kw)).double()


@pytest.mark.parametrize("mid_dim", [0, 32])
@pytest.mark.parametrize("n_heads", [1, 4])
def test_shape_and_fp32_score(mid_dim, n_heads):
    """Scores are ``(B, n_heads, Sk)`` and always fp32.

    fp32 is not incidental: at 32K keys a bf16 score resolves only a couple of hundred
    distinct values, so top-k would break large blocks of ties by index order.
    """
    m = ScalarIndexer(
        ScalarIndexerConfig(hidden_size=64, n_heads=n_heads, mid_dim=mid_dim)
    ).to(torch.bfloat16)
    out = m.score_keys(torch.randn(2, 10, 64, dtype=torch.bfloat16))
    assert out.shape == (2, n_heads, 10)
    assert out.dtype == torch.float32


def test_per_head_scores_are_actually_different():
    """Each KV head must get its own ranking, or per-head is a shared score in disguise.

    The measured reason it is worth having: the eight KV heads of Llama-3-8B agree on only
    14-17% of their top-k, so collapsing them to one score discards a real distinction.
    """
    m = _mod(n_heads=4)
    s = m.score_keys(torch.randn(1, 64, 64, dtype=torch.float64))[0]
    ranks = [set(s[h].topk(16).indices.tolist()) for h in range(4)]
    pairs = [
        len(a & b) / 16 for i, a in enumerate(ranks) for b in ranks[i + 1 :]
    ]
    assert max(pairs) < 0.95, f"heads pick near-identical keys (overlap {max(pairs):.2f})"


def test_input_norm_keeps_the_score_scale_independent_of_hidden_norm():
    """The score must not inherit the hidden state's magnitude.

    Hidden-state norms span two orders of magnitude across depth. Without the input norm the
    score std tracks them (measured 0.009 on a unit-norm stream against 0.887 on a norm-100
    one) while the attention logits it is added to stay at std ~1 -- so a single
    ``GATE_SCALE_INIT`` could not be right for more than one layer.
    """
    m = _mod(n_heads=1)
    base = torch.randn(128, 64, dtype=torch.float64)
    base = base / base.norm(dim=-1, keepdim=True)
    stds = [m.score_keys(base.unsqueeze(0) * scale).std().item() for scale in (1.0, 10.0, 100.0)]
    assert max(stds) / min(stds) < 1.05, f"score scale drifts with input norm: {stds}"


def test_gate_key_query_reproduces_the_score():
    """``qi . ki`` over the width-1 axis is the broadcast score.

    This is what lets gated_attention, its Triton kernel and the sink pin be reused verbatim:
    a per-key scalar is the ``Di = 1`` case of the pairwise gate with the query pinned to one.
    """
    m = _mod(n_heads=1)
    h = torch.randn(2, 9, 64, dtype=torch.float64)
    s = m.score_keys(h).double()
    pair = torch.einsum(
        "bhqd,bkd->bhqk",
        m.gate_query(q_len=9, bsz=2, n_kv_heads=4, dtype=torch.float64),
        m.gate_key(h, dtype=torch.float64),
    )
    torch.testing.assert_close(pair, s.unsqueeze(2).expand(2, 4, 9, 9))


def test_per_head_gate_routes_each_head_to_its_own_score():
    m = _mod(n_heads=4)
    h = torch.randn(2, 9, 64, dtype=torch.float64)
    pair = torch.einsum(
        "bhqd,bkd->bhqk",
        m.gate_query(q_len=9, bsz=2, n_kv_heads=4, dtype=torch.float64),
        m.gate_key(h, dtype=torch.float64),
    )
    torch.testing.assert_close(pair, m.expand_to_pairs(m.score_keys(h).double(), 9))


def test_key_offset_makes_chunked_prefill_match_one_pass():
    """The tilt is absolute, so splitting the prefill must not change any score.

    Without ``key_offset`` the tilt restarts per chunk and the score depends on how the
    prefill happened to be split -- a silent train/inference divergence.
    """
    m = _mod(pos_slope=1e-3)
    h = torch.randn(1, 20, 64, dtype=torch.float64)
    full = m.score_keys(h)
    chunked = torch.cat([m.score_keys(h[:, :12]), m.score_keys(h[:, 12:], key_offset=12)], dim=-1)
    torch.testing.assert_close(chunked, full)
    # and the failure mode is real, not hypothetical
    assert not torch.allclose(torch.cat([m.score_keys(h[:, :12]), m.score_keys(h[:, 12:])], dim=-1), full)


def test_recency_tilt_keeps_topk_irreversible():
    """A key that leaves the top-k never re-enters it.

    SparseK's whole memory argument rests on this: it is what makes dropping a key safe, so
    the KV entry can be freed immediately instead of retained in case a later query wants it.
    A tilt normalised by the current length would break it -- every old key's score would move
    as the sequence grows -- which is why ``pos_slope`` multiplies the absolute position.
    """
    m = _mod(pos_slope=1e-6)
    s = m.score_keys(torch.randn(1, 800, 64, dtype=torch.float64))[0, 0]
    keep = 32
    ever_out: set[int] = set()
    returns = 0
    for n in range(keep + 1, s.numel()):
        prefix = s[:n]
        kth = prefix.sort(descending=True).values[keep - 1]
        selected = set(torch.nonzero(prefix >= kth).flatten().tolist())
        returns += len(ever_out & selected)
        ever_out |= set(range(n)) - selected
        ever_out -= selected
    assert returns == 0, f"{returns} keys re-entered the top-k; eviction would be unsafe"


def test_pin_closes_the_no_op_hole_for_a_scalar_score():
    """A flat gate is a no-op, and the sink pin makes it unreachable here too.

    The pairwise indexer has the same hole (see gate_pin), but it has to be re-checked for a
    query-independent score: this arm is trained with the same end-to-end objective, and
    without the pin the router can satisfy the LM loss by reverting to the frozen dense
    backbone -- SAS's 18.8 against 54.4.
    """
    L, n_pin, dt = 40, 4, torch.float64
    torch.manual_seed(0)
    qk = torch.randn(L, L, dtype=dt)
    causal = torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1)
    pinned = torch.zeros(L, dtype=torch.bool)
    pinned[:n_pin] = True

    def gated(score: torch.Tensor, use_pin: bool) -> torch.Tensor:
        gate = torch.zeros(L, L, dtype=dt)
        for i in range(L):
            history = causal[i] & ~pinned if use_pin else causal[i]
            if history.any():
                gate[i, history] = score[history] - torch.logsumexp(score[history], 0)
            if use_pin:
                gate[i, causal[i] & pinned] = 0.0  # multiplier 1 == log-space 0
        return torch.softmax((qk + gate).masked_fill(~causal, -torch.inf), dim=-1)

    dense = torch.softmax(qk.masked_fill(~causal, -torch.inf), dim=-1)
    flats = (torch.zeros(L, dtype=dt), torch.full((L,), 3.7, dtype=dt))
    # Unpinned: both flat scores land exactly on the dense model -- the hole.
    assert max((gated(f, False) - dense).abs().max() for f in flats) < 1e-12
    # Pinned: neither can reach it.
    assert min((gated(f, True) - dense).abs().max() for f in flats) > 0.1


def test_mask_sends_padding_last_and_leaves_the_rest_alone():
    m = _mod(pos_slope=1e-3)
    h = torch.randn(1, 20, 64, dtype=torch.float64)
    unmasked = m.score_keys(h)
    keep = torch.ones(1, 20, dtype=torch.bool)
    keep[0, 5:8] = False
    out = m.score_keys(h, mask=keep)
    assert (out[0, 0, 5:8] == MASK_NEG).all()
    torch.testing.assert_close(out[0, 0, :5], unmasked[0, 0, :5])
    torch.testing.assert_close(m.score_keys(h, mask=keep.long()), out)  # int masks accepted too


def test_expand_to_pairs_does_not_allocate():
    """Query-independence means every row is identical, so the pairwise view is free."""
    m = _mod(n_heads=4)
    s = m.score_keys(torch.randn(1, 20, 64, dtype=torch.float64))
    wide = m.expand_to_pairs(s, 4096)
    assert wide.shape == (1, 4, 4096, 20)
    assert wide.untyped_storage().size() == s.untyped_storage().size()


def test_every_parameter_receives_gradient():
    m = ScalarIndexer(
        ScalarIndexerConfig(hidden_size=64, n_heads=4, mid_dim=32, gate_scale=True)
    )
    (m.score_keys(torch.randn(2, 16, 64)).sum() * m.require_gate_scale()).backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"parameters with no gradient: {dead}"


def test_require_gate_scale_raises_when_absent():
    with pytest.raises(RuntimeError, match="gate_scale"):
        _mod().require_gate_scale()


@pytest.mark.parametrize(
    "kwargs", [
        {"hidden_size": 0, "n_heads": 1},
        {"hidden_size": 8, "n_heads": 0},
        {"hidden_size": 8, "n_heads": 1, "pos_slope": -1.0},
        {"hidden_size": 8, "n_heads": 1, "mid_dim": -1},
    ]
)
def test_config_rejects_invalid_shapes(kwargs):
    with pytest.raises(ValueError):
        ScalarIndexerConfig(**kwargs)


def test_per_head_gate_query_rejects_head_mismatch():
    with pytest.raises(ValueError, match="KV heads"):
        _mod(n_heads=4).gate_query(q_len=4, bsz=1, n_kv_heads=8)


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="hidden_size"):
        _mod().score_keys(torch.randn(5, 8, dtype=torch.float64))


# ----------------------------------------------------------------------
# Protocol compatibility: the point of the arm is that everything downstream is shared
# ----------------------------------------------------------------------
def test_forward_is_the_pairwise_view_of_score_keys():
    """``forward`` matches GQAIndexer's signature and returns the broadcast, as a view.

    This is what lets the press, the query reductions and the loss helpers run over either
    scorer unchanged -- and it must stay a view, or the arm pays the ``O(Sq * Sk)`` cost it
    exists to avoid.
    """
    m = _mod(n_heads=4)
    h = torch.randn(2, 9, 64, dtype=torch.float64)
    pairs = m(h)  # positional cos/sin/mask, exactly as the press calls it
    assert pairs.shape == (2, 4, 9, 9)
    torch.testing.assert_close(pairs, m.expand_to_pairs(m.score_keys(h), 9))
    assert pairs.untyped_storage().size() == m.score_keys(h).untyped_storage().size()


def test_forward_applies_the_additive_press_mask():
    """The mask is added, not assigned -- matching GQAIndexer, so ``> MASK_NEG / 2`` still
    recovers validity downstream (which is how the press reconstructs it)."""
    m = _mod(n_heads=1)
    h = torch.randn(1, 6, 64, dtype=torch.float64)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.float64)
    mask[..., 3:] = MASK_NEG
    got = m(h, None, None, mask)
    unmasked = m.expand_to_pairs(m.score_keys(h), 6)
    assert (got[..., 3:] < MASK_NEG / 2).all()  # ranks last, and detectable as invalid
    torch.testing.assert_close(got[..., 3:], unmasked[..., 3:] + MASK_NEG)
    torch.testing.assert_close(got[..., :3], unmasked[..., :3])


def test_project_q_k_satisfy_the_trainer_protocol():
    """``indexer_qk`` in the e2e trainer calls exactly these three, positionally."""
    m = _mod(n_heads=4, gate_scale=True)
    h = torch.randn(2, 9, 64, dtype=torch.float64)
    q_idx, k_idx, gate = m.project_q(h, None, None), m.project_k(h, None, None), m.require_gate_scale()
    assert q_idx.shape == (2, 4, 9, 4) and k_idx.shape == (2, 9, 4)
    assert q_idx.dtype == k_idx.dtype == h.dtype  # the gate einsum needs one dtype
    torch.testing.assert_close(
        torch.einsum("bhqd,bkd->bhqk", q_idx, k_idx),
        m.expand_to_pairs(m.score_keys(h).double(), 9),
    )
    assert float(gate.detach()) == 1.0


def test_rope_tables_are_refused_rather_than_ignored():
    """Silently dropping RoPE would train a router against a signal it never sees."""
    m = _mod()
    h = torch.randn(1, 4, 64, dtype=torch.float64)
    tables = torch.ones(1, 4, 8, dtype=torch.float64)
    for call in (
        lambda: m.project_q(h, tables, tables),
        lambda: m.project_k(h, tables, tables),
        lambda: m(h, tables, tables),
    ):
        with pytest.raises(ValueError, match="rope_dim=0"):
            call()


def test_config_reports_rope_dim_zero_so_the_press_skips_rope():
    m = _mod()
    assert m.config.rope_dim == 0
    assert m.rope_dim == 0  # the press reads it off the module, not the config


def test_gated_attention_accepts_the_scalar_router():
    """The existing gate path runs unchanged, and its gradient reaches every parameter."""
    from kvpress.presses.gqa_indexer.gated_attention import gated_attention_reference

    torch.manual_seed(0)
    m = _mod(n_heads=4, mid_dim=256, gate_scale=True)
    h = torch.randn(1, 12, 64, dtype=torch.float64)
    d = 32
    out = gated_attention_reference(
        torch.randn(1, 8, 12, d, dtype=torch.float64),
        torch.randn(1, 4, 12, d, dtype=torch.float64),
        torch.randn(1, 4, 12, d, dtype=torch.float64),
        m.project_q(h, None, None),
        m.project_k(h, None, None),
        gate_scale=m.require_gate_scale(),
        scaling=d**-0.5,
    )
    (out[0] if isinstance(out, tuple) else out).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient through the gate for: {dead}"


# ----------------------------------------------------------------------
# Press and end-to-end trainer wiring: the A/B arms must share one code path
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scorer", ["pairwise", "scalar"])
def test_press_attaches_and_scores_with_either_scorer(unit_test_model, scorer):  # noqa: F811
    """Both arms reach ``token_scores`` through the same press method.

    The comparison is only meaningful if nothing but the router differs, so this asserts the
    press produces the same shape either way rather than that the scalar arm works in isolation.
    """
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import build_position_embeddings
    from kvpress.presses.gqa_indexer.press import get_language_model

    press = GQAIndexerPress(compression_ratio=0.5, scorer=scorer, gate_scale=True)
    # force_reinit: the model fixture is shared, so a previous parametrisation may have left
    # the other scorer attached. Replacing it is exactly what this test wants.
    press.post_init_from_model(unit_test_model, force_reinit=True)
    attn = get_language_model(unit_test_model).layers[0].self_attn
    indexer = press.get_indexer(attn)
    expected = ScalarIndexer if scorer == "scalar" else type(indexer)
    assert isinstance(indexer, expected)
    assert indexer.n_heads == unit_test_model.config.num_key_value_heads

    hidden = torch.randn(1, 12, unit_test_model.config.hidden_size, device=unit_test_model.device)
    kwargs = {} if scorer == "scalar" else {
        "position_embeddings": build_position_embeddings(unit_test_model, hidden)
    }
    assert press.token_scores(attn, hidden, kwargs, k_len=12).shape == (
        1,
        unit_test_model.config.num_key_value_heads,
        12,
    )


@pytest.mark.parametrize("stage,pin", [("dense", "sink"), ("sparse", "none")])
def test_e2e_trainer_trains_the_scalar_router(unit_test_model, stage, pin):  # noqa: F811
    """The LM loss reaches every scalar-router parameter, in both stages.

    ``indexer_qk`` is the trainer's only seam onto the router, and ScalarIndexer satisfies it
    (project_q / project_k / require_gate_scale / rope_dim) rather than the trainer branching on
    scorer type -- so this also asserts that no trainer change was needed.
    """
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer.e2e_trainer import E2EIndexerTrainer
    from kvpress.presses.gqa_indexer.press import get_language_model

    torch.manual_seed(0)
    press = GQAIndexerPress(
        compression_ratio=0.5, scorer="scalar", scalar_mid_dim=32, gate_scale=True
    )
    press.post_init_from_model(unit_test_model, force_reinit=True)
    trainer = E2EIndexerTrainer(press=press, stage=stage, keep_ratio=0.5, pin_mode=pin)
    # The model fixture is session-scoped, so freeze_backbone would leave requires_grad=False
    # on the backbone for every later test. Restore it afterwards.
    was_trainable = {n: p.requires_grad for n, p in unit_test_model.named_parameters()}
    trainer.freeze_backbone(unit_test_model)

    ids = torch.randint(0, unit_test_model.config.vocab_size, (1, 32), device=unit_test_model.device)
    with trainer.hooks(unit_test_model):
        loss = unit_test_model(ids, labels=ids, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss)

    dead = [
        f"layer{i}.{name}"
        for i, layer in enumerate(get_language_model(unit_test_model).layers)
        for name, p in press.get_indexer(layer.self_attn).named_parameters()
        if p.grad is None or not p.grad.any()
    ]
    assert not dead, f"LM loss did not reach: {dead}"
    # and only the router is trained
    trainable = [n for n, p in unit_test_model.named_parameters() if p.requires_grad]
    assert trainable and all(press.scorer_attr in n for n in trainable)

    for n, p in unit_test_model.named_parameters():
        p.requires_grad_(was_trainable.get(n, True))
        p.grad = None


def test_press_rejects_pairwise_only_geometry_for_the_scalar_arm(unit_test_model):  # noqa: F811
    """``head_dim`` / ``rope_dim`` would silently do nothing, so they are refused."""
    from kvpress import GQAIndexerPress

    for name in ("head_dim", "rope_dim"):
        press = GQAIndexerPress(scorer="scalar", **{name: 64})
        with pytest.raises(ValueError, match=name):
            press.post_init_from_model(unit_test_model, force_reinit=True)
