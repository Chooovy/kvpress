# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the LongCE key-token probe.

The probe's whole value is that it decides whether to spend GPU time on the LongCE objective, so the
things that could make it *quietly wrong* are what is tested:

* **the window alignment.** ``per_token_ce`` is next-token-indexed, so the short-context tail is
  ``chunk[trunc_len-1 : ...]`` written at ``short_loss[start+trunc_len-1 : ...]``. An off-by-one here
  shifts every statistic onto neighbouring tokens and is invisible in every aggregate. The test is
  the structural one: the ``start=0`` window sees the true prefix so its loss must *equal* the long
  loss, and every later window is truncated so its must *differ*. Both directions matter -- the first
  catches a shift, the second catches a short context that is not actually short.
* **the ``scored`` mask covers exactly the positions with a counterfactual**, i.e. it excludes the
  first ``trunc_len - 1`` and nothing else. The reference implementation leaves those at weight 1;
  averaging a fabricated 1 over half an 8K sequence would drag every metric towards "no effect".
* **the rank statistics.** ``clamp(max=thre)`` ties a large block of weights at the ceiling, and
  ranking ties arbitrarily would inflate ``|spearman|`` -- manufacturing exactly the correlation the
  gate is meant to measure. Checked against known-answer cases including a fully-tied vector.
* **the gate direction.** A strongly positive ``spearman(w, L_long)`` must produce ``passes=False``.
  If the verdict were inverted the probe would greenlight the failure it exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

from evaluation.probe_longce_key_tokens import (
    average_ranks,
    check_alignment,
    long_context_losses,
    pearson,
    short_context_losses,
    spearman,
    unit_metrics,
    verdict,
)


#: The same toy checkpoint the rest of the suite uses (see tests/presses/test_gqa_indexer_delta_loss.py).
#: fp32 by default, which matters here: the alignment assertion wants the ``start=0`` window to match
#: the full-sequence loss *exactly*, and bf16 flash-attention is not bitwise invariant to sequence
#: length. Falls back to an equivalent config-built Llama when the hub is unreachable -- alignment is
#: a property of the indexing, not of what the model knows.
UNIT_TEST_MODEL = "MaxJeblick/llama2-0b-unit-test"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()
    except OSError:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=512,
            )
        ).eval()
        model.config._attn_implementation = "eager"
        return model


@pytest.fixture(scope="module")
def toy_losses(tiny_model):
    """The probe's two loss vectors at the criterion's stated geometry: L=64, trunc=16, window=8."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 64))
    long_loss = long_context_losses(tiny_model, input_ids, logit_chunk=16)
    short_loss, scored, windows = short_context_losses(
        tiny_model, input_ids, long_loss, trunc_len=16, window=8, logit_chunk=16
    )
    return input_ids, long_loss, short_loss, scored, windows


# ---------------------------------------------------------------------------------------------
# alignment -- the off-by-one that no aggregate would reveal
# ---------------------------------------------------------------------------------------------


def test_start0_window_reproduces_the_long_loss(toy_losses):
    """
    The ``start=0`` window's context *is* the true prefix, so its loss must match bit-for-bit.

    This is the assertion that rules out an off-by-one: shift the slice by one position and this
    window stops agreeing, while every downstream number stays plausible.
    """
    _, _, _, _, windows = toy_losses
    assert windows[0]["start"] == 0
    assert windows[0]["max_abs_diff_vs_long"] == pytest.approx(0.0, abs=1e-6)


def test_truncated_windows_differ_from_the_long_loss(toy_losses):
    """
    Every later window is genuinely truncated, so it must *not* reproduce the long loss.

    Without this direction the test would pass on a bug that made the short context equal to the full
    one -- in which case ``L_short - L_long`` is identically 0 and the weighting is a no-op that still
    reports a clean alignment.
    """
    _, _, _, _, windows = toy_losses
    later = [w["max_abs_diff_vs_long"] for w in windows[1:]]
    assert later, "the geometry should produce more than one window"
    assert min(later) > 1e-4


def test_check_alignment_accepts_the_real_geometry(toy_losses):
    """The runtime check passes on correct windows, and reports both directions as evidence."""
    _, _, _, _, windows = toy_losses
    result = check_alignment(windows, tol=1e-3)
    assert result["start0_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["truncated_max_abs_diff"] > 1e-3
    assert result["n_windows"] == len(windows)


def test_check_alignment_rejects_a_shifted_first_window():
    """A ``start=0`` window that disagrees with the long loss is an off-by-one, and must raise."""
    windows = [
        {"start": 0, "span": 8, "max_abs_diff_vs_long": 0.5},
        {"start": 8, "span": 8, "max_abs_diff_vs_long": 0.7},
    ]
    with pytest.raises(AssertionError, match="off-by-one"):
        check_alignment(windows, tol=1e-3)


def test_check_alignment_rejects_untruncated_windows():
    """If no window differs from the long loss, the short context is not short. Must raise."""
    windows = [
        {"start": 0, "span": 8, "max_abs_diff_vs_long": 0.0},
        {"start": 8, "span": 8, "max_abs_diff_vs_long": 0.0},
    ]
    with pytest.raises(AssertionError, match="no-op"):
        check_alignment(windows, tol=1e-3)


# ---------------------------------------------------------------------------------------------
# the scored mask -- our one deliberate deviation from the reference
# ---------------------------------------------------------------------------------------------


def test_scored_mask_excludes_exactly_the_positions_without_a_counterfactual(toy_losses):
    """
    Scored positions start at ``trunc_len - 1`` on the next-token index and run to the end.

    The reference leaves the earlier positions at weight 1. They have no shorter context to compare
    against, so that 1 is fabricated; at the 8K stage with ``trunc_len=4096`` it would be half the
    sequence, which is why they are masked out instead.
    """
    _, long_loss, _, scored, _ = toy_losses
    trunc_len = 16
    assert scored.shape == long_loss.shape
    assert not scored[: trunc_len - 1].any()
    assert scored[trunc_len - 1 :].all()
    assert int(scored.sum()) == long_loss.numel() - (trunc_len - 1)


def test_windows_tile_the_scored_region_without_gaps_or_overlap(toy_losses):
    """
    The spans must partition the scored region: a gap silently drops tokens, an overlap silently
    overwrites earlier measurements with later ones.
    """
    input_ids, _, _, scored, windows = toy_losses
    trunc_len, length = 16, input_ids.shape[-1]
    covered = []
    for w in windows:
        lo = w["start"] + trunc_len - 1
        covered.extend(range(lo, lo + w["span"]))
    assert len(covered) == len(set(covered)), "windows overlap"
    assert sorted(covered) == list(range(trunc_len - 1, length - 1))
    assert len(covered) == int(scored.sum())


def test_short_context_losses_refuses_a_trunc_len_that_leaves_nothing(tiny_model):
    """``trunc_len >= L`` has no scorable position; better to raise than to return an empty mask."""
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 32))
    long_loss = long_context_losses(tiny_model, input_ids, logit_chunk=16)
    with pytest.raises(ValueError, match="needs a sequence longer"):
        short_context_losses(
            tiny_model, input_ids, long_loss, trunc_len=32, window=8, logit_chunk=16
        )


# ---------------------------------------------------------------------------------------------
# rank statistics -- ties are the whole difficulty
# ---------------------------------------------------------------------------------------------


def test_average_ranks_shares_rank_across_ties():
    """
    Tied values take their mean rank, as ``scipy.stats.rankdata`` does.

    ``clamp(max=thre)`` produces a large tied block at the ceiling; breaking those ties arbitrarily
    would invent an ordering and inflate the correlation the gate reads.
    """
    ranks = average_ranks(torch.tensor([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_a_monotone_map_and_minus_one_when_reversed():
    """Rank correlation ignores the shape of the map, only its order."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, torch.exp(x)) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_correlations_are_zero_for_a_constant_vector():
    """
    A fully-tied vector has no ordering, so the correlation is 0 rather than NaN.

    This is the case a saturated ``clamp`` produces, and a NaN here would propagate into the gate and
    make the verdict unreadable instead of merely uninformative.
    """
    constant = torch.ones(8)
    assert pearson(constant, torch.arange(8.0)) == 0.0
    assert spearman(constant, torch.arange(8.0)) == 0.0


# ---------------------------------------------------------------------------------------------
# metrics: they must match the reference semantics and the trainer's statistic
# ---------------------------------------------------------------------------------------------


def _hand_built(long_values, short_values):
    """One scored unit from explicit losses, with a single unscored leading position."""
    long_loss = torch.tensor([99.0] + list(long_values))
    short_loss = torch.tensor([0.0] + list(short_values))
    scored = torch.tensor([False] + [True] * len(long_values))
    return long_loss, short_loss, scored


def test_weights_follow_loss_weight_exp_then_clamp():
    """``w = clamp(exp(L_short - L_long), max=thre)``, per ``loss_weight``."""
    long_loss, short_loss, scored = _hand_built([1.0, 1.0], [1.0, 4.0])
    m = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    # exp(0)=1 and exp(3)=20.09 -> clamped to 5. Mean 3.0, and half the weights at the ceiling.
    assert m["weight_mean"] == pytest.approx(3.0)
    assert m["weight_at_ceiling_frac"] == pytest.approx(0.5)


def test_metrics_ignore_unscored_positions():
    """
    The masked leading position carries a huge ``L_long``; no statistic may see it.

    If the mask leaked, ``all_loss_mean`` would jump -- which is exactly how the reference's
    fabricated weight-1 region would contaminate the comparison.
    """
    long_loss, short_loss, scored = _hand_built([1.0, 3.0], [1.0, 3.0])
    m = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    assert m["n_scored"] == 2
    assert m["all_loss_mean"] == pytest.approx(2.0)  # not (99+1+3)/3
    assert m["scored_frac"] == pytest.approx(2 / 3)


def test_key_token_criterion_requires_both_conditions():
    """
    ``find_key_token``: discrepancy > alpha **and** ``L_long < -beta``.

    The second condition is the entire reason this probe exists -- it is what the delta weighting
    lacked, and it is what excludes irreducible entropy. Three positions here each fail a different
    way, so dropping either condition changes the count.
    """
    #                       L_long,  L_short  -> discrepancy
    # 0: easy + big gap  -> key
    # 1: hard + big gap  -> rejected by beta (L_long 5.0 >= 2.0)
    # 2: easy + no gap   -> rejected by alpha
    long_loss, short_loss, scored = _hand_built([0.5, 5.0, 0.5], [4.0, 9.0, 0.6])
    m = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    c = m["criteria"]["alpha=2.0,beta=-2.0"]
    assert c["n_key"] == 1
    assert c["key_rate"] == pytest.approx(1 / 3)
    assert c["key_loss_mean"] == pytest.approx(0.5)
    # Two positions clear alpha; beta rejects one of them. Near-zero here would mean beta is inert
    # and the criterion is only a discrepancy threshold after all.
    assert c["beta_rejection_frac"] == pytest.approx(0.5)


def test_key_tokens_are_easier_than_average_under_the_long_context():
    """
    ``key_minus_all_loss`` must be negative: the signature the delta weighting could not produce.

    The ``beta`` condition caps key tokens' ``L_long`` at ``-beta``, so they sit *below* the mean.
    A positive value would mean the criterion selected the loss tail, i.e. the delta failure again.
    """
    long_loss, short_loss, scored = _hand_built([0.5, 6.0, 7.0], [4.0, 6.1, 7.1])
    m = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    c = m["criteria"]["alpha=2.0,beta=-2.0"]
    assert c["n_key"] == 1
    assert c["key_minus_all_loss"] < 0


def test_participation_is_one_for_uniform_weights_and_falls_when_concentrated():
    """
    ``(sum w)^2 / (n sum w^2)`` -- the trainer's statistic, so the numbers are comparable.

    ``1.0`` means the weighting did nothing whatever the loss says; the failed delta run logged
    0.13-0.18. Equal discrepancies give exactly 1.0, and one dominant weight drives it towards 1/n.
    """
    long_loss, short_loss, scored = _hand_built([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    uniform = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    assert uniform["weight_participation"] == pytest.approx(1.0)

    long_loss, short_loss, scored = _hand_built([1.0, 1.0, 1.0], [1.0, 1.0, 4.0])
    spiked = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    assert spiked["weight_participation"] < 0.7


def test_no_key_tokens_reports_none_rather_than_a_fabricated_zero():
    """
    An empty key set has no mean loss. ``None`` says so; ``0.0`` would read as "key tokens are
    trivially easy" and average into the aggregate as a real measurement.
    """
    long_loss, short_loss, scored = _hand_built([1.0, 1.0], [1.0, 1.0])
    m = unit_metrics(long_loss, short_loss, scored, threshold=5.0, criteria=[(2.0, -2.0)])
    c = m["criteria"]["alpha=2.0,beta=-2.0"]
    assert c["n_key"] == 0
    assert c["key_loss_mean"] is None
    assert c["key_minus_all_loss"] is None


# ---------------------------------------------------------------------------------------------
# the gate -- an inverted verdict would greenlight the failure this exists to prevent
# ---------------------------------------------------------------------------------------------


def _aggregate_stub(trunc_len, rho, *, key_rate=0.1, participation=0.5):
    return {
        str(trunc_len): {
            "n_docs": 1,
            "scalars": {
                "spearman_w_vs_long": {"mean": rho, "std": 0.0},
                "weight_participation": {"mean": participation, "std": 0.0},
                "scored_frac": {"mean": 0.5, "std": 0.0},
            },
            "criteria": {"alpha=2.0,beta=-2.0": {"key_rate": {"mean": key_rate}}},
        }
    }


def test_gate_fails_on_a_strongly_positive_correlation():
    """Strongly positive means "weight tracks loss", which is the delta collapse. Must not pass."""
    decision = verdict(_aggregate_stub(1024, 0.72), gate=0.4)
    assert decision["passes"] is False
    assert "STOP" in decision["note"]


def test_gate_passes_when_the_weight_is_decorrelated_from_the_loss():
    """Near zero means LongCE is a different quantity -- the condition for proceeding."""
    decision = verdict(_aggregate_stub(1024, 0.02), gate=0.4)
    assert decision["passes"] is True
    assert "PASS" in decision["note"]


def test_gate_is_decided_by_the_worst_trunc_len_not_the_best():
    """
    ``passes`` reads the maximum across the sweep.

    Taking the minimum would let one favourable ``trunc_len`` license the whole approach, which is
    the "tune it until it looks fine" move this step is meant to rule out.
    """
    aggregates = {**_aggregate_stub(1024, 0.05), **_aggregate_stub(4096, 0.80)}
    decision = verdict(aggregates, gate=0.4)
    assert decision["max_spearman_w_vs_long"] == pytest.approx(0.80)
    assert decision["passes"] is False


def test_recommended_trunc_len_prefers_low_correlation_then_a_usable_key_rate():
    """Lowest correlation wins; ``key_rate`` near 10% breaks a tie."""
    aggregates = {
        **_aggregate_stub(1024, 0.30, key_rate=0.11),
        **_aggregate_stub(2048, 0.05, key_rate=0.09),
        **_aggregate_stub(4096, 0.05, key_rate=0.0001),
    }
    decision = verdict(aggregates, gate=0.4)
    assert decision["recommended_trunc_len"] == 2048
