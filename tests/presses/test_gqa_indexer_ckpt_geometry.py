# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for :func:`press_kwargs_from_checkpoint` -- rebuilding a checkpoint's geometry at load time.

Every evaluation entry point has to reconstruct the *same* router the checkpoint was trained at,
and this is where that is decided. The bug these tests pin: ``scorer='prefix'`` fell through
``evaluate_sparse.py``'s ``if scorer == 'scalar'`` test into the pairwise branch, so it received
none of its own geometry and the press used defaults.

That failed loudly for the dims -- every one is a parameter shape, so ``load_state_dict`` raises
``size mismatch`` -- which is why an evaluation at the default 256/128/128 still produced valid
numbers. ``pos_slope`` is the dangerous half: it is added inside ``score_keys`` and stored nowhere,
so a wrong value mis-scores silently with every tensor loading cleanly. Both halves are covered
below.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.indexer import GQAIndexer, GQAIndexerConfig
from kvpress.presses.gqa_indexer.prefix_indexer import PrefixIndexer, PrefixIndexerConfig
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig
from kvpress.presses.gqa_indexer.train import (
    indexer_state_dict,
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

UNIT_TEST_MODEL = "MaxJeblick/llama2-0b-unit-test"


def _named(module) -> dict:
    """A state dict under the prefix the real checkpoints use."""
    return {f"model.layers.0.self_attn.indexer.{k}": v for k, v in module.state_dict().items()}


@pytest.mark.parametrize(
    "mid_dim,head_dim,value_dim",
    [
        (256, 128, 128),  # the press defaults -- where the old fall-through happened to be right
        (32, 16, 24),  # non-default: what used to build wrong
        (0, 8, 8),  # the linear readout (mid_dim=0), a real configuration
    ],
)
def test_recovers_prefix_geometry_from_weights(mid_dim, head_dim, value_dim):
    """Dims come from the weights, so they cannot disagree with what will load.

    The defaults are parameterized deliberately: at 256/128/128 the buggy path built the correct
    module by coincidence, which is exactly why a working evaluation did not reveal it.
    """
    module = PrefixIndexer(
        PrefixIndexerConfig(
            hidden_size=128, n_heads=4, mid_dim=mid_dim, head_dim=head_dim, value_dim=value_dim
        )
    )
    scorer, kwargs = press_kwargs_from_checkpoint(_named(module), {})
    assert scorer == "prefix"
    assert kwargs["scalar_mid_dim"] == mid_dim
    assert kwargs["prefix_head_dim"] == head_dim
    assert kwargs["prefix_value_dim"] == value_dim


def test_pos_slope_comes_from_config_and_is_absent_when_unrecorded():
    """``pos_slope`` is the one setting weight loading cannot check, so it must be plumbed.

    And when the checkpoint does not record it, it must be *absent* from the kwargs rather than
    defaulted here -- that is what lets the caller warn instead of silently substituting a value it
    has no way to verify.
    """
    state = _named(PrefixIndexer(PrefixIndexerConfig(hidden_size=128, n_heads=4, mid_dim=16)))

    _, kwargs = press_kwargs_from_checkpoint(state, {"scalar_pos_slope": 4e-6})
    assert kwargs["scalar_pos_slope"] == pytest.approx(4e-6)

    _, bare = press_kwargs_from_checkpoint(state, {})
    assert "scalar_pos_slope" not in bare


def test_rejects_a_config_that_contradicts_its_own_weights():
    """An inconsistent checkpoint raises rather than letting one source quietly win."""
    module = PrefixIndexer(
        PrefixIndexerConfig(hidden_size=128, n_heads=4, mid_dim=16, head_dim=16, value_dim=24)
    )
    with pytest.raises(ValueError, match="prefix_value_dim"):
        press_kwargs_from_checkpoint(
            _named(module), {"scorer": "prefix", "prefix_value_dim": 128}
        )


def test_scalar_and_pairwise_checkpoints_still_resolve():
    """The helper must not disturb the two scorers that already worked."""
    scalar = ScalarIndexer(ScalarIndexerConfig(hidden_size=128, n_heads=4, mid_dim=64))
    scorer, kwargs = press_kwargs_from_checkpoint(_named(scalar), {})
    assert scorer == "scalar" and kwargs == {"scalar_mid_dim": 64}

    pairwise = GQAIndexer(GQAIndexerConfig(hidden_size=128, n_heads=4, head_dim=32, rope_dim=16))
    scorer, kwargs = press_kwargs_from_checkpoint(_named(pairwise), {})
    # Nothing is inferred for pairwise: head_dim/rope_dim stay CLI concerns, and rope_dim is not a
    # weight shape at all, so it could not be recovered here even in principle.
    assert scorer == "pairwise" and kwargs == {}


def test_round_trips_a_non_default_prefix_checkpoint_end_to_end():
    """Save at a non-default geometry, reload knowing only the weights, and score identically.

    Equal *scores* rather than equal shapes: a module can carry the right dims and still score
    differently if something unshaped -- ``pos_slope`` -- was dropped in transit.
    """
    from transformers import AutoModelForCausalLM

    geometry = dict(scalar_mid_dim=32, prefix_head_dim=16, prefix_value_dim=24)
    source = AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()
    source_press = GQAIndexerPress(
        compression_ratio=0.5, scorer="prefix", gate_scale=True, **geometry
    )
    source_press.post_init_from_model(source)
    saved = source_press.get_indexer(source.model.layers[0].self_attn)
    with torch.no_grad():
        saved.w_a.weight.normal_(0, 0.3)  # exercise the branch, not the zero-init fallback
    state = indexer_state_dict(source)

    target = AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()
    scorer, kwargs = press_kwargs_from_checkpoint(state, {})
    assert (kwargs["prefix_head_dim"], kwargs["prefix_value_dim"]) == (16, 24)
    target_press = GQAIndexerPress(
        compression_ratio=0.5, scorer=scorer, gate_scale=True, **kwargs
    )
    target_press.post_init_from_model(target)
    load_indexer_state_dict(target, state, "indexer")
    reloaded = target_press.get_indexer(target.model.layers[0].self_attn)

    hidden = torch.randn(1, 24, source.config.hidden_size)
    with torch.no_grad():
        assert torch.equal(saved.score_keys(hidden), reloaded.score_keys(hidden))


def test_default_press_geometry_would_not_load_a_non_default_checkpoint():
    """Documents *why* the dims half of the bug was loud rather than silent.

    Building the press without the checkpoint's dims -- the old fall-through -- yields the defaults
    and then fails on ``load_state_dict``. Worth pinning: it is the reason existing prefix results
    are trustworthy, and if a future refactor ever made this path load successfully, the failure
    would become silent and this test should be the thing that objects.
    """
    from transformers import AutoModelForCausalLM

    source = AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()
    press = GQAIndexerPress(
        compression_ratio=0.5, scorer="prefix", gate_scale=True,
        scalar_mid_dim=32, prefix_head_dim=16, prefix_value_dim=24,
    )
    press.post_init_from_model(source)
    state = indexer_state_dict(source)

    target = AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()
    default_press = GQAIndexerPress(compression_ratio=0.5, scorer="prefix", gate_scale=True)
    default_press.post_init_from_model(target)
    built = default_press.get_indexer(target.model.layers[0].self_attn)
    assert (built.mid_dim, built.head_dim, built.value_dim) == (256, 128, 128)

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_indexer_state_dict(target, state, "indexer")
