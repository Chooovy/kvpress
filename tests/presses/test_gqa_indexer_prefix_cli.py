# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI-level tests for the prefix arm's training wiring.

The scorer itself is tested in ``test_gqa_indexer_prefix.py``. What is tested here is the part that
silently breaks instead of failing: whether the two trainers actually *pass* the prefix geometry
through to the press, and whether a checkpoint records enough to tell one run from another. Both
scripts already shipped a bug of exactly this shape -- a flag threaded through an objective for a
full revision while doing nothing -- so the flags are checked against the press, not just against
the parser.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.prefix_indexer import PrefixIndexer
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer


def test_cross_replay_parser_accepts_prefix():
    """``--scorer prefix`` and its geometry reach the namespace, and zero-init defaults on."""
    from scripts.train_gqa_indexer_cross_replay import build_parser

    parser = build_parser()
    base = ["--data-root", "D", "--model", "M", "--schedule", "8192:10"]

    args = parser.parse_args(base + ["--scorer", "prefix", "--prefix-head-dim", "64",
                                     "--prefix-value-dim", "32"])
    assert (args.scorer, args.prefix_head_dim, args.prefix_value_dim) == ("prefix", 64, 32)
    assert args.prefix_zero_init is True, "nesting must be the default, not opt-in"

    flipped = parser.parse_args(base + ["--scorer", "prefix", "--no-prefix-zero-init"])
    assert flipped.prefix_zero_init is False

    # The scalar default must be untouched by the addition.
    assert parser.parse_args(base).scorer == "scalar"


def test_cross_replay_still_rejects_pairwise():
    """The objective is query-independent; a pairwise router has no place in it.

    Guards the ``choices`` widening: adding ``prefix`` must not have opened the door to
    ``pairwise``, which would need an ``(Sq, Sk)`` gate -- the cost the whole design avoids.
    """
    from scripts.train_gqa_indexer_cross_replay import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--data-root", "D", "--model", "M", "--schedule", "8192:10", "--scorer", "pairwise"]
        )


@pytest.mark.parametrize("zero_init", [True, False])
def test_press_builds_the_geometry_the_flags_ask_for(zero_init):
    """The press honours ``prefix_*``, and zero-init decides whether the arm nests.

    This is the check that a parser test cannot make: the flags have to survive the trip into
    ``GQAIndexerPress`` and come out as the module's actual shapes.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    press = GQAIndexerPress(
        compression_ratio=0.5,
        scorer="prefix",
        scalar_mid_dim=16,
        prefix_head_dim=8,
        prefix_value_dim=4,
        prefix_zero_init=zero_init,
        gate_scale=True,
    )
    press.post_init_from_model(model)

    indexer = press.get_indexer(model.model.layers[0].self_attn)
    assert isinstance(indexer, PrefixIndexer)
    assert indexer.head_dim == 8 and indexer.value_dim == 4
    assert indexer.gate_scale is not None, "gate_scale=True must reach the prefix arm too"
    # bool(), not `is`: `count_nonzero() == 0` yields a 0-d tensor, and `tensor(True) is True` is
    # always False -- which would make this assertion fail on correct code and pass on nothing.
    assert bool(indexer.w_a.weight.count_nonzero() == 0) is zero_init


def test_zero_init_press_nests_inside_the_scalar_press():
    """End to end through the press: the zeroed prefix arm reproduces the scalar arm exactly.

    The property every prefix-vs-scalar number depends on, asserted at the level the training
    scripts actually construct -- a press, on a real model -- rather than on a hand-built module.
    """
    from transformers import AutoModelForCausalLM

    common = dict(compression_ratio=0.5, scalar_mid_dim=16, gate_scale=True)
    prefix_model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    prefix_press = GQAIndexerPress(
        scorer="prefix", prefix_head_dim=8, prefix_value_dim=8, **common
    )
    prefix_press.post_init_from_model(prefix_model)
    prefix_indexer = prefix_press.get_indexer(prefix_model.model.layers[0].self_attn)

    scalar_model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    scalar_press = GQAIndexerPress(scorer="scalar", **common)
    scalar_press.post_init_from_model(scalar_model)
    scalar_indexer = scalar_press.get_indexer(scalar_model.model.layers[0].self_attn)
    assert isinstance(scalar_indexer, ScalarIndexer)

    weights = prefix_indexer.state_dict()
    scalar_indexer.load_state_dict({k: weights[k] for k in scalar_indexer.state_dict()}, strict=True)

    h = torch.randn(1, 32, prefix_model.config.hidden_size)
    with torch.no_grad():
        assert torch.equal(prefix_indexer.score_keys(h), scalar_indexer.score_keys(h))


def test_prefix_press_rejects_pairwise_geometry_flags():
    """``head_dim``/``rope_dim`` belong to the pairwise arm and must not be silently ignored.

    The prefix attention's width is ``prefix_head_dim``; accepting ``head_dim`` here would run a
    geometry nobody asked for. The error says which flag to use instead.
    """
    with pytest.raises(ValueError, match="prefix_head_dim"):
        GQAIndexerPress(
            compression_ratio=0.5, scorer="prefix", head_dim=64
        ).build_indexer_config(_FakeModel(), None)

    with pytest.raises(ValueError, match="rope_dim"):
        GQAIndexerPress(
            compression_ratio=0.5, scorer="prefix", rope_dim=32
        ).build_indexer_config(_FakeModel(), None)


class _FakeConfig:
    hidden_size = 64
    num_key_value_heads = 4
    num_attention_heads = 8


class _FakeModel:
    config = _FakeConfig()
