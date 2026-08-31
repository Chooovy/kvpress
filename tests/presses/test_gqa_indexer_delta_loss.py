# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the delta-weighted LM loss.

Four properties carry it, and each has a test that fails without it:

* **per-token CE agrees with the model's own reduced loss**, at every chunk size. The chunking
  exists to keep the ``(L, vocab)`` logits from being materialized, and it must not change the
  answer -- an off-by-one in the next-token shift would produce a plausible curve trained against
  the wrong targets.
* **``lambda -> inf`` recovers the ordinary mean exactly.** This makes the new objective a strict
  generalization of the old one rather than a different thing that happens to look similar, so
  ``--peak-lr`` transfers and the lambda sweep has a known endpoint.
* **the weights are detached and the gradient is ``w_t / sum w``.** If the gradient reached the
  weights, the objective would be optimizable by making the loss *larger* where the weight is
  large.
* **the dense reference really runs ungated.** If the gate were still installed, every delta would
  be ~0, the objective would silently collapse to the mean, and the loss curve would look fine.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.delta_loss import (
    IGNORE_INDEX,
    delta_weighted_loss,
    delta_weights,
    per_token_ce,
    shift_for_next_token,
    valid_mask,
)

UNIT_TEST_MODEL = "MaxJeblick/llama2-0b-unit-test"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()


@pytest.mark.parametrize("chunk_size", [1024, 17, 8])
def test_per_token_ce_matches_the_models_own_loss(tiny_model, chunk_size):
    """The mean of the per-token losses must equal what the model reports, at any chunk size.

    Both halves matter: agreeing with the model pins the next-token shift, and agreeing across
    chunk sizes pins the chunking. Either alone would let a bug through.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, 3000, (2, 64))
    with torch.no_grad():
        reference = tiny_model(input_ids=ids, labels=ids, use_cache=False).loss
        hidden = tiny_model.model(input_ids=ids, use_cache=False).last_hidden_state
        losses = per_token_ce(
            tiny_model.get_output_embeddings(), hidden, ids, chunk_size=chunk_size
        )
    assert losses.shape == (ids.shape[0] * (ids.shape[1] - 1),)
    assert losses.dtype == torch.float32
    assert torch.allclose(losses[valid_mask(ids)].mean(), reference, atol=1e-6)


def test_per_token_ce_respects_ignore_index(tiny_model):
    """Ignored positions must not enter the mean, and must be identified from the labels.

    ``cross_entropy`` reports ``0.0`` for an ignored row, and a confident position can also be
    ~0 -- so a mask derived from the losses would silently drop the model's best predictions.
    """
    torch.manual_seed(0)
    ids = torch.randint(0, 3000, (1, 32))
    labels = ids.clone()
    labels[:, :16] = IGNORE_INDEX

    with torch.no_grad():
        hidden = tiny_model.model(input_ids=ids, use_cache=False).last_hidden_state
        losses = per_token_ce(tiny_model.get_output_embeddings(), hidden, labels)
        reference = tiny_model(input_ids=ids, labels=labels, use_cache=False).loss

    mask = valid_mask(labels)
    assert int(mask.sum()) == 16  # positions 16..31 predicted from 15..30
    assert torch.allclose(losses[mask].mean(), reference, atol=1e-6)


def test_shift_alignment_is_next_token():
    """Position ``t``'s state predicts token ``t+1`` -- asserted on values, not on shapes."""
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    labels = torch.arange(2 * 5).reshape(2, 5)
    states, targets = shift_for_next_token(hidden, labels)
    assert states.shape == (8, 3) and targets.shape == (8,)
    # Row 0 of the flattened states is hidden[0, 0], and it must line up with labels[0, 1].
    assert torch.equal(states[0], hidden[0, 0])
    assert int(targets[0]) == int(labels[0, 1])
    # The last row of batch element 0 is hidden[0, 3] -> labels[0, 4]; hidden[0, 4] is dropped.
    assert torch.equal(states[3], hidden[0, 3])
    assert int(targets[3]) == int(labels[0, 4])


def test_shift_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="hidden_states must be"):
        shift_for_next_token(torch.zeros(4, 3), torch.zeros(4, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="do not match"):
        shift_for_next_token(torch.zeros(2, 5, 3), torch.zeros(2, 6, dtype=torch.long))


def test_weights_are_detached_and_the_gradient_is_w_over_sum_w():
    """``dL/dL_t = w_t / sum w`` exactly, and no gradient reaches the weights.

    The detachment is not an optimization: a gradient through ``w`` would add a term rewarding a
    *larger* ``L_t`` wherever the weight is large, so the objective could be reduced by getting
    worse.
    """
    torch.manual_seed(0)
    base = torch.rand(200).double() * 3 + 0.5
    sparse = base.clone().requires_grad_(True)
    dense = (base + torch.randn(200).double() * 0.4).detach()
    mask = torch.ones(200, dtype=torch.bool)

    weights = delta_weights(dense, sparse, lam=0.1, mask=mask)
    assert not weights.requires_grad

    loss, _ = delta_weighted_loss(sparse, weights, delta=dense - sparse.detach(), mask=mask)
    loss.backward()
    assert torch.allclose(sparse.grad, weights / weights.sum())


def test_lambda_interpolates_to_the_plain_mean():
    """Large ``lambda`` recovers ``mean(L)``, and participation rises to 1.0 as it does.

    Both are needed: the loss converging says the endpoint is right, and participation converging
    says the *weighting* is what vanished -- the diagnostic and the objective agreeing.
    """
    torch.manual_seed(0)
    sparse = torch.rand(2000).double() * 4 + 0.5
    dense = sparse - torch.randn(2000).double() * 0.5
    mask = torch.ones(2000, dtype=torch.bool)
    mask[:50] = False
    plain = sparse[mask].mean()

    previous_gap = float("inf")
    for lam in (1.0, 10.0, 100.0, 1e4, 1e6):
        weights = delta_weights(dense, sparse, lam=lam, mask=mask)
        loss, stats = delta_weighted_loss(sparse, weights, delta=dense - sparse, mask=mask)
        gap = abs(float(loss) - float(plain))
        assert gap < previous_gap, f"lam={lam} did not move closer to the mean"
        previous_gap = gap
    assert gap < 1e-8
    assert stats["weight_participation"] == pytest.approx(1.0, abs=1e-4)


def test_lambda_zero_drops_solved_positions_which_is_why_the_floor_exists():
    """``lam=0`` removes every position the router has already matched dense on.

    Documents the reason ``lambda`` is not optional: without it those positions receive no gradient
    at all, so nothing maintains them.
    """
    sparse = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    dense = torch.tensor([1.0, 2.5, 3.0, 5.0], dtype=torch.double)  # ahead only at 1 and 3
    mask = torch.ones(4, dtype=torch.bool)

    dropped = delta_weights(dense, sparse, lam=0.0, mask=mask)
    assert int((dropped > 0).sum()) == 2

    kept = delta_weights(dense, sparse, lam=0.1, mask=mask)
    assert int((kept > 0).sum()) == 4


def test_negative_delta_cannot_produce_a_negative_weight():
    """A position where sparse beats dense gets the floor, not a negative weight.

    A negative weight would push that token's loss *up*, which is the opposite of the objective.
    """
    sparse = torch.tensor([1.0, 2.0], dtype=torch.double)
    dense = torch.tensor([0.5, 5.0], dtype=torch.double)  # sparse wins at position 0
    weights = delta_weights(dense, sparse, lam=0.25)
    assert float(weights[0]) == pytest.approx(0.25)
    assert float(weights[1]) == pytest.approx(3.25)


def test_masked_positions_get_zero_weight():
    sparse = torch.rand(10).double()
    dense = sparse + 1.0
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:4] = True
    weights = delta_weights(dense, sparse, lam=0.5, mask=mask)
    assert torch.equal(weights[4:], torch.zeros(6, dtype=weights.dtype))
    assert bool((weights[:4] > 0).all())


def test_zero_weight_sum_raises_rather_than_dividing_by_zero():
    sparse = torch.ones(5).double()
    dense = torch.zeros(5).double()  # delta < 0 everywhere
    weights = delta_weights(dense, sparse, lam=0.0)
    with pytest.raises(RuntimeError, match="delta-lambda"):
        delta_weighted_loss(sparse, weights)


def test_rejects_bad_arguments():
    x = torch.ones(4).double()
    with pytest.raises(ValueError, match="lam must be non-negative"):
        delta_weights(x, x, lam=-1.0)
    with pytest.raises(ValueError, match="disagree"):
        delta_weights(x, torch.ones(5).double(), lam=0.1)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        per_token_ce(torch.nn.Linear(3, 7), torch.zeros(1, 4, 3), torch.zeros(1, 4).long(),
                     chunk_size=0)


# ----------------------------------------------------------------------
# End to end through the trainer
# ----------------------------------------------------------------------


def _trainer_and_press(model):
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import E2EIndexerTrainer

    press = GQAIndexerPress(
        compression_ratio=0.5, scorer="prefix", scalar_mid_dim=16,
        prefix_head_dim=8, prefix_value_dim=8, gate_scale=True,
    )
    press.post_init_from_model(model)
    return E2EIndexerTrainer(press=press, stage="dense", pin_mode="sink", n_sink=4)


def test_step_reduces_to_the_existing_objective_at_large_lambda(tiny_model):
    """The whole objective, end to end, must equal ``e2e_indexer_training_step`` as ``lam -> inf``.

    This is the strongest correctness statement available: the two code paths share nothing except
    the model, and one computes the loss itself in chunks while the other takes the model's own.
    """
    from kvpress.presses.gqa_indexer import (
        e2e_indexer_delta_weighted_step,
        e2e_indexer_training_step,
    )

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    ids = torch.randint(0, 3000, (1, 96))

    with torch.no_grad():
        reference = float(e2e_indexer_training_step(tiny_model, trainer, input_ids=ids))
        weighted, stats = e2e_indexer_delta_weighted_step(
            tiny_model, trainer, input_ids=ids, lam=1e6, logit_chunk=32
        )
    assert float(weighted) == pytest.approx(reference, abs=1e-6)
    assert stats["weight_participation"] == pytest.approx(1.0, abs=1e-3)


def test_step_produces_router_gradient(tiny_model):
    """The gated pass carries the graph, so the indexer parameters must receive gradient."""
    from kvpress.presses.gqa_indexer import e2e_indexer_delta_weighted_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    tiny_model.zero_grad(set_to_none=True)
    loss, _ = e2e_indexer_delta_weighted_step(
        tiny_model, trainer, input_ids=torch.randint(0, 3000, (1, 96)), lam=0.1, logit_chunk=32
    )
    loss.backward()
    grads = [
        p.grad for name, p in tiny_model.named_parameters()
        if "indexer" in name and p.grad is not None
    ]
    assert grads, "no indexer parameter received gradient"
    assert any(float(g.norm()) > 0 for g in grads)


def test_dense_reference_runs_ungated(tiny_model):
    """The dense pass must not run the gate, and the check must survive a stale counter.

    ``layers_gated`` accumulates across calls, so a naive check fires on a *correct* dense pass on
    the second step. Calling the step twice is what pins that.
    """
    from kvpress.presses.gqa_indexer import e2e_indexer_delta_weighted_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    ids = torch.randint(0, 3000, (1, 64))
    with torch.no_grad():
        for _ in range(2):
            loss, stats = e2e_indexer_delta_weighted_step(
                tiny_model, trainer, input_ids=ids, lam=0.1, logit_chunk=32
            )
    # The gated pass ran every layer; the dense one ran none (or the step would have raised).
    assert trainer.layers_gated == len(tiny_model.model.layers)
    assert stats["dense_loss"] > 0 and stats["sparse_loss"] > 0
