# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wiring tests for exact-``K`` chunk-subset indexer training.

The unit-level correctness lives in ``test_gqa_indexer_exact_k.py`` (the estimator) and
``test_gqa_indexer_exact_k_attention.py`` (the attention). This file checks the parts that only
exist once a real model is involved:

* the LM loss reaches **every** indexer parameter, through the frozen backbone;
* the backbone really is frozen, and frozen with ``requires_grad=False`` rather than ``no_grad`` --
  the gradient still has to *flow through* it;
* the attention implementation and the global registry are restored on exit, including on
  exception (the ``_global_mapping`` leak ``teacher_lse`` documents);
* the loss descends, and the diagnostics that a loss curve cannot provide are populated.

The last is the one worth reading: ``test_marginal_entropy_falls_as_the_router_commits`` is the
exact-K analogue of the additive arm's flat-gate check. A loss that falls while entropy stays flat
means the router learned to *use* a random subset rather than to *choose* one.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    ExactKIndexerTrainer,
    GQAIndexerPress,
    exact_k_indexer_training_step,
)


def tiny_model(n_layers=3, n_heads=8, n_kv_heads=4, hidden=64):
    """
    A small real Llama, so these tests exercise HF's actual attention plumbing.

    The config is constructed locally rather than pulled from
    ``hf-internal-testing/tiny-random-LlamaForCausalLM`` (which the rest of this suite uses):
    these tests must run on a box with no network, and a 5-retry HTTP timeout per test is a
    two-minute failure that says nothing about the code.
    """
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=hidden // n_heads,
        max_position_embeddings=512,
    )
    config._attn_implementation = "sdpa"
    return transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval(), config


def make_trainer(model, **kwargs):
    """Press + trainer at a geometry small enough to run on CPU in a test."""
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    defaults = dict(chunk_size=4, query_block=8, n_candidate=4, topk_chunk=2)
    defaults.update(kwargs)
    return press, ExactKIndexerTrainer(press=press, **defaults)


# ----------------------------------------------------------------------- configuration


def test_topk_above_the_pool_is_rejected():
    """``K > M`` is unreachable -- the subset is drawn from the pool -- so it raises."""
    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    with pytest.raises(ValueError, match="exceeds the candidate pool"):
        ExactKIndexerTrainer(press=press, n_candidate=8, topk_chunk=16)


def test_resolved_topk_derives_from_keep_ratio_and_clamps():
    model, _ = tiny_model()
    _, trainer = make_trainer(model, topk_chunk=0, keep_ratio=0.25, n_candidate=8)
    assert trainer.resolved_topk(32) == 8, "0.25 * 32 = 8"
    # Clamped to the pool, and to the chunk count, and never below 1.
    assert trainer.resolved_topk(64) == 8, "K cannot exceed the pool"
    assert trainer.resolved_topk(2) == 1, "K cannot exceed n_chunk, and must be >= 1"


def test_no_exploration_warns(caplog):
    """
    ``explore_frac=0`` is the ablation and says so.

    Without exploration a chunk outside top-M appears nowhere in the graph, so it receives exactly
    zero gradient and can never be promoted -- the same structural dead end that kept the
    selected-gate proxy at 0.0% recall.
    """
    import logging

    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    with caplog.at_level(logging.WARNING, logger="kvpress.presses.gqa_indexer.exact_k_trainer"):
        ExactKIndexerTrainer(press=press, explore_frac=0.0)
    assert any("zero gradient" in record.getMessage() for record in caplog.records)


def test_hard_mode_warns(caplog):
    """``hard=True`` disables exploration in the forward, so it is flagged as an ablation."""
    import logging

    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    with caplog.at_level(logging.WARNING, logger="kvpress.presses.gqa_indexer.exact_k_trainer"):
        ExactKIndexerTrainer(press=press, hard=True)
    assert any("no longer explores" in record.getMessage() for record in caplog.records)


# ----------------------------------------------------------------------- wiring


def test_lm_loss_reaches_every_indexer_parameter():
    """
    The whole point: the router's gradient comes from the model's own next-token loss.

    Every indexer parameter must receive a nonzero gradient. A parameter left at ``None`` means the
    routing path does not depend on it, so no amount of training would move it -- and the loss would
    still descend, via the other layers.
    """
    torch.manual_seed(0)
    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))

    loss = exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    assert torch.isfinite(loss), f"loss is {loss}"
    loss.backward()

    assert trainer.layers_routed == model.config.num_hidden_layers
    scored = 0
    for name, param in model.named_parameters():
        if ".indexer." not in name:
            continue
        if name.endswith("gate_scale"):
            # Deliberately UNUSED by this objective: the score is a routing logit that goes through
            # sigmoid into a Bernoulli probability, so a positive multiplier would be a temperature
            # rather than a scale match. The parameter exists only to keep the checkpoint
            # interchangeable with the gated arm's. Asserted rather than skipped, so that if a
            # future change starts reading it the test says so instead of silently passing.
            assert param.grad is None, (
                "gate_scale received gradient, so this objective now reads it -- either that is "
                "intended (update exact_k_trainer's docstring and give it a purpose) or the routing "
                "path picked it up by accident"
            )
            continue
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
        assert param.grad.abs().sum() > 0, f"{name} gradient is identically zero"
        scored += 1
    # w_q, w_k, and both norms' weight+bias, per layer.
    assert scored == 6 * model.config.num_hidden_layers, f"only {scored} params were trained"


def test_unused_gate_scale_does_not_break_gradient_clipping():
    """
    ``gate_scale`` gets no gradient, and ``clip_grad_norm_`` must tolerate that.

    It is in ``indexer_parameters`` (so the checkpoint round-trips) but off the routing path, so its
    ``.grad`` stays ``None``. ``clip_grad_norm_`` skips ``None`` grads, and ``AdamW`` skips
    ``None``-grad params -- but a run that crashed here would crash on step 1 of a real job, so it
    is worth pinning rather than assuming.
    """
    torch.manual_seed(10)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    assert any(p.numel() == 1 for p in params), "expected the gate_scale scalar in the param list"

    optimizer = torch.optim.AdamW(params, lr=1e-3)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
    exact_k_indexer_training_step(model, trainer, input_ids=input_ids).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert torch.isfinite(norm), "clipping produced a non-finite norm"
    optimizer.step()  # must not raise


def test_backbone_is_frozen_but_still_conducts_gradient():
    """
    Backbone parameters are frozen; the gradient nevertheless flows *through* them.

    That distinction is the reason ``freeze_backbone`` uses ``requires_grad=False`` and never
    ``torch.no_grad()``: the router sits below every layer above it, so a ``no_grad`` region would
    sever the path and the router would receive nothing while everything still looked healthy.
    """
    torch.manual_seed(1)
    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))

    loss = exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    backbone = [(n, p) for n, p in model.named_parameters() if ".indexer." not in n]
    assert backbone, "expected some non-indexer parameters"
    assert all(not p.requires_grad for _, p in backbone), "backbone must be frozen"
    assert all(p.grad is None for _, p in backbone), "a frozen parameter should accumulate no grad"
    # And the router did get gradient, which is only possible if the path through the frozen
    # backbone was live.
    router = model.model.layers[0].self_attn.indexer.w_q.weight
    assert router.grad is not None and router.grad.abs().sum() > 0


def test_freeze_by_module_identity_not_name():
    """
    A backbone parameter whose name contains ``indexer`` must stay frozen.

    ``freeze_backbone`` walks the indexer *modules* and collects parameter ids, rather than
    filtering names for a substring -- a name filter would train the impostor below, silently.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    impostor = torch.nn.Parameter(torch.zeros(4))
    model.register_parameter("indexer_lookalike", impostor)

    trainer.freeze_backbone(model)
    assert not impostor.requires_grad, "a name-substring filter would have trained this"
    assert model.model.layers[0].self_attn.indexer.w_q.weight.requires_grad


def test_attention_implementation_is_restored():
    """The config and the global registry go back to what they were, including on exception."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    before = model.config._attn_implementation
    mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    impl = "kvpress_gqa_indexer_exact_k"
    assert impl not in mapping

    with trainer.hooks(model):
        assert model.config._attn_implementation == impl
    assert model.config._attn_implementation == before
    # The leak teacher_lse documents: register() writes to the CLASS mapping while pop() touches
    # only the instance one, so a naive cleanup leaves this behind forever.
    assert impl not in mapping, "the registry entry leaked"

    with pytest.raises(RuntimeError, match="boom"):
        with trainer.hooks(model):
            raise RuntimeError("boom")
    assert model.config._attn_implementation == before
    assert impl not in mapping, "the registry entry leaked on the exception path"


def test_hooks_are_removed():
    """No forward pre-hook survives the context, or a second run would double-capture."""
    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    counts = lambda: [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
    before = counts()
    with trainer.hooks(model):
        assert all(a > b for a, b in zip(counts(), before))
    assert counts() == before


def test_missing_capture_hook_raises_rather_than_scoring_wrong():
    """
    Calling the routed attention without the pre-hook is an error, not a fallback.

    The attention interface never receives ``hidden_states``, so without the capture there is
    nothing to project the router from. Raising is the point: a silent fallback would train the
    router on a different input than the press scores with.
    """
    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    attn = model.model.layers[0].self_attn
    q = torch.randn(1, 8, 16, 8)
    kv = torch.randn(1, 4, 16, 8)
    with pytest.raises(RuntimeError, match="without its hidden_states"):
        trainer.routed_forward(attn, q, kv, kv, None, None)


def test_step_reports_when_no_layer_routed():
    """
    A model that kept its own attention is a wiring bug, and is reported as one rather than
    training nothing.

    Provoked by pointing the config at a *different* implementation from inside the block, which is
    what a config mismatch (e.g. a VL model whose text_config is the live one) looks like in
    practice: the hooks install fine, the forward runs fine, and the router is simply never called.
    """
    torch.manual_seed(9)
    model, _ = tiny_model(n_layers=1)
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16))

    original = ExactKIndexerTrainer.hooks

    @contextmanager
    def hooks_that_do_not_take_effect(self, m):
        with original(self, m):
            m.config._attn_implementation = "sdpa"  # undo the swap, so no layer routes
            yield self

    with patch.object(ExactKIndexerTrainer, "hooks", hooks_that_do_not_take_effect):
        with pytest.raises(RuntimeError, match="no layer ran the exact-K attention"):
            exact_k_indexer_training_step(model, trainer, input_ids=input_ids)


def test_score_query_tile_is_invariant():
    """
    ``chunk_scores`` gives the same answer at any ``score_query_tile``.

    Tiling exists because the intermediate ``(B, Hkv, Sq, Sk)`` logits are 8 GiB in fp32 at 16K
    against 0.5 MiB of output -- the untiled version OOM'd the first real 16K run. So the tile is a
    memory knob, and the result must not depend on it.

    The trap this pins down is ``query_offset``. Each tile builds its own causal mask, and letting
    that default (``k_len - q_len``) would restart causality at every tile, so each tile's queries
    would see the whole prefix as if they were at the sequence start. Attention would still be
    causal -- its own mask enforces that -- but the ROUTER would have been trained on scores that
    peeked ahead, and nothing downstream would flag it.
    """
    torch.manual_seed(11)
    model, _ = tiny_model(n_layers=1)
    press, trainer = make_trainer(model, chunk_size=4, query_block=4, n_candidate=4, topk_chunk=2)
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 32, model.config.hidden_size)

    # position_embeddings as the layer would supply them, since the indexer has rope_dim > 0.
    position_ids = torch.arange(32).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    kwargs = {"position_embeddings": (cos, sin)}

    reference = None
    # Byte budgets spanning one query_block per tile up to the whole sequence in one.
    for tile in (2048, 8192, 32768, 1 << 30):
        trainer.score_tile_bytes = tile
        scores = trainer.chunk_scores(attn, hidden, kwargs, 32)
        assert scores.shape == (1, model.config.num_key_value_heads, 8, 8)
        if reference is None:
            reference = scores
        else:
            err = (scores - reference).abs().max().item()
            assert err < 1e-5, f"budget={tile} changed the score by {err:.3e}"


def test_tiled_scores_stay_causal():
    """
    A query block's score does not depend on tokens after its own last query.

    The direct check on the ``query_offset`` trap above: perturbing a late token must leave the
    early blocks' scores untouched. If each tile restarted causality, block 0's score would move.
    """
    torch.manual_seed(12)
    model, _ = tiny_model(n_layers=1)
    _, trainer = make_trainer(model, chunk_size=4, query_block=4, n_candidate=4, topk_chunk=2)
    trainer.score_tile_bytes = 2048  # one query_block per tile, so several tiles
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 32, model.config.hidden_size)
    position_ids = torch.arange(32).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    kwargs = {"position_embeddings": (cos, sin)}

    before = trainer.chunk_scores(attn, hidden, kwargs, 32)
    perturbed = hidden.clone()
    perturbed[:, 24:] += 10.0
    after = trainer.chunk_scores(attn, perturbed, kwargs, 32)

    # Blocks 0..5 cover queries 0..23, so none of them may see token 24 onward. Their scores on
    # chunks 0..5 (tokens 0..23) must be unchanged; chunks 6-7 are masked for them anyway.
    assert torch.allclose(before[:, :, :6, :6], after[:, :, :6, :6], atol=1e-5), (
        "an early query block's score moved when a later token changed -- causality leaked, which "
        "is what a defaulted query_offset per tile would do"
    )
    assert not torch.allclose(before[:, :, 7:], after[:, :, 7:]), (
        "the last block should have seen the change, or the test proves nothing"
    )


def test_score_input_is_detached_by_default():
    """
    ``hidden_states`` does not receive gradient through the *score* path, only through attention.

    That second path is a real derivative but a per-layer feedback loop: the gradient it deposits in
    the residual stream is re-amplified by every router below. Measured on the real 36-layer model it
    diverges -- ``grad_norm`` 8.6e13 at 12 layers, ``inf`` at 24, ``nan`` at 36, against 1.1e6
    detached, with a per-layer amplification of 10-50x where the same backbone running dense
    attention shows 1.1x.

    The indexer weights still get gradient, which is what this asserts alongside the detachment: the
    router is trained, it just does not feed its own decision back into the stream.
    """
    torch.manual_seed(13)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model)
    assert trainer.detach_score_input, "detaching must be the default"
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 32, model.config.hidden_size, requires_grad=True)
    position_ids = torch.arange(32).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    scores = trainer.chunk_scores(attn, hidden, {"position_embeddings": (cos, sin)}, 32)
    scores.sum().backward()
    assert hidden.grad is None, "the score path must not reach hidden_states"
    w_q = attn.indexer.w_q.weight
    assert w_q.grad is not None and w_q.grad.abs().sum() > 0, (
        "detaching the INPUT must not detach the WEIGHTS -- the router still has to be trainable"
    )


def test_attached_score_input_does_reach_hidden():
    """
    ``detach_score_input=False`` restores the path, so the flag is doing what it says.

    Kept as an explicit test rather than trusting the default, because the failure mode is silent in
    both directions: attached, a run diverges only at depth; detached, nothing looks wrong at all.
    """
    torch.manual_seed(14)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model, detach_score_input=False)
    attn = model.model.layers[0].self_attn
    hidden = torch.randn(1, 32, model.config.hidden_size, requires_grad=True)
    position_ids = torch.arange(32).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    trainer.chunk_scores(attn, hidden, {"position_embeddings": (cos, sin)}, 32).sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_gradient_norm_does_not_grow_explosively_with_depth():
    """
    The router gradient stays bounded as layers are added.

    A weak version of the real finding (the tiny model is 3 layers and fp32, so it cannot reproduce
    a bf16 overflow), but it pins the *direction*: with the score input attached, the per-layer
    amplification compounds, and deeper is strictly worse. This asserts the detached path does not
    show that growth.
    """
    norms = {}
    for n_layers in (1, 2, 4):
        torch.manual_seed(15)
        model, _ = tiny_model(n_layers=n_layers)
        _, trainer = make_trainer(model)
        trainer.freeze_backbone(model)
        params = trainer.indexer_parameters(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids).backward()
        norms[n_layers] = float(torch.nn.utils.clip_grad_norm_(params, 1e30))
        assert all(
            torch.isfinite(p.grad).all() for p in params if p.grad is not None
        ), f"non-finite gradient at {n_layers} layers"

    # Growth with depth is expected (more parameters contribute), but it must be polynomial-ish,
    # not the 10-50x per layer the attached path showed.
    assert norms[4] < 100 * norms[1], f"gradient grew explosively with depth: {norms}"


# ----------------------------------------------------------------------- diagnostics


def test_diagnostics_are_populated():
    """Entropy, effective budget and (after two steps) Jaccard are recorded per layer."""
    torch.manual_seed(2)
    model, _ = tiny_model()
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))

    exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    n_layers = model.config.num_hidden_layers
    assert len(trainer.marginal_entropy) == n_layers
    assert len(trainer.effective_topk) == n_layers
    assert trainer.mean_marginal_entropy() > 0
    # The first step has nothing to compare against.
    assert trainer.mean_jaccard() is None
    assert not trainer.jaccard

    exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    assert len(trainer.jaccard) == n_layers
    jaccard = trainer.mean_jaccard()
    assert 0.0 <= jaccard <= 1.0, f"Jaccard {jaccard} out of range"


def test_effective_topk_reports_a_shortfall_near_the_diagonal():
    """
    Early query blocks cannot see ``K`` chunks, so the realized budget is below the configured one.

    Correct behaviour -- a block attends to every chunk it can see -- but the *effective* budget is
    then smaller than ``topk_chunk``, and only this statistic says so. A run reporting K=8 while
    actually attending to 3 chunks would otherwise look like a K=8 result.
    """
    torch.manual_seed(3)
    model, _ = tiny_model(n_layers=1)
    # chunk 4, query_block 4 over 16 tokens: block 0 sees one chunk, so a budget of 3 is
    # unreachable there.
    _, trainer = make_trainer(model, chunk_size=4, query_block=4, n_candidate=4, topk_chunk=3)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16))
    exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer.mean_effective_topk() < 3.0, (
        "some blocks cannot reach the budget, so the mean must be below it"
    )


def test_hard_mode_is_deterministic_across_steps():
    """
    ``hard=True`` selects the same subset twice for the same input -- Jaccard 1.0.

    This isolates the sampling: any instability the stochastic mode shows is the sampling's, not
    the scoring's.
    """
    torch.manual_seed(4)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model, hard=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
    with torch.no_grad():
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer.mean_jaccard() == pytest.approx(1.0)


def test_marginal_entropy_falls_as_the_router_commits():
    """
    **The exact-K analogue of the flat-gate check.**

    At init the marginals are uniform at ``K/M``, so entropy is at its maximum. A router whose
    scores have separated has low entropy. If a real run's loss falls while entropy stays at its
    init value, the router has learned to *use* whatever random subset it is handed rather than to
    *choose* one -- which is the one failure mode exact-K does not rule out structurally, and which
    no loss curve would reveal.

    Simulated here by scaling the indexer's own weights, rather than by training: it is the
    diagnostic's response that is under test.
    """
    torch.manual_seed(5)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))

    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.indexer.w_q.weight.mul_(0.0)  # every score identical -> uniform mu
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
        flat = trainer.mean_marginal_entropy()

        for layer in model.model.layers:
            layer.self_attn.indexer.w_q.weight.normal_(0.0, 20.0)  # strongly separated scores
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
        committed = trainer.mean_marginal_entropy()

    assert committed < flat, (
        f"a committed router must show lower marginal entropy: {committed:.4f} vs flat {flat:.4f}"
    )


def test_loss_descends():
    """
    A short optimization actually reduces the loss on a fixed batch.

    Deliberately overfitting one batch: the claim is only that the gradient points somewhere useful,
    which is the minimum bar for the estimator being wired up correctly. The forward is stochastic,
    so the comparison is between averages rather than between single steps.
    """
    torch.manual_seed(6)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=1e-2)

    losses = []
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(loss.item())

    first, last = sum(losses[:6]) / 6, sum(losses[-6:]) / 6
    assert last < first, f"loss did not descend: {first:.4f} -> {last:.4f}"


def test_reset_keeps_the_cross_step_selection():
    """
    ``reset`` clears the per-pass diagnostics but not the previous selection.

    Jaccard is a *between-step* quantity, so the state it needs deliberately outlives the reset that
    runs at the start of every ``hooks()`` block. Clearing it would make the metric permanently
    ``None`` and nothing would say why.
    """
    torch.manual_seed(7)
    model, _ = tiny_model(n_layers=1)
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16))
    with torch.no_grad():
        exact_k_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer._previous_selection, "the selection must be stored for the next step"
    trainer.reset()
    assert not trainer.marginal_entropy and trainer.layers_routed == 0
    assert trainer._previous_selection, "reset must not clear the cross-step selection"


def test_gradient_flows_to_deeper_layers_too():
    """
    Every layer's router is trained, not just the last one.

    The gradient reaches layer 0's router only by travelling back through every layer above it, so
    a broken path shows up as "the first layers have tiny or absent gradients" -- which a mean over
    layers would hide.
    """
    torch.manual_seed(8)
    model, _ = tiny_model(n_layers=4)
    _, trainer = make_trainer(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
    exact_k_indexer_training_step(model, trainer, input_ids=input_ids).backward()

    norms = [
        float(layer.self_attn.indexer.w_q.weight.grad.norm()) for layer in model.model.layers
    ]
    assert all(n > 0 for n in norms), f"a layer received no router gradient: {norms}"
