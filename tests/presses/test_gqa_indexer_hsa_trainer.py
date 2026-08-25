# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wiring tests for two-level (HSA) chunk-attention indexer training.

Operator correctness lives in ``test_gqa_indexer_hsa.py``. This file checks what only exists once a
real model is involved:

* the LM loss reaches **every** indexer parameter, through the frozen backbone;
* the backbone is frozen with ``requires_grad=False`` rather than ``no_grad`` -- the gradient still
  has to *flow through* it, or the router receives nothing;
* the attention implementation and the global registry are restored on exit, including on exception
  (the ``_global_mapping`` leak ``teacher_lse`` documents);
* the loss descends and the diagnostics populate.

Two tests carry the argument for this objective over the other three:

``test_entropy_falls_as_the_router_commits`` is this arm's analogue of the additive arm's flat-gate
check -- a loss that falls while entropy stays at 1.0 means the router learned to *average* rather
than to *choose*.

``test_score_lse_correlation_rises_with_training`` has no counterpart in any other arm: the optimal
score is known in closed form (``ROUTER_LEARNABILITY.md`` §6), so progress is measurable directly
rather than through an oracle or a swap experiment.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    GQAIndexerPress,
    HSAIndexerTrainer,
    hsa_indexer_training_step,
)


def tiny_model(n_layers=3, n_heads=8, n_kv_heads=4, hidden=64):
    """
    A small real Llama, so these tests exercise HF's actual attention plumbing.

    Config built locally rather than pulled from ``hf-internal-testing/...``: this box has no
    network, and a 5-retry HTTP timeout per test is a two-minute failure that says nothing.
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
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    defaults = dict(chunk_size=4)
    defaults.update(kwargs)
    return press, HSAIndexerTrainer(press=press, **defaults)


# ----------------------------------------------------------------------- configuration


def test_chunk_size_one_is_rejected():
    """
    ``chunk_size=1`` degenerates the objective, so it raises rather than running.

    At ``chunk_size=1`` the within-chunk softmax is ``softmax(one element) = 1``, so ``out = sum_j
    w_j v_j``: the ``q . k`` term vanishes and the router must learn the entire attention
    distribution itself, discarding the frozen backbone's prior. ``ROUTER_LEARNABILITY.md`` §6 calls
    this structural rather than a tuning matter -- so accepting the value and running a degenerate
    objective would be the wrong kindness.
    """
    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    with pytest.raises(ValueError, match="chunk_size must be > 1"):
        HSAIndexerTrainer(press=press, chunk_size=1)


def test_score_scale_defaults_to_the_indexer_scale():
    """
    Default ``score_scale`` is ``head_dim ** -0.5``, shared with ``GATE_SCALE_INIT``.

    Not a saturation fix (softmax has nothing to saturate against) but an *initialization* fix: an
    unscaled ``qi . ki`` has std ~sqrt(head_dim), which starts ``w`` nearly one-hot on a randomly
    chosen chunk -- far from the frozen backbone's output, which is the prior this objective exists
    to preserve.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    indexer = press.get_indexer(model.model.layers[0].self_attn)
    assert trainer.resolved_score_scale(indexer) == pytest.approx(indexer.head_dim**-0.5)
    assert HSAIndexerTrainer(press=press, chunk_size=4, score_scale=2.0).resolved_score_scale(
        indexer
    ) == 2.0


def test_default_aggregate_is_lse():
    """
    The default must be ``lse``, not ``mean``.

    Pinned as its own test because the two preceding tests establish *why* lse is right but both pass
    a mode explicitly, so neither would notice the default silently reverting -- which is the exact
    regression that produced the mean-pooled run (77.98 on 8K RULER, losing 59.5 points on
    niah_multikey_2 against the gated arm).
    """
    model, _ = tiny_model(n_layers=1)
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    assert HSAIndexerTrainer(press=press, chunk_size=4).chunk_aggregate == "lse"


def test_rejects_a_bad_aggregate():
    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    with pytest.raises(ValueError, match="chunk_aggregate must be"):
        HSAIndexerTrainer(press=press, chunk_size=4, chunk_aggregate="bogus")


# ----------------------------------------------------------------------- gradient plumbing


def test_loss_reaches_every_indexer_parameter():
    """
    The LM loss must produce a nonzero gradient on every indexer parameter **that this objective
    uses** -- and ``gate_scale``, which it does not, must stay at ``None``.

    The property that makes this end-to-end rather than distillation: the gradient travels from the
    LM head, back through the frozen backbone, into the chunk weights. A layer whose indexer gets no
    gradient is silently untrained and the loss curve would look identical.

    ``gate_scale`` is asserted ``None`` rather than skipped, following the exact-K arm: it exists only
    to keep checkpoints interchangeable with the gated arm's, and a softmax weight has no use for a
    positive multiplier (it would be a temperature, and one the loss could shrink to weaken its own
    gradient). If a future change starts reading it, this says so instead of passing quietly.
    """
    torch.manual_seed(0)
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    ids = torch.randint(0, 256, (1, 32))

    loss = hsa_indexer_training_step(model, trainer, input_ids=ids)
    assert torch.isfinite(loss), f"loss is {loss}"
    loss.backward()

    scored = 0
    for name, param in model.named_parameters():
        if ".indexer." not in name:
            continue
        if name.endswith("gate_scale"):
            assert param.grad is None, (
                "gate_scale received gradient, so this objective now reads it -- either that is "
                "intended (update hsa_trainer's docstring and give it a purpose) or the routing "
                "path picked it up by accident"
            )
            continue
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
        assert param.grad.abs().sum() > 0, f"{name} gradient is identically zero"
        scored += 1
    # w_q, w_k, and both norms' weight+bias, per layer.
    assert scored == 6 * model.config.num_hidden_layers, f"only {scored} params were trained"
    assert trainer.layers_routed == 3


def test_unused_gate_scale_does_not_break_gradient_clipping():
    """
    ``gate_scale`` gets no gradient, and ``clip_grad_norm_`` must tolerate that.

    It is in ``indexer_parameters`` (so the checkpoint round-trips) but off this objective's path, so
    its ``.grad`` stays ``None``. Both ``clip_grad_norm_`` and ``AdamW`` skip ``None`` grads -- but a
    run that crashed here would crash on step 1 of a real job, so it is pinned rather than assumed.
    """
    torch.manual_seed(10)
    model, _ = tiny_model(n_layers=2)
    _, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    assert any(p.numel() == 1 for p in params), "expected the gate_scale scalar in the param list"

    optimizer = torch.optim.AdamW(params, lr=1e-3)
    hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 32))).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert torch.isfinite(norm), "clipping produced a non-finite norm"
    optimizer.step()  # must not raise


def test_backbone_is_frozen_but_the_gradient_flows_through_it():
    """
    ``requires_grad=False`` on the backbone, **not** ``no_grad``.

    The distinction is the whole objective: ``no_grad`` around the backbone would sever the path from
    the LM loss to the router, and the router would receive exactly nothing while the code still
    ran. So this asserts both halves -- no backbone gradient is accumulated, and the router's is
    nonzero anyway.
    """
    torch.manual_seed(1)
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    indexer_ids = {id(p) for p in trainer.indexer_parameters(model)}
    backbone = [p for p in model.parameters() if id(p) not in indexer_ids]
    assert backbone
    assert all(not p.requires_grad for p in backbone)

    loss = hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 24)))
    loss.backward()
    assert all(p.grad is None for p in backbone), "a frozen backbone parameter accumulated gradient"
    assert any(
        p.grad is not None and bool((p.grad != 0).any()) for p in trainer.indexer_parameters(model)
    ), "the router got no gradient -- the path through the backbone is severed"


def test_detach_score_input_severs_only_the_feedback_path():
    """
    ``detach_score_input`` must still leave every indexer parameter trained.

    It cuts ``d(score)/d(hidden)``, not ``d(score)/d(indexer weights)``. Worth pinning because the
    obvious implementation of "detach" -- wrapping the score computation in ``no_grad`` -- would cut
    both and silently train nothing, while the loss would still descend (the backbone alone can do
    that).
    """
    torch.manual_seed(2)
    for detach in (True, False):
        model, _ = tiny_model(n_layers=2)
        press, trainer = make_trainer(model, detach_score_input=detach)
        trainer.freeze_backbone(model)
        loss = hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 24)))
        loss.backward()
        dead = [
            n for n, p in model.named_parameters()
            if ".indexer." in n and not n.endswith("gate_scale")
            and (p.grad is None or not bool((p.grad != 0).any()))
        ]
        assert not dead, f"detach={detach}: {dead} got no gradient"


def test_checkpointing_does_not_change_the_gradient():
    """Both checkpoints are memory knobs; the gradient must be identical without them."""
    grads = []
    for flags in ({"checkpoint_attention": True, "checkpoint_scores": True},
                  {"checkpoint_attention": False, "checkpoint_scores": False}):
        torch.manual_seed(3)
        model, _ = tiny_model(n_layers=2)
        press, trainer = make_trainer(model, **flags)
        trainer.freeze_backbone(model)
        loss = hsa_indexer_training_step(model, trainer, input_ids=torch.arange(24).view(1, 24))
        loss.backward()
        grads.append(
            # gate_scale is unused by this objective, so its grad is None; skipped rather than
            # crashed on. Filtering by `is not None` (not by shape) so that if a future change gives
            # it a gradient this test starts comparing it instead of quietly ignoring it.
            torch.cat([
                p.grad.flatten()
                for p in trainer.indexer_parameters(model)
                if p.grad is not None
            ])
        )
    rel = (grads[0] - grads[1]).norm() / grads[1].norm()
    assert rel < 1e-5, f"checkpointing changed the gradient by {rel:.2e} relative"


# ----------------------------------------------------------------------- hook hygiene


def test_attention_implementation_is_restored():
    model, config = tiny_model()
    press, trainer = make_trainer(model)
    before = config._attn_implementation
    with trainer.hooks(model):
        assert config._attn_implementation == "kvpress_gqa_indexer_hsa"
    assert config._attn_implementation == before


def test_registry_is_restored_even_on_exception():
    """
    The ``_global_mapping`` entry must not leak.

    ``register()`` writes to the class-level ``_global_mapping`` while ``pop()`` only touches the
    instance mapping, so the naive cleanup leaves the entry behind forever -- the trap
    ``capture_teacher_lse`` documents. A leaked entry is invisible until an unrelated later run picks
    up a stale closure over a dead trainer.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    assert "kvpress_gqa_indexer_hsa" not in mapping
    with pytest.raises(RuntimeError, match="boom"):
        with trainer.hooks(model):
            assert "kvpress_gqa_indexer_hsa" in mapping
            raise RuntimeError("boom")
    assert "kvpress_gqa_indexer_hsa" not in mapping, "the registry entry leaked"


def test_hooks_are_removed():
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    attn = model.model.layers[0].self_attn
    before = len(attn._forward_pre_hooks)
    with trainer.hooks(model):
        assert len(attn._forward_pre_hooks) == before + 1
    assert len(attn._forward_pre_hooks) == before


def test_a_layer_that_never_routes_raises():
    """
    If no layer runs the replacement attention the step must fail loudly.

    Otherwise the run trains nothing at full speed: the model would use its own attention, the loss
    would descend (the backbone is strong), and the router would get no gradient. Silent.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    # Point the config back to sdpa from inside the block, so the swap does not take effect.
    original = trainer.hooks

    @contextmanager
    def sabotaged(m):
        with original(m) as t:
            for cfg in [m.config]:
                cfg._attn_implementation = "sdpa"
            yield t

    trainer.hooks = sabotaged
    with pytest.raises(RuntimeError, match="no layer ran the HSA attention"):
        hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 16)))


# ----------------------------------------------------------------------- learning + diagnostics


def test_loss_descends():
    torch.manual_seed(4)
    model, _ = tiny_model(n_layers=2)
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    opt = torch.optim.Adam(trainer.indexer_parameters(model), lr=1e-2)
    ids = torch.randint(0, 256, (1, 48))

    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = hsa_indexer_training_step(model, trainer, input_ids=ids)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_entropy_falls_as_the_router_commits():
    """
    Normalized chunk entropy must **fall** over training.

    This arm's analogue of the flat-gate check. Uniform mixing over chunks is a legitimate operator,
    so the LM loss can descend without the router ever learning a ranking -- it would just be using
    a blunt average. Entropy is the only readout that separates "learned to choose" from "learned to
    use whatever mixture it produces".
    """
    torch.manual_seed(5)
    model, _ = tiny_model(n_layers=2)
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    opt = torch.optim.Adam(trainer.indexer_parameters(model), lr=5e-2)
    ids = torch.randint(0, 256, (1, 64))

    first, last = None, None
    for step in range(20):
        opt.zero_grad()
        loss = hsa_indexer_training_step(model, trainer, input_ids=ids)
        loss.backward()
        opt.step()
        ent = trainer.mean_chunk_entropy()
        if step == 0:
            first = ent
        last = ent
    assert first is not None and last is not None
    # At init the score is scaled to ~unit std, so w starts diffuse -- entropy near its maximum.
    assert first > 0.7, f"entropy did not start diffuse: {first:.3f}"
    assert last < first, f"entropy did not fall: {first:.3f} -> {last:.3f}"


def test_lse_aggregation_hits_the_closed_form_optimum_and_mean_does_not():
    """
    **Why ``chunk_aggregate`` defaults to ``lse``.** Fed a *perfect* token scorer, the ``lse`` path
    reproduces the closed-form optimal chunk score exactly; ``mean`` and ``max`` cannot.

    ``ROUTER_LEARNABILITY.md`` §6 verifies that for a frozen backbone the true chunk mass is
    ``softmax_c(LSE_c)`` (5.55e-17), and :func:`hsa_chunk_attention` makes ``w = softmax_c(s_chunk)``
    *be* the realized mass (5.6e-17). Equating those fixes the target: ``s_chunk = LSE_c``. Since the
    indexer's token score imitates the backbone's attention *logit*, the aggregation must be the same
    functional, or the chunk level cannot match **even with a perfect token scorer** -- which is what
    this test isolates by supplying one.

    Run through the real :func:`_score_tile` (via a stub indexer) rather than a reimplementation, so
    it also pins that the scale is applied *before* the reduction: with the multiplier outside,
    ``logsumexp`` operates on the raw dot's std ~11.3, degenerates to ``max``, and scores 0.65.
    """
    from kvpress.presses.gqa_indexer.hsa_attention import chunk_lse
    from kvpress.presses.gqa_indexer.hsa_trainer import _score_tile

    class StubIndexer:
        """Returns a fixed token-score tensor, i.e. a token scorer that is exactly right."""

        def __init__(self, scores):
            self.scores = scores

        def __call__(self, *a, **k):
            return self.scores

    def spearman(x, y):
        rx = x.argsort(-1).argsort(-1).double()
        ry = y.argsort(-1).argsort(-1).double()
        rx = rx - rx.mean(-1, keepdim=True)
        ry = ry - ry.mean(-1, keepdim=True)
        denom = (rx.norm(dim=-1) * ry.norm(dim=-1)).clamp_min(1e-12)
        return float(((rx * ry).sum(-1) / denom).mean())

    cs, head_dim, n_chunk = 64, 128, 16
    scale = head_dim**-0.5
    got = {"lse": [], "mean": [], "max": []}
    for seed in range(20):
        torch.manual_seed(seed)
        # Raw indexer dot at its natural magnitude: IndexerNorm leaves unit variance per channel, so
        # the dot has std ~ sqrt(head_dim) = 11.3. That scale is exactly what makes the placement of
        # the multiplier matter.
        raw = torch.randn(1, 1, 1, n_chunk * cs, dtype=torch.float64) * head_dim**0.5
        target = chunk_lse(raw * scale, cs)  # the closed-form optimum
        for mode in got:
            s = _score_tile(
                StubIndexer(raw), None, None, None, None, None, None, None, cs, mode, scale
            )
            got[mode].append(spearman(s.reshape(-1, n_chunk), target.reshape(-1, n_chunk)))

    means = {k: sum(v) / len(v) for k, v in got.items()}
    assert means["lse"] > 0.999, (
        f"lse aggregation should reproduce the closed-form optimum exactly, got {means['lse']:.4f}. "
        f"If this is ~0.65 the scale is being applied AFTER the reduction, so logsumexp degenerated "
        f"to max."
    )
    # And the alternatives genuinely cannot -- so the default is a correctness choice, not a preference.
    assert means["mean"] < 0.85, f"mean unexpectedly optimal ({means['mean']:.4f})"
    assert means["max"] < 0.85, f"max unexpectedly optimal ({means['max']:.4f})"
    assert means["lse"] > means["mean"] > means["max"], means


def test_lse_recovers_a_lone_needle_that_mean_dilutes():
    """
    The failure mode the aggregation change is meant to fix, stated as a property.

    A chunk holding **one** high-logit token among 63 background tokens is the needle-retrieval case.
    ``mean`` divides that single token's evidence by ~64 and loses the chunk; ``lse`` is dominated by
    the max and keeps it. Measured needle recall at top-4 chunks: **lse 1.000, mean 0.533** -- which
    lines up with where the mean-pooled run actually lost on RULER (niah_multikey_2 -59.5,
    multikey_3 -26.1 against the gated arm).
    """
    from kvpress.presses.gqa_indexer.hsa_trainer import _score_tile

    class StubIndexer:
        def __init__(self, scores):
            self.scores = scores

        def __call__(self, *a, **k):
            return self.scores

    cs, n_chunk, head_dim = 64, 32, 128
    scale = head_dim**-0.5
    recall = {"lse": [], "mean": []}
    for seed in range(30):
        gen = torch.Generator().manual_seed(seed)
        a = torch.randn(n_chunk, cs, generator=gen, dtype=torch.float64) * 0.5
        needles = torch.randperm(n_chunk, generator=gen)[:4]
        for c in needles.tolist():
            a[c, torch.randint(cs, (1,), generator=gen)] += 6.0
        raw = (a.reshape(1, 1, 1, -1) / scale)  # undo the scale the tile will re-apply
        want = set(needles.tolist())
        for mode in recall:
            s = _score_tile(
                StubIndexer(raw), None, None, None, None, None, None, None, cs, mode, scale
            ).reshape(-1)
            top = set(s.topk(4).indices.tolist())
            recall[mode].append(len(top & want) / 4)

    lse = sum(recall["lse"]) / len(recall["lse"])
    mean = sum(recall["mean"]) / len(recall["mean"])
    assert lse > 0.95, f"lse lost needles it should keep: recall {lse:.3f}"
    assert lse > mean + 0.2, (
        f"lse ({lse:.3f}) did not clearly beat mean ({mean:.3f}) at recovering a lone needle -- the "
        f"premise of using logsumexp"
    )


def test_lse_correlation_is_one_when_the_score_IS_the_lse():
    """
    Feeding the true chunk LSE as the score must give correlation ~1.

    Calibrates the diagnostic: an implementation that reported ~0 for a perfect router would make
    every real measurement unreadable, and the sign of a correlation is easy to get backwards (it
    was, in the exact-K arm's swap oracle). Patched at ``chunk_scores`` so the rest of the path is
    untouched.
    """
    torch.manual_seed(7)
    model, _ = tiny_model(n_layers=1)
    press, trainer = make_trainer(model, measure_lse_corr=True)
    trainer.freeze_backbone(model)

    from kvpress.presses.gqa_indexer.hsa_attention import chunk_lse

    real_routed = trainer.routed_forward
    captured = {}

    def routed(module, query, key, value, attention_mask, scaling):
        # Replace the router's score with the exact optimum for this layer.
        b, n_heads, q_len, head_dim = query.shape
        n_kv, k_len = key.shape[1], key.shape[2]
        group = n_heads // n_kv
        scale = head_dim**-0.5 if scaling is None else float(scaling)
        logits = torch.einsum(
            "bhqd,bhsd->bhqs", query.view(b, n_kv, group, q_len, head_dim)[:, :, 0], key
        ) * scale
        causal = torch.arange(k_len).view(1, k_len) <= (
            torch.arange(q_len) + (k_len - q_len)
        ).view(q_len, 1)
        want = chunk_lse(logits, trainer.chunk_size, valid=causal.view(1, 1, q_len, k_len))
        captured["want"] = want
        with patch.object(
            type(trainer), "chunk_scores", lambda self, *a, **kw: want.clone().requires_grad_(True)
        ):
            return real_routed(module, query, key, value, attention_mask, scaling)

    trainer.routed_forward = routed
    loss = hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 48)))
    assert torch.isfinite(loss)
    corr = trainer.mean_score_lse_corr()
    assert corr > 0.999, f"a perfect router measured {corr:+.4f}, so the diagnostic is miscalibrated"


def test_diagnostics_cover_every_layer():
    torch.manual_seed(8)
    model, _ = tiny_model(n_layers=3)
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 32)))
    for name in ("chunk_entropy", "mass_top1", "mass_topquarter", "score_lse_corr"):
        got = getattr(trainer, name)
        assert set(got) == {0, 1, 2}, f"{name} covered {sorted(got)}, expected all 3 layers"
        assert all(v == v for v in got.values()), f"{name} has NaN: {got}"


def test_measure_lse_corr_off_skips_it():
    torch.manual_seed(9)
    model, _ = tiny_model(n_layers=1)
    press, trainer = make_trainer(model, measure_lse_corr=False)
    trainer.freeze_backbone(model)
    hsa_indexer_training_step(model, trainer, input_ids=torch.randint(0, 256, (1, 32)))
    assert trainer.score_lse_corr == {}
    assert trainer.mean_score_lse_corr() is None
    assert trainer.chunk_entropy  # the cheap ones still run
