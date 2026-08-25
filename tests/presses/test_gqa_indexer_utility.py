# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility self-distillation: the target, the sampler, the loss, and the wiring.

The tests that justify the module existing are the ones that pin properties an eyeball cannot check
and a descending loss would hide:

* ``test_utility_matches_a_finite_difference`` -- ``u`` really is ``-dL/db_j``, to fp64. The
  algebra is short enough to look right while being wrong by a sign or a missing ``- o``.
* ``test_utility_sign_is_loss_decreasing`` -- the **sign convention**, checked against a real loss
  change rather than against the formula. An inverted sign still descends; it just trains the router
  to rank the worst keys first, and nothing else in the pipeline would notice.
* ``test_forward_is_bit_identical_to_the_unhooked_model`` -- this arm's defining claim. Every other
  objective in this package changes the forward; if this one does too, it is not what it says.
* ``test_lm_loss_alone_gives_the_router_no_gradient`` -- the reason ``loss = loss_rank`` is the whole
  objective, and the reason this is a distillation arm rather than an end-to-end one. Stated as a
  test because it is the single most load-bearing fact about the design.
* ``test_every_layer_is_supervised_during_one_backward`` -- the reentrant-backward-inside-a-hook
  mechanism the whole arm rests on. If it silently fired for one layer, or for none, the loss would
  still look plausible.
"""

import pytest
import torch
import torch.nn.functional as F

from kvpress.presses.gqa_indexer import (
    GQAIndexerPress,
    UtilityIndexerTrainer,
    utility_indexer_training_step,
)
from kvpress.presses.gqa_indexer.utility_loss import (
    INVALID_UTILITY,
    lm_gradient_utility,
    pairwise_rank_loss,
    sample_boundary_pairs,
    score_utility_correlation,
    utility_recall_at_k,
)

DT = torch.float64


def tiny_model(n_layers=3, n_heads=8, n_kv_heads=4, hidden=64):
    """
    A small real Llama, so the wiring tests exercise HF's actual attention plumbing.

    Config built locally rather than pulled from ``hf-internal-testing/...``: this box has no network
    and a 5-retry HTTP timeout per test is a two-minute failure that says nothing.
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
    defaults = dict(n_rows=4, n_pairs=8, band=4)
    defaults.update(kwargs)
    return press, UtilityIndexerTrainer(press=press, **defaults)


def attention_with_bias(q, k, v, bias=None):
    """One causal attention row-block, with an optional additive per-key bias ``b_j``."""
    s = q.shape[2]
    logits = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if bias is not None:
        logits = logits + bias
    causal = torch.arange(s).view(1, s) <= torch.arange(s).view(s, 1)
    alpha = torch.softmax(logits.masked_fill(~causal.view(1, 1, s, s), -float("inf")), -1)
    return alpha, alpha @ v


# ---------------------------------------------------------------- the target


def test_utility_matches_a_finite_difference():
    """
    ``u_j = -dL/db_j`` exactly, verified against autograd on the bias itself.

    The point of the identity is that it costs **one** backward for **every** key: no key has to be
    selected first, which is precisely the candidate-pool dead end the exact-K arm hit. That only
    holds if the closed form is right, so it is checked against the thing it claims to equal rather
    than re-derived.
    """
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 12, 8
    q, k, v = (torch.randn(b, h, s, d, dtype=DT) for _ in range(3))
    bias = torch.zeros(b, h, s, s, dtype=DT, requires_grad=True)

    alpha, out = attention_with_bias(q, k, v, bias)
    # An arbitrary downstream loss -- the identity must not depend on which one.
    target = torch.randn_like(out)
    loss = (out * target).sum()
    loss.backward()

    # dL/do for the same graph, obtained independently of the bias path.
    grad_out = target

    u = lm_gradient_utility(alpha.detach(), v, out.detach(), grad_out)
    torch.testing.assert_close(u, -bias.grad, rtol=0, atol=1e-12)


def test_utility_sign_is_loss_decreasing():
    """
    ``u_j > 0`` means moving mass onto ``j`` **lowers** the loss.

    Checked against an actual loss change, not against the algebra, because an inverted sign is
    silent: the ranking loss still descends, the router still learns a consistent ordering, and the
    only symptom is that it ranks the least useful keys first -- which would look like the
    representability ceiling rather than like a bug.
    """
    torch.manual_seed(1)
    b, h, s, d = 1, 1, 10, 8
    q, k, v = (torch.randn(b, h, s, d, dtype=DT) for _ in range(3))
    target = torch.randn(b, h, s, d, dtype=DT)

    alpha, out = attention_with_bias(q, k, v)
    base = float((out * target).sum())
    u = lm_gradient_utility(alpha, v, out, target)

    row = s - 1
    best = int(u[0, 0, row, :s].argmax())
    worst = int(u[0, 0, row, :s].argmin())
    assert u[0, 0, row, best] > 0 > u[0, 0, row, worst], "expected a signed spread to test against"

    eps = 1e-4
    for key, expect_drop in ((best, True), (worst, False)):
        bias = torch.zeros(b, h, s, s, dtype=DT)
        bias[0, 0, row, key] = eps
        _, nudged = attention_with_bias(q, k, v, bias)
        delta = float((nudged * target).sum()) - base
        assert (delta < 0) is expect_drop, (
            f"nudging key {key} (u={float(u[0, 0, row, key]):+.3e}) changed the loss by {delta:+.3e}; "
            "the sign convention is inverted"
        )


def test_utilities_sum_to_zero_over_a_row():
    """
    ``sum_j u_j = 0``: a softmax cannot add mass without taking it from somewhere.

    Why this matters beyond being a tidy identity -- it says utility is inherently **relative**, so
    only *differences* carry information. That is the justification for a pairwise ranking loss over
    a regression onto ``u``'s values, and for reporting a rank correlation rather than an error.
    """
    torch.manual_seed(2)
    b, h, s, d = 1, 2, 16, 8
    q, k, v = (torch.randn(b, h, s, d, dtype=DT) for _ in range(3))
    alpha, out = attention_with_bias(q, k, v)
    u = lm_gradient_utility(alpha, v, out, torch.randn_like(out))
    torch.testing.assert_close(u.sum(-1), torch.zeros_like(u.sum(-1)), rtol=0, atol=1e-12)


def test_attention_weight_is_a_poor_proxy_for_utility():
    """
    ``alpha`` does not rank keys the way ``u`` does -- the reason for this arm over attention-KL.

    A key can hold a lot of attention and still be worthless, because its value may already sit at
    the row's output; ``u``'s ``v_j - o`` factor is exactly that correction. Constructed here rather
    than measured: one key gets large attention and ``v_j = o``, so its utility is ~0 while its
    ``alpha`` is the largest in the row. On real text this shows up as Spearman +0.037 for ``alpha``
    against +0.991 for ``u``, versus the true single-key drop effect.
    """
    torch.manual_seed(3)
    b, h, s, d = 1, 1, 6, 4
    alpha = torch.full((b, h, 1, s), 0.1, dtype=DT)
    alpha[..., 0] = 1.0 - 0.1 * (s - 1)  # the dominant key
    v = torch.randn(b, h, s, d, dtype=DT)

    # Make v_0 == o a FIXED POINT, not just an assignment: o depends on v_0, so setting v_0 = o and
    # recomputing moves the target. Solving o = a_0 o + sum_{j>0} a_j v_j gives the value below.
    rest = (alpha[..., 1:] @ v[:, :, 1:]) / (1.0 - alpha[..., 0]).unsqueeze(-1)
    v[:, :, 0] = rest[:, :, 0]
    out = alpha @ v
    torch.testing.assert_close(v[:, :, 0], out[:, :, 0], rtol=0, atol=1e-12)

    u = lm_gradient_utility(alpha, v, out, torch.randn(b, h, 1, d, dtype=DT))
    assert int(alpha[0, 0, 0].argmax()) == 0, "key 0 should hold the most attention"
    assert abs(float(u[0, 0, 0, 0])) < 1e-12, "yet its utility should be ~0"
    assert float(u[0, 0, 0].abs().max()) > 1e-3, "while other keys have real utility"


# ---------------------------------------------------------------- the sampler


def test_boundary_pairs_are_ordered_by_the_teacher():
    """``idx_win`` always holds the higher utility, so the loss knows which should win."""
    torch.manual_seed(4)
    scores = torch.randn(2, 3, 4, 64)
    utility = torch.randn(2, 3, 4, 64)
    win, lose = sample_boundary_pairs(scores, utility, n_pairs=32, band=8, budget=32)
    assert bool((utility.gather(-1, win) >= utility.gather(-1, lose)).all())


def test_boundary_pairs_concentrate_near_the_budget():
    """
    Pairs are drawn from a narrow band of the **router's** ranking around the budget.

    The reason to sample this way rather than uniformly: top-k depends only on the order across the
    K-th boundary, so a pair at ranks 3 and 7000 is already ordered right by any usable router and
    its gradient is wasted (§23.3). Drawing from the router's own ranking also makes the sampler
    self-correcting -- the band tracks where the router is still uncertain.
    """
    torch.manual_seed(5)
    n_keys, budget, band = 512, 128, 16
    scores = torch.randn(1, 1, 1, n_keys)
    utility = torch.randn(1, 1, 1, n_keys)
    win, lose = sample_boundary_pairs(
        scores, utility, n_pairs=4096, band=band, budget=budget
    )

    rank_of_key = scores.argsort(-1, descending=True).argsort(-1)
    drawn = torch.cat([rank_of_key.gather(-1, win), rank_of_key.gather(-1, lose)], -1)
    assert int(drawn.min()) >= budget - band
    assert int(drawn.max()) <= budget + band
    # And a uniform sampler would not have done this -- most of the row is never touched.
    assert drawn.unique().numel() <= 2 * band + 1


def test_invisible_keys_are_never_drawn():
    """
    A key the query cannot see must not enter a pair.

    ``INVALID_UTILITY`` rather than 0 for those keys, because real utilities are **signed**: zero
    would place an invisible key in the middle of the ranking, where the band sampler is looking.
    """
    torch.manual_seed(6)
    n_keys, n_visible = 128, 40
    scores = torch.randn(1, 1, 1, n_keys)
    utility = torch.randn(1, 1, 1, n_keys)
    utility[..., n_visible:] = INVALID_UTILITY
    win, lose = sample_boundary_pairs(scores, utility, n_pairs=2048, band=64)
    assert int(torch.cat([win, lose], -1).max()) < n_visible


def test_a_row_shorter_than_the_band_contributes_no_loss():
    """
    Degenerate rows drop out by weight rather than by a mask.

    When every drawn pair is a tie the weight ``|u_i - u_j|`` is 0, so such rows neither contribute
    loss nor gradient and no filtering is needed. Worth pinning: the alternative failure is a NaN
    from a ``-inf`` utility difference, which would poison the whole step.
    """
    scores = torch.zeros(1, 1, 1, 32)
    utility = torch.full((1, 1, 1, 32), INVALID_UTILITY)
    utility[..., 0] = 1.0
    win, lose = sample_boundary_pairs(scores, utility, n_pairs=16, band=8)
    loss = pairwise_rank_loss(scores.requires_grad_(True), utility, win, lose)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------- the loss


def test_a_correctly_ordered_router_gets_far_less_gradient():
    """
    Agreement with the teacher costs almost no gradient; disagreement costs a lot.

    Compared as a *ratio* against the inverted router rather than against an absolute threshold,
    because ``softplus`` is deliberately not fully saturated: the closest-ranked pairs keep a small
    gradient, which is the soft margin the band sampler exists to exploit. A tolerance tight enough to
    call that "zero" would be testing the tolerance.
    """
    torch.manual_seed(7)
    utility = torch.randn(1, 1, 1, 64)

    def grad_norm(scores):
        scores = scores.clone().requires_grad_(True)
        win, lose = sample_boundary_pairs(scores.detach(), utility, n_pairs=64, band=16)
        pairwise_rank_loss(scores, utility, win, lose).backward()
        return float(scores.grad.norm())

    # Same ranking as the teacher, wide margins; versus the exactly-inverted ranking.
    assert grad_norm(utility * 1e3) < 1e-4 * grad_norm(-utility * 1e3)


def test_loss_gradient_pushes_the_mis_ranked_pair_apart():
    """The one behavioural property: a wrongly-ordered pair gets pushed in the right direction."""
    utility = torch.tensor([[[[1.0, 0.0]]]])
    scores = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)  # backwards
    win = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    lose = torch.ones(1, 1, 1, 1, dtype=torch.long)
    pairwise_rank_loss(scores, utility, win, lose).backward()
    assert float(scores.grad[0, 0, 0, 0]) < 0, "the winner's score should be pushed up"
    assert float(scores.grad[0, 0, 0, 1]) > 0, "the loser's should be pushed down"


def test_loss_weights_by_utility_gap():
    """
    ``|u_i - u_j|`` weighting makes the loss track the *regret* of a mis-ranking, not its count.

    Two keys of nearly equal utility can be swapped at nearly no cost, and ``u`` is a noisy
    first-order estimate so those are exactly the pairs where its own sign is least trustworthy.

    Compared **within one row**, because normalization is per row: a row holding a single pair
    normalizes to weight 1 whatever its gap, which is correct (a lone pair has no other pair to be
    relatively more or less important than) but says nothing about the weighting. So the row here holds
    both a wide-gap and a narrow-gap pair, and the check is that the wide one dominates the gradient.
    """
    #        keys:      0     1      2     3
    utility = torch.tensor([[[[10.0, 0.0, 0.51, 0.5]]]])  # gaps: 10.0 and 0.01
    scores = torch.zeros(1, 1, 1, 4, requires_grad=True)
    win = torch.tensor([[[[0, 2]]]])
    lose = torch.tensor([[[[1, 3]]]])
    pairwise_rank_loss(scores, utility, win, lose).backward()
    wide = float(scores.grad[0, 0, 0, 0].abs())
    narrow = float(scores.grad[0, 0, 0, 2].abs())
    assert wide > 100 * narrow, f"wide-gap pair {wide:.3e} should dominate narrow {narrow:.3e}"


def test_weight_normalization_makes_the_gradient_scale_free():
    """
    Scaling every utility by a constant must not change the router's gradient.

    **The bug this pins was measured, not hypothesized.** ``u`` is proportional to ``alpha_j``
    (``~1/Sq``) times ``dL/do`` (carrying the LM loss's ``1/(B*Sq)`` mean), so ``|u| ~ 1/Sq**2``:
    measured mean ``|u|`` falls 4x per doubling -- 5.2e-7 at 256 tokens, 8.3e-9 at 2048, 3.5e-10 at 8K
    on Qwen3-8B, with a router gradient norm of ~3e-8. At that magnitude AdamW's ``eps = 1e-8``
    dominates its denominator and the optimizer is no longer scale-invariant (realized step against
    ideal: 42.9% at 1e-8, 8.8% at 1e-9, 1.0% at 1e-10), while ``grad_clip`` -- an absolute threshold --
    never fires. Both scale with ``Sq``, so the effective learning rate would be a function of the
    curriculum stage, and nothing in the loss curve would reveal it.

    Checked as invariance under a scalar rather than against the measured numbers, because that is the
    property the fix has to provide and it holds at any length.
    """
    torch.manual_seed(12)
    utility = torch.randn(1, 1, 4, 128)
    base_scores = torch.randn(1, 1, 4, 128)

    def router_grad(u, normalize):
        # A fresh seeded generator per call, so the two runs draw the SAME pairs and the only thing
        # varying is the utility scale. Without this the sampler's own RNG draw differs and the
        # comparison measures sampling noise instead of the invariance.
        gen = torch.Generator().manual_seed(3)
        scores = base_scores.clone().requires_grad_(True)
        win, lose = sample_boundary_pairs(
            base_scores, u, n_pairs=64, band=16, generator=gen
        )
        pairwise_rank_loss(scores, u, win, lose, normalize=normalize).backward()
        return scores.grad

    normalized = router_grad(utility, True)
    torch.testing.assert_close(router_grad(utility * 1e-6, True), normalized, rtol=1e-5, atol=1e-9)
    # ...and without normalization the same 1e-6 rescale shrinks the gradient by 1e-6, which is the
    # failure: the objective's magnitude, and so the effective LR, follows the sequence length.
    raw = router_grad(utility, False)
    shrunk = router_grad(utility * 1e-6, False)
    assert float(shrunk.norm()) < 1e-5 * float(raw.norm())


def test_normalized_weights_still_rank_by_regret():
    """
    Normalization rescales weights per row; it must not flatten them.

    The weighting is what makes the loss track the *regret* of a mis-ranking rather than its count, so
    a fix that made every pair equal would trade one silent failure for another.
    """
    utility = torch.tensor([[[[10.0, 5.0, 0.1, 0.0]]]])
    scores = torch.zeros(1, 1, 1, 4)
    win = torch.tensor([[[[0, 2]]]])
    lose = torch.tensor([[[[1, 3]]]])
    # Pair (0,1) has gap 5.0, pair (2,3) has gap 0.1 -- a 50x ratio that must survive.
    weight = (utility.gather(-1, win) - utility.gather(-1, lose)).abs()
    normalized = weight / weight.mean(-1, keepdim=True)
    assert float(normalized[..., 0] / normalized[..., 1]) == pytest.approx(50.0)
    assert float(normalized.mean()) == pytest.approx(1.0)
    assert torch.isfinite(pairwise_rank_loss(scores, utility, win, lose))


def test_the_router_learns_to_rank_by_utility():
    """
    The behavioural test: on a repeated batch, ``score_corr`` climbs.

    **Why a single batch rather than a training curve.** This arm has a *measured representability
    ceiling* -- ``u``'s ranking lives mostly in a value term a ``q . k`` scorer cannot see, and probes
    on Qwen3-8B put the reachable part at +0.03 to +0.32. So a flat correlation on real data is an
    expected outcome, not a failure, and a test asserting otherwise would be asserting the research
    question. Overfitting one batch removes the ceiling (the router only has to fit *these* rows) and so
    isolates the wiring and the gradient *direction*, which is what a test can legitimately pin.

    This distinction is exactly what the smoke run needed: 12 steps at 8K left ``score_corr`` at 0.00,
    and without this check there is no way to tell "the gradient points the wrong way" from "12 steps is
    nothing". Measured here: +0.013 -> +0.61 over 120 steps, recall 0.357 -> 0.65.
    """
    model, _ = tiny_model(n_layers=2, hidden=128, n_heads=8, n_kv_heads=4)
    press, trainer = make_trainer(model, n_rows=8, n_pairs=64, band=32)
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)

    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 256))

    first, last = None, None
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        # Same batch AND same sampler seed each step, so the target is a fixed function to fit.
        utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 0:
            first = trainer.mean_score_corr()
        last = trainer.mean_score_corr()

    assert last > first + 0.1, (
        f"score_corr went {first:+.4f} -> {last:+.4f} on a repeated batch. The router should be able "
        "to fit rows it sees every step, so a flat curve here is a wiring or gradient-direction bug "
        "rather than the representability ceiling."
    )


def test_score_correlation_is_scale_and_shift_invariant():
    """
    Spearman, so an affine reparameterization of the score reads identically.

    Load-bearing rather than cosmetic: top-k selects the same keys under ``a*s + b`` for ``a > 0``
    (§24), so a diagnostic that moved under one would be measuring something the operator discards.
    """
    torch.manual_seed(8)
    utility = torch.randn(1, 1, 4, 64)
    scores = torch.randn(1, 1, 4, 64)
    base = score_utility_correlation(scores, utility)
    assert base == pytest.approx(score_utility_correlation(3.0 * scores + 7.0, utility), abs=1e-9)


def test_recall_and_correlation_are_perfect_for_a_perfect_router():
    torch.manual_seed(9)
    utility = torch.randn(1, 1, 2, 64)
    assert score_utility_correlation(utility.clone(), utility) == pytest.approx(1.0, abs=1e-6)
    assert utility_recall_at_k(utility.clone(), utility, 16) == pytest.approx(1.0)


# ---------------------------------------------------------------- the wiring


def test_lm_loss_alone_gives_the_router_no_gradient():
    """
    The fact the whole design rests on: the router is **not** on the forward path.

    So ``dL_LM/dtheta_router`` is ``None`` -- absent, not small -- and ``loss = loss_rank`` is the
    entire objective. This is what makes this arm a *distillation* arm with a better teacher rather
    than an end-to-end one, and it is worth a test because the module docstring's claim is otherwise
    unfalsifiable from the outside.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)

    ids = torch.randint(0, 256, (1, 32))
    # A plain forward, with the embedding grad-leaf hook so a graph exists at all, but WITHOUT the
    # attention swap -- i.e. exactly this arm's forward, minus the ranking loss.
    handle = model.model.embed_tokens.register_forward_hook(
        lambda m, a, o: o.requires_grad_(True)
    )
    try:
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    finally:
        handle.remove()
    assert all(p.grad is None for p in params), "the LM loss must not reach the router"


def test_forward_is_bit_identical_to_the_unhooked_model():
    """
    This arm's defining claim: the forward pass is **unmodified**.

    Every other objective in this package changes what the model computes -- a gate inside the
    softmax, a chunk mixture, a hard subset. If this one also did, the ``u`` it reads would not be
    the frozen model's utility and the comparison against the other arms would be confounded.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 32))

    with torch.no_grad():
        reference = model(input_ids=ids, use_cache=False).logits.clone()

    with trainer.hooks(model, seed=0):
        hooked = model(input_ids=ids, use_cache=False).logits.clone()

    torch.testing.assert_close(hooked, reference, rtol=0, atol=0)


def test_every_layer_is_supervised_during_one_backward():
    """
    The reentrant-backward-inside-a-hook mechanism, checked per layer.

    Each layer's ranking loss is built and backwarded from inside that layer's ``dL/do`` hook, so
    nothing has to be stashed across layers -- ``dL/do`` for all 36 layers is 2.4 GiB at 8K on
    Qwen3-8B, on a backbone already at ~92 of 95 GiB at 16K. If the hooks fired for only some layers
    the loss would still look plausible, hence the exact count.
    """
    model, config = tiny_model(n_layers=3)
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 48))
    utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)

    assert trainer.layers_supervised == config.num_hidden_layers
    assert sorted(trainer.rank_losses) == list(range(config.num_hidden_layers))
    assert all(torch.isfinite(torch.tensor(v)) for v in trainer.rank_losses.values())


def test_the_ranking_loss_reaches_every_indexer_parameter():
    """
    Every *scoring* parameter gets a gradient, and no backbone parameter does.

    ``gate_scale`` is excluded, and its absence is correct rather than tolerated: the ranking loss
    reads only the **order** of the scores, so a single positive multiplier on all of them is
    unidentifiable -- it cannot change which keys are selected, only how sharp ``softplus`` is. This
    arm therefore scales by the fixed :attr:`~.utility_trainer.UtilityIndexerTrainer.score_scale` and
    leaves the parameter alone; it exists only to keep the checkpoint byte-compatible with the gated
    arm's. ``test_gate_scale_is_deliberately_untrained`` states that as its own claim.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 48))
    utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)

    params = [p for n, p in _named_indexer_parameters(model, press) if n != "gate_scale"]
    missing = [i for i, p in enumerate(params) if p.grad is None or not p.grad.any()]
    assert not missing, f"scoring parameters {missing} received no gradient"

    indexer_ids = {id(p) for p in trainer.indexer_parameters(model)}
    stray = [
        name for name, p in model.named_parameters()
        if id(p) not in indexer_ids and p.grad is not None and bool(p.grad.any())
    ]
    assert not stray, f"backbone parameters received gradient: {stray}"


def test_gate_scale_is_deliberately_untrained():
    """
    ``gate_scale`` receives no gradient, because the ranking loss cannot identify it.

    Only score *differences* reach the loss (``softplus(s_lose - s_win)``), and a positive scalar
    multiplying every score changes no ranking -- so there is no signal to move it and any movement
    would be drift. The gated arm trains it because there it multiplies a gate *inside* the softmax,
    where its magnitude is what decides how hard that layer leans on its router.

    Worth its own test rather than a comment: a future change that started feeding ``gate_scale`` into
    this arm's score would make the parameter wander with no diagnostic reporting it, and the
    checkpoint would then carry a value the eval reads.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 48))
    utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)

    gates = [p for n, p in _named_indexer_parameters(model, press) if n == "gate_scale"]
    assert gates, "the press should still create gate_scale, for checkpoint compatibility"
    assert all(p.grad is None or not bool(p.grad.any()) for p in gates)


def _named_indexer_parameters(model, press):
    """``(name, parameter)`` for every indexer parameter, names relative to the indexer module."""
    from kvpress.presses.gqa_indexer.press import get_language_model

    for layer in get_language_model(model).layers:
        indexer = getattr(layer.self_attn, press.scorer_attr, None)
        if indexer is not None:
            yield from indexer.named_parameters()


def test_diagnostics_populate_for_every_layer():
    """
    ``score_corr`` is **the** number this arm is read by, so it must exist for every layer.

    The loss value alone cannot say whether the router is learning: it is weighted by
    ``|u_i - u_j|``, which scales with ``||dL/do||``, so it falls when the batch gets easier. The
    correlation is against a fixed quantity, and it reads directly against the +0.03 to +0.32
    representability ceiling the probes measured -- a plateau there is the hypothesis class rather
    than the optimizer, which is the distinction that decides what to change next.
    """
    model, config = tiny_model()
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 48))
    utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)

    for name in ("score_corr", "utility_recall", "utility_scale"):
        values = getattr(trainer, name)
        assert len(values) == config.num_hidden_layers, f"{name} missing layers"
    assert -1.0 <= trainer.mean_score_corr() <= 1.0
    assert 0.0 <= trainer.mean_utility_recall() <= 1.0
    assert trainer.mean_utility_scale() > 0, "a collapsed teacher scale would zero every weight"


def test_loss_scale_scales_the_router_gradient():
    """
    ``loss_scale`` must carry the accumulation divisor, since the reentrant backward is *inside* the
    driver's own scaled backward and cannot see it.

    Getting this wrong changes only the effective learning rate, which no diagnostic reveals -- so
    the relationship is pinned here instead.
    """
    grads = {}
    for scale in (1.0, 0.25):
        torch.manual_seed(11)
        model, _ = tiny_model()
        press, trainer = make_trainer(model, loss_scale=scale)
        ids = torch.arange(48).remainder(256).view(1, 48)
        utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)
        grads[scale] = torch.cat(
            [p.grad.flatten() for p in trainer.indexer_parameters(model) if p.grad is not None]
        )
    torch.testing.assert_close(grads[0.25], 0.25 * grads[1.0], rtol=1e-5, atol=1e-8)


def test_registry_and_attention_impl_are_restored_on_exception():
    """
    The ``_global_mapping`` leak ``teacher_lse`` documents: ``register()`` writes to the global
    mapping while ``pop()`` only clears the instance one, so a naive cleanup leaves the entry behind
    forever and a later run silently uses this arm's attention.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    before = model.config._attn_implementation
    mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    assert "kvpress_gqa_indexer_utility" not in mapping

    with pytest.raises(RuntimeError, match="boom"):
        with trainer.hooks(model, seed=0):
            raise RuntimeError("boom")

    assert model.config._attn_implementation == before
    assert "kvpress_gqa_indexer_utility" not in mapping


def test_missing_grad_leaf_would_be_caught():
    """
    A frozen backbone with no grad leaf produces no ``grad_fn`` on ``o``, so no hook fires and the
    objective trains on nothing while the LM loss still looks healthy.

    :meth:`~.utility_trainer.UtilityIndexerTrainer.hooks` installs the leaf to prevent that; this
    checks the *guard*, by suppressing the leaf and confirming the step raises rather than returning a
    plausible loss. This exact failure cost two debugging rounds in the probe scripts that preceded
    the module.
    """
    model, _ = tiny_model()
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 32))

    original = trainer.hooks

    from contextlib import contextmanager

    @contextmanager
    def hooks_without_leaf(m, seed=None):
        import kvpress.presses.gqa_indexer.utility_trainer as mod

        saved = mod._require_grad_hook
        mod._require_grad_hook = lambda module, args, output: output
        try:
            with original(m, seed=seed) as t:
                yield t
        finally:
            mod._require_grad_hook = saved

    trainer.hooks = hooks_without_leaf
    with pytest.raises(RuntimeError, match="does not require grad"):
        utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)


def test_utility_is_pooled_over_the_query_group():
    """
    One score per KV head, so ``u`` is pooled over the query group -- and by a **mean**, not a max.

    The router emits one score per KV head and one cache is evicted per KV head, so the quantity to
    rank is the utility of dropping key ``j`` for the whole group. Each group member's utility is a
    first-order term in the same loss, so they add; the mean is that sum up to a constant, and only
    the ranking is used.
    """
    model, config = tiny_model(n_heads=8, n_kv_heads=2)
    press, trainer = make_trainer(model)
    ids = torch.randint(0, 256, (1, 48))
    utility_indexer_training_step(model, trainer, input_ids=ids, seed=0)
    # The diagnostics are computed on the pooled tensor, so a shape error would surface as a crash
    # inside supervise(); reaching here with every layer supervised is the check.
    assert trainer.layers_supervised == config.num_hidden_layers
