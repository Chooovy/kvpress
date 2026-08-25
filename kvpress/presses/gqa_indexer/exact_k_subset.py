# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Exact-K subset routing: sample a genuine k-subset in the forward, differentiate the exact
marginals in the backward.

This is SIMPLE (Ahmed et al., ICLR 2023) as adopted by ProbMoE (arXiv:2606.01509), ported from
``ProbMoE_V1_olmoe_exact.py`` and batched over arbitrary leading dims instead of their single
``(batch, n_experts)``. It exists because every *additive* gate this package has tried can go
**flat** along the key axis, and a flat additive gate is inert -- softmax is shift-invariant, so
the model reverts to the frozen backbone and the LM loss is satisfied with no ranking learned.
:mod:`~kvpress.presses.gqa_indexer.gate_pin` patches that hole by exempting some keys from the
gate's normalizer. Exact-K removes it structurally: the forward commits to exactly ``k`` items,
so there is no configuration of the scores under which "do nothing" reproduces dense attention.
See ``ROUTER_LEARNABILITY.md`` §7 -- this is the "STE fwd-sparse" row, but with a *principled*
backward rather than a hard mask.

The three pieces
----------------
1. :func:`exact_k_marginals` -- ``mu_i = P(z_i = 1 | sum(z) = k)`` under independent Bernoullis
   with logits ``s``. Obtained from an ``O(nk)`` log-domain DP over the Poisson-binomial
   distribution plus a "probe" autograd trick: ``d log P(sum = k) / d log p_i`` *is* the marginal,
   so one ``torch.autograd.grad`` call against a zero probe recovers all n of them at once. With
   ``create_graph=True`` the marginals stay differentiable in ``s``.
2. :func:`sample_k_subset` -- ancestral sampling from the same DP table, ``O(n)`` and always
   exactly ``k`` items. Runs under ``no_grad``; the sample is a discrete draw, not a relaxation.
3. :func:`straight_through_mask` -- ``g = (z - mu).detach() + mu``. Forward value is the hard 0/1
   mask; backward gradient is ``d mu / d s``, which is nonzero for **every** item including the
   ones not selected. That last property is the whole point (see "Boundary credit" below).

Why the multiplicative form, not an additive gate
-------------------------------------------------
``g`` **multiplies** the softmax weights::

    alpha_j = g_j * exp(a_j) / sum_i g_i * exp(a_i)

With ``g`` the straight-through mask this is *exactly* sparse attention over the sampled subset
(verified 1.11e-16 in fp64), so there is no train/inference gap in the forward -- unlike the
dense-forward gated path, which gates every key at train time and hard-selects at eval. ProbMoE
does the same thing (``w_full = g_full * softmax_probs``); it is **not** added to the logits.

Boundary credit, and why it is the reason to bother
---------------------------------------------------
The naive "score reweights only the selected set" proxy has gradient *identically zero* on
unselected items, so a key outside the top-k can never be promoted, no matter how much the loss
would benefit. Measured on an adversarial retrieval toy where the needed key starts outside
top-k: that proxy goes 0.0% -> 0.0% recall while exact-K goes 0.0% -> 93.8% (random = 12.5%).
The marginals are what buy this: ``mu_i`` depends on *every* score through the DP's normalizer,
so credit reaches items that were not sampled.

Two numerical details that are load-bearing, not cosmetic
---------------------------------------------------------
Both are deliberate in ProbMoE's source, and omitting either breaks this code outright:

* ``log_sigmoid`` is clamped to ``max=-1e-7``. Without it a saturated score gives ``log p == 0``,
  hence ``log(1 - p) = -inf``, and the DP's "not selected" branch is dead everywhere: the
  marginals come out ranked by *position* and summing to ``k + 1``, with a **NaN** gradient. No
  exception is raised. See :data:`LOG_P_MAX` for the measured numbers, and
  ``test_clamp_is_required``.
* The DP sentinel is ``-300.0``, **not** ``-inf``, because ``logaddexp(-inf, -inf)`` computes
  ``-inf - (-inf) = NaN`` in the shifted form used here. ``-300`` is far below any reachable
  log-probability and cannot poison a subtraction.

Memory
------
The DP retains ``n + 1`` states of width ``k + 2``, and ``create_graph=True`` keeps all of them
alive for the outer backward -- ``O(n k)`` per row. :func:`exact_k_marginals` therefore accepts
``checkpoint=True``, which recomputes the DP in the backward and keeps only ``O(k)``. That is not
a micro-optimization at our row counts: the caller's row count is
``B * n_kv_heads * n_query_blocks``, which is thousands, and ``n`` is the number of candidate
chunks.

Row count is the whole feasibility question
-------------------------------------------
The DP is sequential in ``n``, so cost is ``O(rows * n * k)`` with a Python-level loop of length
``n``. In attention there is one row per ``(query, kv-head)`` and ``n`` in the thousands, against
ProbMoE's one row per token and ``n = 64`` experts. Three levers make it fit, and
:mod:`~kvpress.presses.gqa_indexer.exact_k_attention` applies all three: chunk granularity
(not token), query-block sharing (one subset per block of queries), and candidate restriction
(``n`` is a candidate pool of size ``M``, not the full key axis). Measured GPU numbers are in
that module's docstring.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Log-domain sentinel for an impossible DP state.
#:
#: **-300, not -inf.** The DP's recurrence adds two log-probabilities and combines them with a
#: shifted ``logaddexp``; that form evaluates ``x1 - x2``, and ``-inf - (-inf)`` is NaN. A NaN
#: introduced in one unreachable state propagates through every subsequent one and the whole row
#: is lost. -300 is ~130 orders of magnitude below any log-probability an fp32 DP can represent as
#: nonzero, so it is "impossible" for every practical purpose while staying an ordinary number.
#: This is ProbMoE's own value and their comment records the same reason.
NEG_INF = -300.0

#: Upper clamp on ``log_sigmoid``, i.e. the largest allowed ``log p``.
#:
#: **Required, and the failure it prevents is silent.** ``p = sigmoid(s)`` saturates to exactly 1
#: in fp32 at ``s ~= 104`` (bf16 ~100, fp16 ~20, fp64 ~745), at which point ``log p == 0`` and
#: ``log_q = log1mexp(0) = -inf``. Every DP transition that does *not* select an item then carries
#: ``-inf``, so the only reachable terminal state is "all n selected" and
#: ``log P(sum = k) = NEG_INF`` for every ``k < n``. Measured consequence at ``n=8, k=3``, all
#: scores 110:
#:
#: * marginals become ``[0,0,0,0,1,1,1,1]`` -- they sum to **4**, not 3, and rank by *position*
#:   rather than by score;
#: * ``d mu / d s`` is **NaN** for every coordinate.
#:
#: So the forward keeps producing plausible-looking numbers while the backward poisons the whole
#: optimizer. Clamping ``log p`` to ``-1e-7`` keeps ``1 - p ~= 1e-7``, which is representable, and
#: costs nothing elsewhere: it perturbs ``p`` by 1e-7 only for scores that had already saturated.
#:
#: ProbMoE carries the same clamp. Their reported symptom is a ``torch.bernoulli`` range error
#: (``p_in >= 0 && p_in <= 1``) rather than this one; **that symptom did not reproduce here** --
#: the ancestral sampler in :func:`sample_k_subset` survives saturation because its
#: ``remaining == 0`` guard replaces the degenerate ratio with the sentinel before
#: :func:`log1mexp` sees it. The marginals are what break. Same clamp, different failure --
#: worth knowing, because the wrong-marginals failure has no traceback attached to it.
LOG_P_MAX = -1e-7


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    The dtype the DP runs in: at least fp32, but never *narrower* than the input.

    ProbMoE hardcodes ``.float()``, which is right for a bf16 caller -- the DP is a long chain of
    ``logaddexp`` over ``n`` steps and 8 mantissa bits do not survive it. But it silently narrows
    an fp64 caller, and fp64 is exactly what the reference tests use so their tolerances measure
    floating-point noise rather than the DP's own error. Same rule, and the same reasoning, as
    :func:`~.fused_loss.accumulation_dtype`.
    """
    return torch.float32 if dtype.itemsize < 4 else dtype


def log_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """
    ``log sigmoid(logits)``, clamped away from 0 so that ``log(1 - p)`` stays finite.

    See :data:`LOG_P_MAX` for why the clamp is load-bearing. ``min=-inf`` in the clamp is
    ProbMoE's own no-op lower bound, kept so the two functions read identically; the lower tail
    needs no protection because ``F.logsigmoid`` is already stable there (it is ``-softplus(-x)``).
    """
    return torch.clamp(F.logsigmoid(logits), max=LOG_P_MAX)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    ``log(1 - exp(-|x|))``, in the numerically stable two-branch form.

    Used to get ``log(1 - p)`` from ``log p``. The branch at ``log(0.5)`` is what keeps both tails
    accurate: for ``x`` near 0 the naive ``log1p(-exp(x))`` loses all precision to cancellation, so
    that side uses ``log(-expm1(x))`` instead, and vice versa.
    """
    x = -x.abs()
    return torch.where(
        x > -0.693147180559945309417232121458,  # log(0.5)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def logaddexp(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    ``log(exp(x1) + exp(x2))``, in the TF-style form ProbMoE uses.

    ``torch.logaddexp`` would do, except for its gradient at ``x1 == x2 == NEG_INF``: the
    ``where`` here forces the difference to exactly 0 on ties before it is ever formed, so a pair
    of sentinels produces ``NEG_INF + log 2`` rather than a NaN from ``-300 - -300`` underflowing
    in the general path. Keeping ProbMoE's exact expression also keeps the DP bit-comparable
    against their implementation, which is what makes the port checkable.
    """
    delta = torch.where(x1 == x2, torch.zeros_like(x1), x1 - x2)
    return torch.maximum(x1, x2) + F.softplus(-torch.abs(delta))


def log_pr_exactly_k(log_p: torch.Tensor, log_q: torch.Tensor, k: int) -> torch.Tensor:
    """
    Log-domain DP over the Poisson-binomial distribution, returning **all** intermediate states.

    Parameters
    ----------
    log_p, log_q : torch.Tensor
        ``(rows, n)`` log-probabilities of selecting / not selecting each item. ``log_q`` must be
        ``log1mexp(log_p)``; it is passed separately rather than derived here because
        :func:`exact_k_marginals` needs the probe added to ``log_p`` alone.
    k : int
        Target cardinality.

    Returns
    -------
    torch.Tensor
        ``(rows, n + 1, k + 2)``. Entry ``[r, i, j]`` is ``log P(exactly j - 1 of the first i
        items are selected)`` for row ``r``. The layout carries two deliberate offsets:

        * **the ``j`` axis is shifted by one**, so ``j = 0`` is a permanently-impossible
          "``-1`` selected" column. That makes the "select item ``i``" term a plain slice
          (``state[:, :-1]``) instead of a pad-and-shift, which is where an off-by-one would
          otherwise hide.
        * **the ``i`` axis has ``n + 1`` entries**, state ``i`` being "after considering ``i``
          items". :func:`sample_k_subset` walks it backwards and needs every one.

        Width is ``k + 2`` rather than ``k + 1`` for the same shift; counts above ``k`` are never
        needed because the recurrence only ever reads downwards.
    """
    rows, n = log_p.shape
    dtype = log_p.dtype

    state = torch.full((rows, k + 2), NEG_INF, device=log_p.device, dtype=dtype)
    state[:, 1] = 0.0  # log P(0 of 0 items selected) = log 1 = 0

    all_states = [state]
    for i in range(1, n + 1):
        state = torch.cat(
            [
                torch.full((rows, 1), NEG_INF, device=log_p.device, dtype=dtype),
                logaddexp(
                    state[:, :-1] + log_p[:, i - 1 : i],  # item i-1 selected
                    state[:, 1:] + log_q[:, i - 1 : i],  # item i-1 not selected
                ),
            ],
            dim=1,
        )
        all_states.append(state)
    return torch.stack(all_states, dim=1)


def _marginals_from_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The probe trick: ``(marginals, dp_table)`` for ``(rows, n)`` logits.

    Why a probe rather than a closed form. The marginal is
    ``mu_i = d log P(sum = k) / d log p_i`` -- an identity, not an approximation, because
    ``log P(sum = k)`` is linear in each ``p_i`` and the derivative picks out exactly the subsets
    containing ``i``. So adding a zero tensor ``probe`` to ``log_p``, running the DP, and asking
    autograd for ``d/d probe`` returns all ``n`` marginals from **one** backward over the DP.
    Writing them out by hand would need a second (reverse) DP.

    ``create_graph=True`` is what keeps the result differentiable in ``logits``, which is the
    whole point: the outer loss's gradient reaches the scores *through* the marginals.

    ``log_q`` is derived from ``log_p`` **before** the probe is added, matching ProbMoE. The probe
    is a device for extracting a derivative, not a perturbation of the model: it must appear in
    exactly the places ``log p_i`` appears as "item i selected", and ``log_q`` is the complement.

    ``enable_grad`` is unconditional because the probe's graph is *how the value is computed*, not
    an optional extra -- under an outer ``no_grad`` there would be no graph to differentiate and
    ``autograd.grad`` would raise. Whether the **result** stays differentiable in ``logits`` is
    decided separately, by ``create_graph``, which follows the caller's grad mode. So a
    ``no_grad`` caller still gets correct marginals, just detached ones.
    """
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        logits32 = logits.to(accumulation_dtype(logits.dtype))
        log_p = log_sigmoid(logits32)
        log_q = log1mexp(log_p)

        probe = torch.zeros_like(log_p, requires_grad=True)
        table = log_pr_exactly_k(log_p + probe, log_q, k)
        log_pr = table[:, -1, k + 1]

        (marginals,) = torch.autograd.grad(
            outputs=log_pr.sum(),
            inputs=probe,
            create_graph=create_graph,
        )
    return marginals, table


class _CheckpointedMarginals(torch.autograd.Function):
    """
    :func:`exact_k_marginals` with the DP recomputed in the backward instead of retained.

    Why this is not the default path's job. The probe trick already runs one backward *inside* the
    forward, and ``create_graph=True`` keeps every one of the DP's ``n + 1`` states of width
    ``k + 2`` alive so the outer backward can traverse them again -- ``O(n k)`` per row.
    Recomputing costs one extra DP pass and holds ``O(k)``.

    Measured on an H20, ``rows = 2048``, ``n = 128``, ``k = 32``, replicated over 36 layers as a
    real training step would:

    ==============  =============  ===============
    path            peak (fwd)     peak (fwd+bwd)
    ==============  =============  ===============
    retained        11.13 GiB      11.13 GiB
    checkpointed    **0.28 GiB**   **0.45 GiB**
    ==============  =============  ===============

    A 25x cut, and 11 GiB is not affordable next to a frozen 8B backbone's own activations. So
    ``checkpoint=True`` is what the trainer uses; the retained path exists to check it against.

    ``torch.utils.checkpoint`` cannot express this. It reruns the forward under
    ``enable_grad`` and expects a graph it can back-propagate through, but our forward's output
    already *is* a gradient (of the probe), so what backward needs is a double-backward through
    the DP. That is written out explicitly here: recompute the DP with both ``probe`` and
    ``logits`` requiring grad, re-derive the marginals with ``create_graph=True``, then take a
    second ``autograd.grad`` of ``sum(marginals * grad_out)`` with respect to ``logits``.
    """

    @staticmethod
    def forward(ctx, logits, k):
        with torch.no_grad():
            marginals, _ = _marginals_from_logits(logits, k)
        ctx.save_for_backward(logits)
        ctx.k = k
        return marginals

    @staticmethod
    def backward(ctx, grad_out):
        (logits,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        with torch.enable_grad():
            leaf = logits.detach().requires_grad_(True)
            marginals, _ = _marginals_from_logits(leaf, ctx.k)
            (grad,) = torch.autograd.grad(
                outputs=(marginals * grad_out.to(marginals.dtype)).sum(), inputs=leaf
            )
        return grad.to(logits.dtype), None


def _flatten_rows(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate ``(..., n)`` logits and view them as ``(rows, n)``. Returns the leading shape."""
    if logits.dim() < 1:
        raise ValueError(f"logits must have at least one dim (..., n), got {tuple(logits.shape)}")
    n = logits.shape[-1]
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n] with n={n}, got {k}")
    return logits.reshape(-1, n), tuple(logits.shape[:-1])


def exact_k_marginals(
    logits: torch.Tensor, k: int, *, checkpoint: bool = False
) -> torch.Tensor:
    """
    Exact inclusion marginals ``mu_i = P(z_i = 1 | sum(z) = k)``, differentiable in ``logits``.

    Parameters
    ----------
    logits : torch.Tensor
        ``(..., n)`` scores. Each item's independent selection probability is
        ``sigmoid(logits_i)``; the conditioning on ``sum(z) = k`` is what couples them.
    k : int
        Target cardinality, in ``[0, n]``.
    checkpoint : bool
        Recompute the DP in the backward instead of retaining it -- ``O(k)`` memory per row
        instead of ``O(n k)``, at the cost of one extra DP pass. See
        :class:`_CheckpointedMarginals`. Numerically identical, which
        ``test_checkpointed_matches_retained`` pins down.

    Returns
    -------
    torch.Tensor
        ``(..., n)`` marginals, in fp32 (or the input dtype if that is wider -- see
        :func:`accumulation_dtype`). ``sum(mu) == k`` exactly (to rounding), and every ``mu_i`` is
        a differentiable function of **every** score -- which is what lets an unselected item
        receive gradient.

    Notes
    -----
    Equal scores give ``mu_i = k/n`` for all ``i``, so a router that has learned nothing produces
    a *uniform* marginal rather than a degenerate one -- and the forward still commits to exactly
    ``k`` items. There is no "flat is inert" escape here, which is the structural difference from
    the additive gate this module exists to replace.
    """
    flat, lead = _flatten_rows(logits, k)
    if checkpoint:
        marginals = _CheckpointedMarginals.apply(flat, k)
    else:
        marginals, _ = _marginals_from_logits(flat, k)
    return marginals.reshape(*lead, logits.shape[-1])


@torch.no_grad()
def sample_k_subset(logits: torch.Tensor, k: int, *, generator=None) -> torch.Tensor:
    """
    Draw a subset of **exactly** ``k`` items, from the conditional distribution the marginals
    describe.

    Ancestral sampling over the same DP table: walk items from last to first, and select item
    ``i`` with the conditional probability that the remaining budget can still be met by the items
    before it. ``O(n)`` after the DP.

    Parameters
    ----------
    logits : torch.Tensor
        ``(..., n)`` scores.
    k : int
        Cardinality. Every returned row has exactly this many ones (``k = 0`` returns all zeros).
    generator : torch.Generator, optional
        For reproducible draws. ``None`` uses the global RNG.

    Returns
    -------
    torch.Tensor
        ``(..., n)`` fp32 0/1 mask.

    Notes
    -----
    ``no_grad`` throughout, and deliberately: the sample is a discrete draw with no useful
    pathwise derivative. The gradient comes from the marginals instead, which is exactly the
    division of labour SIMPLE proposes -- discrete forward, exact-marginal backward, no relaxation
    anywhere. (A secondary source described ProbMoE as using Gumbel-softmax; their README and
    source say otherwise.)
    """
    flat, lead = _flatten_rows(logits, k)
    rows, n = flat.shape
    if k == 0:
        return flat.new_zeros((rows, n)).reshape(*lead, n)

    logits32 = flat.to(accumulation_dtype(flat.dtype))
    log_p = log_sigmoid(logits32)
    log_q = log1mexp(log_p)
    table = log_pr_exactly_k(log_p, log_q, k)

    row_idx = torch.arange(rows, device=flat.device)
    remaining = torch.full((rows,), k, device=flat.device, dtype=torch.long)
    picks = []
    for i in range(n, 0, -1):
        exhausted = remaining == 0
        # P(item i-1 selected | budget `remaining` over the first i items)
        #   = P(remaining-1 over first i-1) * p_{i-1} / P(remaining over first i)
        # written in logs. The table's j axis is shifted by one, hence `remaining` and
        # `remaining + 1` rather than `remaining - 1` and `remaining`.
        numerator = table[row_idx, i - 1, remaining] + log_p[:, i - 1]
        denominator = table[row_idx, i, remaining + 1]
        log_ratio = numerator - denominator
        # A row with no budget left must not select, and its ratio is a 0/0 in logs -- force it to
        # the sentinel before log1mexp sees it rather than masking the NaN afterwards.
        log_ratio = torch.where(exhausted, torch.full_like(log_ratio, NEG_INF), log_ratio)
        prob = torch.sigmoid(log_ratio - log1mexp(log_ratio))
        pick = torch.bernoulli(prob, generator=generator)
        pick = torch.where(exhausted, torch.zeros_like(pick), pick)
        remaining = torch.where(pick > 0, remaining - 1, remaining)
        picks.append(pick)

    return torch.stack(picks[::-1], dim=1).reshape(*lead, n)


def straight_through_mask(
    logits: torch.Tensor,
    k: int,
    *,
    checkpoint: bool = False,
    generator=None,
    hard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ``g = (z - mu).detach() + mu``: hard ``k``-subset in the forward, exact marginals in the
    backward.

    Parameters
    ----------
    logits : torch.Tensor
        ``(..., n)`` scores.
    k : int
        Cardinality.
    checkpoint : bool
        Passed to :func:`exact_k_marginals`.
    generator : torch.Generator, optional
        RNG for the sample.
    hard : bool
        Take the deterministic top-``k`` instead of a stochastic sample. This is what inference
        does (ProbMoE's ``deterministic_routing``), and it is available at train time only to
        measure the variance the sampling adds -- using it *for* training reintroduces the
        selected-set-only gradient problem, because the top-k of a fixed score does not explore.

    Returns
    -------
    g : torch.Tensor
        ``(..., n)`` fp32. Values are exactly 0 or 1; the gradient is ``d mu / d logits``.
    z : torch.Tensor
        The 0/1 sample itself, detached. Callers need it to build the gather indices.
    mu : torch.Tensor
        The marginals, still attached to the graph -- exposed for diagnostics (their entropy and
        their agreement with an oracle are the informative readouts, not the loss alone).
    """
    mu = exact_k_marginals(logits, k, checkpoint=checkpoint)
    if hard:
        z = torch.zeros_like(mu)
        if k > 0:
            z.scatter_(-1, logits.float().topk(k, dim=-1).indices, 1.0)
    else:
        z = sample_k_subset(logits, k, generator=generator)
    return (z - mu).detach() + mu, z, mu


def subset_indices(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Turn a 0/1 mask into ascending selected positions, ``(..., k)`` int64.

    ``topk`` on the mask itself would return the ``k`` ones in arbitrary order (they are all tied),
    and downstream gathers here want them **ascending** -- the same convention
    :func:`~.sparse_support.sort_support` emits, so the indices feed
    :func:`~.sparse_attention.sparse_gqa_attention_reference` with no adapter. Sorting the raw
    positions rather than re-ranking by score is what keeps them ascending regardless of ``z``'s
    provenance.

    Raises if any row does not hold exactly ``k`` ones -- a row that does not is a bug in the
    sampler, and silently padding it would make the attention that follows quietly attend to
    fewer keys than the budget it was measured at.
    """
    counts = z.sum(-1)
    if k > 0 and not bool((counts == k).all()):
        bad = int((counts != k).sum())
        raise RuntimeError(
            f"{bad} row(s) do not hold exactly k={k} selected items (counts range "
            f"{float(counts.min())}..{float(counts.max())}). sample_k_subset guarantees exact "
            "cardinality, so this is a bug rather than an input problem."
        )
    return z.nonzero(as_tuple=False)[:, -1].reshape(*z.shape[:-1], k) if k > 0 else z.new_zeros(
        (*z.shape[:-1], 0), dtype=torch.long
    )
