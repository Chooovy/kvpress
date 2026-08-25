# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cross-replay LM loss for a query-independent indexer.

The objective. Prefill a context ``C`` densely, then replay the *same* tokens as ``C'`` against
``KV(C)`` alone -- ``C' -> KV(C)`` but ``C' -/-> KV(C')`` -- and take the LM loss on ``C'``. The
indexer's per-key score enters the replay attention as an additive gate, so ``dL/ds`` arrives
through the ordinary attention softmax::

    A_ji = softmax_i( q'_j . k_i / sqrt(d) + g_i )
    dL/ds_i = sum_j A_ji <dL/do_j, v_i - o_j>

The sum over ``j`` runs over **every** replay query, which is the point: supervision goes from a
causal triangle to a full cross-context rectangle, and one score is forced to serve many queries at
once. That is the query-agnostic reuse value eviction needs, whereas the ordinary causal LM loss
measures value to the natural continuation only.

Design notes, with the verification behind each claim, are in ``cross_replay_e2e.md``. What matters
for reading this module:

* **The gate never enters the KV cache.** It is an additive term on the logits, so ``K_C``/``V_C``
  are bit-identical across the two passes. Nothing here rewrites the cache.
* **Query-independence makes the gate one value per key.** ``s_i = f(h_i)`` does not depend on the
  query, so the gate is one ``(B, H_kv, N)`` vector shared by every replay query. Neither
  :mod:`~.triton_gated_attention` nor :func:`~.gate_pin.history_lse` is needed -- both exist to keep a
  *pairwise* gate at ``O(L)`` memory, and a per-key gate has no ``O(Sq * Sk)`` term to begin with.
* **But it must not reach SDPA as an ``attn_mask``.** That is algebraically correct and was measured at
  **46.7 GiB** of pure waste: a mask that requires grad, under GQA, leaves SDPA no fused backend at
  all, so it runs MATH and retains the entire ``(B, H, Sq, Sk)`` score matrix. The gate goes through
  ``flex_attention``'s ``score_mod`` instead -- see :meth:`CrossReplayTrainer._flex_attention` for the
  backend table and ``cross_replay_e2e.md`` §6.3. Still no hand-written kernel.
* **Pass 1 is dense, ungated and under ``no_grad``.** Gating it would train streaming prefill rather
  than eviction, and would close a loop (``gate -> h -> s -> gate``) since ``h`` at layer ``l``
  depends on earlier layers' attention. ``dL/dw = sum_i (dL/ds_i) h_i`` needs only the *values* of
  ``h``, so no graph is built for pass 1.

Three failure modes here are **silent** -- they produce a clean loss curve and an untrained router --
so each is checked rather than trusted:

1. ``pin_mode="self"`` pins **zero** keys in this geometry (the diagonal key of replay query ``j``
   lies in the masked-out ``C'`` block). With no pins the normalizer degenerates to a plain
   log-softmax, which :mod:`~.gate_pin` shows is exactly interchangeable with a raw score -- the
   flat-gate no-op is reachable again. Rejected in :meth:`CrossReplayTrainer.__post_init__`.
2. An implicit mask gives a causal triangle, not a rectangle. With ``q_len == k_len`` SDPA takes its
   ``is_causal`` fast path, and :func:`~.gated_attention._visible` intersects with a causal mask
   unconditionally. The rectangle mask is therefore always built explicitly.
3. ``d_max = 2N - 1`` can exceed the model's trained position range, putting replay queries at
   positions never seen in pretraining. Warned in :meth:`CrossReplayTrainer.hooks`.

The gate carries a **budget** term, ``log B`` on the gated keys, and it is load-bearing: the identity
``sum_{j gated} exp(g_j) = B`` makes ``B`` the number of sink-equivalents the whole history is worth.
It cancels *within* the gated softmax -- so at fixed parameters it changes neither the ranking nor the
participation -- but not against the pinned sinks, so it decides how concentrated the gate becomes at
convergence. **Leave it at 1**: "set it to the inference top-k" was this module's advice and is
retracted -- measured, ``B=topk`` costs 27.8 RULER points against ``B=1``. See
:attr:`CrossReplayTrainer.log_budget` and ``cross_replay_e2e.md`` §15.3.

Two knobs exist for §16's follow-ups, both defaulting to today's behaviour:
:attr:`~CrossReplayTrainer.demand_reduce` (how the replay queries' demands on one key combine into
``dL/ds`` -- ``max`` approximates KVzip's max-attention label) and
:attr:`~CrossReplayTrainer.lookahead` (a bound on how far past its own position a replay row may see,
where ``0`` is the causal triangle the e2e loss trains on). See ``cross_replay_e2e.md`` §17.
"""

from __future__ import annotations

import functools
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import DynamicCache

from kvpress.presses.gqa_indexer.gate_pin import check_pin_mode, pins_self, pins_sink
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model

logger = logging.getLogger(__name__)

#: Replay queries are padded up to a multiple of this before ``flex_attention``.
#:
#: Two reasons, one of them a hard failure. Inductor's autotuner has **no valid config** for
#: ``64 <= Sq < 128`` at ``Sk=8192, head_dim=128`` -- every candidate exceeds the H20's 232448-byte
#: shared-memory limit, and the call raises ``No valid triton configs`` rather than falling back
#: (measured: Sq = 63, 64, 65, 96, 100, 127 all raise; 17 and 128 are fine). A ragged final replay
#: chunk lands in that band for any ``|C| % query_chunk`` in it. Padding also collapses the number of
#: distinct shapes, which matters because of ``_note_flex_shape`` below.
#:
#: Padding is exact: the padded rows' outputs are sliced off, so their cotangent is zero and they
#: contribute nothing to ``dL/ds``. Verified against the unpadded call where the unpadded call works
#: (Sq = 129: dg maxabs 93.4822 vs 93.4823; Sq = 1000: 414.4854 vs 414.4850).
#:
#: Load-bearing, verified by mutation: setting this to 1 makes ``--context-len 8292 --query-chunk
#: 1024`` (final chunk 100) raise ``No valid triton configs``, and the same run passes at 128.
_FLEX_Q_ALIGN = 128

#: Reasons already warned about, so the fallback is loud once rather than 36 times per chunk.
_FLEX_FALLBACK_WARNED: set[str] = set()


@functools.lru_cache(maxsize=1)
def compiled_flex_attention():
    """
    ``torch.compile(flex_attention)``, compiled once per process.

    **The compile is not an optimization, it is the whole point.** Eager ``flex_attention`` falls back
    to a materializing reference implementation: measured at ``Sq=1024, Sk=8192`` on Qwen3-8B's
    geometry, eager peaks at **18730 MiB** for a single layer against **40 MiB** compiled -- worse
    than the MATH backend this replaces. Anything that silently bypasses the compiled path
    (``torch._dynamo.disable``, a recompile-limit bail-out, ``TORCHDYNAMO_DISABLE=1``) turns the fix
    into a 460x regression, which is why :meth:`CrossReplayTrainer._note_flex_shape` watches for it.

    ``dynamic=False``: the replay runs a handful of fixed shapes, and a dynamic-shape kernel is slower
    for no benefit here.

    Cached with ``lru_cache`` so every layer and every chunk shares one dynamo cache entry per shape.

    ``donated_buffer=False`` is required, not tuning. Inductor's donated-buffer optimization frees a
    compiled backward's inputs as it consumes them, and then *asserts* that no backward is called with
    ``retain_graph=True``::

        RuntimeError: This backward function was compiled with non-empty donated buffers which
        requires create_graph=False and retain_graph=False.

    ``logit_chunk`` backwards each row block with ``retain_graph=True`` (the chunk's transformer graph
    is shared across row blocks), so every ``--logit-chunk`` run raises without this. It is off by one
    knob's interaction and was found only by sweeping the two chunk sizes together -- ``query_chunk``
    alone never trips it.
    """
    from torch.nn.attention.flex_attention import flex_attention

    torch._functorch.config.donated_buffer = False
    return torch.compile(flex_attention, dynamic=False)


def flex_fallback_reason(
    query: torch.Tensor, real_mask: torch.Tensor | None, dropout: float
) -> str | None:
    """
    Why ``flex_attention`` cannot serve this call, or ``None`` if it can.

    Split out as a plain function so the dispatch can be tested without a GPU -- the branch that
    decides between a 48 MiB kernel and a 1288 MiB one is worth checking directly rather than
    inferring from a memory number on hardware not everyone has.
    """
    try:
        compiled_flex_attention()
    except Exception as exc:  # pragma: no cover - torch too old to have flex_attention
        return f"flex_attention is unavailable in torch {torch.__version__} ({exc})"
    if not query.is_cuda:
        # Measured: the compiled CPU path raises `IndexError: tuple index out of range` in inductor.
        return "flex_attention's kernels are CUDA-only"
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        # fp64 raises NYI in the inductor lowering; the CPU test models run fp32/fp64.
        return f"flex_attention does not support {query.dtype}"
    if query.shape[-1] < 16:
        return f"flex_attention needs head_dim >= 16, got {query.shape[-1]}"
    if real_mask is not None:
        # This objective only ever passes the all-zero rectangle, which is dropped. A real mask would
        # have to enter score_mod, and no caller here does that -- so defer rather than guess.
        return "a non-rectangle attention mask was supplied"
    if dropout:
        return f"flex_attention takes no dropout, got dropout_p={dropout}"
    return None


def _warn_flex_fallback(reason: str, on_cuda: bool) -> None:
    """
    Log the fallback to SDPA once per reason, loudly when it costs real memory.

    On CUDA this branch is the **46.7 GiB bug** this module was fixed for, so it must never be quiet:
    the mask disqualifies flash, the GQA mismatch disqualifies mem-efficient, and the gate's
    ``requires_grad`` disqualifies cuDNN, so SDPA lands on MATH and keeps the full ``(B, H, Sq, Sk)``
    score matrix. Off CUDA it is expected and uninteresting.
    """
    if reason in _FLEX_FALLBACK_WARNED:
        return
    _FLEX_FALLBACK_WARNED.add(reason)
    if not on_cuda:
        logger.info("cross-replay attention using SDPA rather than flex_attention: %s.", reason)
        return
    logger.warning(
        "cross-replay attention fell back to SDPA on CUDA: %s. This is EXPENSIVE, not cosmetic: the "
        "attn_mask disqualifies flash, the GQA head mismatch disqualifies mem-efficient, and the "
        "gate's requires_grad disqualifies cuDNN, so SDPA runs the MATH backend and retains the full "
        "(B, H, Sq, Sk) score matrix -- measured 1288 MiB per layer at Sq=1024, N=8192 on Qwen3-8B, "
        "i.e. 46.7 GiB over 36 layers, against 48 MiB per layer for flex_attention.",
        reason,
    )


class ReadOnlyCache(DynamicCache):
    """
    A cache that serves ``KV(C)`` and refuses to grow.

    Pass 2 must attend to ``KV(C)`` and *only* that. Two layouts express this identically -- append
    ``C'``'s keys and mask them out (``k_len = 2N``), or never append them (``k_len = N``) -- and
    they were verified equal in loss (``0.0e+00``) and in gate gradient (maxdiff ``5.8e-10``). This
    is the second: half the attention work, and ``C'``'s K/V are never even computed.

    ``use_cache=False`` does **not** achieve this. Transformers still calls ``update()`` and the
    cache still grows from ``N`` to ``2N`` (verified), which silently doubles the key axis and admits
    ``C'`` to the mask's causal region. Overriding ``update`` is what actually pins the key set.

    Wraps the layers of a prefilled cache by reference -- no copy, so ``KV(C)`` is not duplicated.
    """

    def __init__(self, prefilled: DynamicCache):
        super().__init__()
        # Referenced, not copied: KV(C) is 2.25 GiB at 16K on Qwen3-8B.
        self.layers = prefilled.layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Return the prefilled entries and discard the incoming ones."""
        layer = self.layers[layer_idx]
        return layer.keys, layer.values


def rectangle_mask(
    q_len: int, k_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    The all-visible ``(1, 1, q_len, k_len)`` additive mask: every replay query sees every ``C`` key.

    Explicit by necessity, not by preference. Passing ``None`` here would be wrong in two
    independent ways: transformers/SDPA take an ``is_causal`` fast path when ``q_len == k_len``
    (which a read-only cache guarantees), and :func:`~.gated_attention._visible` intersects with a
    bottom-right causal mask unconditionally. Either one silently yields a causal triangle -- the
    ordinary LM objective -- while the caller believes it configured a rectangle. Measured
    divergence from the intended result in that case: 1.44, not a rounding error.

    All zeros, so it adds nothing numerically; its job is to *exist*, so that mask creation is not
    skipped. A 4D mask is returned as-is by ``masking_utils._preprocess_mask_arguments``, which is
    what lets it reach attention unmodified.

    Tagged with ``_kvpress_all_zero`` so :meth:`CrossReplayTrainer._attention` can drop it rather
    than add it. That matters more than it sounds: adding a ``(1, 1, Sq, N)`` zero mask to the
    ``(B, H, 1, N)`` gate broadcasts the result into a materialized ``(B, H, Sq, N)`` tensor, which
    measured **73.1 GiB** peak at 8K instead of the estimated 23.9. The tag is checked instead of the
    values because ``(mask != 0).any()`` would sync the device on every layer of every chunk.
    """
    mask = torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)
    mask._kvpress_all_zero = True
    return mask


def replay_horizon_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    query_offset: int = 0,
    lookahead: int | None = None,
) -> torch.Tensor:
    """
    The replay mask with a bounded **lookahead**: row ``j`` sees keys ``<= j + lookahead``.

    Generalises :func:`rectangle_mask`, which is the ``lookahead=None`` (unbounded) case and stays the
    default everywhere. This exists to test §16.3's mechanism: the full rectangle makes every replay
    row choose among all ``N`` keys from step 0, so one ``s_i`` must satisfy ``N`` heterogeneous
    demands under a ``sum exp(g) = B`` constraint, and the cheapest solution is to concentrate on the
    few keys every query wants. Every cross-replay arm converged that way (participation 0.007-0.062)
    against the causal e2e arm's 0.165, across 26x capacity and 2048x budget -- see §16.3's table.

    ``lookahead`` interpolates between the two objectives' supervision shapes:

    * ``None``  -- the full rectangle. Today's cross-replay. Row ``j`` sees all ``k_len`` keys.
    * ``0``     -- row ``j`` sees ``[0, j]``, i.e. the causal triangle the e2e LM loss trains on.
      Note the target ``C'[j+1]`` then lies outside the visible set, matching e2e exactly.
    * ``m > 0`` -- ``[0, j + m]``. A ramp: the candidate set still grows with ``j``, so the
      difficulty curriculum §16.3 identifies is preserved, but each row sees ``m`` keys of context
      beyond its own position.

    ``query_offset`` is the absolute position of row 0 within ``C'``, needed because
    :func:`cross_replay_training_step` chunks the replay: chunk ``[start, stop)`` has
    ``query_offset=start``, so its rows index the same key axis the unchunked pass would.

    Returns an additive ``(1, 1, q_len, k_len)`` mask. Unlike :func:`rectangle_mask` this one carries
    real ``-inf`` entries, so it is **not** tagged ``_kvpress_all_zero`` and
    :meth:`CrossReplayTrainer._attention` will not drop it -- which also means it disqualifies the
    flex path (``flex_fallback_reason`` refuses a non-rectangle mask) and lands on SDPA MATH. That is
    the 46.7 GiB retention path, so a bounded-lookahead run needs a smaller ``query_chunk``; the
    alternative is to express the horizon inside ``score_mod``, which is the follow-up if the ablation
    proves worth keeping.
    """
    if lookahead is None:
        return rectangle_mask(q_len, k_len, device, dtype)
    if lookahead < 0:
        raise ValueError(
            f"lookahead must be non-negative or None, got {lookahead}: a negative horizon would "
            "hide keys at and before the query's own position, which no objective here wants"
        )
    rows = torch.arange(query_offset, query_offset + q_len, device=device).view(q_len, 1)
    keys = torch.arange(k_len, device=device).view(1, k_len)
    visible = keys <= rows + lookahead
    if not bool(visible.any()):
        raise ValueError(
            f"lookahead={lookahead} at query_offset={query_offset} leaves some row with no visible "
            "keys; every replay row must see at least its own position"
        )
    mask = torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)
    mask.masked_fill_(~visible.view(1, 1, q_len, k_len), torch.finfo(dtype).min)
    # Carry the horizon so :meth:`CrossReplayTrainer._attention` can reproduce it inside
    # ``score_mod`` and keep the flex path, instead of letting a real mask force SDPA MATH (46.7 GiB,
    # §6.3). The mask itself stays the single source of truth: the flex branch recomputes exactly
    # this predicate, and ``test_lookahead_flex_and_sdpa_paths_agree`` pins the two together.
    mask._kvpress_lookahead = int(lookahead)
    mask._kvpress_query_offset = int(query_offset)
    return mask


@dataclass
class CrossReplayTrainer:
    """
    Train a query-independent indexer from the cross-replay LM loss.

    Usage mirrors :class:`~.e2e_trainer.E2EIndexerTrainer`: enter :meth:`hooks`, call
    :func:`cross_replay_training_step`, then ``.backward()`` on the returned loss. No auxiliary
    term is produced -- the router's gradient comes through the attention softmax.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers. Must hold :class:`~.scalar_indexer.ScalarIndexer` modules
        built with ``gate_scale=True``: this objective trains a *query-independent* score, and a
        pairwise indexer would reintroduce the ``O(Sq * Sk)`` gate this module is built to avoid.
    pin_mode : str
        Which keys are exempt from the gate's normalizer, so a flat gate cannot be a no-op.
        Only ``"sink"`` is permitted here -- see :meth:`__post_init__` for why ``"self"`` is a
        silent failure in this geometry, and ``cross_replay_e2e.md`` §3.
    n_sink : int, optional
        Leading keys to pin. Defaults to the press's own ``n_sink``, so the keys the gate exempts in
        training are the keys the press protects at inference. Must be > 0: with zero pins the
        normalizer is inert and the no-op reopens.
    query_chunk : int, optional
        Replay queries per forward. ``None`` runs them all at once.

        This is the memory knob, and it is **exact**: with ``C'`` masked from itself the replay
        queries are independent (verified -- a subset of positions gives hidden states matching the
        full pass to 8.3e-07), so chunking and accumulating gradients reproduces the unchunked run
        (verified: gate-gradient maxdiff 7.5e-09, relative 5.0e-08 -- floating-point accumulation
        noise). Every chunk still attends to the **whole** key axis, so the rectangle is preserved.

        Note this is *unlike* KVzip's chunking, which also restricts the keys to the chunk and so
        yields block-diagonal supervision rather than a rectangle (``cross_replay_e2e.md`` §7.2).

        Peak activation is ``O(query_chunk)`` instead of ``O(N)``: on Qwen3-8B at 16K, ~10 GiB at
        ``query_chunk=1024`` against ~65 GiB unchunked.
    log_budget : float, optional
        ``log B`` in the gate ``s_j - LSE(s) + log B``, applied to the gated keys only.

        Sets **how concentrated the gate is at convergence**, which is a different thing from the
        ranking it learns. The identity is ``sum_{j gated} exp(g_j) = B``: the gated keys share a
        total multiplier of ``B`` while each pinned sink sits at 1. So ``B`` is the number of sink-
        equivalents the whole history is worth, and with a flat score every gated key gets
        ``B / n_gated``.

        ⚠️ **RETRACTED: "set it to the inference top-k."** That was this docstring's advice, argued
        from representability -- a hard top-k gate keeps ``topk`` keys at multiplier 1 and drops the
        rest, whose ``sum exp(g)`` is exactly ``topk``, so ``B = topk`` is the only exactly
        representable value. The 2x2 grid was completed and **representability is not the property
        that matters**. Isolating ``B`` at fixed ``mid_dim=256``, RULER 8K, ``fraction=0.100``,
        step 600: ``B=1`` scores **48.20** and ``B=2048`` scores **20.43** -- so ``B = topk`` costs
        **27.8 points**, against an 18.0-point objective gap and 3.4 points for 26x the scorer
        capacity. It is the largest single effect measured on this objective and its sign is negative.
        **Leave it at 1.** See ``cross_replay_e2e.md`` §15.3.

        Also retracted: that ``B = 1`` *caused* the first run's low participation. The e2e arm -- the
        best of the four -- also trains at ``B = 1`` (``E2EIndexerTrainer.gate_budget`` defaults to
        ``1.0``) with 24x the participation at the same ``B``. Concentration is set by the loss
        geometry, not by ``B`` (§15.2, §16.3).

        * ``B = 1`` (what omitting the term gives) makes the entire history worth **one** sink. At
          ``n_gated = 16384`` a flat gate is then ``6.1e-05`` per key. This was believed to be the
          cause of the first run's collapse; it is not, and it is the measured-best setting.
        * ``B = n_gated`` is the flat-gate no-op point (``g_j = 0`` for all ``j``), where the router
          can satisfy the loss having learned nothing. That is what pinning exists to prevent, so it
          is the diagnostic reference rather than a setting -- and it is what ``None`` resolves to.
          This end of the range is **not** retracted.
    lookahead : int, optional
        Bound on how far past its own position a replay row may see, in keys. ``None`` (the default)
        is the unbounded rectangle -- today's objective, and what every measured arm trained on.

        This is the knob for §16.3's mechanism: under the rectangle every row chooses among all ``N``
        keys from step 0, so one ``s_i`` must satisfy ``N`` heterogeneous demands under
        ``sum exp(g) = B``, and the cheap solution is to concentrate on the few keys every query
        wants. All three cross-replay arms converged there (participation 0.007-0.062) against the
        causal e2e arm's 0.165, across 26x capacity and 2048x budget. ``0`` reproduces e2e's causal
        triangle; a small positive value keeps a growing candidate set (hence the difficulty ramp the
        rectangle lacks) while letting each row see a little context beyond itself.

        **Keeps the flex path.** A bounded horizon is a real mask, which would disqualify every fused
        backend and land on SDPA MATH (46.7 GiB, §6.3) -- so :meth:`_attention` lifts it out of the
        mask and re-expresses it inside ``score_mod`` instead. That matters for the ablation as much as
        for memory: it means a lookahead run needs **no change to** ``query_chunk``, so it stays
        single-variable against the unbounded arm. The non-flex branch (CPU, no-flex torch) still
        applies the real mask, and ``test_lookahead_flex_and_sdpa_paths_agree`` pins the two together.
    demand_reduce : str
        How the replay queries' demands on one key are combined into ``dL/ds_i``. ``"sum"`` (default)
        is plain autograd accumulation and is what every measured arm trained with.

        The motivation is the KVzip contrast (``cross_replay_e2e.md`` §16.4). §4's gradient is
        ``dL/ds_i = sum_j A_ji <dL/do_j, v_i - o_j>`` -- an average over all ``N`` replay queries,
        which §16.3 argues is what drives every cross-replay arm to over-concentrate. KVzip takes the
        **max** attention over queries instead, and a max is not a compromise: each key keeps its
        *best* query's demand rather than the mean of ``N``, which is what a retrieval task needs and
        what an average destroys. KVzip gets this for free because it never differentiates.

        A true per-query max would need one backward per query. The affordable granularity is the
        **query chunk**: ``"max"`` harvests and zeroes ``leaves[idx].grad`` after each chunk, then
        combines the per-chunk demands, so ``N / query_chunk`` demand groups compete rather than all
        ``N`` queries averaging. Requires at least 2 chunks and raises otherwise -- with one chunk the
        reduction is arithmetically inert and would be a silently dead knob.

        ``"mean"`` is the null control: it is ``sum`` divided by the chunk count, i.e. the same
        direction at a smaller magnitude, so it separates "the reduction changed the direction" from
        "the reduction changed the effective learning rate". ``"max"`` is rescaled by the chunk count
        for the same reason.

        This parameter did not exist for one revision, on the reasoning that ``+log B`` is added to
        every gated key alike and therefore cancels. It *does* cancel **within** the gated softmax,
        so at fixed parameters it changes neither the ranking nor the participation -- but it does
        not cancel against the pinned sinks, so it moves where training converges. Measured on
        Qwen3-8B/0.6B at 4K, only ``B`` varying: final participation 0.254/0.236 at ``B=1`` against
        0.710/0.951 raw. See ``cross_replay_e2e.md`` §2.5.
    freeze : bool
        Freeze every non-indexer parameter on :meth:`hooks` entry.
    """

    press: GQAIndexerPress
    pin_mode: str = "sink"
    n_sink: int | None = None
    query_chunk: int | None = None
    log_budget: float | None = None
    lookahead: int | None = None
    demand_reduce: str = "sum"
    freeze: bool = True

    #: Layer index -> ``gate_scale`` on the last pass, for logging.
    gate_scales: dict[int, float] = field(default_factory=dict)
    #: Layers that actually ran the gated attention, as a wiring check.
    layers_gated: int = field(default=0, init=False)

    #: Per-layer per-key scores for the current replay pass, ``(B, H_kv, N)``. Set by
    #: :meth:`score_context` and read by the attention override. Kept as scores rather than as
    #: hidden states so the attention path cannot accidentally rescore under a different offset.
    _scores: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _context_len: int | None = field(default=None, init=False, repr=False)
    #: Distinct ``(Sq, Sk)`` shapes flex_attention has been asked to compile, to catch a dynamo
    #: recompile-limit bail-out before it silently reverts to the materializing eager path. Not reset
    #: per pass: the limit is per code object and accumulates over the whole process.
    _flex_shapes: set[tuple[int, int]] = field(default_factory=set, init=False, repr=False)
    #: The attention implementation the model had before :meth:`hooks` swapped it, so
    #: :meth:`ungated` can put it back for pass 1. ``None`` outside the hooks block.
    _original_impl: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        check_pin_mode(self.pin_mode)
        if self.demand_reduce not in ("sum", "max", "mean"):
            raise ValueError(
                f"demand_reduce must be 'sum', 'max' or 'mean', got {self.demand_reduce!r}"
            )
        if self.lookahead is not None and self.lookahead < 0:
            raise ValueError(
                f"lookahead must be non-negative or None, got {self.lookahead}: a negative horizon "
                "would hide the query's own position and earlier keys"
            )
        if pins_self(self.pin_mode):
            # The decisive geometric fact, and the reason this is an error rather than a warning:
            # under [C ; C'] the diagonal key of replay query j sits in the C' block, which this
            # objective masks out entirely. So a "self" pin pins nothing that the attention can
            # see (measured: 0 pins inside C, all 16 inside the masked region at N=16), the
            # normalizer degenerates to a plain log-softmax, and gate_pin.py's own analysis then
            # makes it exactly interchangeable with a raw score -- the flat-gate no-op is reachable
            # again. A run configured this way trains cleanly and learns no ranking.
            raise ValueError(
                f"pin_mode={self.pin_mode!r} pins nothing under cross-replay: query j's diagonal "
                "key lies in the C' block, which this objective masks out, so the pinned set inside "
                "C is empty and the gate can flatten into a no-op. Use pin_mode='sink'."
            )
        if not pins_sink(self.pin_mode):
            raise ValueError(
                f"pin_mode={self.pin_mode!r} leaves the gate able to flatten into a no-op, which "
                "recovers the frozen dense model and satisfies the loss with no ranking learned. "
                "Use pin_mode='sink'."
            )
        if self.n_sink is not None and self.n_sink <= 0:
            raise ValueError(
                f"n_sink must be positive, got {self.n_sink}: pinning zero keys makes the gate's "
                "normalizer inert (a per-row constant cancels in the softmax), which reopens the "
                "no-op the pin exists to close."
            )
        if self.query_chunk is not None and self.query_chunk <= 0:
            raise ValueError(f"query_chunk must be positive, got {self.query_chunk}")
        if self.log_budget is None:
            # Not an error: B = n_gated is the flat-gate no-op point, which is the right *reference*
            # for a diagnostic and the wrong setting for a run. Warned rather than defaulted away,
            # because a silent B=n_gated trains a router that can satisfy the loss having learned
            # nothing -- the exact hole pinning exists to close.
            logger.warning(
                "log_budget is unset, so it resolves to log(n_gated) -- the flat-gate NO-OP point, "
                "where the router can satisfy the loss without learning any ranking. Pass "
                "log_budget=math.log(topk) to train at the budget inference will evict at."
            )
        elif self.log_budget <= 0.0:
            # log B <= 0 means B <= 1. This USED to warn and tell the caller to pass log(topk); that
            # advice is retracted -- the 2x2 grid measured B=1 at 48.20 RULER against B=2048's 20.43,
            # so B=topk costs 27.8 points and B=1 is the best measured setting (§15.3). Kept as an
            # info-level note rather than a warning, because B<1 (strictly) is still worth flagging:
            # nothing has measured it, and the identity makes the whole history worth less than one
            # sink. Silence at exactly B=1, which is the recommended value.
            if self.log_budget < 0.0:
                logger.info(
                    "log_budget=%.4f means B<1: the entire gated history is worth less than one "
                    "pinned sink. B=1 (log_budget=0) is the measured-best setting; values below it "
                    "are unmeasured. See cross_replay_e2e.md §15.3.",
                    self.log_budget,
                )

    @property
    def sink_count(self) -> int:
        """Leading keys to pin, defaulting to the press's own ``n_sink``."""
        return self.press.n_sink if self.n_sink is None else self.n_sink

    def reset(self) -> None:
        """Drop per-pass state."""
        self.gate_scales = {}
        self.layers_gated = 0
        self._scores.clear()
        self._context_len = None

    @contextmanager
    def ungated(self, model: nn.Module):
        """
        Temporarily restore the model's own attention, for pass 1.

        Required because :meth:`prefill` is called from inside :meth:`hooks`, where the attention
        implementation is already the gated one. Without this, pass 1 would run *gated* -- which is
        exactly the failure the design forbids: ``KV(C)`` would no longer be the dense values
        inference produces, and the score would feed back into the hidden states it is computed
        from. It also cannot work at all, since the scores do not exist yet when pass 1 runs.

        A no-op outside :meth:`hooks`, so :meth:`prefill` can also be called standalone.
        """
        if self._original_impl is None:
            yield
            return
        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        swapped = [cfg._attn_implementation for cfg in configs]
        try:
            for cfg in configs:
                cfg._attn_implementation = self._original_impl
            yield
        finally:
            for cfg, previous in zip(configs, swapped):
                cfg._attn_implementation = previous

    def indexer_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """Every indexer parameter, in layer order -- what the optimizer should be given."""
        params: list[nn.Parameter] = []
        for layer in get_language_model(model).layers:
            indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
            if indexer is not None:
                params.extend(indexer.parameters())
        return params

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Put every non-indexer parameter at ``requires_grad=False``, upcasting ``gate_scale`` to fp32.

        Delegates both to :class:`~.e2e_trainer.E2EIndexerTrainer`, whose implementation identifies
        indexers by module identity (not by name substring) and documents why the bf16 ``gate_scale``
        would otherwise stay frozen at initialization under a warmup learning rate. Shared rather
        than reimplemented so the two objectives cannot drift on what "frozen" means.
        """
        from kvpress.presses.gqa_indexer.e2e_trainer import E2EIndexerTrainer

        E2EIndexerTrainer(press=self.press, pin_mode="sink").freeze_backbone(model)

    # ------------------------------------------------------------------
    # Pass 1
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, model: nn.Module, input_ids: torch.Tensor) -> tuple[DynamicCache, dict]:
        """
        Dense, ungated prefill of ``C``. Returns ``(cache, {layer_idx: hidden_states})``.

        Dense and ungated because the inference path is dense prefill -> evict -> decode, so
        ``KV(C)`` must hold the values the deployed model would produce. Gating here would also be
        circular: ``h`` at layer ``l`` depends on earlier layers' attention output (verified --
        perturbing layer 0's output by 5% moves later layers' ``h``), and ``s = f(h)``, so a gate in
        pass 1 feeds back into its own score.

        ``no_grad`` throughout, and the hidden states are detached. ``dL/dw = sum_i (dL/ds_i) h_i``
        needs only ``h``'s *values*: the indexer weights are the leaves, so re-feeding a detached
        ``h`` still yields a full gradient (verified). Pass 1 therefore builds no autograd graph at
        all, which is what keeps it off the memory peak -- see ``cross_replay_e2e.md`` §6.1, where
        the retained ``h_C`` turns out to be ~10% of the cost and pass 2's own graph the rest.
        """
        hidden: dict[int, torch.Tensor] = {}

        def capture(module, args, kwargs):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None:
                return None
            h = kwargs.get("hidden_states")
            if h is None and args:
                h = args[0]
            # Detached deliberately: see the docstring. Keeping the graph would retain the whole
            # backbone's activations for a pass whose gradient nothing needs.
            hidden[int(layer_idx)] = h.detach()
            return None

        language_model = get_language_model(model)
        handles = [
            layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
            for layer in language_model.layers
        ]
        cache = DynamicCache()
        try:
            # Under the model's OWN attention, never the gated one: see `ungated`. Pass 1 must
            # produce the dense KV inference would produce, and the scores do not exist yet.
            with self.ungated(model):
                language_model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()

        missing = [
            int(layer.self_attn.layer_idx)
            for layer in language_model.layers
            if int(layer.self_attn.layer_idx) not in hidden
        ]
        if missing:
            raise RuntimeError(
                f"prefill captured no hidden states for layer(s) {missing}; the indexer cannot be "
                "scored without them. The pre-hook must be registered on every attention module."
            )
        return cache, hidden

    def score_context(self, model: nn.Module, hidden: dict[int, torch.Tensor]) -> None:
        """
        Score every ``C`` key from its own hidden state, once, with gradients.

        ``(B, H_kv, N)`` per layer -- no query axis is ever formed, which is both the ``O(L)``
        property the query-independent arm exists for and the reason the gate can be a broadcast
        mask.

        Scored here rather than inside the attention override so that a chunked replay scores ``C``
        **once** and reuses it across chunks. Rescoring per chunk would be wasted work and would
        also risk each chunk passing a different ``key_offset``.
        """
        self._scores.clear()
        for layer in get_language_model(model).layers:
            indexer = self.press.get_indexer(layer.self_attn)
            if not hasattr(indexer, "score_keys"):
                raise TypeError(
                    f"{type(indexer).__name__} has no score_keys(): cross-replay trains a "
                    "query-independent score. A pairwise GQAIndexer would need an (Sq, Sk) gate, "
                    "which is the cost this objective avoids. Build the press with "
                    "scorer='scalar'."
                )
            layer_idx = int(layer.self_attn.layer_idx)
            # key_offset=0: these are the C keys and C starts at absolute position 0. The recency
            # tilt must be measured from there, or it restarts and the score depends on framing.
            self._scores[layer_idx] = indexer.score_keys(hidden[layer_idx], key_offset=0)

    # ------------------------------------------------------------------
    # Pass 2
    # ------------------------------------------------------------------
    def gate(self, layer_idx: int, n_kv_heads: int) -> torch.Tensor:
        """
        The additive log-gate over ``C``'s keys, ``(B, H_kv, N)``.

        ``score - logsumexp(score over gated keys) + log B`` on the gated keys and ``0`` on the
        pinned leading keys, matching :func:`~.gate_pin.gate_from_score` plus the budget term --
        reproduced here rather than called because that helper takes the materialized
        ``(B, h, Sq, Sk)`` layout this objective never builds.

        The pin is what makes the normalizer load-bearing. Normalizing *without* it is inert: the
        logsumexp is one constant per row, so it cancels in the attention softmax along with
        everything else. Exempting the leading keys breaks that symmetry -- they sit at multiplier
        1 while the gated keys share a total of ``B`` -- so the router can only choose *which* keys
        receive that budget, and that choice is the ranking.

        The budget term is what :attr:`log_budget` sets; see it for why omitting it is not neutral.
        """
        scores = self._scores[layer_idx]
        if scores.shape[1] != n_kv_heads:
            raise ValueError(
                f"layer {layer_idx} scored {scores.shape[1]} heads but attention has {n_kv_heads} "
                "KV heads; per-head scores must match for each head to gate its own keys."
            )
        k_len = scores.shape[-1]
        n_sink = min(self.sink_count, k_len)
        gated = torch.arange(k_len, device=scores.device) >= n_sink
        if not bool(gated.any()):
            raise ValueError(
                f"n_sink={n_sink} pins every one of the {k_len} keys, leaving nothing gated: the "
                "router would have no keys to rank."
            )
        lse = torch.logsumexp(
            scores.masked_fill(~gated, -float("inf")), dim=-1, keepdim=True
        )
        normalized = scores - lse + self.resolve_log_budget(k_len - n_sink)
        return torch.where(gated, normalized, torch.zeros_like(scores))

    def resolve_log_budget(self, n_gated: int) -> float:
        """
        ``log B`` for this row, resolving :attr:`log_budget`'s sentinels.

        ``None`` -> ``log(n_gated)``, the flat-gate no-op point, which is the *wrong* default for
        training and is offered only as the diagnostic reference. Negative sentinel is not used;
        a numeric value is taken as ``log B`` directly.
        """
        if self.log_budget is None:
            return math.log(max(n_gated, 1))
        return float(self.log_budget)

    def _note_flex_shape(self, query: torch.Tensor, key: torch.Tensor) -> None:
        """
        Count the distinct ``(Sq, Sk)`` shapes flex is compiled for, and warn before dynamo bails.

        A test for the thing the optimization optimizes, wired into the run itself. Dynamo's
        ``recompile_limit`` is 8 per code object; on the ninth distinct shape it stops compiling and
        **falls back to eager**, which for ``flex_attention`` means the materializing reference
        implementation -- 18730 MiB against 40 MiB for one layer at ``Sq=1024, Sk=8192``. That is
        worse than the MATH backend this replaces, and nothing about the loss would show it.

        Query padding to :data:`_FLEX_Q_ALIGN` keeps the count at 1-2 for a normal run (one full-chunk
        shape, plus one if ``|C| % query_chunk`` is nonzero), so crossing 8 means something is varying
        that should not be.
        """
        shape = (int(query.shape[2]), int(key.shape[2]))
        if shape in self._flex_shapes:
            return
        self._flex_shapes.add(shape)
        if len(self._flex_shapes) == 8:
            logger.warning(
                "flex_attention has now been compiled for %d distinct (Sq, Sk) shapes: %s. Dynamo's "
                "recompile limit is 8, and on the next new shape it stops compiling and runs "
                "flex_attention EAGER -- which materializes the score matrix (measured 18730 MiB vs "
                "40 MiB compiled for one layer at Sq=1024, N=8192). Keep query_chunk fixed across "
                "steps, or the memory fix silently inverts.",
                len(self._flex_shapes), sorted(self._flex_shapes),
            )

    def _flex_attention(self, query, key, value, gate, group_size, scale, horizon=None):
        """
        Gated attention through ``flex_attention``: the gate enters as a ``score_mod``, never a mask.

        This is the fix for the objective's dominant memory cost, and it is a backend-selection
        problem rather than a layout one. The gate has to reach the logits somehow, and as an
        ``attn_mask`` it disqualifies every fused SDPA backend at once -- measured on an H20 at
        ``B=1, H=32, H_kv=8, Sq=1024, Sk=8192, D=128`` bf16:

        ==================================================  =====  =======  =====  ========  ========
        SDPA call                                           flash  mem_eff  cudnn  runs      retained
        ==================================================  =====  =======  =====  ========  ========
        ``(B,H,1,N)`` mask, gqa, **requires_grad**          no     no       no     **MATH**  1288 MiB
        ``(B,H,1,N)`` mask, gqa, detached                   no     no       yes    cudnn      8.1 MiB
        ``attn_mask=None``, gqa                             yes    no       yes    flash       16 MiB
        ``(B,H,1,N)`` mask, K/V replicated (no gqa)         no     yes      yes    mem_eff    177 MiB
        ``flex_attention`` + ``score_mod``                  --     --       --     fused    **48 MiB**
        ==================================================  =====  =======  =====  ========  ========

        **Three independent conditions have to coincide, and this objective supplies all three.**
        Flash rejects *any* ``attn_mask`` ("Flash Attention does not support non-null attn_mask").
        Mem-efficient rejects the GQA head mismatch once a dense mask is present ("both fused kernels
        require query, key and value to have the same num_heads"). And cuDNN -- which otherwise
        handles this call fine, row 2 -- rejects a mask that **requires grad**. Remove any one and a
        fused kernel survives; together they leave only MATH, which retains the full
        ``(B, H, Sq, Sk)`` scores *and* the softmax output. 1288 MiB x 36 layers = **46.7 GiB**, the
        gap between the measured 73.2 GiB peak at 8K and the 23.9 GiB the arithmetic predicted.

        Row 2 is worth staring at, because it is how this stayed hidden: a backend probe written with a
        detached gate reports cuDNN available and 8.1 MiB retained, i.e. no bug. But ``dL/ds`` arriving
        through the mask *is* the objective, so the gate always requires grad and row 1 is always the
        row that runs. Nothing about the loss curve distinguishes them.

        Row 4 was measured as the cheap stopgap and rejected: replicating K/V costs ``group_size`` x
        the cache and still retains 3.7x what flex does. ``score_mod`` expresses
        ``score + gate[h, kv_idx]`` exactly and fuses it, so no mask tensor of any shape is built.
        Still no hand-written kernel -- see ``cross_replay_e2e.md`` §1.1.

        Queries are padded to :data:`_FLEX_Q_ALIGN` and sliced back; see that constant for the
        autotuner failure that forces it.

        ``horizon``, when given, is ``(lookahead, query_offset)`` from
        :func:`replay_horizon_mask`, and the bound is applied **inside** ``score_mod`` rather than as
        an ``attn_mask``. That is what lets a bounded-lookahead run keep this 48 MiB path instead of
        falling to the 1288 MiB MATH row -- and, just as importantly, it means the ablation needs no
        change to ``query_chunk``, so it stays single-variable against the unbounded arm.

        Expressed as ``-inf`` on out-of-horizon pairs rather than a ``mask_mod``/``BlockMask``: the
        block mask would skip whole tiles and be faster, but ``score_mod`` alone keeps this method's
        signature and its verified gradient path, and the horizon is an ablation rather than the
        production configuration. Note the padded query rows (see above) get positions past the real
        ``q_len``; their cotangent is zero either way, so a horizon computed for them is harmless.
        """
        flex = compiled_flex_attention()
        # score_mod is traced, so this closes over `gate` by reference: no (Sq, N) tensor is formed
        # and the gradient flows back into `gate` through the compiled backward (verified against
        # SDPA: forward maxdiff 5.6e-16 fp64, dL/dgate maxdiff 4.8e-07 -- the fp32 floor this gate
        # path already sits at, since ScalarIndexer.score_keys returns .float()).
        if horizon is None:
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + gate[b, h // group_size, kv_idx].to(score.dtype)
        else:
            lookahead, query_offset = horizon
            # Same predicate as replay_horizon_mask's `keys <= rows + lookahead`, with `rows` the
            # ABSOLUTE replay position (query_offset + q_idx) so a chunked run matches the unchunked
            # one.
            #
            # The bound is a 0-d TENSOR, not a Python int, and that is load-bearing. As an int it is
            # traced as a compile-time constant, so every chunk's distinct `query_offset` is a fresh
            # dynamo cache entry: measured 42 recompile events at 4 chunks, and at 16 chunks it blows
            # through dynamo's recompile_limit of 8 and **falls back to eager flex_attention** -- the
            # 460x regression (18730 MiB vs 40 MiB per layer) that `compiled_flex_attention`'s
            # docstring exists to warn about, arriving silently as a mere UserWarning. As a tensor the
            # value is an input rather than a guard, so all chunks share one compiled kernel.
            bound = torch.tensor(
                query_offset + lookahead, device=query.device, dtype=torch.int32
            )

            def score_mod(score, b, h, q_idx, kv_idx):
                gated = score + gate[b, h // group_size, kv_idx].to(score.dtype)
                visible = kv_idx <= q_idx + bound
                return torch.where(visible, gated, torch.full_like(gated, float("-inf")))

        q_len = query.shape[2]
        pad = (-q_len) % _FLEX_Q_ALIGN
        if pad:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad))
        out = flex(
            query, key, value, score_mod=score_mod, scale=scale, enable_gqa=group_size != 1
        )
        # The padded rows never reach the loss, so their cotangent is zero and dL/ds is unaffected.
        return out[:, :, :q_len] if pad else out

    def _attention(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_):
        """
        Replacement attention for the replay pass: add the per-key gate to the logits.

        Because the score is query-independent, the gate is one value per ``(KV head, key)``. It
        reaches the logits either as a ``flex_attention`` ``score_mod`` (the CUDA path, and the only
        one that is not ruinously expensive -- see :meth:`_flex_attention`) or, where flex cannot run,
        as a ``(B, H, 1, N)`` additive mask broadcast over queries. The mask route is verified
        bit-exact against an explicit materialized gate in forward and gradient (maxdiff ``0.0e+00``),
        and flex is verified against the mask route to the gate path's fp32 floor (4.8e-07), so all
        three agree; they differ only in what they retain.
        """
        layer_idx = int(module.layer_idx)
        n_heads, k_len = query.shape[1], key.shape[2]
        n_kv_heads = key.shape[1]

        if self._context_len is not None and k_len != self._context_len:
            # The read-only cache is what holds the key axis at |C|. If this fires, C' was appended
            # after all, so the mask's causal region now admits C' and the objective has silently
            # become ordinary causal LM on a doubled cache.
            raise RuntimeError(
                f"layer {layer_idx} attended to {k_len} keys but C has {self._context_len}: the "
                "replay pass must see KV(C) only. Use ReadOnlyCache -- use_cache=False does not "
                "prevent the cache from growing."
            )

        indexer = self.press.get_indexer(module)
        gate_scale = indexer.require_gate_scale()
        self.gate_scales[layer_idx] = float(gate_scale.detach())
        self.layers_gated += 1

        gate = self.gate(layer_idx, n_kv_heads) * gate_scale
        group_size = n_heads // n_kv_heads
        scale = query.shape[-1] ** -0.5 if scaling is None else float(scaling)

        # The all-zero rectangle exists so that mask creation is not skipped (see rectangle_mask); it
        # adds nothing numerically, so it is dropped rather than combined with the gate. Recognized by
        # tag, not by inspecting values: `(mask != 0).any()` would be a device sync on every layer of
        # every chunk.
        #
        # NOTE, so nobody repeats the mistake: dropping it here was originally written believing the
        # broadcast `bias + rectangle` was the cause of the 73.1 GiB peak at 8K. It was NOT -- that fix
        # left the peak at 73.1 GiB, unchanged. The real cause was the SDPA backend, fixed by
        # `_flex_attention`. This dropping is kept because avoiding a materialized (B, H, Sq, N)
        # tensor is right on its own terms, and because it is what lets the flex path run at all: a
        # real mask would have to enter score_mod.
        real_mask = attention_mask
        if real_mask is not None and getattr(real_mask, "_kvpress_all_zero", False):
            real_mask = None

        # A bounded replay horizon (replay_horizon_mask with lookahead set) is a REAL mask, so leaving
        # it here would disqualify flex and land on the 1288 MiB MATH row. It is instead re-expressed
        # inside score_mod from the (lookahead, query_offset) the mask carries, which keeps the 48 MiB
        # path and -- the reason it matters for the ablation -- keeps query_chunk unchanged, so the
        # comparison against the unbounded arm stays single-variable. The SDPA branch below still
        # applies the real mask, and the two are pinned together by
        # `test_lookahead_flex_and_sdpa_paths_agree`.
        horizon = None
        if real_mask is not None and hasattr(real_mask, "_kvpress_lookahead"):
            horizon = (real_mask._kvpress_lookahead, real_mask._kvpress_query_offset)
            flex_mask, real_mask = real_mask, None
        else:
            flex_mask = None

        reason = flex_fallback_reason(query, real_mask, dropout)
        if reason is None:
            self._note_flex_shape(query, key)
            out = self._flex_attention(
                query, key, value, gate, group_size, scale, horizon=horizon
            )
        else:
            _warn_flex_fallback(reason, query.is_cuda)
            # (B, H_kv, N) -> (B, H, 1, N): one row, broadcast over every query. On CUDA this route
            # is the 46.7 GiB MATH fallback documented in `_flex_attention`; off CUDA (the fp32/fp64
            # test models) it is correct and cheap enough, and it is the reference the flex path is
            # verified against.
            #
            # A horizon mask was moved out of `real_mask` above so flex could take it; put it back
            # here, since this branch has no score_mod to fold it into. Forgetting this would silently
            # train the UNBOUNDED rectangle on every non-flex run (CPU tests, no-flex torch) while the
            # config said otherwise -- the exact failure shape §9 catalogues.
            if flex_mask is not None:
                real_mask = flex_mask
            bias = gate.repeat_interleave(group_size, dim=1).unsqueeze(2).to(query.dtype)
            mask = bias if real_mask is None else bias + real_mask.to(bias.dtype)
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                scale=scale,
                dropout_p=dropout,
                enable_gqa=group_size != 1,
            )
        # The attention interface contract is (B, Sq, H, D); the layer reshapes to (B, Sq, H*D).
        return out.transpose(1, 2).contiguous(), None

    @contextmanager
    def hooks(self, model: nn.Module):
        """
        Point the model's attention at :meth:`_attention` for the duration of the block.

        Only the attention implementation is swapped -- unlike
        :class:`~.e2e_trainer.E2EIndexerTrainer` no per-layer pre-hook is needed, because the
        scores are computed from pass 1's hidden states before the replay starts rather than from
        the tensors flowing through pass 2.

        Cleanup goes through ``_global_mapping`` directly: ``register()`` writes there while
        ``pop()`` only touches the instance mapping, so the naive removal leaves the entry behind
        forever. Same reason :func:`~.teacher_lse.capture_teacher_lse` does it this way.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        if self.freeze:
            self.freeze_backbone(model)

        impl_name = "kvpress_cross_replay_gated"
        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        previous_impls = [cfg._attn_implementation for cfg in configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        had_previous = impl_name in global_mapping
        previous_fn = global_mapping.get(impl_name)
        ALL_ATTENTION_FUNCTIONS.register(impl_name, self._attention)

        try:
            for cfg in configs:
                cfg._attn_implementation = impl_name
            self._original_impl = previous_impls[0]
            yield self
        finally:
            self._original_impl = None
            for cfg, previous in zip(configs, previous_impls):
                cfg._attn_implementation = previous
            if had_previous:
                global_mapping[impl_name] = previous_fn
            else:
                global_mapping.pop(impl_name, None)
            self._scores.clear()
            self._context_len = None

    def check_positions(self, model: nn.Module, context_len: int) -> None:
        """
        Warn when the replay's furthest relative distance leaves the trained position range.

        Replay query ``j`` sits at absolute position ``N + j`` and reads keys at ``0..N-1``, so the
        largest relative distance is ``2N - 1``. Past ``max_position_embeddings`` the replay queries
        occupy positions the model was never trained on, which no loss curve would flag. On
        Qwen3-8B (``max_position_embeddings=40960``) this is clear at ``N <= 16384`` and exceeded by
        1.6x at ``N = 32768``.

        A warning rather than an error: RoPE still evaluates there, and someone deliberately probing
        length extrapolation should not be blocked.
        """
        config = getattr(model, "config", None)
        limit = getattr(config, "max_position_embeddings", None)
        if limit is None:
            return
        d_max = 2 * context_len - 1
        if d_max > limit:
            logger.warning(
                "cross-replay at N=%d puts replay queries up to position %d, past this model's "
                "max_position_embeddings=%d (%.1fx). Early keys are then read from untrained "
                "positions, which the loss curve will not reveal. Reduce the context length, or "
                "accept that this run also measures length extrapolation.",
                context_len, d_max, limit, d_max / limit,
            )


def cross_replay_training_step(
    model: nn.Module,
    trainer: CrossReplayTrainer,
    *,
    input_ids: torch.Tensor,
    replay_ids: torch.Tensor | None = None,
    logit_chunk: int | None = None,
    backward: bool = True,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """
    One cross-replay step: dense prefill of ``C``, gated replay of ``C'``, LM loss on ``C'``.

    Must be called inside ``trainer.hooks(model)``.

    Parameters
    ----------
    input_ids : torch.Tensor
        ``(1, N)`` context ``C``. Batch size 1 only: pass 1's cache and the per-key scores are
        indexed per layer without a batch axis to reconcile, and the memory profile is set by ``N``
        rather than by the batch.
    replay_ids : torch.Tensor, optional
        ``(1, N)`` replay text ``C'``. Defaults to ``input_ids`` -- replaying the context against its
        own KV is the reconstruction objective. Supplying different tokens turns this into a
        cross-document control, which is the natural null: a score trained on the rectangle should
        do worse when the replay text is unrelated to ``C``.
    logit_chunk : int, optional
        Rows of ``lm_head`` output to materialize at a time. ``None`` builds the whole chunk's
        logits at once.

        A separate knob from ``query_chunk`` because the logits are a first-order memory term that
        the attention chunking does not touch: at ``vocab = 151936`` one replay chunk costs
        ``chunk * vocab`` in the model dtype **plus** the fp32 copy the cross-entropy needs -- 0.87
        GiB at ``query_chunk=1024`` and 6.96 GiB at 8192 on Qwen3-8B. Splitting the ``lm_head`` +
        cross-entropy over row blocks bounds that independently, at the cost of one backward per
        block.

        Deliberately *not* Liger's ``skip_logits``. That fuses ``lm_head`` into the loss inside
        ``*ForCausalLM.forward`` and needs ``labels`` passed to it; this objective calls the **base**
        model (it must, to control the cache and mask) and computes the loss itself, so the flag
        would be silently ignored -- it was accepted and threaded through here for one revision
        before that was noticed.
    backward : bool
        Run ``backward()`` **inside** this function, chunk by chunk. Default True, and required for
        ``query_chunk`` to save anything.

        This is not a convenience flag. Accumulating the chunks' losses and calling ``backward()``
        once afterwards keeps every chunk's graph alive until the end, so peak memory is the same as
        the unchunked run -- measured: retention *grew* slightly (1164 KiB unchunked to 1195 KiB at
        8 chunks) because chunking adds per-chunk overhead without releasing anything. The saving
        comes only from freeing each chunk's graph as soon as its gradient is taken.

        ``False`` returns a graph-carrying loss for the caller to backward, which is only meaningful
        unchunked; it is rejected when chunking would actually split the sequence -- unless grad is
        disabled, where there is no graph to hold and evaluating a chunked loss is legitimate (the
        shuffle control does exactly that).
    loss_scale : float
        Multiplier applied to the loss **before** the internal backward, so it reaches the gradients.
        Use ``1 / accum_steps`` for gradient accumulation.

        This parameter exists because the obvious way to accumulate is **silently wrong here**. With
        ``backward=True`` this function differentiates internally and returns a *detached* scalar, so
        the usual ``(loss / accum_steps).backward()`` -- or any scaling of the return value -- divides
        a number that no longer has a graph attached. The gradients would be ``accum_steps`` times too
        large while the logged loss looked perfectly correct, which is the exact shape of the four
        bugs recorded in ``cross_replay_e2e.md`` §9: right-looking output, wrong quantity. Scaling has
        to happen inside, hence a parameter rather than caller arithmetic.

        Applied to each chunk's and each row block's loss, so it composes with ``query_chunk`` and
        ``logit_chunk``; the returned value is scaled too, so a caller summing the returns across an
        accumulation group gets the mean it expects.

    Returns
    -------
    torch.Tensor
        With ``backward=True``: the **detached** mean replay loss, already backwarded. Gradients are
        accumulated into the indexer parameters, so an outer loop may keep accumulating across
        micro-batches as usual.

        With ``backward=False``: the loss with its graph attached, for the caller to backward.

    Notes
    -----
    **Why the scores become leaves.** The per-key scores are shared by every chunk, so a per-chunk
    ``backward()`` would free their graph on the first chunk and fail on the second. Instead ``s`` is
    detached into a leaf for the replay, each chunk accumulates ``dL/ds`` into it, and one final
    ``backward`` pushes the accumulated cotangent through ``s = f(h)`` into the indexer weights. This
    is the two-stage structure of ``cross_replay_e2e.md`` §6.1(b), which was verified bit-exact
    against holding the whole graph; here it is load-bearing rather than an optimization.

    ``gate_scale`` needs no such treatment: it is applied inside each chunk's own graph, so its
    gradient accumulates directly.

    ⚠️ **That also means ``demand_reduce`` does not reach ``gate_scale``.** The reduction operates on
    the per-key cotangent ``dL/ds`` harvested from the score leaves, and ``gate_scale``'s gradient
    never passes through them -- so it keeps summing over chunks whatever the reduction is. Measured:
    under ``demand_reduce="mean"`` every indexer weight's gradient scales by exactly ``1/n_chunks``
    while ``gate_scale``'s scales by ``1.0``. This is a real inconsistency, not a rounding artefact,
    and it is left in deliberately: ``gate_scale`` is one scalar per layer whose job is the *magnitude*
    of the gate, not the ranking, and the reduction is about which keys the ranking favours. Recorded
    because a reader comparing gradient norms across reductions will otherwise find it and wonder,
    and because if the ablation is ever extended to ``gate_scale`` this is the line to change.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be (1, N), got {tuple(input_ids.shape)}")
    replay_ids = input_ids if replay_ids is None else replay_ids
    if replay_ids.shape != input_ids.shape:
        raise ValueError(
            f"replay_ids {tuple(replay_ids.shape)} must match input_ids {tuple(input_ids.shape)}"
        )

    if loss_scale <= 0:
        # Zero would look like a working step -- a finite loss, no error -- while contributing no
        # gradient at all, and 1/accum_steps is the only intended source of this value.
        raise ValueError(f"loss_scale must be positive, got {loss_scale}")

    context_len = input_ids.shape[1]
    trainer.check_positions(model, context_len)

    chunk = trainer.query_chunk or context_len
    # Chunk spans and their target counts, computed up front: the loss must be divided by the TOTAL
    # token count before any chunk is backwarded, or the chunks would be weighted unequally and the
    # gradient would not match the unchunked run.
    spans = []
    for start in range(0, context_len, chunk):
        stop = min(start + chunk, context_len)
        label_stop = min(stop + 1, context_len)
        n_targets = max(label_stop - start - 1, 0)
        if n_targets:
            spans.append((start, stop, label_stop, n_targets))
    n_tokens = sum(span[3] for span in spans)
    if not spans or n_tokens == 0:
        raise ValueError(
            f"context length {context_len} yields no next-token targets; need at least 2 tokens."
        )
    if not backward and len(spans) > 1 and torch.is_grad_enabled():
        raise ValueError(
            f"backward=False with query_chunk={trainer.query_chunk} would hold all {len(spans)} "
            "chunks' graphs at once, which is exactly the memory chunking exists to avoid (measured: "
            "no saving at all). Use backward=True, leave query_chunk unset, or wrap the call in "
            "torch.no_grad() if you only want the loss value."
        )

    cache, hidden = trainer.prefill(model, input_ids)
    trainer.score_context(model, hidden)
    # h is referenced by the score graph from here on, so dropping the dict costs nothing and
    # keeps a second reference from pinning L * N * hidden_size.
    hidden.clear()

    # Detach the scores into leaves for the replay; see the Notes above. Without this a per-chunk
    # backward frees s's graph on the first chunk. Skipped when grad is off: requires_grad_ raises
    # inside no_grad, and there is nothing to accumulate anyway.
    scored = trainer._scores
    stage_two = backward and torch.is_grad_enabled()
    if stage_two:
        leaves = {idx: s.detach().requires_grad_(True) for idx, s in scored.items()}
        trainer._scores = leaves
    else:
        leaves = {}

    trainer._context_len = context_len
    language_model = get_language_model(model)

    # Chunk-level demand aggregation (`demand_reduce="max"`). The cotangent that reaches `s` is
    # `sum_j` over replay queries -- §16.3's averaging, and the thing KVzip replaces with a `max`.
    # A true per-query max would need one backward per query, so the affordable granularity is the
    # query chunk: `leaves[idx].grad` is read and zeroed after each chunk, giving that chunk's own
    # demand, and the per-chunk demands are combined at the end instead of summed by autograd.
    per_chunk_demand: dict[int, list[torch.Tensor]] = {}
    collect_per_chunk = stage_two and trainer.demand_reduce != "sum"
    if collect_per_chunk and len(spans) < 2:
        # One chunk means one demand group, so `max` and `sum` coincide and the knob is silently
        # inert -- exactly the "looks configured and is not" failure this file keeps recording.
        raise ValueError(
            f"demand_reduce={trainer.demand_reduce!r} needs at least 2 query chunks to aggregate "
            f"over, but query_chunk={trainer.query_chunk} gives {len(spans)} for context length "
            f"{context_len}. Set query_chunk to at most {context_len // 2} (it also bounds the "
            "per-chunk granularity of the reduction: N/query_chunk demand groups)."
        )

    total = 0.0
    graph_loss = None
    for start, stop, label_stop, _ in spans:
        # A fresh view per chunk: the cache must not accumulate, and each chunk carries its own
        # absolute positions so the replay sits at [N + start, N + stop) as the geometry requires.
        read_only = ReadOnlyCache(cache)
        positions = torch.arange(
            context_len + start, context_len + stop, device=input_ids.device
        ).unsqueeze(0)
        out = language_model(
            input_ids=replay_ids[:, start:stop],
            past_key_values=read_only,
            position_ids=positions,
            # Explicit, always: see rectangle_mask. None here would silently give a causal triangle.
            # query_offset=start so a chunked run indexes the key axis exactly as the unchunked pass
            # would -- without it every chunk's horizon would restart at 0 and later chunks would see
            # far less than they should.
            attention_mask=replay_horizon_mask(
                stop - start,
                context_len,
                input_ids.device,
                model.dtype,
                query_offset=start,
                lookahead=trainer.lookahead,
            ),
            use_cache=True,
        )
        hidden_out = out.last_hidden_state
        targets = replay_ids[0, start + 1 : label_stop]
        n_rows = targets.shape[0]
        rows = logit_chunk or n_rows

        for row_start in range(0, n_rows, rows):
            row_stop = min(row_start + rows, n_rows)
            logits = model.lm_head(hidden_out[0, row_start:row_stop])
            # promote_types, not .float(): upcasting bf16 logits to fp32 keeps the summed
            # cross-entropy precise, but a hard .float() would DOWNCAST an fp64 test model.
            acc_dtype = torch.promote_types(logits.dtype, torch.float32)
            # loss_scale enters HERE, before any backward, which is the whole point of it being a
            # parameter: scaling the returned scalar instead would touch a detached tensor and leave
            # the gradients unscaled. See the docstring.
            block_loss = loss_scale * torch.nn.functional.cross_entropy(
                logits.to(acc_dtype), targets[row_start:row_stop], reduction="sum"
            ) / n_tokens
            if stage_two:
                # Immediately, so this block's logits and this chunk's activations are released
                # before the next is built. retain_graph because the chunk's transformer graph is
                # shared by every row block; it is freed on the last one.
                block_loss.backward(retain_graph=row_stop < n_rows)
                total += float(block_loss.detach())
            elif backward:
                # backward=True under no_grad: nothing to differentiate, accumulate the value.
                total += float(block_loss.detach())
            else:
                graph_loss = block_loss if graph_loss is None else graph_loss + block_loss

        if collect_per_chunk:
            # Harvest this chunk's own dL/ds and zero the leaf, so the next chunk's backward
            # accumulates from 0 rather than on top of this one. Cloned because .grad is reused.
            for idx, leaf in leaves.items():
                if leaf.grad is not None:
                    per_chunk_demand.setdefault(idx, []).append(leaf.grad.detach().clone())
                    leaf.grad = None

    if trainer.layers_gated == 0:
        raise RuntimeError(
            "no layer ran the gated attention: the model kept its own attention implementation. "
            "This usually means the model's config is not the one CrossReplayTrainer pointed at "
            f"{model.config._attn_implementation!r}."
        )

    if not backward:
        trainer._scores = scored
        return graph_loss

    if stage_two:
        # Stage B: push the accumulated dL/ds through s = f(h) into the indexer weights, once.
        if collect_per_chunk:
            # Combine the per-chunk demands with the chosen reduction instead of letting autograd
            # sum them. `sum` is what plain accumulation already gives, so it never lands here.
            #
            # Sign convention matters and is easy to get backwards: the cotangent is dL/ds, so a
            # NEGATIVE entry is a key the loss wants raised. "The demand of the chunk that needs this
            # key most" is therefore the most negative entry, i.e. amin over chunks -- taking amax
            # would keep the chunk that wants the key GONE the most, which is the opposite selection
            # and would still train to a plausible-looking loss.
            reduced = {}
            for idx, demands in per_chunk_demand.items():
                stacked = torch.stack(demands)  # (n_chunks, B, H_kv, N)
                if trainer.demand_reduce == "max":
                    # Scaled to keep the gradient magnitude comparable to `sum`, so the effective
                    # learning rate does not silently change with the reduction. Without this, `max`
                    # would also be an LR ablation and the comparison would be confounded.
                    reduced[idx] = stacked.amin(dim=0) * stacked.shape[0]
                else:  # "mean" -- the average demand, i.e. sum/n_chunks. Kept as the null control.
                    reduced[idx] = stacked.mean(dim=0)
            tensors = [scored[idx] for idx in reduced]
            cotangents = [reduced[idx] for idx in reduced]
        else:
            tensors = [scored[idx] for idx in leaves if leaves[idx].grad is not None]
            cotangents = [leaves[idx].grad for idx in leaves if leaves[idx].grad is not None]
        if tensors:
            torch.autograd.backward(tensors, cotangents)
    trainer._scores = scored
    return torch.tensor(total, device=input_ids.device)


def gate_participation(gate: torch.Tensor, n_sink: int) -> float:
    """
    Effective fraction of keys the gate spreads its mass over -- **the readout that says whether the
    router learned anything at all**.

    ``PR = 1 / sum(r^2)`` over the gated keys, where ``r`` is the gate's mass **normalized to sum to
    1**: ``PR = n`` for a flat distribution over ``n`` keys and ``1`` when a single key takes
    everything. Divided by the gated count to give a scale-free number comparable across layers,
    lengths, and budgets.

    Read it, not the loss. ``~1.0`` means the gate is **flat**, i.e. no ranking was learned, however
    healthy the loss curve looks -- that is the reachable no-op the whole pin mechanism exists to close
    (``cross_replay_e2e.md`` §0, §3). Falling towards 0 is the concentration eviction needs, and the
    target is the eval budget ``topk/N``, not 0.

    **The normalization is load-bearing, and omitting it silently broke this metric for one run.**
    ``exp(gate)`` sums to ``B``, not to 1 (that is what the budget *is*), so ``1 / sum(exp(gate)^2)``
    is the true participation scaled by ``1 / B^2``. At ``B = 1`` the two agree exactly, so the bug
    was invisible until the budget term was added -- and then the ``B = 2048`` run reported
    ``participation = 0.0000`` at every step from step 0, which reads as "totally collapsed" when the
    truth was 0.927 falling to 0.062. Normalizing first makes the metric budget-invariant, which is
    the property it needs: it must compare runs whose budgets differ.

    Lives here rather than in a script because two callers need it -- the smoke check and the training
    driver -- and a second copy of the one number that distinguishes "trained" from "trained-looking"
    is exactly the drift worth preventing. Same reason :meth:`CrossReplayTrainer.freeze_backbone`
    delegates to :class:`~.e2e_trainer.E2EIndexerTrainer` instead of reimplementing.
    """
    p = gate[..., n_sink:].detach().exp()
    n_gated = p.shape[-1]
    if n_gated == 0:
        return float("nan")
    # Normalize to a distribution before squaring: sum(exp(gate)) is B, not 1.
    p = p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
    return float((1.0 / (p**2).sum(-1)).mean() / n_gated)


@contextmanager
def shuffled_scores(trainer: CrossReplayTrainer, perm: torch.Tensor):
    """
    Replace the trainer's per-key scores with a permutation of themselves, for the shuffle control.

    **The control that separates "trained" from "trained-looking".** A score that ranks keys usefully
    must do worse when its ranking is destroyed; if permuting it along the key axis costs nothing, the
    gate carries no usable ordering and the objective is measuring nothing -- and no loss curve would
    say so. On random token ids, shuffling cost +1.95 nats/token after five steps.

    Implemented by replacing :meth:`~CrossReplayTrainer.score_context` for the duration, because
    :func:`cross_replay_training_step` calls it internally and would otherwise overwrite the shuffled
    scores with freshly computed ones. A context manager rather than inline monkey-patching so the
    restore happens on the exception path too -- a leaked patch would leave every later step training
    against permuted scores, which is a silent and total corruption of the run.

    Call the step under ``torch.no_grad()`` with ``backward=False``: this measures a loss, it must not
    contribute gradient.

    Restores by **deleting the instance attribute** rather than assigning the bound method back. The
    difference matters: ``trainer.score_context`` builds a *new* bound-method object on every access,
    so reassigning one leaves the attribute shadowing the class's method forever with an object that
    also holds a reference back to the trainer. Deleting it re-exposes the descriptor, which is the
    true original. Same ``had_previous`` / restore-or-remove shape as
    :meth:`CrossReplayTrainer.hooks` uses for ``ALL_ATTENTION_FUNCTIONS``, for the same reason.
    """
    learned = {idx: s.detach().clone() for idx, s in trainer._scores.items()}
    had_override = "score_context" in trainer.__dict__
    previous = trainer.__dict__.get("score_context")

    def inject(*_args, **_kwargs):
        trainer._scores.update({idx: s[..., perm] for idx, s in learned.items()})

    trainer.score_context = inject
    try:
        yield learned
    finally:
        if had_override:
            trainer.score_context = previous
        else:
            del trainer.score_context
