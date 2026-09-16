# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run the GQA indexer as *sparse attention* at inference.

The eviction press (:class:`~kvpress.presses.gqa_indexer.press.GQAIndexerPress`) drops keys from
the cache. This does the other thing the indexer enables, and the thing DSA ships: keep the whole
cache and let **each query attend only to its own top-k keys**, via
:func:`~kvpress.presses.gqa_indexer.triton_sparse_attention.sparse_gqa_attention`. Selection only --
the softmax is plain over the selected keys, with **no gate term** (that is the end-to-end training
path, :class:`~kvpress.presses.gqa_indexer.e2e_trainer.E2EIndexerTrainer`).

Mechanism
---------
The wiring mirrors ``E2EIndexerTrainer.hooks()`` exactly: a forward pre-hook on every attention
module stashes the ``hidden_states`` used by hidden-state scorers, while the attention interface
provides the actual projected values used by value scorers. A temporary entry in
``ALL_ATTENTION_FUNCTIONS`` that ``config._attn_implementation`` is pointed at replaces the
attention itself. Both are removed on exit.

The one thing training does not need and this does: an **indexer key-cache**. During training a
forward sees the whole sequence at once, so ``project_k`` produces every key. At inference each
step contributes only the *new* hidden states or projected values, so we accumulate the per-layer
indexer keys ourselves -- initialize on prefill, append on each decode step -- and assert the cache
length stays in lockstep with the model's own ``key.shape[2]``. The cache is tiny (one projected
key, or one score per KV head, per token and layer) and lives only for the duration of the ``with``
block, so a fresh context per generation gives a fresh cache.

Batch-1 assumption
------------------
Eval processes one context at a time, so there is no padding and the causal-arithmetic path of
:func:`~kvpress.presses.gqa_indexer.sparse_support.streaming_topk_support` (``mask=None``) is both
correct and cheap. For a padded batch the selector would need the additive mask built by
``build_indexer_mask``; that is deliberately not wired here.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.qi_flex_attention import (
    FLEX_BLOCK,
    HAS_FLEX,
    deadlines,
    qi_sparse_attention,
)
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support
from kvpress.presses.gqa_indexer.triton_sparse_attention import sparse_gqa_attention

logger = logging.getLogger(__name__)

#: The name the sparse attention is registered under in ``ALL_ATTENTION_FUNCTIONS``.
IMPL_NAME = "kvpress_gqa_indexer_sparse"


class SparseAttentionContext:
    """
    Replace every attention layer with indexer-driven sparse attention for the block's duration.

    Parameters
    ----------
    model : nn.Module
        The causal LM. Its attention layers are swapped on entry and restored on exit.
    press : GQAIndexerPress
        Holds the per-layer indexers (via ``post_init_from_model``) and the RoPE narrowing. The
        press's ``compression_ratio`` is irrelevant here -- nothing evicts; the indexer is used
        only to *select* per-query top-k keys.
    topk : int
        Support size per query, including the forced sink/local slots. Fixed, not a ratio: a ratio
        makes the retained support ``O(L^2)``.

        For a benchmark whose documents vary in length, see ``topk_ratio``, which resolves a
        per-document ``topk`` from the actual context length. A fixed ``topk`` on such a benchmark
        silently stops compressing on every document shorter than it -- measured on LongBench,
        whose median context is 10156 tokens: at ``topk=16384`` **80% of documents evict nothing**,
        so a nominal "retain 50%" is really "retain 100% four times out of five", and the 50% and
        25% arms scored 47.07 against 47.06.
    topk_ratio : float, optional
        Retain this FRACTION of each document's own context instead of a fixed count, i.e.
        ``topk = max(ceil(ratio * context_length), force_sink + force_local + 1)``. This is what
        makes a "retain 50%" claim mean the same thing on every row of a length-heterogeneous
        benchmark. Set through :meth:`set_context_length`, which the generation pipeline calls
        once per document; ``topk`` is then ignored.
    force_sink, force_local : int
        Slots reserved per row for the leading keys and the row's own most-recent keys, matching
        what the sparse training stage used.
    block_k : int
        Triton topk-tile; power of two, ``>= 16``. Memory/throughput knob only.
    causal : bool
        Mask slots past each query's diagonal. Redundant for causally-selected indices and cheap.
    precision : str
        ``tl.dot`` precision. ``"tf32"`` by default here, unlike the kernel's own ``"ieee"``
        default, because at inference the operands are the model's own bf16 q/k/v: every bf16
        value is exactly representable in tf32 (10 mantissa bits against bf16's 8), so the QK
        dot is *bit-identical* either way, and the PV dot's only genuine fp32 operand -- the
        softmax weights -- rounds to ~2e-4 relative, some 30x below the bf16 epsilon the output
        is stored at. Measured on an H20 at ``L=8192, topk=2048``: 67.0 s per prefill under
        ``"ieee"`` against 9.4 s under ``"tf32"``, for identical error against the fp32
        reference (7.52e-3 both, i.e. bf16 output rounding alone).

        The reason the gap is so large is that ``"ieee"`` fp32 does not use tensor cores at all,
        so ``BLOCK_G`` -- padded up to 16 on Triton 3.3, which requires ``M >= 16`` -- becomes
        real work rather than lanes the hardware was going to occupy regardless: M scales the
        kernel ~linearly under ``"ieee"`` (1.89x for 2x M, measured) and barely at all under
        ``"tf32"`` (1.09x). Pass ``"ieee"`` to reproduce the fp32 reference exactly, which is
        what the tests do; it is the wrong default for a bf16 model at length.
    memory : bool
        Fold the linear memory over the evicted keys into the same softmax as the retained ones
        (:mod:`~kvpress.presses.gqa_indexer.memory`). Requires a press built with ``memory=True``,
        its weights loaded, and ``flex_attention`` -- the fusion reads the retained branch's ``lse``,
        which the gather kernel does not return, so this forces the flex path for decode rows too.

    Usage
    -----
    >>> with SparseAttentionContext(model, press, topk=512, force_sink=4, force_local=64):
    ...     model.model(input_ids=context_ids, past_key_values=cache)   # sparse prefill
    ...     out = model(input_ids=question_ids, past_key_values=cache)   # sparse decode
    """

    def __init__(
        self,
        model: nn.Module,
        press: GQAIndexerPress,
        *,
        topk: int,
        force_sink: int = 0,
        force_local: int = 0,
        block_k: int = 64,
        causal: bool = True,
        precision: str = "tf32",
        query_independent: bool | None = None,
        memory: bool = False,
        cmp_slots: int = 0,
        cmp_delta: float = 0.0,
        cmp_mass: str = "count",
        cmp_space: str = "post_rope",
        cmp_mass_ckpt: str = "",
        head_budget: str = "uniform",
        head_budget_floor: int = 0,
        head_budget_table: str = "",
        topk_ratio: float | None = None,
    ):
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if force_sink < 0 or force_local < 0:
            raise ValueError("force_sink and force_local must be non-negative")
        if force_sink + force_local > topk:
            raise ValueError(
                f"force_sink + force_local = {force_sink + force_local} exceeds topk = {topk}; "
                "the forced keys would be truncated. Lower them or raise topk."
            )
        self.model = model
        self.press = press
        self.memory = bool(memory)
        # CMP slots (training-free). Paid for out of `topk`, not added to it: the row still reads
        # `topk` entries, of which `cmp_slots` are cluster summaries of what it dropped. An unmatched
        # budget would measure the budget rather than the idea.
        self.cmp_slots = int(cmp_slots)
        self.cmp_delta = float(cmp_delta)
        if cmp_mass not in ("count", "count+var"):
            raise ValueError(f"cmp_mass must be 'count' or 'count+var', got {cmp_mass!r}")
        self.cmp_mass = cmp_mass
        # Which space the k-means ASSIGNMENT runs in. "pre_rope" un-rotates the cache first, so the
        # clustering sees content with the positional phase removed; the slot's k/v are still means
        # of post-RoPE keys either way (see cluster_evicted). A clean single-variable ablation: only
        # the partition changes.
        if cmp_space not in ("post_rope", "pre_rope"):
            raise ValueError(f"cmp_space must be 'post_rope' or 'pre_rope', got {cmp_space!r}")
        self.cmp_space = cmp_space
        # Trained CMPMassHead per layer (scripts/train_cmp_mass.py). Supersedes cmp_mass/cmp_delta:
        # b_r = count_coef*log n_r + var_coef*(1/2 Var_r) + bias, 3 scalars per KV head.
        self._cmp_mass_heads: dict[int, object] = {}
        if cmp_mass_ckpt:
            from kvpress.presses.gqa_indexer.cmp_slots import CMPMassHead

            payload = torch.load(cmp_mass_ckpt, map_location="cpu", weights_only=False)
            cfg = payload.get("config", {})
            if int(cfg.get("slots", self.cmp_slots)) != self.cmp_slots:
                # Not fatal, but it changes what log n_r means: the same document clustered into a
                # different R gives different populations, and count_coef was fitted against one of
                # them. Warn rather than raise so an R sweep against one head is still possible.
                logger.warning(
                    "cmp_mass_ckpt was trained at R=%s but this run uses R=%s; the learned "
                    "count_coef was fitted against different cluster populations.",
                    cfg.get("slots"), self.cmp_slots,
                )
            if cfg.get("space") and cfg["space"] != self.cmp_space:
                logger.warning(
                    "cmp_mass_ckpt was trained with cmp_space=%r but this run uses %r; the "
                    "partition differs, so the fitted coefficients are off-distribution.",
                    cfg["space"], self.cmp_space,
                )
            n_kv_ck = int(cfg.get("n_kv_heads", 0)) or None
            for k_, sd_ in payload["mass_heads"].items():
                head = CMPMassHead(n_kv_ck or sd_["count_coef"].numel())
                head.load_state_dict(sd_)
                self._cmp_mass_heads[int(k_)] = head.eval()
            logger.info(
                "Loaded %d CMP mass heads from %s (R=%s, space=%s)",
                len(self._cmp_mass_heads), cmp_mass_ckpt, cfg.get("slots"), cfg.get("space"),
            )
            # The head weights the 1/2 Var term, so the variance must actually be computed.
            self.cmp_mass = "count+var"
        if self.cmp_slots < 0:
            raise ValueError(f"cmp_slots must be non-negative, got {cmp_slots}")
        if self.cmp_slots and self.memory:
            raise ValueError(
                "cmp_slots and memory=True both compensate the evicted set and would double-count "
                "it; pick one."
            )
        if topk_ratio is None and self.cmp_slots >= topk - force_sink - force_local:
            raise ValueError(
                f"cmp_slots={cmp_slots} leaves no top-k budget at topk={topk}, "
                f"force_sink={force_sink}, force_local={force_local}: the slots are funded out of "
                "topk, so they must be a fraction of it."
            )
        # Per-layer CMP state, built once at the end of the context prefill.
        self._cmp: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # Cache length at which each layer's slots were built. A later forward may read them only if
        # its FIRST row sits at or after this point, which is the causality condition -- keeping the
        # build position rather than a bool is what lets the question forward (Sq ~ 28, all rows
        # behind the context) use the slots while a prefill row never can.
        self._cmp_at: dict[int, int] = {}
        # Slots actually used per layer. Equals `cmp_slots` normally; under `topk_ratio` a short
        # document can force it lower (or to 0) -- see `_cmp_take`. The attend path must read the
        # SAME value the build used, or the two branches disagree about how many summaries exist.
        self._cmp_r: dict[int, int] = {}
        if self.memory and not HAS_FLEX:
            raise RuntimeError(
                "memory=True needs flex_attention: the fusion reads the retained branch's lse, "
                "which the gather kernel does not return."
            )
        self.topk = int(topk)
        # Per-document budget. `self.topk` becomes the RESOLVED count for the current document,
        # recomputed by set_context_length(); `_topk_nominal` keeps the constructor's value so a
        # ratio run can be told apart from a fixed-topk one after the fact.
        self._topk_nominal = int(topk)
        if topk_ratio is not None and not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        self.topk_ratio = None if topk_ratio is None else float(topk_ratio)
        # How the per-layer budget `n_kv_heads * topk` is split across KV heads.
        #
        # "uniform" is today's behaviour and the only allocation the arm was trained under.
        #
        # "mass" splits it so every head reaches the same RETAINED ATTENTION MASS. That currency is
        # forced, not chosen: the trained gate is `score - lse` on history and `0` on pinned keys,
        # so adding a constant to a whole (layer, head)'s score vector leaves the forward BITWISE
        # unchanged -- the score's absolute level is a gauge freedom. Any rule that pools raw scores
        # across heads (AdaKV's `scores.reshape(bsz, -1)`) is therefore ranking on an
        # unidentifiable quantity; retained softmax mass is invariant to it. Measured on the
        # fwkl_ce01 router at 8K: mass retained under the uniform budget spreads 0.268 across heads
        # within a layer (max 0.606), and matching lifts the worst head by +0.154 on average.
        #
        # The total is conserved exactly (see `allocate_by_mass`), so this reallocates the cache
        # rather than enlarging it.
        if head_budget not in ("uniform", "mass", "shuffle", "static"):
            raise ValueError(
                f"head_budget must be 'uniform', 'mass', 'shuffle' or 'static', "
                f"got {head_budget!r}"
            )
        self.head_budget = head_budget
        # Minimum evictable budget each head keeps before the split, TrimKV's
        # `min_tokens_per_head`. A pooled allocation can otherwise starve a head down to its pins
        # on the strength of one reference row. 0 disables it.
        self.head_budget_floor = int(head_budget_floor)
        # (n_layers, n_kv_heads) table for head_budget="static": budgets fitted OFFLINE, once, and
        # read here. Motivated by measurement, not convenience -- the per-head demand ranking is as
        # stable across documents as across reference rows within one document (Spearman 0.59
        # mass / 0.64 participation, scratch/diag_skeleton.py), so much of "which heads need keys"
        # is a property of the model rather than of the input. A constant table costs nothing at
        # prefill and, because it never depends on the input, cannot drift during a long decode.
        self.head_budget_table: torch.Tensor | None = None
        if head_budget_table:
            payload = torch.load(head_budget_table, map_location="cpu", weights_only=False)
            table = payload["table"] if isinstance(payload, dict) else payload
            self.head_budget_table = torch.as_tensor(table, dtype=torch.int64)
            fitted = payload.get("topk") if isinstance(payload, dict) else None
            if fitted is not None and int(fitted) != int(topk):
                # Fatal: the table's rows sum to the topk it was fitted at, so reading it at a
                # different topk silently changes the total cache the run uses -- the one thing
                # that would make an A/B against the uniform baseline meaningless.
                raise ValueError(
                    f"head_budget_table was fitted at topk={fitted} but this run uses "
                    f"topk={topk}; its rows sum to {fitted} * n_kv_heads, so it would change "
                    "the total budget. Refit at this topk."
                )
        elif head_budget == "static":
            raise ValueError("head_budget='static' needs head_budget_table=<path>")
        # Per-layer (n_kv_heads,) int64 budgets, computed once at the context prefill and reused by
        # every later row -- decode must not re-derive them, both because a 1-row forward has no
        # attention distribution to measure and because a budget that moved between prefill and
        # decode would evict keys the prefill had already committed to.
        self._head_topk: dict[int, torch.Tensor] = {}
        self.force_sink = int(force_sink)
        self.force_local = int(force_local)
        self.block_k = int(block_k)
        self.causal = bool(causal)
        if precision not in ("ieee", "tf32"):
            raise ValueError(f"precision must be 'ieee' or 'tf32', got {precision!r}")
        self.precision = precision
        # Resolved lazily in __enter__: it depends on the scorer the press holds, which
        # post_init_from_model attaches there. None means "use the fast path when the scorer
        # declares itself query-independent and this torch has flex_attention".
        self._query_independent = query_independent
        self._use_qi = False
        self._decay_active = False

        # Per-layer state, all keyed by layer_idx and reset on entry.
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._kwargs: dict[int, dict] = {}
        self._k_idx: dict[int, torch.Tensor] = {}  # the indexer key-cache, (B, Sk, Di)

        self._handles: list = []
        self._configs: list = []
        self._previous_impls: list = []
        self._registry_restore = None  # (had_previous, previous_fn)

    # ------------------------------------------------------------------
    # Hooks (mirrors E2EIndexerTrainer._capture_hook)
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """Stash this layer's hidden_states + kwargs before its attention runs."""
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        self._hidden_states[int(layer_idx)] = hidden_states
        self._kwargs[int(layer_idx)] = kwargs
        return None

    # ------------------------------------------------------------------
    # The sparse attention itself
    # ------------------------------------------------------------------
    def _attend(self, module, query, key, value, scaling):
        """
        Select each query's top-k keys with the indexer, then attend only to them.

        ``query``/``key``/``value`` arrive post-RoPE and post-cache-update from the layer, so
        ``key.shape[2]`` is the full ``k_len`` and the KV tensors carry ``n_kv_heads`` heads --
        exactly what :func:`sparse_gqa_attention` expects.
        """
        layer_idx = int(module.layer_idx)
        hidden_states = self._hidden_states.get(layer_idx)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached sparse attention without its hidden_states captured. "
                "The pre-hook must be installed on the same modules as the attention swap."
            )
        kwargs = self._kwargs.get(layer_idx, {})

        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        previous = self._k_idx.get(layer_idx)
        previous_len = 0 if previous is None else previous.shape[1]
        # Absolute position of the first NEW token: 0 at prefill, the cache length at each decode
        # step. A decay-carrying scorer folds -log_beta_j * j / ref into its key, so an offset of 0
        # at decode would score every generated token as if it sat at position 0 -- its whole
        # history would look infinitely old to it.
        q_kwargs = {}
        if self._decay_active:
            q_kwargs["query_offset"] = previous_len
            q_kwargs["n_kv_heads"] = key.shape[1]
        # Same inputs the training path uses: hidden-state scorers ignore value_states, while DMA
        # consumes the actual post-projection values for only this step's new tokens.
        q_idx = indexer.project_q(hidden_states, cos, sin, **q_kwargs)  # (B, h, Sq, Di)
        value_states_new = value[:, :, previous_len:, :]
        if value_states_new.shape[2] != hidden_states.shape[1]:
            raise RuntimeError(
                f"layer {layer_idx}: expected {hidden_states.shape[1]} newly appended values but "
                f"found {value_states_new.shape[2]}. SparseAttentionContext requires an "
                "append-only KV cache."
            )
        k_idx_new = indexer.project_k(
            hidden_states,
            cos,
            sin,
            value_states=value_states_new,
            **({"key_offset": previous_len} if self._decay_active else {}),
        )  # (B, Sq, Di)

        # Indexer key-cache: initialize on prefill, append on each decode step. Entries stay in
        # position order, so the cache mirrors the model's own KV cache.
        k_idx = k_idx_new if previous is None else torch.cat([previous, k_idx_new], dim=1)
        self._k_idx[layer_idx] = k_idx

        k_len = key.shape[2]
        if k_idx.shape[1] != k_len:
            raise RuntimeError(
                f"layer {layer_idx}: indexer key-cache has {k_idx.shape[1]} keys but the model "
                f"cache holds {k_len}. The two must stay in lockstep -- use one "
                "SparseAttentionContext per generation (its cache resets on entry)."
            )

        # A cache shorter than the forced slots has nothing to select: force_sink + force_local
        # already covers every key, so the support is the whole sequence and sparse attention IS
        # dense attention. Short-circuiting is a correctness requirement, not an optimization --
        # streaming_topk_support clamps topk down to k_len and *then* rejects
        # force_sink + force_local > topk, so this case raises instead of trivially keeping
        # everything. Reasoning benchmarks hit it constantly rather than as a corner case:
        # math500's context is a single space (4 tokens once the chat template is applied), so 20
        # of the 50 rows sampled at fraction 0.1 begin decoding at k_len < 68 = 4 + 64 and the run
        # dies on its first question instead of producing a number.
        #
        # Deliberately narrow: k_len <= topk is also a no-op selection, but the gather path handles
        # it correctly (topk is clamped to k_len), so it stays on the kernel and the precision
        # plumbing that test_precision_reaches_the_kernel pins remains observable.
        if k_len < self.force_sink + self.force_local:
            return self._attend_dense(query, key, value, scaling)

        if self.memory:
            return self._attend_with_memory(
                module, query, key, value, scaling, q_idx, k_idx, k_len
            )

        # CMP slots. Built once, at the end of the CONTEXT prefill, and read by every later forward
        # whose rows all sit after the clustered keys -- see the causality note in
        # :mod:`~.cmp_slots`.
        #
        # The read condition is "this forward starts at or after the build point", NOT "q_len == 1".
        # The pipeline runs two multi-token forwards (context, then the question), and the question's
        # rows are all behind every context key, so they are safe -- and they matter most, since the
        # last question row emits the first answer token, which is the whole of a needle score.
        # Gating on q_len == 1 would forfeit that for no gain in safety.
        if self.cmp_slots:
            built_at = self._cmp_at.get(layer_idx)
            if built_at is None:
                self._build_cmp(module, query, key, value, q_idx, k_idx, k_len, scaling)
            elif k_len - q_idx.shape[2] >= built_at and layer_idx in self._cmp:
                # `layer_idx in self._cmp` is load-bearing, not defensive. `_build_cmp` records
                # `_cmp_at` even when NOTHING was evicted (so the layer is not re-clustered on every
                # later forward), and in that case it returns without writing `_cmp`. A context
                # shorter than the budget evicts nothing -- which is the common case whenever topk
                # is a large fraction of the context, e.g. LongBench at topk=16384 where many
                # documents are under 16K. Without this guard the next forward raised
                # `KeyError: <layer>` from `_attend_with_cmp`. Falling through is also the correct
                # behaviour: with no evicted keys there is nothing for the slots to summarize, so
                # the exact branch alone IS the full attention.
                return self._attend_with_cmp(
                    module, query, key, value, scaling, q_idx, k_idx, k_len
                )

        # Query-independent scorers take the flex_attention path: the score is a fixed per-key
        # vector, so each key is selected by one contiguous interval of query rows and the whole
        # support is a per-key deadline instead of a (B, h, Sq, topk) index tensor. Same selection,
        # block-sparse contiguous reads instead of gathers -- measured 2.31x at L=8030 and 4.25x at
        # L=4096 for the select+attend pair. Only worth it when there is a query axis to amortize
        # the block-mask build over, so decode (Sq == 1) stays on the gather path.
        if self._use_qi and q_idx.shape[2] > 1:
            # The deadline path needs ONE frozen per-key ranking. Without decay any query row
            # gives it, since every row is identical. With decay the rows genuinely differ, so a
            # row has to be chosen and the selection becomes an approximation of the exact
            # per-row top-k the gather path below computes.
            #
            # The MIDDLE row, not row 0. Row 0 sees every key at age ~0, which is the one position
            # where the lifetime term contributes nothing -- it throws the feature away and ranks
            # purely by magnitude. Measured deadline-mask error against the exact per-row top-k
            # (log_beta ~ U(-1,0) nats/ref, Sk=4096, take=512): row 0 4.52%, middle 1.64%, last
            # 2.59%. The middle row minimizes the worst-case age error over the row range.
            ref_row = q_idx.shape[2] // 2 if self._decay_active else 0
            router_scores = torch.einsum(
                "bhqd,bkd->bhk", q_idx[:, :, ref_row : ref_row + 1], k_idx
            )
            out = qi_sparse_attention(
                query,
                key,
                value,
                # The per-key score IS one row of the score matrix; take it from q_idx/k_idx
                # rather than calling score_keys again, so this path cannot drift from what the
                # gather path would score.
                router_scores,
                self._budget_for(module, query, key, router_scores, scaling, layer_idx),
                force_sink=self.force_sink,
                force_local=self.force_local,
                scaling=scaling,
            )  # (B, H, Sq, Dv)
            return out.transpose(1, 2).contiguous()

        # query_offset defaults to k_len - Sq in both calls (bottom-right), correct for prefill
        # (Sq == k_len) and decode (Sq == 1) alike -- so it is never passed explicitly.
        # Per-head budget for the gather path. Goes through `_budget_for` rather than reading
        # `_head_topk` directly, which had two consequences that only showed up on a benchmark
        # with no long prefill (math500/aime25, whose whole problem is in `question` and whose
        # context is a single space):
        #
        # * `head_budget="static"` was silently downgraded to the uniform scalar here -- the old
        #   condition tested `== "mass"`, so the fitted table was never consulted at decode. That
        #   is precisely the mode that is supposed to work without a query axis, so the one
        #   setting able to allocate during generation was the one being ignored.
        # * `mass` reached this line only ever pre-populated by an earlier multi-row prefill. With
        #   a 1-row prefill `_budget_for` bails to the scalar (its distribution is not measurable
        #   from a single row), so `_head_topk` stays empty and the run is uniform -- measured: 0
        #   of 36 layers allocated on math500. Calling `_budget_for` makes that fallback explicit
        #   and identical in both paths instead of implicit in a dict lookup.
        #
        # `mass` needs a per-key score row to build its mass curve from. On this path the score is
        # not necessarily row-invariant (that is why this path exists), so one row has to be
        # chosen -- the middle one, matching the qi path's reasoning about the decay term. It is
        # computed only when it will actually be consumed: at decode, and for `static`/`uniform`,
        # `_budget_for` returns before looking at it, and a (1, Hkv, Sk) einsum per layer per step
        # would be pure overhead.
        gather_scores = None
        if self.head_budget in ("mass", "shuffle") and q_idx.shape[2] > 1 and layer_idx not in self._head_topk:
            mid = q_idx.shape[2] // 2
            gather_scores = torch.einsum("bhqd,bkd->bhk", q_idx[:, :, mid : mid + 1], k_idx)
        budget = self._budget_for(module, query, key, gather_scores, scaling, layer_idx)
        if not torch.is_tensor(budget):
            support, _ = streaming_topk_support(
                q_idx,
                k_idx,
                self.topk,
                mask=None,
                force_sink=self.force_sink,
                force_local=self.force_local,
            )  # (B, h, Sq, topk) int32, ascending, -1 empty
        else:
            # Decode under a per-head budget. The gather path's support is a rectangular
            # (B, h, Sq, topk) tensor, so ragged budgets are expressed by padding each head out to
            # the widest one with the -1 "empty" sentinel the kernel already skips.
            #
            # Per head rather than one call at max(budget): decode MUST read the same support the
            # prefill committed to. Letting it fall back to the uniform topk would quietly hand the
            # starved heads back their keys exactly when the answer is being generated, which is
            # the one place the allocation has to hold.
            widest = int(budget.max())
            parts = []
            for h in range(q_idx.shape[1]):
                one, _ = streaming_topk_support(
                    q_idx[:, h : h + 1],
                    k_idx,
                    int(budget[h]),
                    mask=None,
                    force_sink=self.force_sink,
                    force_local=self.force_local,
                )
                if one.shape[-1] < widest:
                    pad = one.new_full((*one.shape[:-1], widest - one.shape[-1]), -1)
                    one = torch.cat([one, pad], dim=-1)
                parts.append(one)
            support = torch.cat(parts, dim=1)
        out, _ = sparse_gqa_attention(
            query,
            key,
            value,
            support,
            scaling=scaling,
            causal=self.causal,
            block_k=self.block_k,
            precision=self.precision,
        )  # (B, H, Sq, Dv)
        # The attention interface contract is (B, Sq, H, D); our op returns (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Per-head budget allocation
    # ------------------------------------------------------------------
    def set_context_length(self, context_length: int) -> int:
        """
        Resolve this document's ``topk`` from :attr:`topk_ratio`. Returns the value in force.

        A no-op when ``topk_ratio`` is None, so a fixed-``topk`` run is bit-identical whether or
        not the caller invokes this.

        Call it BEFORE the context prefill. Two pieces of per-layer state are derived from
        ``topk`` and are cleared here, because neither is detectable later if it goes stale:

        * ``_head_topk`` -- the mass-matched per-head split. Its entries sum to
          ``topk * n_kv_heads`` for the topk they were fitted at, so reusing them under a
          different budget would silently change the total cache the run uses.
        * ``_cmp``/``_cmp_at`` -- the CMP slots summarize the keys the *old* budget evicted.

        The floor keeps at least one evictable slot: ``force_sink + force_local + 1``. Without it
        a short document could resolve to a budget the pins alone consume, and the top-k would
        have nothing to rank -- which is a different regime, not a tighter one.
        """
        if self.topk_ratio is None:
            return self.topk
        import math

        floor = self.force_sink + self.force_local + 1
        resolved = max(int(math.ceil(self.topk_ratio * int(context_length))), floor)
        if resolved != self.topk:
            self._head_topk.clear()
            self._cmp.clear()
            self._cmp_at.clear()
        self.topk = resolved
        return resolved

    def _budget_for(self, module, query, key, router_scores, scaling, layer_idx):
        """
        This layer's budget: the scalar ``topk``, or a ``(n_kv_heads,)`` mass-matched split.

        Computed **once**, on the forward that first sees a multi-row prefill for this layer, and
        cached. Two reasons it must not be recomputed:

        * a later forward (the question, or decode) has a different -- often much shorter -- query
          axis, so its attention distribution is not the one the cache was filled against;
        * the deadline mask is monotone by construction, and a budget that shrank between forwards
          would try to un-evict keys the earlier rows already committed to.

        Falls back to the uniform scalar whenever the allocation is not well-posed (a single-row
        forward, or a context short enough that every head can keep its whole pool), so the
        feature can never make a short prompt behave differently from the baseline.
        """
        if self.head_budget == "uniform":
            return self.topk
        cached = self._head_topk.get(layer_idx)
        if cached is not None:
            return cached
        if self.head_budget == "static":
            # No measurement at all: read the row and cache it. Deliberately NOT gated on the
            # prefill shape -- a static table is valid for a 1-row forward too, which is exactly
            # what makes it immune to the decode drift the measured modes have to live with.
            budgets = self.head_budget_table[layer_idx].to(key.device)
            self._head_topk[layer_idx] = budgets
            return budgets
        if query.shape[2] <= 1 or key.shape[2] <= self.topk:
            return self.topk

        from kvpress.presses.gqa_indexer.head_budget import mass_head_budgets

        budgets = mass_head_budgets(
            query, key, router_scores[0],
            topk=self.topk,
            force_sink=self.force_sink,
            force_local=self.force_local,
            scaling=scaling,
            floor=self.head_budget_floor,
        )
        if self.head_budget == "shuffle":
            # THE CONTROL. Same multiset of budgets, permuted so each head gets some OTHER head's
            # allocation. Conserves the total and reproduces the ragged shape exactly, but destroys
            # the head<->budget correspondence. If shuffling scores as well as "mass", then the gain
            # is from raggedness itself (an implicit budget increase somewhere, or a lucky
            # interaction with the deadline mask) and NOT from measuring per-head demand -- which
            # would invalidate the paper's claim while leaving the RULER number intact.
            g = torch.Generator(device="cpu").manual_seed(1234 + layer_idx)
            budgets = budgets[torch.randperm(budgets.numel(), generator=g).to(budgets.device)]
        self._head_topk[layer_idx] = budgets
        if layer_idx == 0:
            logger.info(
                "head_budget=mass layer 0: %s (sum %d, uniform would be %d)",
                budgets.tolist(), int(budgets.sum()), self.topk * budgets.numel(),
            )
        return budgets

    # ------------------------------------------------------------------
    # CMP slots (training-free compensation for the evicted set)
    # ------------------------------------------------------------------
    def _cmp_take(self, module, query, key, scaling, layer_idx, router_scores):
        """
        The budget the exact branch reads once ``cmp_slots`` are funded out of it.

        Scalar ``topk - R`` under a uniform budget; ``(n_kv_heads,)`` ``budget_h - R`` under a
        per-head one. Both CMP call sites go through here so the slot build and the attend cannot
        disagree about which keys were evicted -- if they did, a key could land in both branches
        (double-counted in the softmax) or in neither (dropped silently). ``_budget_for`` caches
        per layer, so every call after the build returns the same vector.

        ``router_scores`` is ``(1, Hkv, Sk)`` -- the same one row of the score matrix the deadline
        is built from, so the allocation and the selection are measured against one ranking.

        The floor guard is per head rather than the scalar check ``__init__`` can make: with a
        ragged allocation the smallest head is the binding one, and it is not known until the
        budget has been measured.
        """
        budget = self._budget_for(module, query, key, router_scores, scaling, layer_idx)

        if not self.cmp_slots:
            return budget
        evictable = (
            int(budget.min()) if isinstance(budget, torch.Tensor) else int(budget)
        ) - self.force_sink - self.force_local
        if self.cmp_slots < evictable:
            return budget - self.cmp_slots
        # R does not fit. Under a FIXED topk that is a misconfiguration and refusing is right --
        # the user asked for something impossible. Under `topk_ratio` it is expected: the budget
        # tracks each document's length, so a short row legitimately resolves to a small topk
        # (LongBench `multi_news` has a 1898-token median -> topk=475, leaving as little as 1
        # evictable slot per head after the 132 pins), and aborting the eval because one row is
        # short would make the ratio arm unrunnable. Shrink R to half the smallest head's
        # evictable budget so the exact branch still reads at least as many real keys as
        # summaries; 0 disables the slots for this document.
        if self.topk_ratio is None:
            raise ValueError(
                f"cmp_slots={self.cmp_slots} leaves no top-k budget for the smallest head at "
                f"layer {layer_idx}: its allocation is "
                f"{int(budget.min()) if isinstance(budget, torch.Tensor) else int(budget)} against "
                f"force_sink={self.force_sink} + force_local={self.force_local}, i.e. "
                f"{evictable} evictable slots. The slots are funded out of each head's own "
                "budget, so R must be below the SMALLEST allocation -- lower R, raise "
                "--head_budget_floor, or raise topk."
            )
        usable = max(0, min(self.cmp_slots, (evictable - 1) // 2))
        self._cmp_r[layer_idx] = usable
        return budget - usable if usable else budget

    def _cmp_scores(self, q_idx, k_idx):
        """One row of the score matrix -- the per-key score, for a query-independent scorer.

        Taken from ``q_idx``/``k_idx`` rather than by calling ``score_keys`` again, for the same
        reason the memory path does it: the slots and the attention mask must be derived from the
        *same* ranking or a key can end up in both branches (double-counted) or neither.
        """
        ref = q_idx.shape[2] // 2 if self._decay_active else 0
        return torch.einsum("bhqd,bkd->bhk", q_idx[:, :, ref : ref + 1], k_idx)[0].float()

    @torch.no_grad()
    def _build_cmp(self, module, query, key, value, q_idx, k_idx, k_len, scaling) -> None:
        """
        Cluster the keys the *last* prefill row drops, and stash the slots for decode.

        The horizon is taken at the final row (``k_len - 1 - force_local``), which is the evicted set
        every subsequent decode row inherits -- eviction is monotone in the horizon, so a key dropped
        here is dropped for good (the irreversibility the position tilt buys: 0 re-entries over 1500
        steps). Nothing about the future enters, because every clustered key sits at a position the
        last prefill row has already passed.
        """
        from kvpress.presses.gqa_indexer.cmp_slots import (
            cluster_evicted,
            evicted_from_deadline,
        )
        from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines

        layer_idx = int(module.layer_idx)
        scores = self._cmp_scores(q_idx, k_idx)
        # take = topk - R: the deadline must be computed at the budget the row will ACTUALLY read
        # exactly, otherwise the "evicted" set is the wrong one and the slots summarize keys the
        # top-k is still holding.
        #
        # Under a per-head budget this is a VECTOR, and it has to be: the exact branch below reads
        # budget_h - R keys from head h, so a scalar here would mark the wrong keys evicted for
        # every head whose allocation differs from topk -- the slots would summarize keys the head
        # still holds while keys it actually dropped went unsummarized. `deadlines` takes an
        # (n_heads,) tensor, so the two stay consistent by construction.
        dl = deadlines(
            scores,
            self._cmp_take(
                module, query, key, scaling, layer_idx, scores.unsqueeze(0)
            ),
            force_sink=self.force_sink,
            force_local=self.force_local,
        )
        horizon = max(k_len - 1 - self.force_local, 0)
        evicted = evicted_from_deadline(dl, horizon)
        # Record the build point even when nothing was evicted, so the layer is not re-clustered on
        # every subsequent forward (which would rebuild against a growing cache and silently make
        # the slots a different summary at each step).
        self._cmp_at[layer_idx] = k_len
        if not bool(evicted.any()):
            return
        # `_cmp_take` may have shrunk R to 0 for this document (a short row under `topk_ratio`
        # whose smallest head cannot fund any summaries). Record the build point above so the
        # layer is not reconsidered, but skip the clustering -- `cluster_evicted` rejects
        # n_slots=0, and with no slots the exact branch alone is the whole attention anyway.
        if self._cmp_r.get(layer_idx, self.cmp_slots) <= 0:
            return
        n_kv = key.shape[1]
        group = query.shape[1] // n_kv
        cluster_keys = None
        if self.cmp_space == "pre_rope":
            from kvpress.presses.gqa_indexer.cmp_slots import unrotate

            kwargs = self._kwargs.get(layer_idx, {})
            pe = kwargs.get("position_embeddings")
            if pe is None:
                raise RuntimeError(
                    f"layer {layer_idx}: cmp_space='pre_rope' needs the layer's RoPE tables, but "
                    "position_embeddings was not in its kwargs. The cache stores post-RoPE keys, so "
                    "the pre-RoPE space cannot be recovered without them."
                )
            cos, sin = pe
            # The tables cover only THIS forward's rows, while the cache holds the whole history --
            # so for the last-row build the two lengths differ and the cache's leading keys have no
            # table entry. Rebuild over the full cache from the layer's own rotary module instead of
            # padding, which would un-rotate the history at the wrong angles.
            if cos.shape[-2] != k_len:
                rope = getattr(get_language_model(self.model), "rotary_emb", None)
                if rope is None:
                    raise RuntimeError(
                        "cmp_space='pre_rope' needs model.model.rotary_emb to rebuild the tables "
                        f"over the full cache ({k_len} keys vs {cos.shape[-2]} supplied)."
                    )
                pos = torch.arange(k_len, device=key.device).unsqueeze(0)
                cos, sin = rope(key, pos)
            cluster_keys = unrotate(key[0].float(), cos[0].float(), sin[0].float())
        mass_head = self._cmp_mass_heads.get(layer_idx)
        if mass_head is not None:
            mass_head = mass_head.to(key.device)
        q_kv = None
        if self.cmp_mass == "count+var" or mass_head is not None:
            # Average the group's queries into one per KV head: the slot is a KV entry, so its mass
            # is shared by every query head that reads that head.
            q_kv = query.view(1, n_kv, group, query.shape[2], -1).mean(2)[0].float()
        self._cmp[layer_idx] = cluster_evicted(
            key[0].float(),
            value[0].float(),
            evicted,
            self._cmp_r.get(layer_idx, self.cmp_slots),
            cluster_keys=cluster_keys,
            mass_head=mass_head,
            queries=q_kv,
            scaling=float(scaling),
            delta=self.cmp_delta,
        )

    def _attend_with_cmp(
        self, module, query, key, value, scaling, q_idx, k_idx, k_len
    ) -> torch.Tensor:
        """
        Read ``[topk - R retained keys ; R CMP slots]`` through ONE softmax.

        Single softmax rather than two branches plus a fusion weight, and that is the design's main
        simplification: the normalizer is shared, so there is nothing to calibrate between the two
        sets. The slots are just extra keys carrying an additive log-mass.

        Implemented on top of the existing kernel plus :func:`~.memory.fuse_memory` rather than by
        concatenating and re-softmaxing, which is an identity, not an approximation::

            o = (N_S + n) / (D_S + d),   n = sum_r e^{l_r} v_r,   d = sum_r e^{l_r}

        and ``fuse_memory`` computes exactly that from ``(o_S, lse_S)``. Two reasons to reuse it
        instead of materializing the joint logits: the kernel keeps its block-sparse reads (a
        gathered ``(B, H, Sq, topk, D)`` is ~1.8 GB at the question forward, against streaming), and
        the ``exp(-lse_S)`` overflow that path needs clamping for is already handled there -- a row
        holding one sink key has an ``lse`` far below zero, and ``inf * 0`` on an empty slot set is
        NaN. That was a real failure once, silent until step 260.
        """
        from kvpress.presses.gqa_indexer.memory import fuse_memory
        from kvpress.presses.gqa_indexer.qi_flex_attention import _flex, qi_block_mask

        layer_idx = int(module.layer_idx)
        k_cmp, v_cmp, b_cmp = self._cmp[layer_idx]  # (Hkv, R, D), (Hkv, R, D), (Hkv, R)
        bsz, n_q_heads, q_len, head_dim = query.shape
        n_kv = key.shape[1]
        group = n_q_heads // n_kv

        # Exact branch at the REDUCED budget, so the total read count still equals topk (per head,
        # when the budget is a vector). Same deadline the slots were built from, so a key is in
        # exactly one of the two branches -- `_cmp_take` returns the cached per-layer budget on
        # every call after the build, so this cannot drift from it.
        scores = self._cmp_scores(q_idx, k_idx)
        dl = deadlines(
            scores,
            self._cmp_take(
                module, query, key, scaling, layer_idx, scores.unsqueeze(0)
            ),
            force_sink=self.force_sink,
            force_local=self.force_local,
        )
        block_mask = qi_block_mask(
            dl,
            q_len=q_len,
            k_len=k_len,
            n_q_heads=n_q_heads,
            force_sink=self.force_sink,
            force_local=self.force_local,
            device=query.device,
        )
        o_s, lse_s = _flex()(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            block_mask=block_mask,
            scale=scaling,
            return_lse=True,
        )

        # The slots' contribution as an unnormalized (numerator, denominator) pair.
        #
        # Per QUERY head, not per KV head: the slot is a KV entry so its k/v are shared across the
        # group, but the logit q.k_cmp is not -- each query head brings its own q. Expanding the
        # slots and passing group=1 is therefore the correct call, and it is cheap because R is 64
        # (the weight tensor is (B, H, Sq, 64), not (B, H, Sq, k_len)).
        kc = k_cmp.to(query.device).float().repeat_interleave(group, 0)  # (H, R, D)
        vc = v_cmp.to(query.device).float().repeat_interleave(group, 0)
        bc = b_cmp.to(query.device).float().repeat_interleave(group, 0)  # (H, R)
        l_cmp = (
            torch.einsum("bhqd,hrd->bhqr", query.float(), kc) * scaling
            + bc.view(1, n_q_heads, 1, -1)
        )
        w = l_cmp.exp()  # -inf on silenced slots -> exactly 0, no NaN
        n = torch.einsum("bhqr,hrd->bhqd", w, vc)
        d = w.sum(-1)
        out = fuse_memory(o_s, lse_s, n, d, group=1)
        return out.transpose(1, 2).contiguous()

    def _attend_dense(self, query, key, value, scaling) -> torch.Tensor:
        """
        Plain causal attention, for a cache that already fits inside the support budget.

        This is the *exact* answer, not an approximation: when ``k_len <= topk`` every key is
        selected, so the sparse softmax and the dense one run over the same set. Routed through
        SDPA rather than the gather kernel because the kernel's ``force_sink + force_local <= topk``
        precondition is violated exactly when the sequence is this short (see the caller).

        The mask is built explicitly and aligned BOTTOM-RIGHT (query row ``i`` sees key ``j`` iff
        ``j <= k_len - q_len + i``), which is what ``is_causal=True`` does *not* do: SDPA aligns its
        built-in mask top-left, so any call with ``Sq != Sk`` gets the wrong triangle. That case is
        the common one here, not a corner -- the pipeline prefills the context and the question in
        two separate forwards, so answering a 28-token question over a 4-token context arrives as
        ``Sq=28, Sk=32``. Under top-left alignment its first row would attend to key 0 alone, and the
        model degenerates into repeating text instead of answering. ``streaming_topk_support``
        defaults to the same bottom-right convention, so this matches the path it replaces.
        """
        group = query.shape[1] // key.shape[1]
        q_len, k_len = query.shape[2], key.shape[2]
        key_idx = torch.arange(k_len, device=query.device)
        q_pos = torch.arange(q_len, device=query.device).unsqueeze(-1) + (k_len - q_len)
        attn_mask = key_idx <= q_pos  # (Sq, Sk), True = visible
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            attn_mask=attn_mask,
            scale=scaling,
        )  # (B, H, Sq, Dv)
        return out.transpose(1, 2).contiguous()

    def _attend_with_memory(
        self, module, query, key, value, scaling, q_idx, k_idx, k_len: int
    ) -> torch.Tensor:
        """
        Sparse attention plus the linear memory over the keys this row would have evicted.

        Prefill and decode take different state constructions, and the difference is causality
        rather than performance:

        * **Prefill** (``Sq > 1``) builds a per-query-block state with
          :func:`~.memory_schedule.block_memory_states`, the same function training uses. A single
          state shared by every row would let row ``t`` read keys at positions ``> t``. That is not a
          mild approximation -- measured at ``L=8192``, row 0 saw a state built from 6208 future
          keys, and it took RULER 8K from 73.71 to **4.00** while the (causal) training curve looked
          healthy throughout.
        * **Decode** (``Sq == 1``) uses one state over the whole evicted set, which is causal by
          construction: the single query sits after every key in the cache.

        Either way the state is rebuilt from the current cache each call rather than carried across
        steps. That is deliberate at eval scale: ``psi`` is ``O(L)``, and maintaining ``(H, z, W)``
        incrementally means tracking exactly which keys crossed their deadline since the last step --
        a second bookkeeping path that has to agree with ``deadlines()`` exactly or the two branches
        double-count. The incremental form is a decode optimization, not a different model.
        """
        from kvpress.presses.gqa_indexer.memory import fuse_memory
        from kvpress.presses.gqa_indexer.qi_flex_attention import _flex, qi_block_mask

        memory = self.press.get_memory(module)
        bsz, n_q_heads, q_len, _ = query.shape
        n_kv_heads = key.shape[1]
        group = n_q_heads // n_kv_heads
        if bsz != 1:
            raise NotImplementedError(
                f"memory=True supports batch 1, got {bsz}: qi_block_mask is built with B=None."
            )

        # One row of the score matrix IS the per-key score for a query-independent scorer. Taken
        # from q_idx/k_idx rather than by calling score_keys again, so this path cannot drift from
        # what the selection uses -- the same reason the non-memory branch does it this way.
        scores = torch.einsum("bhqd,bkd->bhk", q_idx[:, :, :1], k_idx)[0].float()
        dl = deadlines(
            scores, self.topk, force_sink=self.force_sink, force_local=self.force_local
        )
        block_mask = qi_block_mask(
            dl,
            q_len=q_len,
            k_len=k_len,
            n_q_heads=n_q_heads,
            force_sink=self.force_sink,
            force_local=self.force_local,
            device=query.device,
        )
        o_s, lse_s = _flex()(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            block_mask=block_mask,
            scale=scaling,
            return_lse=True,
        )

        q_kv = query.view(bsz, n_kv_heads, group, q_len, query.shape[-1]).mean(2)
        offset = k_len - q_len
        if q_len > 1:
            # PREFILL: a per-query-block state, exactly as training builds it. A single state shared
            # by every row is **not** valid here -- it would let row t read keys at positions > t.
            # Measured: at L=8192 row 0 saw a state built from 6208 future keys, and RULER 8K went
            # from 73.71 to 4.00 while the training curve (which is causal) looked healthy. The
            # streaming construction is the same code training uses, so the two cannot drift.
            from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
            from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer

            H, z, W, counts = block_memory_states(
                memory,
                key,
                value,
                dl,
                q_len=q_len,
                block=FLEX_BLOCK,
                n_local=self.force_local,
                scores=scores,
            )
            n, d = MemoryTrainer._read_per_block(
                memory, q_kv, H, z, W, counts, block=FLEX_BLOCK
            )
        else:
            # DECODE: one row, and it sits after every key in the cache, so a single state over the
            # whole evicted set is causal by construction -- there is no future to leak.
            enter = dl.to(torch.int64) + 1
            weights = memory.ingest_weights(enter, k_len, scores=scores)
            state = memory.ingest(key, value, weights)
            counts = (enter <= k_len - 1).sum(-1)  # (Hkv,)
            n_evicted = counts.view(1, -1, 1).expand(bsz, -1, q_len).float()
            n, d = memory.read(q_kv, state, n_evicted)

        out = fuse_memory(o_s, lse_s, n, d, group=group)
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Context management (mirrors E2EIndexerTrainer.hooks)
    # ------------------------------------------------------------------
    def _set_prefix_cache(self, enabled: bool) -> None:
        for layer in get_language_model(self.model).layers:
            indexer = self.press.get_indexer(layer.self_attn)
            setter = getattr(indexer, "enable_cache" if enabled else "disable_cache", None)
            if setter is not None:
                setter()

    def reset(self) -> None:
        self._hidden_states.clear()
        self._kwargs.clear()
        self._k_idx.clear()
        # Stale slots would be read against a different document's cache, so this must be cleared
        # with the rest of the per-generation state rather than rebuilt opportunistically.
        self._cmp.clear()
        self._cmp_at.clear()
        self._cmp_r.clear()
        # Budgets are fitted to ONE document's attention distribution, so carrying them into the
        # next context would allocate against the wrong demand -- and silently, since the shapes
        # still line up.
        self._head_topk.clear()
        self._set_prefix_cache(True)

    def __enter__(self) -> "SparseAttentionContext":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(self.model)
        self.reset()

        # Decide the selection path now that the indexers exist. Keyed off a declared capability
        # rather than isinstance, so a third scorer only has to set the attribute.
        layers = get_language_model(self.model).layers
        scorer_is_qi = bool(
            getattr(self.press.get_indexer(layers[0].self_attn), "is_query_independent", False)
        )
        # A decay-carrying scalar scorer is still query-INDEPENDENT in the sense that matters
        # (the query side carries only position, no content), so the flex path stays available.
        # But its score is no longer constant along the query axis, which the deadline
        # construction assumes, so the selection becomes approximate and the offsets have to be
        # threaded. Both are gated on this flag.
        self._decay_active = bool(
            getattr(self.press.get_indexer(layers[0].self_attn), "decay", False)
        )
        if self._query_independent is None:
            self._use_qi = scorer_is_qi and HAS_FLEX
            if scorer_is_qi and not HAS_FLEX:
                logger.warning(
                    "scorer is query-independent but this torch has no flex_attention; falling "
                    "back to the gather path (correct, just slower)."
                )
        else:
            self._use_qi = bool(self._query_independent)
            if self._use_qi and not scorer_is_qi:
                # The fast path reads one row of the score matrix and applies it to every query. For
                # a pairwise scorer that is a different (wrong) support, not a slower one.
                raise ValueError(
                    "query_independent=True but the indexer's score depends on the query "
                    f"({type(self.press.get_indexer(layers[0].self_attn)).__name__}). The flex path "
                    "would attend over the wrong keys."
                )
            if self._use_qi and not HAS_FLEX:
                raise RuntimeError("query_independent=True but this torch has no flex_attention")
        logger.info("sparse selection path: %s", "flex (query-independent)" if self._use_qi else "gather")

        def sparse_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self._attend(module, query, key, value, scaling), None

        self._configs = [self.model.config]
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None:
            self._configs.append(text_config)
        self._previous_impls = [cfg._attn_implementation for cfg in self._configs]

        # Register through the mapping, and remember what to restore. register() writes to the
        # class-level _global_mapping while pop() only touches the instance mapping, so the naive
        # removal would leak the entry -- the same care capture_teacher_lse documents.
        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        self._registry_restore = (IMPL_NAME in global_mapping, global_mapping.get(IMPL_NAME))
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, sparse_attention_impl)

        self._handles = []
        for layer in get_language_model(self.model).layers:
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
            )
        for cfg in self._configs:
            cfg._attn_implementation = IMPL_NAME
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        for handle in self._handles:
            handle.remove()
        self._handles = []
        for cfg, previous in zip(self._configs, self._previous_impls):
            cfg._attn_implementation = previous
        if self._registry_restore is not None:
            global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
            had_previous, previous_fn = self._registry_restore
            if had_previous:
                global_mapping[IMPL_NAME] = previous_fn
            else:
                global_mapping.pop(IMPL_NAME, None)
            self._registry_restore = None
        self._hidden_states.clear()
        self._kwargs.clear()
        self._k_idx.clear()
        self._set_prefix_cache(False)
