# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Drive a model through :class:`~.evict_cache.EvictPagedPool`: prefill, compress, decode.

:class:`~.sparse_inference.SparseAttentionContext` keeps the whole cache and masks it. This runs
the other arm -- the cache is physically compressed once the context is prefilled, and every decode
step reads ``budget`` keys instead of ``O(L)``. The two select the *same* keys (proven as a set
identity in ``tests/presses/test_gqa_indexer_evict_cache.py``), so a number from this path is
comparable to one from the mask path; what changes is memory and decode cost.

The three phases
----------------
1. **Prefill, one sequence at a time**, under the ordinary mask path. Batching it would need a
   per-sequence block mask, which :func:`~.qi_flex_attention.qi_sparse_attention` refuses -- and
   there is nothing to gain, since prefill is compute-bound while decode is memory-bound.
2. **Commit**: each layer's dense cache is compressed into the pool at the last prefill row's
   ranking, and the dense cache is released.
3. **Decode, batched**, entirely inside the pool. No mask, no block mask, no gather.

Why the attention implementation owns the eviction
--------------------------------------------------
The eviction step needs the *router's score* for the arriving token, which comes from the hidden
state. A ``Cache`` never sees one, so :class:`~.evict_cache.EvictCacheLayer.update` is a
pass-through and the real work happens here, where the pre-hook has already stashed the hidden
states.

One consequence shapes the design: the per-token bookkeeping is **deferred to the end of the
step** and applied for all layers in a single :meth:`~.evict_cache.EvictPagedPool.ingest` call.
That is not tidiness. One ingest issues ~90 small CUDA kernels whose total device time is 204 us,
so per-layer calls are bound by CPU launch dispatch and cost 36x more -- measured 32 ms/token
against 0.9 ms. At 3.3 ms/token for the attention itself, the per-layer form would have been the
dominant cost of decoding.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from kvpress.presses.gqa_indexer.evict_cache import EvictCache, EvictPagedPool
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model

logger = logging.getLogger(__name__)

#: Registered attention implementation name for the eviction path.
IMPL_NAME = "kvpress_gqa_indexer_evict"


def split_router_key(k_idx: torch.Tensor, decay: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Unpack :meth:`~.scalar_indexer.ScalarIndexer.project_k`'s output into ``(mag, log_beta)``.

    ``project_k`` returns ``(B, S, Di)``. Without decay ``Di == n_heads`` and the whole thing *is*
    the magnitude. With decay ``Di == 2 * n_heads`` and head ``h`` occupies columns ``2h``
    (``s_j - log_beta_j * j / ref``) and ``2h+1`` (``log_beta_j``) -- the interleaving ``gate_key``
    documents. Both come back as ``(B, n_heads, S)``, which is the layout the pool stores.
    """
    if decay:
        mag = k_idx[..., 0::2].transpose(1, 2).contiguous().float()
        log_beta = k_idx[..., 1::2].transpose(1, 2).contiguous().float()
        return mag, log_beta
    return k_idx.transpose(1, 2).contiguous().float(), None


def shift_router_key(
    mag: torch.Tensor,
    log_beta: torch.Tensor | None,
    *,
    offset: torch.Tensor | float,
    pos_slope: float,
    decay_ref: float | None,
) -> torch.Tensor:
    """
    Re-base a router magnitude computed at ``key_offset=0`` onto absolute position ``offset``.

    Both position terms are **linear in the key index** -- the recency tilt is ``pos_slope * j``
    and the decay fold contributes ``-log_beta_j * j / decay_ref`` -- so a single scorer call plus
    this correction is exactly equal to calling it at the real offset. Verified to fp32 rounding
    (0 without decay, <= 1.9e-6 at j = 99999 with it).

    That equality is what makes **batched decode** possible at all: ``project_k`` takes one scalar
    ``key_offset``, but a batch has a different absolute position per sequence, so the alternative
    would be one scorer call per sequence.
    """
    off = torch.as_tensor(offset, dtype=torch.float32, device=mag.device)
    # `mag` is (B, Hkv, Sq) and the offset is per (sequence, token), i.e. (B, Sq) -- so the HEAD
    # axis is the one to insert, not a trailing one. Unsqueezing at the end would line the token
    # axis up against the heads and broadcast a different position into every head.
    if off.dim() == 2:
        off = off.unsqueeze(1)
    while off.dim() < mag.dim():
        off = off.unsqueeze(0)
    shifted = mag + off * pos_slope
    if log_beta is not None:
        shifted = shifted - log_beta * (off / float(decay_ref))
    return shifted


#: Rows whose logits went non-finite, counted across the process. **Greedy decoding cannot detect a
#: NaN**: `argmax` has no NaN check and silently returns the NaN's index, so a corrupted trace keeps
#: generating and looks plausible, while `multinomial` raises a device-side assert. That asymmetry is
#: why the sampled arm crashed and the greedy arm did not -- greedy was never clean, it was blind.
#: Counting here makes a greedy run auditable: a nonzero count means some traces are corrupt even
#: though the run completed. Read it via :func:`nonfinite_logit_count`.
_NONFINITE_LOGIT_ROWS = 0


def nonfinite_logit_count() -> int:
    """How many decode rows have had non-finite logits since process start. 0 == clean."""
    return _NONFINITE_LOGIT_ROWS


def _pick(logits: torch.Tensor, sampling: dict | None) -> torch.Tensor:
    """Next token per row, ``(B,)``. Greedy when ``sampling`` is None.

    Greedy is bitwise the old path when ``sampling`` is None **and the logits are finite**, which
    is every healthy step -- `nan_to_num` is a no-op on finite input, so enabling sampling cannot
    perturb a result produced without it. On a NON-finite step the behaviour deliberately DIFFERS
    from the old path: the old path fed the NaN to `argmax`. The filter order is the standard one
    (temperature, then top-k, then top-p) and matches what ``generate`` does, since the reasoning
    benchmarks are only comparable to the literature at Qwen3's shipped settings.

    Non-finite logits are **counted, not raised on**, in both modes. Raising would turn a partially
    corrupt run into no run at all; counting lets the run finish and reports how much of it to
    distrust. One `.any()` per step on a `(B, V)` tensor is negligible against the forward itself.
    """
    global _NONFINITE_LOGIT_ROWS
    bad = ~torch.isfinite(logits)
    if bool(bad.any()):
        _NONFINITE_LOGIT_ROWS += int(bad.any(-1).sum())
        # Neutralize so greedy picks a real token instead of the NaN's index, and so `multinomial`
        # does not abort the whole shard. The trace is already unreliable either way -- this only
        # keeps the process alive long enough to report the count.
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=-1e4, neginf=-1e4)
    if not sampling:
        return logits.argmax(-1)
    x = logits.float()
    temperature = float(sampling.get("temperature") or 1.0)
    if temperature > 0 and temperature != 1.0:
        x = x / temperature
    top_k = sampling.get("top_k")
    if top_k:
        k = min(int(top_k), x.shape[-1])
        kth = x.topk(k, dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    top_p = sampling.get("top_p")
    if top_p and float(top_p) < 1.0:
        order = x.argsort(dim=-1, descending=True)
        sorted_x = x.gather(-1, order)
        cum = sorted_x.softmax(-1).cumsum(-1)
        # Keep the smallest prefix whose mass reaches top_p: shifting the comparison keeps the
        # first token always, so a peaked distribution cannot mask out everything.
        drop = cum - sorted_x.softmax(-1) > float(top_p)
        x = x.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return torch.multinomial(x.softmax(-1), 1).squeeze(-1)


class EvictInferenceContext:
    """
    Run a model with a physically compressed KV cache.

    Parameters
    ----------
    model : nn.Module
        The causal LM. Its attention is swapped on entry and restored on exit.
    press : GQAIndexerPress
        Holds the per-layer indexers. Must carry a **query-independent** scorer: hard eviction
        rests on a key's rank being frozen once it leaves the top-k, which a query-aware score
        does not give (a later query may need exactly the key an earlier one dropped -- with a
        mask that key is merely unselected, here it is gone).
    budgets : torch.Tensor
        ``(n_layers, n_kv_heads)`` slots per head. Shared by every sequence in a batch.
    n_sink, n_local : int
        Pin geometry, matching what the router was trained and evaluated under.
    """

    def __init__(
        self,
        model: nn.Module,
        press: GQAIndexerPress,        *,
        budgets: torch.Tensor | None = None,
        n_sink: int,
        n_local: int,
        batch_size: int = 1,
        cmp_slots: int = 0,
        cmp_reseed_below: int = 0,
    ):
        self.model = model
        self.press = press
        # None means "read the budget off the prefill", which is what makes --topk_ratio and
        # --head_budget work: both are resolved only during the context prefill.
        self._explicit_budgets = (
            None if budgets is None else torch.as_tensor(budgets, dtype=torch.int64)
        )
        self._pool_budgets: torch.Tensor | None = None
        self.n_sink = int(n_sink)
        self.n_local = int(n_local)
        self.batch_size = int(batch_size)
        self.pool: EvictPagedPool | None = None
        # CMP slots, maintained incrementally. Funded out of the read budget rather than added to
        # it: the exact branch reads `topk - cmp_slots` keys, so a run with slots and one without
        # read the same number of entries and the A/B measures the idea, not the budget.
        self.cmp_slots = int(cmp_slots)
        self.cmp_reseed_below = int(cmp_reseed_below)
        self.cmp: "StreamingCMP | None" = None

        self._hidden: dict[int, torch.Tensor] = {}
        self._kwargs: dict[int, dict] = {}
        #: Per-layer router state for the tokens arriving in THIS step, drained by
        #: :meth:`finish_step` into one batched ingest. See the module docstring on why this is
        #: deferred rather than applied per layer.
        self._pending: dict[int, tuple] = {}
        #: Set while a sequence is being prefilled under the mask path; None during decode.
        self._prefill_seq: int | None = None
        #: Which sequences the current forward covers. Not None only during the question phase,
        #: where they are run one at a time because their lengths differ.
        self._active_seqs: torch.Tensor | None = None
        self._decay = False
        self._pos_slope = 0.0
        self._handles: list = []
        self._configs: list = []
        self._previous_impls: list = []
        self._registry_restore = None

    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        self._hidden[int(layer_idx)] = hidden
        self._kwargs[int(layer_idx)] = kwargs
        return None

    def _router_state(self, module, layer_idx: int, *, key_offset):
        """This step's ``(mag, log_beta)``, ``(B, n_kv_heads, Sq)`` fp32.

        Scored once at ``key_offset=0`` and re-based by :func:`shift_router_key`, so a batch whose
        sequences sit at different absolute positions still costs one scorer call.
        """
        indexer = self.press.get_indexer(module)
        hidden = self._hidden[layer_idx]
        cos, sin = self.press.get_rope_tables(indexer, self._kwargs.get(layer_idx, {}))
        k_idx = indexer.project_k(hidden, cos, sin, value_states=None, key_offset=0)
        mag, log_beta = split_router_key(k_idx, self._decay)
        mag = shift_router_key(
            mag, log_beta, offset=key_offset,
            pos_slope=self._pos_slope, decay_ref=self.pool.decay_ref,
        )
        return mag, log_beta

    def _attend(self, module, query, key, value, scaling):
        """Attention for one layer, in whichever phase the runner is in."""
        layer_idx = int(module.layer_idx)
        if self._prefill_seq is not None:
            # Phase 1: the ordinary mask path builds the dense cache; nothing is evicted yet.
            raise RuntimeError(
                "prefill must run under SparseAttentionContext, not this implementation"
            )

        bsz, _, q_len, _ = query.shape
        # `_active_seqs` is set while the question forwards run one sequence at a time; during
        # batched decode it is None and every sequence steps.
        seqs = self._active_seqs
        if seqs is None:
            seqs = torch.arange(bsz, device=self.pool.device)
        seen = self.pool.seen[seqs]
        # Position of each arriving token, per sequence. The pool's `seen` is the LOGICAL length,
        # which is what these positions must be based on -- the physical slot count stopped
        # growing at the budget.
        base = seen.view(-1, 1) + torch.arange(q_len, device=seen.device).view(1, -1)
        mag, log_beta = self._router_state(module, layer_idx, key_offset=base)
        self._pending[layer_idx] = (key, value, mag, log_beta)

        # The arriving tokens are attended as the SECOND branch, always -- including a single
        # decode token, which must see ITSELF.
        #
        # This is the causal diagonal, and dropping it is not a small error. The pool is updated by
        # `finish_step` *after* the forward, so at step t it holds [0, t-1]; passing new_key=None
        # here would have the query at t attend to its history but not to its own key, while the
        # masking arm (whose cache is updated inside the attention, before the softmax) sees
        # [0, t]. Measured cost of getting this wrong on RULER 4096: cwe 95.71 -> 24.29,
        # vt 100.00 -> 40.00, mean 79.39 -> 60.82, with the pure-needle tasks untouched -- the
        # generation degenerates into repetition ("1. band 2. band 3. band").
        extra = None
        if self.cmp is not None:
            cmp_rows = (
                layer_idx * self.batch_size * self.n_kv_heads
                + seqs.view(-1, 1) * self.n_kv_heads
                + torch.arange(self.n_kv_heads, device=self.pool.device).view(1, -1)
            ).reshape(-1)
            extra = self.cmp.read(
                cmp_rows, query, group=query.shape[1] // self.n_kv_heads,
                scaling=scaling if scaling is not None else query.shape[-1] ** -0.5,
            )
        return self.pool.attend(
            layer_idx, query, seqs=seqs, scaling=scaling,
            new_key=key, new_value=value, extra=extra,
        )

    def finish_step(self, seqs: torch.Tensor | None = None) -> None:
        """Absorb this step's tokens into the pool, for **every layer at once**.

        Call once after each forward. Splitting this per layer costs 36x -- see the module
        docstring; the pending state exists for no other reason.

        ``seqs`` names the sequences the forward covered, for the question phase where they are
        run one at a time.
        """
        if not self._pending:
            return
        layers = sorted(self._pending)
        keys = torch.stack([self._pending[i][0] for i in layers])      # (L, B, Hkv, Sq, D)
        values = torch.stack([self._pending[i][1] for i in layers])
        mags = torch.stack([self._pending[i][2] for i in layers])      # (L, B, Hkv, Sq)
        betas = (
            torch.stack([self._pending[i][3] for i in layers])
            if self._pending[layers[0]][3] is not None
            else None
        )
        self._pending.clear()

        q_len = keys.shape[3]
        if seqs is None:
            seqs = torch.arange(self.batch_size, device=self.pool.device)
        seqs = seqs.to(self.pool.device)
        seen = self.pool.seen[seqs].clone()
        # Tokens enter one position at a time: the eviction rule is defined per token, and a
        # multi-row forward (the question) must not let its later rows displace keys its earlier
        # rows could still see.
        for t in range(q_len):
            dropped = self.pool.ingest(
                layers,
                key=keys[:, :, :, t, :],
                value=values[:, :, :, t, :],
                mag=mags[:, :, :, t],
                log_beta=None if betas is None else betas[:, :, :, t],
                positions=seen + t,
                seqs=seqs,
            )
            if self.cmp is not None and dropped is not None:
                # Summarize what the cache just threw away. This is the whole streaming idea:
                # the key is gone from the cache after this call, so it has to be folded into a
                # centroid now or never.
                self.cmp.ingest(
                    dropped["rows"], dropped["key"], dropped["value"],
                    active=dropped["evicted"],
                )
        self.pool.seen[seqs] = seen + q_len

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _ensure_pool(self, ctx, layers) -> None:
        """Allocate the pool from the budgets a completed prefill resolved, once.

        ``budgets`` may be given explicitly at construction; otherwise it is read off the mask
        context, which is the only place ``topk_ratio`` and ``head_budget`` have been resolved.
        """
        resolved = torch.empty((len(layers), self.n_kv_heads), dtype=torch.int64)
        # `head_budget="static"` is read from its TABLE, not from what the prefill happened to
        # resolve. RE-APPLIED 2026-09-14 23:5x after a rewrite of this file dropped it; the bug it
        # fixes is silent, so it is worth restating why.
        #
        # `_attend` short-circuits to dense attention when `k_len < force_sink + force_local`, and
        # that branch never calls `_budget_for` -- so `_head_topk` stays EMPTY and the loop below
        # falls back to the uniform scalar while the config still says `head_budget: static`. On
        # math500/aime25 the whole problem lives in `question` and `context` is a single space, so
        # EVERY document takes that short-circuit. Measured before the original fix: the pool was
        # allocated [1024]*8 while the fitted table asked for
        # [2074, 400, 389, 821, 463, 945, 1610, 1490] at layer 0 -- i.e. the head-budget component
        # was absent from the run, and nothing reported it.
        #
        # Legitimate because a static table depends on nothing the prefill observes; the measured
        # modes (`mass`, `shuffle`) genuinely cannot be recovered this way and still come from the
        # context below. Verified after the original fix: pool == table exactly (ragged 389..2074,
        # total conserved 8192) and `head_budget=uniform` still allocates [1024]*8 unchanged.
        static_table = None
        if getattr(ctx, "head_budget", "uniform") == "static":
            static_table = getattr(ctx, "head_budget_table", None)
            if static_table is None:
                raise ValueError(
                    "head_budget='static' without a loaded head_budget_table; the pool has no "
                    "budget to allocate from."
                )
        for layer_idx in range(len(layers)):
            if static_table is not None:
                resolved[layer_idx] = static_table[layer_idx].to("cpu", torch.int64)
                continue
            per_head = ctx._head_topk.get(layer_idx)
            if per_head is None:
                resolved[layer_idx] = int(ctx.topk)
            else:
                resolved[layer_idx] = per_head.to("cpu", torch.int64)
        if self._explicit_budgets is not None:
            resolved = self._explicit_budgets.clone()
        # The CMP slots are paid for here, out of every head's own budget, so the physical cache
        # stays at `topk * n_kv_heads` whether or not slots are enabled. Applied BEFORE the
        # consistency check below, so both sides of that comparison are the same quantity --
        # subtracting afterwards would make every document after the first look like a mismatch.
        #
        # Done here rather than by lowering `topk` because a `head_budget=static` table's rows sum
        # to the topk it was fitted at, and the loader refuses any other value (correctly: the row
        # sums ARE the budget).
        if self.cmp_slots:
            resolved = resolved - self.cmp_slots

        if self.pool is not None:
            if not torch.equal(resolved, self._pool_budgets):
                raise ValueError(
                    "this document resolved a different per-head budget than the one the pool was "
                    f"allocated for (layer 0: {resolved[0].tolist()} against "
                    f"{self._pool_budgets[0].tolist()}). The paged block table is fixed at "
                    "allocation and cannot be re-laid-out, so one pool serves one budget. Use "
                    "head_budget=uniform/static (input-independent) for a batch, or batch_size=1."
                )
            return

        self._pool_budgets = resolved.clone()
        param = next(self.model.parameters())
        indexer = self.press.get_indexer(layers[0].self_attn)
        self.pool = EvictPagedPool(
            resolved,
            batch_size=self.batch_size,
            n_layers=len(layers),
            n_kv_heads=self.n_kv_heads,
            n_sink=self.n_sink,
            n_local=self.n_local,
            head_dim=self.head_dim,
            device=param.device,
            dtype=param.dtype,
            decay_ref=float(indexer.decay_ref) if self._decay else None,
        )
        logger.info("evict pool: %s", self.pool.summary())
        if self.cmp_slots:
            from kvpress.presses.gqa_indexer.streaming_cmp import StreamingCMP

            self.cmp = StreamingCMP(
                self.pool.rows,
                self.cmp_slots,
                self.head_dim,
                device=param.device,
                reseed_below=self.cmp_reseed_below,
            )
            logger.info("streaming CMP: %s", self.cmp.summary())

    @torch.no_grad()
    def prefill_and_commit(self, input_ids: torch.Tensor, seq: int, sparse_kwargs: dict) -> None:
        """
        Prefill one sequence under the mask path, then compress its cache into the pool.

        The dense cache is built and released inside this call, so peak memory is one context --
        not ``batch_size`` of them. That is what lets a batch decode against contexts whose dense
        form would not have fit simultaneously.
        """
        from transformers import DynamicCache

        from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext

        if not self._handles:
            raise RuntimeError("enter the context before prefilling")
        # `memory` stays refused: it reads the retained branch's lse, which the paged decode
        # kernel does not return, and it summarizes keys this path has deleted. `cmp_slots` IS
        # supported here, through the streaming centroid update -- see :mod:`~.streaming_cmp`.
        if sparse_kwargs.get("memory"):
            raise ValueError(
                "memory=True reads the retained branch's lse, which the paged decode kernel does "
                "not return, and summarizes keys hard eviction has deleted. Use the masking path "
                "(SparseAttentionContext), which keeps the whole cache."
            )
        # The mask path would build FROZEN slots during this prefill and then never refresh them.
        # Here they are maintained incrementally, so its batch build is bypassed and only its
        # k-means result is taken as a seed (when the context was long enough to evict anything).
        cmp_slots = int(sparse_kwargs.get("cmp_slots") or 0)
        if cmp_slots and cmp_slots != self.cmp_slots:
            raise ValueError(
                f"cmp_slots={cmp_slots} in the prefill kwargs but the context was built with "
                f"cmp_slots={self.cmp_slots}. The slot state is allocated with the pool, so the "
                "two must agree -- pass it to EvictInferenceContext."
            )
        sparse_kwargs = {k: v for k, v in sparse_kwargs.items() if k != "cmp_slots"}
        if self.cmp_slots:
            # Fund the slots OUT OF the budget, so a run with slots reads the same number of
            # entries as one without and the A/B measures the compensation rather than a larger
            # cache.
            #
            # Deliberately NOT by lowering `topk`: under `head_budget=static` the table's rows sum
            # to `topk * n_kv_heads` and the loader refuses a mismatch (correctly -- the row sums
            # ARE the budget). So the reduction is applied to the resolved per-head budgets
            # instead, inside `_ensure_pool`, where both the uniform and the table-driven cases
            # end up. `topk` itself is left alone so the prefill's mask and the table agree.
            floor = self.n_sink + self.n_local + 1
            if int(sparse_kwargs["topk"]) - self.cmp_slots < floor:
                raise ValueError(
                    f"cmp_slots={self.cmp_slots} would leave "
                    f"{int(sparse_kwargs['topk']) - self.cmp_slots} of topk="
                    f"{sparse_kwargs['topk']} after the pins (n_sink={self.n_sink} + "
                    f"n_local={self.n_local}), i.e. nothing for the top-k to rank. Lower "
                    "cmp_slots or raise topk."
                )
        layers = get_language_model(self.model).layers
        cache = DynamicCache()
        with SparseAttentionContext(self.model, self.press, **sparse_kwargs) as ctx:
            # Per-document budget, when --topk_ratio is set. Must precede the prefill, exactly as
            # in SparseGenerationPipeline: the head-budget split is derived from topk and
            # set_context_length clears it when it changes.
            setter = getattr(ctx, "set_context_length", None)
            if setter is not None:
                setter(int(input_ids.shape[1]))
            self.model.model(input_ids=input_ids, past_key_values=cache)
            k_len = input_ids.shape[1]
            # Allocate the pool from the budgets THIS prefill actually resolved. Deferred to here
            # rather than done in __enter__ because two features settle the budget only during the
            # prefill, and both would otherwise be silently ignored:
            #
            # * `topk_ratio` resolves topk from the document's own length;
            # * `head_budget` in {mass, static, shuffle} fills `_head_topk` per layer.
            #
            # Once allocated the pool is FROZEN, so every later sequence must resolve the same
            # budget -- checked below, because a ragged block table cannot be re-laid-out and a
            # quiet mismatch would evict against the wrong capacity.
            self._ensure_pool(ctx, layers)
            for layer_idx, layer in enumerate(layers):
                k = cache.layers[layer_idx].keys[0]     # (Hkv, Sk, D)
                v = cache.layers[layer_idx].values[0]
                # The router key-cache SparseAttentionContext already built for this prefill, so
                # the committed ranking is the one the prefill actually attended under.
                k_idx = ctx._k_idx[layer_idx]           # (B, Sk, Di)
                mag, log_beta = split_router_key(k_idx, self._decay)
                self.pool.commit(
                    layer_idx, seq,
                    key=k, value=v, mag=mag[0],
                    log_beta=None if log_beta is None else log_beta[0],
                    k_len=k_len,
                )
                if self.cmp is not None:
                    self._seed_cmp(layer_idx, seq, k, v, mag[0], log_beta, k_len, ctx)
        del cache
        self.pool.seen[seq] = k_len

    @torch.no_grad()
    def _seed_cmp(self, layer_idx, seq, key, value, mag, log_beta, k_len, ctx) -> None:
        """Seed this (layer, sequence)'s slots from the keys the CONTEXT prefill evicted.

        Real k-means where there is a batch to run it on, streaming from then on -- the two
        compose. On a CoT benchmark (math500's context is a single space) the evicted set here is
        empty and this is a no-op, which is precisely the case frozen CMP could never recover: it
        would record "built" and produce nothing for the whole generation.
        """
        from kvpress.presses.gqa_indexer.cmp_slots import cluster_evicted, evicted_from_deadline
        from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines

        rows = self.pool.rows_for(layer_idx, seq)
        budget = self.pool.row_budget[rows]
        scores = mag if log_beta is None else mag + log_beta[0] * (float(k_len - 1) / self.pool.decay_ref)
        dl = deadlines(scores, budget, force_sink=self.n_sink, force_local=self.n_local)
        evicted = evicted_from_deadline(dl, max(k_len - 1 - self.n_local, 0))
        if not bool(evicted.any()):
            return
        k_cmp, v_cmp, b_cmp = cluster_evicted(
            key.float(), value.float(), evicted, self.cmp_slots
        )
        # cluster_evicted may return fewer than R slots when the evicted set is smaller than R.
        got = k_cmp.shape[1]
        if got < self.cmp_slots:
            pad = self.cmp_slots - got
            k_cmp = torch.cat([k_cmp, torch.zeros_like(k_cmp[:, :pad])], 1)
            v_cmp = torch.cat([v_cmp, torch.zeros_like(v_cmp[:, :pad])], 1)
            b_cmp = torch.cat(
                [b_cmp, torch.full_like(b_cmp[:, :pad], -float("inf"))], 1
            )
        self.cmp.load_batch(rows, k_cmp, v_cmp, b_cmp)

    def replicate(self, src: int, dst: int) -> None:
        """Copy a committed sequence onto another row-set, pool **and** CMP slots.

        The slots have to travel with the pool: they summarize the keys that row's own budget
        dropped, so a replica reading another sequence's slots would be compensating for an
        eviction that never happened to it.
        """
        self.pool.replicate_seq(src, dst)
        if self.cmp is not None:
            for layer_idx in range(self.pool.n_layers):
                s = self.pool.rows_for(layer_idx, src)
                d = self.pool.rows_for(layer_idx, dst)
                self.cmp.k_cmp[d] = self.cmp.k_cmp[s]
                self.cmp.v_cmp[d] = self.cmp.v_cmp[s]
                self.cmp.pop[d] = self.cmp.pop[s]

    def new_cache(self) -> EvictCache:
        return EvictCache(self.pool)

    @torch.no_grad()
    def generate(
        self,
        question_ids: list[torch.Tensor],
        *,
        max_new_tokens: int,
        eos_token_ids: list[int] | None = None,
        sampling: dict | None = None,
    ) -> list[list[int]]:
        """
        Greedy-decode every committed sequence **as one batch**.

        Each sequence asks its own question, so the question forwards are run one at a time (they
        differ in length, and a padded batch would put padding inside the local window). The
        answer tokens are then generated together, which is where the batching pays: decode is
        memory-bound, and every sequence reads only its own ``budget`` slots.

        Sequences that emit EOS stop contributing tokens but keep stepping, so the batch stays
        rectangular; their extra tokens are discarded. Simpler than compacting the batch, and the
        wasted work is bounded by the spread in answer lengths.

        Parameters
        ----------
        question_ids : list of torch.Tensor
            One ``(1, Sq)`` question per committed sequence.
        max_new_tokens : int
            Cap on generated tokens.
        eos_token_ids : list of int, optional
            Stop tokens; defaults to the model's generation config.
        sampling : dict, optional
            ``{"temperature", "top_p", "top_k"}``. ``None`` is greedy, bitwise -- which is what
            every result before the reasoning benchmarks used. Set for math500/aime25, where a
            greedy trace is not comparable to the literature (Qwen3 ships temperature 0.6 /
            top_p 0.95). Matches the pipeline's ``self.sampling`` contract.

        Returns
        -------
        list of list of int
            The generated token ids per sequence, EOS-terminated and trimmed.
        """
        batch = len(question_ids)
        if batch != self.batch_size:
            raise ValueError(
                f"{batch} questions for a pool built with batch_size={self.batch_size}"
            )
        if eos_token_ids is None:
            eos = self.model.generation_config.eos_token_id
            eos_token_ids = eos if isinstance(eos, list) else [eos]
        device = self.pool.device
        self.activate()
        cache = self.new_cache()

        # --- the question forwards, one sequence at a time --------------------------------
        # Not a batch: the questions differ in length, and left-padding them would put padding
        # tokens inside the local window, which is pinned and therefore unevictable.
        first = []
        for seq, q_ids in enumerate(question_ids):
            q_ids = q_ids.to(device)
            start = int(self.pool.seen[seq])
            pos = torch.arange(start, start + q_ids.shape[1], device=device).unsqueeze(0)
            only = torch.tensor([seq], device=device)
            self._active_seqs = only
            try:
                logits = self.model(
                    input_ids=q_ids, past_key_values=cache, position_ids=pos,
                    num_logits_to_keep=1,
                ).logits
                self.finish_step(seqs=only)
            finally:
                self._active_seqs = None
            first.append(int(_pick(logits[:, -1], sampling)[0]))

        # --- the answer, batched ----------------------------------------------------------
        out: list[list[int]] = [[t] for t in first]
        done = [t in eos_token_ids for t in first]
        nxt = torch.tensor(first, device=device).view(batch, 1)
        for _ in range(max_new_tokens - 1):
            if all(done):
                break
            pos = self.pool.seen[:batch].view(batch, 1)
            logits = self.model(
                input_ids=nxt, past_key_values=cache, position_ids=pos
            ).logits
            self.finish_step()
            nxt = _pick(logits[:, -1], sampling).view(batch, 1)
            for b in range(batch):
                if done[b]:
                    continue
                token = int(nxt[b])
                out[b].append(token)
                if token in eos_token_ids:
                    done[b] = True
        return out

    # ------------------------------------------------------------------
    def __enter__(self) -> "EvictInferenceContext":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(self.model)
        layers = get_language_model(self.model).layers
        indexer = self.press.get_indexer(layers[0].self_attn)
        if not getattr(indexer, "is_query_independent", False):
            raise ValueError(
                f"{type(indexer).__name__} scores each (query, key) pair, so a key's rank is not "
                "frozen and 'it left the top-k' does not mean 'no future query wants it'. Hard "
                "eviction would delete keys a later query needs. Use --scorer scalar/kvzip, or "
                "the masking path (SparseAttentionContext), which keeps everything."
            )
        self._decay = bool(getattr(indexer, "decay", False))
        self._pos_slope = float(getattr(indexer, "pos_slope", 0.0))

        cfg = self.model.config
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        # The pool is allocated by the FIRST prefill (see _ensure_pool), not here: `topk_ratio`
        # and `head_budget` only settle the per-head budget while the context is being prefilled.

        def evict_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self._attend(module, query, key, value, scaling), None

        self._configs = [self.model.config]
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None:
            self._configs.append(text_config)
        self._previous_impls = [c._attn_implementation for c in self._configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        self._registry_restore = (IMPL_NAME in global_mapping, global_mapping.get(IMPL_NAME))
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, evict_attention_impl)

        self._handles = [
            layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
            for layer in layers
        ]
        return self

    def activate(self) -> None:
        """Point the model's attention at the eviction path (after prefills are committed)."""
        for cfg in self._configs:
            cfg._attn_implementation = IMPL_NAME

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
        self._hidden.clear()
        self._kwargs.clear()
        self._pending.clear()
        self.pool = None
