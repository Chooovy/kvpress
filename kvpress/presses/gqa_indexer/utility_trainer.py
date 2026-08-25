# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training wiring for utility self-distillation (``differentiable_topk_for_sparse_attention.md`` §31).

The fifth objective in this package, and the only one whose **forward pass is unmodified**. Attention
runs exactly as the frozen backbone would run it -- no gate, no chunk routing, no candidate pool, no
straight-through estimator. The router is supervised by a target read *out of the backbone's own
backward pass*:

    u_j = -dL/db_j = -alpha_j * <dL/do, v_j - o>

See :mod:`~.utility_loss` for what ``u`` is, why it is a far better teacher than the attention
weights (Spearman +0.991 against +0.037 versus the true drop effect), and for the measured ceiling
that this arm is expected to run into.

The consequence of an unmodified forward, stated plainly
--------------------------------------------------------
The router is not on the forward path, so ``dL_LM/dtheta_router`` is **None** -- not small, absent.
``loss = loss_rank`` is the entire objective and the LM loss is never returned to the driver as
something to descend. This is a **distillation** arm with a much better teacher, not an end-to-end
one; :class:`~.e2e_trainer.E2EIndexerTrainer` and :class:`~.hsa_trainer.HSAIndexerTrainer` are the
end-to-end arms. Calling it "e2e" because a gradient of the LM loss appears in the target would be a
category error: the target is a *number* here, detached, and the router's gradient is entirely the
ranking loss's.

How it is made affordable
-------------------------
Two things, and neither is a memory concession that changes the objective:

1. **The ranking loss is built and backwarded inside the tensor hook that delivers ``dL/do``.** When
   that hook fires, mid-backward, this layer's ``q``/``k``/``v``/``hidden_states`` are still alive as
   the LM backward's own saved tensors, so reading them costs nothing and *nothing has to be stashed
   across layers*. The alternative -- retaining ``dL/do`` for all 36 layers and looping afterwards --
   is 2.4 GiB at 8K on Qwen3-8B, on a backbone that already sits at ~92 of 95 GiB at 16K. A reentrant
   ``backward()`` inside a hook is legal and was verified to fire for every layer before this module
   was written.

2. **``alpha`` is recomputed for sampled query rows only, never materialized.** ``(B, H, Sq, Sk)``
   fp32 is 8 GiB per layer at 16K; :attr:`n_rows` rows of it are a few MiB. The forward keeps using
   the backbone's own fused SDPA, so this arm's forward is not merely equivalent to dense attention,
   it *is* dense attention -- ``test_forward_is_bit_identical_to_the_unhooked_model`` pins that.

Row subsampling is not a compromise here the way a candidate pool was for the exact-K arm: every
*key* of a sampled row gets a utility, which is the property §31 exists for. Only the set of queries
supervising the router is thinned, and the router's parameters are shared across all of them.

The one wiring subtlety
-----------------------
A frozen backbone means nothing in the graph requires grad, so ``o`` has no ``grad_fn``, no hook
fires, and the objective silently trains on nothing. :meth:`UtilityIndexerTrainer.hooks` therefore
makes the embedding output a grad leaf. This cost two debugging rounds in the probe scripts that
preceded this module, which is why :attr:`layers_supervised` is checked and raises rather than
warning.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import build_indexer_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.utility_loss import (
    INVALID_UTILITY,
    lm_gradient_utility,
    pairwise_rank_loss,
    sample_boundary_pairs,
    score_utility_correlation,
    utility_recall_at_k,
)

logger = logging.getLogger(__name__)


@dataclass
class UtilityIndexerTrainer:
    """
    Train the indexer to rank keys by their LM-gradient utility.

    Register with :meth:`hooks`, run the model, call ``.backward()`` on its LM loss. The ranking
    loss is produced and backwarded *during* that call, so there is no second loss for the driver to
    add -- read :meth:`mean_rank_loss` afterwards for logging.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers and :meth:`~.press.GQAIndexerPress.get_rope_tables`, so the
        supervised score is computed by the same function the press scores with at inference.
    n_rows : int
        Query rows sampled per layer per step. The utility is exact for every *key* of a sampled row;
        this only thins which queries supervise the router. 16 rows x 36 layers x 8 KV heads is ~4600
        supervised rankings per step, each over the row's whole visible history.
    n_pairs : int
        Boundary pairs drawn per row -- see :func:`~.utility_loss.sample_boundary_pairs`.
    band : int
        Rank half-width of the sampling window around the budget. Small on purpose.
    budget : int, optional
        Rank the top-k boundary sits at. ``None`` uses each row's midpoint, appropriate when the eval
        budget is a compression *ratio*.
    loss_scale : float
        Multiplier on the ranking loss before its reentrant ``backward()``. **The driver must set
        this to ``1 / accum_steps``.** The reentrant backward happens inside the LM loss's backward,
        so the driver's own ``(loss / accum_steps).backward()`` scaling cannot reach it, and a run
        with ``--accum-steps 8`` would otherwise apply 8x the intended gradient. There is no way for
        this class to infer the value, and getting it wrong changes only the effective learning rate
        -- which no diagnostic would reveal.
    """

    press: GQAIndexerPress
    n_rows: int = 16
    n_pairs: int = 64
    band: int = 32
    budget: int | None = None
    loss_scale: float = 1.0

    #: Multiplier on the router's raw ``qi . ki`` dot before the ranking loss. ``None`` uses
    #: ``head_dim ** -0.5``.
    #:
    #: The ranking loss reads only the *order* of the scores, so this cannot change what the router is
    #: asked to learn. It matters for ``softplus``: ``IndexerNorm`` leaves the raw dot at std
    #: ``~sqrt(head_dim)`` = 11.3, where ``softplus(s_lose - s_win)`` is effectively a hinge that is
    #: either saturated flat (no gradient on badly-ordered pairs) or linear, losing the soft margin
    #: near the boundary that the band sampler exists to exploit. Reuses the same
    #: ``GATE_SCALE_INIT`` the other arms scale by, so the arms cannot drift apart on what the
    #: score's natural magnitude is.
    score_scale: float | None = None

    #: Rescale each row's pair weights to mean 1 before the loss.
    #:
    #: **Leave True. Without it this objective does not train**, and the loss curve does not say so.
    #: ``u`` is proportional to ``alpha_j`` (``~1/Sq``) times ``dL/do`` (which carries the LM loss's
    #: ``1/(B*Sq)`` mean), so ``|u| ~ 1/Sq**2``: measured mean ``|u|`` falls 4x per doubling of the
    #: sequence, reaching **3.5e-10 at 8K on Qwen3-8B** with a router gradient norm of ~3e-8. At that
    #: magnitude AdamW's ``eps = 1e-8`` dominates its denominator and the optimizer stops being
    #: scale-invariant -- realized step size against the ideal is 42.9% at gradient 1e-8, 8.8% at 1e-9,
    #: 1.0% at 1e-10 -- while ``grad_clip``, an absolute threshold, never fires. Both effects scale with
    #: ``Sq``, so the *effective learning rate becomes a function of the curriculum stage*: a 16x change
    #: between 8K and 32K, silently.
    #:
    #: Normalizing is correct rather than a workaround: only *relative* weights within a row carry
    #: information, since the loss asks "which of these two keys matters more" and a per-row constant
    #: does not change that. Same argument that makes this a ranking loss instead of a regression,
    #: applied to the weights. ``False`` reproduces the un-normalized behaviour for the ablation.
    normalize_weights: bool = True

    #: Cut the score's gradient path back into ``hidden_states``.
    #:
    #: **Leave True**, though for a different and simpler reason than the other arms. There the score
    #: feeds the forward, so attaching it created a per-layer feedback loop that drove ``grad_norm``
    #: to ``nan`` at 36 layers. Here the score feeds only the loss, so there is no loop -- but letting
    #: the ranking loss deposit gradient into the residual stream would make the *backbone's* backward
    #: depend on the router's, mid-backward, while that same backward is still running. The result
    #: would depend on hook ordering. Detaching keeps the LM backward exactly the frozen model's.
    detach_score_input: bool = True

    #: Also compute :attr:`utility_recall`. Cheap, but it is a second top-k per sampled row.
    measure_recall: bool = True

    #: Layer index -> the ranking loss on the last step. The objective's own value.
    rank_losses: dict[int, float] = field(default_factory=dict)
    #: Layer index -> ``spearman(router score, u)`` over visible keys. **The diagnostic to watch**:
    #: the loss value moves with ``||dL/do||`` and so can fall on an easier batch alone, while this is
    #: measured against a fixed quantity. Read it against the +0.03 to +0.32 ceiling the
    #: representability probes measured -- a plateau there is the hypothesis class, not the optimizer.
    score_corr: dict[int, float] = field(default_factory=dict)
    #: Layer index -> fraction of the teacher's top-k that the router's top-k keeps.
    utility_recall: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean ``|u|`` over sampled rows, as a scale readout. If this collapses the
    #: teacher has stopped distinguishing keys and the loss weights go to 0, which would look like
    #: convergence.
    utility_scale: dict[int, float] = field(default_factory=dict)
    #: Layers that produced a ranking loss, as a wiring check. Zero means the ``dL/do`` hooks never
    #: fired -- the silent failure this arm is most exposed to.
    layers_supervised: int = field(default=0, init=False)

    _hidden_states: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _kwargs: dict[int, dict] = field(default_factory=dict, init=False, repr=False)
    _generator: torch.Generator | None = field(default=None, init=False, repr=False)
    _seed: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.n_rows < 1:
            raise ValueError(f"n_rows must be >= 1, got {self.n_rows}")
        if self.n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {self.n_pairs}")
        if self.band < 1:
            raise ValueError(f"band must be >= 1, got {self.band}")

    def resolved_score_scale(self, indexer) -> float:
        """:attr:`score_scale`, defaulting to ``head_dim ** -0.5``. See that attribute."""
        if self.score_scale is not None:
            return float(self.score_scale)
        head_dim = getattr(indexer, "head_dim", None)
        if not head_dim:
            return 1.0
        return float(type(indexer).GATE_SCALE_INIT(head_dim))

    def reset(self) -> None:
        self.rank_losses = {}
        self.score_corr = {}
        self.utility_recall = {}
        self.utility_scale = {}
        self.layers_supervised = 0
        self._hidden_states.clear()
        self._kwargs.clear()

    # ------------------------------------------------------------------
    # Parameters -- identical to the other trainers, so the arms are comparable
    # ------------------------------------------------------------------
    def indexer_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """
        Every indexer parameter, in layer order -- what the optimizer should be given.

        Includes ``gate_scale``, which this objective **never trains**: the ranking loss reads only
        score *differences*, so a positive scalar multiplying every score is unidentifiable -- it
        cannot change a ranking. It stays in the list so the optimizer state and the checkpoint layout
        match the other arms byte for byte (``--init-from`` works in both directions, and
        ``evaluate_sparse.py``'s ``has_gate`` detection behaves the same for all of them); AdamW simply
        never sees a gradient for it. This arm scales instead by the fixed :attr:`score_scale`.
        """
        params = []
        for layer in get_language_model(model).layers:
            indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
            if indexer is not None:
                params.extend(indexer.parameters())
        return params

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Put every non-indexer parameter at ``requires_grad=False``.

        Identifies indexers by **module identity**, not by a parameter-name substring: a name filter
        would also catch a backbone parameter containing the attribute name and silently train it.

        Unlike the other arms this does *not* need the gradient to flow through the backbone to reach
        the router -- the router is not in the forward. The backbone's backward is still run in full,
        because ``dL/do`` at every layer is the target and there is no way to obtain it without one.
        """
        indexer_params = {id(p) for p in self.indexer_parameters(model)}
        if not indexer_params:
            raise RuntimeError(
                f"no {self.press.scorer_attr!r} modules found on the model; call "
                "press.post_init_from_model(model) first"
            )
        for param in model.parameters():
            param.requires_grad = id(param) in indexer_params

    # ------------------------------------------------------------------
    # The objective
    # ------------------------------------------------------------------
    def sample_rows(self, q_len: int, k_len: int, device) -> torch.Tensor:
        """
        Query rows to supervise, drawn from the second half of the sequence.

        Early rows see a handful of keys, so a ranking over them is nearly vacuous and a rank
        correlation on 2 visible keys is +-1 by construction -- it would inflate the diagnostic while
        teaching the router nothing. Deterministic (``linspace``) rather than random: the metric is
        compared across steps, and resampling rows every step would add variance to the one number
        this arm is read by.
        """
        n = min(self.n_rows, max(1, q_len // 2))
        return torch.linspace(q_len // 2, q_len - 1, n, device=device).long().unique()

    def row_utility(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        grad_out: torch.Tensor,
        rows: torch.Tensor,
        scaling: float | None,
        attention_mask,
    ) -> torch.Tensor:
        """
        ``u`` for the sampled rows, pooled to KV heads: ``(B, Hkv, r, Sk)``.

        ``alpha`` is **recomputed** for these rows rather than captured from the forward, which is
        what keeps the forward on the backbone's fused kernel and the memory at a few MiB instead of
        the 8 GiB per layer a full ``(B, H, Sq, Sk)`` fp32 ``alpha`` costs at 16K. The recomputation
        is exact -- same ``q``, ``k``, ``scaling`` and causal geometry the kernel used.

        Pooled to KV heads by **mean over the query group**, because the router emits one score per
        KV head and one cache is evicted per KV head: the quantity to rank is the utility of dropping
        key ``j`` *for that whole group*, and each group member's utility is a first-order term in the
        same loss, so they add. The mean is that sum up to the constant ``group_size``, and only the
        ranking is used.

        Everything is fp32. In bf16 ``<g, v_j> - <g, o>`` subtracts two numbers of similar magnitude
        and its **sign** is the entire signal.
        """
        b, n_heads, q_len, head_dim = query.shape
        n_kv, k_len = key.shape[1], key.shape[2]
        group = n_heads // n_kv
        scale = head_dim**-0.5 if scaling is None else float(scaling)
        query_offset = k_len - q_len

        q_rows = query[:, :, rows].float()
        k_full = key.float()
        v_full = value.float()
        if group != 1:
            k_full = k_full.repeat_interleave(group, dim=1)
            v_full = v_full.repeat_interleave(group, dim=1)

        logits = torch.einsum("bhrd,bhsd->bhrs", q_rows, k_full) * scale
        q_pos = rows + query_offset
        visible = torch.arange(k_len, device=query.device).view(1, k_len) <= q_pos.view(-1, 1)
        if attention_mask is not None:
            # Padding as well as causality. build_indexer_mask returns an ADDITIVE mask; only its
            # zero entries are valid pairs.
            pad = build_indexer_mask(
                q_len, k_len, query.device, attention_mask=attention_mask
            )
            visible = visible & (pad[0, 0][rows] == 0)
        visible = visible.view(1, 1, len(rows), k_len)

        alpha = torch.softmax(logits.masked_fill(~visible, -float("inf")), -1)
        out = torch.einsum("bhrs,bhsd->bhrd", alpha, v_full)
        g = grad_out[:, :, rows].float()

        u = lm_gradient_utility(alpha, v_full, out, g)
        if group != 1:
            u = u.view(b, n_kv, group, len(rows), k_len).mean(2)
        # INVALID_UTILITY rather than 0: an invisible key must sort below every real utility, and real
        # utilities are signed, so 0 would place invisible keys in the middle of the ranking.
        return u.masked_fill(~visible, INVALID_UTILITY)

    def row_scores(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict, rows: torch.Tensor
    ) -> torch.Tensor:
        """
        Router scores for the sampled rows against every key: ``(B, Hkv, r, Sk)``, with gradients.

        Built from the same ``hidden_states`` and RoPE tables the layer itself used, via the press's
        own :meth:`~.press.GQAIndexerPress.get_rope_tables` -- the invariant that keeps the trained
        score and the scored-at-inference score the same function.

        No mask is applied: the loss only reads the scores at sampled pair indices, and those come
        from :func:`~.utility_loss.sample_boundary_pairs`, which draws only from keys the *utility*
        marked visible. Masking here would additionally put ``-inf`` into a tensor the loss
        differentiates, for no benefit.
        """
        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        if self.detach_score_input:
            hidden_states = hidden_states.detach()

        q_idx = indexer.project_q(
            hidden_states[:, rows],
            None if cos is None else cos[:, rows],
            None if sin is None else sin[:, rows],
        )
        k_idx = indexer.project_k(hidden_states, cos, sin)
        scale = self.resolved_score_scale(indexer)
        return torch.einsum("bhrd,bkd->bhrk", q_idx.float(), k_idx.float()) * scale

    def _pair_generator(self, device) -> torch.Generator | None:
        """
        The pair sampler's RNG, built lazily on the device the scores live on.

        ``torch.rand(generator=...)`` requires the generator's device to match the tensor's, and this
        class is constructed before any device is known -- so it cannot be made in ``__post_init__``.
        Cached per device rather than per call: a fresh generator each time would restart the stream
        from the seed on every layer, making all 36 layers draw *identical* pair ranks.
        """
        if self._seed is None:
            return None
        if self._generator is None or self._generator.device != torch.device(device):
            self._generator = torch.Generator(device=device).manual_seed(self._seed)
        return self._generator

    def supervise(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        grad_out: torch.Tensor,
        scaling: float | None,
    ) -> None:
        """
        Build the ranking loss for one layer and backward it, from inside that layer's ``dL/do`` hook.

        Called mid-backward. Every tensor it reads is already alive as the LM backward's saved state,
        so the peak memory this adds is the ``(B, Hkv, r, Sk)`` score and utility -- a few MiB.

        The reentrant ``backward()`` accumulates into the indexer parameters alongside whatever the
        driver's own ``backward()`` is depositing elsewhere, which is why :attr:`loss_scale` has to
        carry the accumulation factor: this call is *inside* the driver's scaled backward and cannot
        see its divisor.
        """
        layer_idx = int(module.layer_idx)
        hidden_states = self._hidden_states.get(layer_idx)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached the utility hook without its hidden_states being "
                "captured. The pre-hook that records them must be registered on the same modules the "
                "attention hook is -- use UtilityIndexerTrainer.hooks()."
            )
        kwargs = self._kwargs.get(layer_idx, {})
        rows = self.sample_rows(query.shape[2], key.shape[2], query.device)

        with torch.no_grad():
            utility = self.row_utility(
                query, key, value, grad_out, rows, scaling, kwargs.get("attention_mask")
            )

        # enable_grad because we are inside a backward, where grad mode is off by default. Only the
        # score's graph is built; `utility` was produced under no_grad and is a plain number here.
        with torch.enable_grad():
            scores = self.row_scores(module, hidden_states, kwargs, rows)
            idx_win, idx_lose = sample_boundary_pairs(
                scores.detach(),
                utility,
                n_pairs=self.n_pairs,
                band=self.band,
                budget=self.budget,
                generator=self._pair_generator(scores.device),
            )
            loss = pairwise_rank_loss(
                scores, utility, idx_win, idx_lose, normalize=self.normalize_weights
            )
            if self.loss_scale != 1.0:
                loss = loss * self.loss_scale
            loss.backward()

        self.rank_losses[layer_idx] = float(loss.detach()) / max(self.loss_scale, 1e-12)
        self.score_corr[layer_idx] = score_utility_correlation(scores.detach(), utility)
        self.utility_scale[layer_idx] = float(
            utility.masked_fill(utility <= INVALID_UTILITY / 2, 0.0).abs().mean()
        )
        if self.measure_recall:
            budget = self.budget or max(1, key.shape[2] // 4)
            self.utility_recall[layer_idx] = utility_recall_at_k(
                scores.detach(), utility, budget
            )
        self.layers_supervised += 1

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """
        Stash this layer's ``hidden_states`` and kwargs -- the attention interface never sees them.

        Unlike the other trainers these are **not popped** on use: the consumer runs during the
        *backward*, so the entry has to survive the whole forward. That costs no memory beyond a
        Python reference, because the layer's own ``q_proj``/``k_proj`` backward already saves the
        same tensor; the dict is cleared on :meth:`hooks` exit.
        """
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        self._hidden_states[int(layer_idx)] = hidden_states
        self._kwargs[int(layer_idx)] = kwargs
        return None

    def attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float | None,
        dropout: float,
    ) -> torch.Tensor:
        """
        Plain SDPA, plus a hook on the output that will build the ranking loss during the backward.

        The forward is **the backbone's own attention** -- this is the arm's defining property, so it
        delegates to ``scaled_dot_product_attention`` rather than to anything in this package. The
        only addition is ``register_hook``, which does not alter the value.
        """
        is_causal = attention_mask is None
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None if attention_mask is None else attention_mask[..., : key.shape[2]],
            is_causal=is_causal,
            scale=scaling,
            dropout_p=dropout,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        if out.requires_grad:
            # dL/do arrives here mid-backward, with q/k/v still alive as saved tensors.
            out.register_hook(
                lambda g: self.supervise(module, query, key, value, g, scaling)
            )
        return out.transpose(1, 2).contiguous()

    @contextmanager
    def hooks(self, model: nn.Module, seed: int | None = None):
        """
        Install the capture pre-hooks, the attention swap, and the embedding grad leaf.

        The **grad leaf** is the subtlety. Every backbone parameter is frozen, so with an ordinary
        forward nothing in the graph requires grad: ``o`` has no ``grad_fn``, ``register_hook`` is
        never called, and the objective trains on nothing while the LM loss still looks healthy. A
        forward hook on the embedding calls ``requires_grad_(True)`` on its output, which is a leaf
        (its producer's weight is frozen), and that makes the whole activation graph exist. This is
        the same fix the probe scripts needed twice; :func:`utility_indexer_training_step` asserts
        :attr:`layers_supervised` afterwards so the failure cannot be silent.

        The registry cleanup goes through ``_global_mapping`` directly for the reason
        :func:`~.teacher_lse.capture_teacher_lse` documents: ``register()`` writes there while
        ``pop()`` only touches the instance mapping, so the naive removal leaks the entry forever.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        self.freeze_backbone(model)
        if seed is not None:
            self._seed = seed

        impl_name = "kvpress_gqa_indexer_utility"

        def utility_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self.attention_forward(
                module, query, key, value, attention_mask, scaling, dropout
            ), None

        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        previous_impls = [cfg._attn_implementation for cfg in configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        had_previous = impl_name in global_mapping
        previous_fn = global_mapping.get(impl_name)
        ALL_ATTENTION_FUNCTIONS.register(impl_name, utility_attention_impl)

        language_model = get_language_model(model)
        handles = []
        try:
            for layer in language_model.layers:
                handles.append(
                    layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
                )
            handles.append(
                language_model.embed_tokens.register_forward_hook(_require_grad_hook)
            )
            for cfg in configs:
                cfg._attn_implementation = impl_name
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for cfg, previous in zip(configs, previous_impls):
                cfg._attn_implementation = previous
            if had_previous:
                global_mapping[impl_name] = previous_fn
            else:
                global_mapping.pop(impl_name, None)
            self._hidden_states.clear()
            self._kwargs.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def mean_rank_loss(self) -> float | None:
        """Mean ranking loss over the layers that ran. The objective's own value."""
        return _mean(self.rank_losses)

    def mean_score_corr(self) -> float | None:
        """Mean ``spearman(score, u)``. **The** number this arm is read by -- see :attr:`score_corr`."""
        return _mean(self.score_corr)

    def mean_utility_recall(self) -> float | None:
        return _mean(self.utility_recall)

    def mean_utility_scale(self) -> float | None:
        return _mean(self.utility_scale)


def _require_grad_hook(module, args, output):
    """Make the embedding output a grad leaf, so the frozen backbone still builds a graph."""
    if isinstance(output, torch.Tensor) and not output.requires_grad:
        output.requires_grad_(True)
    return output


def _mean(values: dict[int, float]) -> float | None:
    finite = [v for v in values.values() if v is not None and v == v]  # v == v drops NaN
    if not finite:
        return None
    return sum(finite) / len(finite)


def utility_indexer_training_step(
    model: nn.Module,
    trainer: UtilityIndexerTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    skip_logits: bool | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    One forward and one backward. Returns the LM loss **already backwarded**, for logging only.

    This signature deliberately differs from the other arms' training steps, and the difference is
    the objective's: they return a loss for the driver to call ``.backward()`` on, because the router
    is in their forward. Here the router is not, so ``dL_LM/dtheta_router`` is ``None`` and the
    router's entire gradient is deposited by the reentrant backwards that happen *during* the LM
    backward this function runs. A driver that called ``.backward()`` on the return value would run
    the whole backbone backward a second time -- and every ranking loss with it, doubling the
    router's gradient.

    Set ``trainer.loss_scale = 1 / accum_steps`` before calling under gradient accumulation.

    ``skip_logits`` must **not** be passed as True here even under Liger, and the reason is specific
    to this arm: the fused path needs no ``(L, vocab)`` logits, but the LM loss still has to be
    differentiable back to ``dL/do``, which it is either way. It is accepted and forwarded for
    interface parity with the other steps.
    """
    extra = {} if skip_logits is None else {"skip_logits": skip_logits}
    with trainer.hooks(model, seed=seed):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids if labels is None else labels,
            use_cache=False,
            **extra,
        )
        if not out.loss.requires_grad:
            # Checked before backward, because autograd's own message here is "element 0 of tensors
            # does not require grad", which says nothing about the cause or the fix. Every backbone
            # parameter is frozen, so the graph only exists because of the embedding grad leaf that
            # hooks() installs -- without it there is no dL/do to read and the objective has no target.
            raise RuntimeError(
                "the LM loss does not require grad, so there is no dL/do to build the utility target "
                "from. Every backbone parameter is frozen by design, so the activation graph exists "
                "only because of the embedding grad-leaf hook installed by "
                "UtilityIndexerTrainer.hooks(); this means that hook did not take effect."
            )
        # The backward is what produces the objective: each layer's dL/do hook builds and backwards
        # that layer's ranking loss. Run inside the context so the hooks are still installed.
        out.loss.backward()
        if trainer.layers_supervised == 0:
            raise RuntimeError(
                "no layer produced a ranking loss: the dL/do hooks never fired even though the LM "
                "loss was differentiable. This usually means the model kept its own attention "
                f"implementation (config is {model.config._attn_implementation!r}), so the swapped "
                "attention that registers the hook never ran."
            )
        return out.loss.detach()
