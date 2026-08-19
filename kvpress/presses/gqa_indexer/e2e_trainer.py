# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end indexer training: the LM loss trains the router directly.

:class:`~kvpress.presses.gqa_indexer.fused_trainer.FusedIndexerTrainer` distills -- it builds
a teacher from the frozen model's own attention and pushes the indexer's scores towards it. The
router never touches the model's forward pass, so the LM loss cannot reach it, and the
supervision is a surrogate: it teaches the router to rank keys by *where the dense model
attends*, which is not the same as *which keys the model's prediction actually needs* under a
fixed budget. SAS reports this gap directly -- their distillation baseline covers more
attention mass yet scores lower downstream (96.8% mass / 79.5% accuracy against 79.5% / 79.5%
at K=64).

This trainer removes the surrogate. The indexer score is added inside the attention softmax::

    out = softmax(scale * q @ k^T + gate_scale * qi @ ki^T) @ v

so the router sits on the forward path and ``dL/dscore`` comes from the ordinary attention
backward. There is no teacher, no KL, and no second forward pass: the loss is the model's own.

Pinning is not optional
-----------------------
Adding the same number to every key of a row cancels in the softmax, so a gate that is **flat
along the key axis** does nothing and the model falls back to the frozen dense backbone -- which
is already strong. The router can reach that point at zero cost (``qi = 0``), satisfying the LM
loss without having learned any ranking. Under a positive fixed or ratio budget, ``pin_mode``
exempts some keys from the gate's normalizer, which makes a flat gate arithmetically impossible; see
:mod:`~kvpress.presses.gqa_indexer.gate_pin` for the mechanism and
``test_pin_closes_the_no_op_hole`` for the property. This is the difference between 18.8 and
54.4 in SAS's ablation, so ``pin_mode="none"`` warns.

Comparability with distillation
-------------------------------
Deliberately mirrors :class:`FusedIndexerTrainer`'s two stages, so the two objectives can be
compared at matched budget rather than at matched wall-clock:

* ``stage="dense"`` -> full scope. Every key is gated, so every key's score gets a
  content-dependent gradient. ``O(L^2)``, like stage 1 of distillation.
* ``stage="sparse"`` -> sparse scope. Only each row's own top-k is gated, at the same
  ``keep_ratio`` the press will evict at.

The stage names are shared with the distillation trainer on purpose; ``scope`` is the more
accurate word for what changes here and is accepted as an alias.

What is trained
---------------
Only the indexers. :meth:`E2EIndexerTrainer.freeze_backbone` puts every non-indexer parameter
at ``requires_grad=False``, matching SAS (which freezes the LLM and trains the selector alone)
and matching what distillation does -- otherwise the comparison would confound "end-to-end
gradient" with "more trainable parameters".

Note the gradient still *flows through* the frozen backbone to reach the router, which is the
point: the router is told what the model's prediction needed, not merely what its attention did.

Train/inference mismatch is intentional
---------------------------------------
Under ``stage="dense"`` training gates all keys while inference hard-selects top-k. That looks
like a bug and is the design: SAS tests the "consistent" alternative (STE, hard top-k in the
forward) and it is consistently worse -- 61.30 -> 51.48 on AIME25 at budget 4096 -- because the
hard forward collapses the score into a selected/unselected bit and discards the ranking
information near the top-k boundary. ``stage="sparse"`` is the consistent variant, available for
exactly that comparison.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.gate_pin import (
    check_pin_mode,
    history_lse,
    history_mask,
    pinned_mask,
    pins_self,
)
from kvpress.presses.gqa_indexer.triton_fused_loss import HAS_TRITON
from kvpress.presses.gqa_indexer.gated_attention import gated_attention
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.sparse_support import resolve_topk, streaming_topk_support

logger = logging.getLogger(__name__)

STAGES = {"dense": "full", "sparse": "sparse"}


def _sink_history_attention_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    row_lse: torch.Tensor,
    *,
    scaling: float | None,
    n_sink: int,
    query_offset: int | None = None,
) -> float | None:
    """Mean attention probability on non-sink history, without materializing ``(Sq, Sk)``."""
    q_len, k_len = q.shape[2], k.shape[2]
    if query_offset is None:
        query_offset = k_len - q_len
    sink_count = min(n_sink, k_len)
    if sink_count == 0:
        return None

    with torch.no_grad():
        group_size = q.shape[1] // k.shape[1]
        sink_key = k[:, :, :sink_count]
        if group_size > 1:
            sink_key = sink_key.repeat_interleave(group_size, dim=1)
        scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
        sink_logits = torch.einsum(
            "bhqd,bhsd->bhqs", q.float(), sink_key.float()
        ) * scale

        q_pos = torch.arange(q_len, device=q.device) + query_offset
        sink_pos = torch.arange(sink_count, device=q.device)
        visible_sink = sink_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        sink_mass = torch.where(
            visible_sink, torch.exp(sink_logits - row_lse.unsqueeze(-1)), 0.0
        ).sum(-1)

        visible_keys = (q_pos + 1).clamp(max=k_len)
        valid = visible_keys > visible_sink.sum(-1)
        if not bool(valid.any()):
            return None
        return float((1.0 - sink_mass)[..., valid].mean())


@dataclass
class E2EIndexerTrainer:
    """
    Train the indexer end-to-end by gating the model's own attention.

    Register with :meth:`hooks`, then run the model and call ``.backward()`` on its LM loss --
    no auxiliary objective is produced, so unlike :class:`FusedIndexerTrainer` there is no
    ``total_loss()`` to read.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers and the RoPE narrowing. Must have been built with
        ``gate_scale=True``.
    stage : str
        ``dense`` (full scope) or ``sparse`` (top-k scope). See "Comparability with
        distillation" above.
    topk : int, optional
        Sparse-stage support size. ``None`` derives it from ``keep_ratio``.
    keep_ratio : float
        Used when ``topk`` is None. Set to ``1 - compression_ratio`` so the sparse stage trains
        at the eviction budget used at eval.
    force_sink, force_local : int
        Support slots reserved per row for the leading keys and the row's own most recent keys.
    pin_mode : str
        Which keys are exempt from a positive fixed or ratio budget, so that a flat gate cannot
        become a no-op -- see :mod:`~kvpress.presses.gqa_indexer.gate_pin`. Applies to the dense
        stage only; the sparse stage needs no pin because its forward pass is already restricted
        to the selected keys, and leaving this at its default resolves to ``"none"`` there.

        ``"self"`` mirrors SAS's always-retained current block most closely; ``"sink"`` exempts
        the leading keys instead. Both run on the fused kernel
        (:mod:`~kvpress.presses.gqa_indexer.triton_gated_attention`) at ``O(L)`` memory, so the
        choice is now about which keys deserve the exemption rather than what it costs. Without
        Triton, ``self`` falls back to an ``O(Sq * Sk)`` path and ``sink`` to the concat form.

        ``"none"`` reproduces the un-pinned behaviour and exists as the ablation baseline: it
        leaves the no-op reachable, which is the failure SAS measures at 18.8 against 54.4.
    n_sink : int
        Leading keys to pin under ``sink`` / ``self+sink``. Defaults to the press's own
        ``n_sink`` when left at ``None``, so the keys the press protects at inference are the
        keys the gate exempts during training.
    gate_budget : float
        ``0`` uses raw history gates. A positive value fixes history's total multiplier relative
        to unit-gated pinned keys in the dense stage.
    gate_budget_ratio : float, optional
        Instead fix each query row's history multiplier budget to this ratio times the number of
        history keys visible to that row.
    key_tile : int
        Key tile for the streaming history ``logsumexp``.
    select_grad : bool
        Whether the sparse stage's *selection* pass carries gradients. Leave False: the top-k
        itself is not differentiable, so the pass only wastes memory building a graph that
        ``torch.topk`` severs anyway. The gate values inside the selected set are what carry
        the signal, and those are recomputed with gradients.
    freeze : bool
        Freeze every non-indexer parameter on :meth:`hooks` entry. Leave True -- see "What is
        trained".
    """

    press: GQAIndexerPress
    stage: str = "dense"

    # Sparse stage only
    topk: int | None = None
    keep_ratio: float = 0.25
    force_sink: int = 0
    force_local: int = 0
    select_grad: bool = False

    # Gate pinning (dense stage only). None resolves per stage: "self" for dense, "none" for
    # sparse -- so `stage="sparse"` needs no second flag to say what the scope already implies.
    pin_mode: str | None = None
    n_sink: int | None = None
    gate_budget: float = 1.0
    gate_budget_ratio: float | None = None
    key_tile: int = 1024

    freeze: bool = True

    #: Layer index -> the gate_scale seen on the last forward pass, for logging. A rising value
    #: means the layer is leaning harder on its router; one that collapses towards 0 means that
    #: layer's router is not earning its place, which no loss curve would tell you.
    gate_scales: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean *participation ratio* of the gate, as a fraction of the history
    #: length. This is the readout on whether the router learned to be SELECTIVE, which
    #: ``gate_scales`` alone cannot say: a layer can lean hard on a router that still spreads its
    #: attention over everything. Only populated when :attr:`measure_sparsity` is on, since it
    #: costs a second streaming pass over the keys. See :meth:`_gate_participation`.
    gate_sparsity: dict[int, float] = field(default_factory=dict)
    #: Layer index -> actual attention probability assigned to visible, non-pinned history.
    #: Reuses the fused forward's row log-normalizer and scores only the pinned sink keys, so
    #: collection is ``O(Sq * n_sink)`` rather than ``O(Sq * Sk)``.
    history_attention_mass: dict[int, float | None] = field(default_factory=dict)
    #: Compute :attr:`gate_sparsity` on this forward. The driver turns it on only for the steps
    #: it logs. This also collects :attr:`history_attention_mass` when the fused sink-pinned
    #: full-scope path exposes its row log-normalizer.
    measure_sparsity: bool = field(default=False)
    #: Layer index -> number of layers that actually ran, as a wiring check.
    layers_gated: int = field(default=0, init=False)

    _hidden_states: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _kwargs: dict[int, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {tuple(STAGES)}, got {self.stage!r}")
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.force_sink < 0 or self.force_local < 0:
            raise ValueError("force_sink and force_local must be non-negative")
        if self.pin_mode is not None:
            check_pin_mode(self.pin_mode)
        if self.n_sink is not None and self.n_sink < 0:
            raise ValueError(f"n_sink must be non-negative, got {self.n_sink}")
        if self.stage == "sparse" and self.pin_mode not in (None, "none"):
            # Not silently ignored: pinning under the sparse scope would be a no-op that the
            # loss curve cannot reveal, and a run configured for it would report having pinned.
            raise ValueError(
                f"pin_mode={self.pin_mode!r} does not apply to stage='sparse': the forward pass "
                "is already restricted to the selected keys, so a flat gate cannot recover dense "
                "attention and there is nothing to pin against. Leave pin_mode unset (it resolves "
                "to 'none' for the sparse stage) or pass pin_mode='none'."
            )
        if self.pin_mode == "none" and self.stage == "dense":
            logger.warning(
                "pin_mode='none' leaves the gate able to flatten into a no-op, which recovers "
                "the frozen dense model and lets the router satisfy the loss without learning any "
                "ranking. This is the ablation baseline; use pin_mode='self' (or 'sink') to train."
            )
        if pins_self(self.gate_pin_mode) and not HAS_TRITON:
            # Only a problem without the kernel: a query-dependent pin cannot fold into the
            # concatenated QK, so the fallback builds the logits explicitly and no tile knob
            # reduces that. Said at construction rather than left to an OOM traceback.
            logger.warning(
                "pin_mode=%r without Triton falls back to explicit (Sq, Sk) logits per layer, "
                "because a query-dependent pin cannot fold into the concatenated QK. Expect OOM "
                "much above ~8K. Install Triton (the fused kernel handles this pin at O(L)) or "
                "use pin_mode='sink'.",
                self.gate_pin_mode,
            )

    @property
    def gate_pin_mode(self) -> str:
        """
        The pin mode actually applied, resolving ``None`` from the stage.

        The sparse scope makes pinning inert, so the stage already determines the answer; the
        default resolves rather than forcing every sparse-stage caller to restate it.
        """
        if self.pin_mode is not None:
            return self.pin_mode
        return "none" if self.stage == "sparse" else "self"

    @property
    def sink_count(self) -> int:
        """Leading keys to pin, defaulting to the press's own ``n_sink``."""
        return self.press.n_sink if self.n_sink is None else self.n_sink

    @property
    def scope(self) -> str:
        """The gating scope this stage maps to: ``full`` or ``sparse``."""
        return STAGES[self.stage]

    def reset(self) -> None:
        """Drop the per-pass state from the previous forward."""
        self.gate_scales = {}
        self.gate_sparsity = {}
        self.history_attention_mass = {}
        self.layers_gated = 0
        self._hidden_states.clear()
        self._kwargs.clear()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def indexer_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """Every indexer parameter, in layer order -- what the optimizer should be given."""
        params = []
        for layer in get_language_model(model).layers:
            indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
            if indexer is not None:
                params.extend(indexer.parameters())
        return params

    def upcast_gate_scales(self, model: nn.Module) -> tuple[int, list[float]]:
        """
        Put every ``gate_scale`` scalar in fp32.

        Returns ``(how many were converted, their values)``.

        **Why this is required, not an optimization.** ``post_init_from_model`` builds the whole
        indexer at the model's dtype, which is bf16 for this run. bf16 has 8 mantissa bits, so
        around the init value ``head_dim**-0.5 = 0.0884`` the representable values are ~3.0e-4
        apart. AdamW's update is ``lr * m_hat / (sqrt(v_hat) + eps)``, whose magnitude is ~``lr``
        for a stable gradient -- so during warmup (``lr`` 1.3e-5 rising to 1.5e-4) every single
        step rounds straight back to the value it started from. The scalar is **frozen at its
        initialization**, and it stays that way until ``lr`` alone exceeds the spacing.

        Observed exactly that: ``gate_scale_mean`` pinned at 0.08837890625 -- bf16's rendering of
        ``1/sqrt(128)`` -- identically across all 36 layers for 30 steps, while the loss fell 4.52
        -> 2.42 and ``grad_norm`` moved 6.4 -> 0.80. The other indexer weights are unaffected:
        their elements sit near 0.02 where bf16 spacing is ~1e-4, and they receive updates far
        larger than one step's ``lr``. It is only the scalar, and only because 0.0884 is a large
        value to be nudged by 1e-5 increments.

        Left unfixed the run still trains ``w_q``/``w_k`` and the loss still falls, so nothing
        looks wrong -- but the per-layer gate strength, which is the readout on whether a layer's
        router earns its place at all, is a constant. The stored value would also be a rounded
        one, so a resumed run inherits the same freeze.

        fp32 is safe everywhere the scalar is consumed: the Triton kernel loads it with
        ``.to(tl.float32)`` and accumulates its gradient into an fp32 buffer, and the reference
        path multiplies it into an fp32 accumulator. In the concat path it divides ``q_idx``
        before a ``.to(q.dtype)`` cast, so the product lands back in bf16 either way.
        """
        converted = 0
        values: list[float] = []
        for layer in get_language_model(model).layers:
            indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
            gate_scale = getattr(indexer, "gate_scale", None) if indexer is not None else None
            if gate_scale is None or gate_scale.dtype == torch.float32:
                continue
            # A new leaf Parameter, so this must happen BEFORE the optimizer is built -- an
            # optimizer already holding the bf16 tensor would keep stepping that one.
            indexer.gate_scale = nn.Parameter(
                gate_scale.detach().to(torch.float32), requires_grad=gate_scale.requires_grad
            )
            values.append(float(indexer.gate_scale.detach()))
            converted += 1
        return converted, values

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Put every non-indexer parameter at ``requires_grad=False``.

        Identifies the indexers by module identity rather than by parameter-name substring: a
        name filter would also catch a backbone parameter that happens to contain the attribute
        name, and would silently train it.

        Upcasts the ``gate_scale`` scalars to fp32 on the way through -- in bf16 they cannot be
        moved by a warmup-sized learning rate at all. See :meth:`upcast_gate_scales`. Done here
        because this is where the trainable set is decided, and it must precede the optimizer.
        """
        converted, values = self.upcast_gate_scales(model)
        if converted:
            logger.info(
                "upcast %d gate_scale scalar(s) to fp32 (value %.6f): at bf16's ~3.0e-4 spacing "
                "there, a warmup learning rate rounds every step back and the gate would stay "
                "frozen at initialization.",
                converted, values[0],
            )
        indexer_params = {id(p) for p in self.indexer_parameters(model)}
        if not indexer_params:
            raise RuntimeError(
                f"no {self.press.scorer_attr!r} modules found on the model; call "
                "press.post_init_from_model(model) first"
            )
        for param in model.parameters():
            param.requires_grad = id(param) in indexer_params

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------
    def indexer_qk(self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict) -> tuple:
        """
        The indexer's ``(q_idx, k_idx)`` for this layer, with gradients.

        Built from the same ``hidden_states`` and the same RoPE tables the layer itself is
        using, via the press's own :meth:`~.press.GQAIndexerPress.get_rope_tables` -- so the
        gate is scored by exactly the function the press will score with at inference. That
        shared path is the invariant worth protecting: a divergence here trains the router for
        a scoring function it never runs under, and nothing downstream would flag it.
        """
        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        return (
            indexer.project_q(hidden_states, cos, sin),
            indexer.project_k(hidden_states, cos, sin),
            indexer.require_gate_scale(),
        )

    def select_support(
        self, q_idx: torch.Tensor, k_idx: torch.Tensor, k_len: int, attention_mask
    ) -> torch.Tensor:
        """
        The sparse stage's per-row top-k support, ``(B, Hkv, Sq, topk)``.

        Uses the same :func:`~.sparse_support.streaming_topk_support` the eviction path and the
        distillation trainer's stage 2 use, so all three select identically.

        ``mask=None`` is passed deliberately when there is no padding: that takes the streaming
        selector's causal-arithmetic path and avoids materializing an ``O(Sq * Sk)`` mask, which
        would defeat the point of a sparse stage.
        """
        topk = resolve_topk(k_len, self.topk, self.keep_ratio)
        mask = None
        if attention_mask is not None:
            from kvpress.presses.gqa_indexer.indexer import build_indexer_mask

            mask = build_indexer_mask(
                q_idx.shape[2], k_len, q_idx.device, attention_mask=attention_mask
            )
        support, _ = streaming_topk_support(
            q_idx,
            k_idx,
            topk,
            mask=mask,
            force_sink=self.force_sink,
            force_local=self.force_local,
        )
        return support

    def gated_forward(
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
        Replacement attention for one layer: gate the logits with the indexer score.

        ``query``/``key``/``value`` arrive post-RoPE and post-cache-update from the layer, so
        this sees exactly the tensors the original attention would have.
        """
        layer_idx = int(module.layer_idx)
        # POPPED, not read. The pre-hook fires just before this call, so exactly one layer's entry
        # is ever needed -- but leaving it in the dict pins that layer's (B, L, hidden) tensor for
        # the rest of the forward *and* the whole backward, since the context only clears on exit.
        # At L=16384 on Qwen3-8B that is 36 x 0.125 GiB kept reachable, which also stops autograd
        # from releasing each layer's activations as backward walks up the stack. Popping keeps the
        # peak at one layer.
        hidden_states = self._hidden_states.pop(layer_idx, None)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached the gated attention without its hidden_states being "
                "captured. The pre-hook that records them must be registered on the same "
                "modules as the attention swap -- use E2EIndexerTrainer.hooks(). A second call "
                "for the same layer in one forward (gradient checkpointing recomputes a block) "
                "also lands here, because the entry is consumed on first use."
            )
        kwargs = self._kwargs.pop(layer_idx, {})

        q_idx, k_idx, gate_scale = self.indexer_qk(module, hidden_states, kwargs)
        self.gate_scales[layer_idx] = float(gate_scale.detach())
        self.layers_gated += 1

        if self.measure_sparsity:
            self.gate_sparsity[layer_idx] = self._gate_participation(
                q_idx, k_idx, gate_scale, key.shape[2], attention_mask
            )

        indices = None
        if self.scope == "sparse":
            ctx = torch.enable_grad() if self.select_grad else torch.no_grad()
            with ctx:
                indices = self.select_support(
                    q_idx if self.select_grad else q_idx.detach(),
                    k_idx if self.select_grad else k_idx.detach(),
                    key.shape[2],
                    attention_mask,
                )

        measure_history_mass = (
            self.measure_sparsity and self.scope == "full" and self.gate_pin_mode == "sink"
        )
        attention = gated_attention(
            query,
            key,
            value,
            q_idx,
            k_idx,
            scope=self.scope,
            indices=indices,
            scaling=scaling,
            gate_scale=gate_scale,
            gate_budget=self.gate_budget,
            gate_budget_ratio=self.gate_budget_ratio,
            # The sparse scope takes its masking from `indices`; the full scope needs the
            # layer's mask, and `None` there means plain causal.
            mask=None if self.scope == "sparse" else attention_mask,
            dropout_p=dropout if self.scope == "full" else 0.0,
            pin_mode="none" if self.scope == "sparse" else self.gate_pin_mode,
            n_sink=self.sink_count,
            key_tile=self.key_tile,
            return_row_lse=measure_history_mass,
        )
        if measure_history_mass:
            out, row_lse = attention
            self.history_attention_mass[layer_idx] = self._history_attention_mass(
                query, key, row_lse, scaling
            )
        else:
            out = attention
        # The attention interface contract is (B, Sq, H, D) -- the layer reshapes to
        # (B, Sq, H*D) itself. Our ops return (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """
        Stash this layer's ``hidden_states`` and kwargs before its attention runs.

        Needed because the attention *interface* is called with q/k/v only -- it never sees
        ``hidden_states``, which is what the indexer projects from. A forward pre-hook on the
        attention module is the earliest place both are available.
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

    @contextmanager
    def hooks(self, model: nn.Module):
        """
        Gate every attention layer for the duration of the block.

        Two things are installed: a pre-hook per attention module that captures
        ``hidden_states``, and a temporary entry in ``ALL_ATTENTION_FUNCTIONS`` that the model's
        ``config._attn_implementation`` is pointed at. Both are removed on exit, including on
        exception.

        The registry cleanup goes through ``_global_mapping`` directly for the reason
        :func:`~.teacher_lse.capture_teacher_lse` documents: ``register()`` writes there, while
        ``pop()`` only touches the instance mapping, so the naive removal leaves the entry
        behind forever.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        if self.freeze:
            self.freeze_backbone(model)

        impl_name = "kvpress_gqa_indexer_gated"

        def gated_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self.gated_forward(
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
        ALL_ATTENTION_FUNCTIONS.register(impl_name, gated_attention_impl)

        handles = []
        try:
            for layer in get_language_model(model).layers:
                handles.append(
                    layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
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

    def _gate_participation(
        self, q_idx, k_idx, gate_scale, k_len: int, attention_mask
    ) -> float | None:
        """
        Mean **participation ratio** of the gate, as a fraction of each row's history length.

        What it measures. The diagnostic normalizes each row's raw history scores as
        ``p_j = exp(s_j - lse)`` even when ``gate_budget=0`` leaves the training gate unnormalized.
        Its participation ratio
        ``PR = 1 / sum_j p_j^2`` is the standard effective-support size of such a distribution:
        ``PR = n`` for a flat distribution over ``n`` keys, ``PR = 1`` when one key takes
        everything. Dividing by the history length gives a scale-free number:

        * ``~1.0`` -- the gate is flat. The router is NOT selective, whatever ``gate_scale`` says.
        * ``~0.5`` -- half the history is doing the work.
        * ``->0``  -- strongly peaked, which is the behaviour eviction needs.

        Why PR rather than counting keys above a threshold: no threshold has to be chosen, and it
        is invariant to how the probability mass is permuted across keys, so it compares across
        layers and sequence lengths. It is also exactly the quantity that predicts eviction
        quality -- a top-k of size ``k`` captures most of the mass only if ``PR`` is around ``k``.

        Computed with the streaming identity ``sum_j p_j^2 = exp(2*lse - lse2)`` where ``lse2`` is
        the logsumexp of ``2*s``. Since ``s = gate_scale * qi.ki``, ``lse2`` is just
        :func:`history_lse` called with ``2 * gate_scale`` -- so this reuses the same ``O(L)``
        streaming kernel instead of materializing the ``(Sq, Sk)`` score matrix, which at 16K
        would be 1 GiB per layer per head.

        Returns ``None`` under the sparse scope, where the gate is not normalized over a history
        (the forward is already restricted to the router's top-k, so this ratio is not defined
        against the same denominator and comparing the two would be misleading).
        """
        if self.scope == "sparse":
            return None

        pinned = pinned_mask(
            self.gate_pin_mode, q_idx.shape[2], k_len, q_idx.device, n_sink=self.sink_count
        )
        if pinned is None:
            # pin_mode="none": every visible key is history. history_lse still needs a mask
            # tensor, so build the all-false one it expects.
            pinned = torch.zeros(
                (q_idx.shape[2], k_len), dtype=torch.bool, device=q_idx.device
            )

        # no_grad throughout: this is a diagnostic. Leaving it in the graph would retain the
        # streaming kernel's backward state for a term the objective never uses.
        with torch.no_grad():
            causal_keep = None
            if attention_mask is not None:
                from kvpress.presses.gqa_indexer.indexer import build_indexer_mask

                causal_keep = build_indexer_mask(
                    q_idx.shape[2], k_len, q_idx.device, attention_mask=attention_mask
                ) == 0
            common = dict(pinned=pinned, causal_keep=causal_keep, key_tile=self.key_tile)
            lse = history_lse(q_idx, k_idx, gate_scale=gate_scale, **common)
            lse2 = history_lse(q_idx, k_idx, gate_scale=2.0 * gate_scale, **common)

            # sum p^2 = exp(lse2 - 2*lse), so PR = exp(2*lse - lse2).
            pr = torch.exp(2.0 * lse - lse2)

            # Rows with no history get lse=lse2=0 (documented in history_lse), hence PR=1 against
            # a history of 0 -- meaningless, so they are excluded rather than counted as "flat".
            # history_mask is the SAME function the positive-budget normalizer uses, so the
            # diagnostic's denominator cannot drift from that objective.
            history_len = history_mask(
                pinned, causal_keep, q_idx.shape[2], k_len, q_idx.device
            ).sum(-1)
            valid = history_len > 0
            if not bool(valid.any()):
                return None
            fraction = pr / history_len.clamp(min=1).to(pr.dtype)
            return float(fraction[..., valid].mean())

    def _history_attention_mass(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_lse: torch.Tensor | None,
        scaling: float | None,
    ) -> float | None:
        """Actual non-pinned attention mass for the fused, sink-pinned full-scope path."""
        if row_lse is None or self.gate_pin_mode != "sink":
            return None
        return _sink_history_attention_mass(
            q, k, row_lse, scaling=scaling, n_sink=self.sink_count
        )

    def mean_gate_sparsity(self) -> float | None:
        """Mean gate participation fraction over the layers that measured it."""
        values = [v for v in self.gate_sparsity.values() if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def mean_history_attention_mass(self) -> float | None:
        """Mean actual history attention mass over layers that exposed the fused row LSE."""
        values = [v for v in self.history_attention_mass.values() if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def mean_gate_scale(self) -> float | None:
        """Mean ``gate_scale`` over the layers that ran, or ``None`` before the first pass."""
        if not self.gate_scales:
            return None
        return sum(self.gate_scales.values()) / len(self.gate_scales)


def e2e_indexer_training_step(
    model: nn.Module,
    trainer: E2EIndexerTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    skip_logits: bool | None = None,
) -> torch.Tensor:
    """
    One gated forward pass, returning the model's own LM loss.

    ``labels`` defaults to ``input_ids`` (ordinary next-token prediction; the model shifts them
    internally). ``use_cache=False`` -- nothing here reads the cache, and building one during
    training only costs memory.

    ``skip_logits=True`` asks a Liger-patched model to fuse ``lm_head`` into the cross-entropy so
    the ``(L, vocab)`` logits are never materialized -- 7.0 GiB at ``L=8192`` on Qwen3-8B, and it
    scales with ``L``. It must be passed **explicitly** here: Liger's own default is
    ``self.training and labels is not None``, and this backbone is deliberately left in ``eval()``
    to keep dropout off, so the default resolves to ``False`` and the patch silently saves nothing.
    Ignored by an unpatched model, which does not accept the kwarg -- hence the ``None`` default
    rather than ``False``.

    Returns
    -------
    torch.Tensor
        The LM loss, ready for ``backward()``. The router's gradient arrives through the
        attention softmax rather than from an auxiliary term, so there is no second loss to add.
    """
    extra = {} if skip_logits is None else {"skip_logits": skip_logits}
    with trainer.hooks(model):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids if labels is None else labels,
            use_cache=False,
            **extra,
        )
        if trainer.layers_gated == 0:
            raise RuntimeError(
                "no layer ran the gated attention: the model kept its own attention "
                "implementation. This usually means the model's config is not the one "
                f"E2EIndexerTrainer pointed at {model.config._attn_implementation!r}."
            )
        return out.loss
