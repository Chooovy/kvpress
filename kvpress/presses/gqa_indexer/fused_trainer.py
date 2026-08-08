# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-layer driver for the tiled indexer loss.

:mod:`kvpress.presses.gqa_indexer.train` collects every layer's dense attention first and
then computes the loss, which is fine at warmup scale but needs the whole
``(B, H, Sq, Sk)`` teacher. The tiled loss in
:mod:`kvpress.presses.gqa_indexer.fused_loss` removes that, but only if the teacher's Q/K
are *also* never all resident at once: on Llama-3.1-8B they are 320 MiB per layer at
L=32K, so keeping all 32 layers would cost 10 GiB (40 GiB at L=128K).

So the loss is computed **inside** each attention layer's forward, via a hook, and only
the resulting scalar is kept. One layer's teacher tensors are alive at a time; per-layer
state that survives is a single float.

Teacher reconstruction
----------------------
The hook fires after attention has run, so the post-RoPE keys are already in the cache --
no need to recompute them. Queries are rebuilt from ``hidden_states`` with the layer's own
``q_proj`` and the same ``position_embeddings`` the layer just used, which is exactly what
the other kvpress scorers do (see ``SnapKVPress.compute_window_attention``). Everything is
detached: the teacher is a frozen reference.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.gqa_indexer.fused_loss import (
    fused_indexer_loss,
    make_recompute_teacher,
    teacher_lse_from_qk,
)
from kvpress.presses.gqa_indexer.fused_sparse_loss import (
    TEACHER_MODES,
    fused_sparse_indexer_loss,
    make_sparse_recompute_teacher,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG, build_indexer_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.sparse_support import resolve_topk, streaming_topk_support
from kvpress.presses.gqa_indexer.teacher_lse import capture_teacher_lse
from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    decompose_mask,
    kernels_available,
    triton_indexer_loss,
    triton_interpret_enabled,
)
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


def teacher_query_states(
    module: nn.Module, hidden_states: torch.Tensor, position_embeddings: tuple
) -> torch.Tensor:
    """
    Rebuild the layer's post-RoPE queries, (B, H, Sq, d).

    Uses the layer's own ``q_proj`` (via :func:`kvpress.utils.get_prerope_query_states`,
    which also applies QK-norm on the models that have it) and the same cos/sin the layer
    just used, so the result matches what attention actually saw.
    """
    query_states = get_prerope_query_states(module, hidden_states)
    cos, sin = position_embeddings
    if cos.dim() == 4:  # some models emit (B, 1, S, D)
        cos, sin = cos.squeeze(1), sin.squeeze(1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (query_states * cos) + (rotate_half(query_states) * sin)


def attention_scaling(module: nn.Module) -> float:
    """Softmax scale for the layer, preferring its own attribute over ``head_dim ** -0.5``."""
    scaling = getattr(module, "scaling", None)
    if scaling is not None:
        return float(scaling)
    return float(module.head_dim**-0.5)


@dataclass
class FusedIndexerTrainer:
    """
    Accumulate the tiled indexer loss layer by layer during one forward pass.

    Register with :meth:`hooks`, run the frozen model, then read :attr:`per_layer_losses`
    or call :meth:`total_loss`. The base model must be frozen and its attention left on
    whatever fast kernel it normally uses -- unlike the dense path, nothing here needs
    ``output_attentions=True``.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers and the RoPE narrowing.
    stage : str
        ``dense`` (stage 1) streams the full key axis; ``sparse`` (stage 2) restricts the
        objective to each row's own top-k support, turning the teacher recompute from
        ``H * L^2 * d`` into ``H * L * topk * d`` -- 64x at ``L=32K, topk=512``. Run dense
        first: stage 2's support is only meaningful once the indexer roughly knows where to
        look, and MiniMax MSA reports the same warmup ordering.
    key_tile, query_tile : int
        Tile sizes for the fused loss. Peak tile memory is ``O(query_tile * key_tile)``, so
        both must be bounded for the footprint to stay flat in sequence length.
    topk_tile : int
        Support-axis tile width in stage 2. Stage 2's scratch carries a gathered-key tensor
        of ``O(query_tile * topk_tile * D)``, so it wants smaller tiles than stage 1.
    loss_coeff : float
        Scalar multiplier on each layer's loss.
    skip_sink_in_loss : int
        Exclude the first N keys from the objective; they are protected at inference
        regardless, so target mass spent on them teaches nothing.
    detach_teacher : bool
        Keep the teacher out of the autograd graph. Leave True: MiniMax M3 reports that
        letting KL gradients reach the backbone lets it lower the loss by *simplifying its
        own attention* instead of improving the indexer, which shows up as gradient-norm
        spikes and short-context regression.
    topk : int, optional
        Stage-2 support size. ``None`` derives it from ``keep_ratio``.
    keep_ratio : float
        Used when ``topk`` is None: ``topk = max(1, int(k_len * keep_ratio))``. Set this to
        ``1 - compression_ratio`` so stage 2 trains at the eviction budget used at eval.
    teacher_mode : str
        ``global`` or ``support``; see
        :mod:`kvpress.presses.gqa_indexer.fused_sparse_loss`. These are different
        objectives, not two ways to compute one. ``global`` keeps the teacher fixed across
        steps (so the loss curve means the same thing throughout) at the cost of a dense
        ``teacher_lse``; ``support`` drops that cost and makes stage 2 ``O(L * topk)`` end to
        end.
    force_sink, force_local : int
        Stage-2 slots reserved per query row for the leading keys and the row's own most
        recent keys, mirroring MSA's always-selected local block. Distinct from the press's
        ``n_sink``/``n_local``, which protect keys globally after the query axis is reduced.
    backend : str
        ``auto`` (default) takes the Triton kernels when they can run and the mask
        decomposes, else PyTorch. ``torch`` forces the reference path. ``triton`` forces the
        kernels and *raises* rather than falling back, so a benchmark cannot silently measure
        the wrong thing. Stage 2 is torch-only for now.
    block_m, block_n : int
        Triton tile sizes; must be powers of two.
    capture_lse : str
        Where the teacher's logsumexp comes from. ``never`` (default) recomputes it with
        ``teacher_lse_from_qk``. ``auto`` reuses the value flash-attention already computed
        during the model's own forward pass -- free, but only valid when the loss's mask is
        purely causal, so it falls back per layer otherwise. ``always`` refuses to start unless
        the capture is usable, so a benchmark cannot silently measure the recompute. See
        :meth:`teacher_lse`.
    """

    press: GQAIndexerPress
    stage: str = "dense"
    key_tile: int = 512
    query_tile: int = 512
    topk_tile: int = 512
    loss_coeff: float = 1.0
    skip_sink_in_loss: int = 0
    detach_teacher: bool = True

    # Stage 2 only
    topk: int | None = None
    keep_ratio: float = 0.25
    teacher_mode: str = "global"
    force_sink: int = 0
    force_local: int = 0

    # Kernel backend
    backend: str = "auto"
    block_m: int = 64
    block_n: int = 64

    # Reuse flash-attention's logsumexp instead of recomputing it. "auto" uses it when
    # flash-attn is importable AND the mask is purely causal, else recomputes; "never" always
    # recomputes; "always" refuses to start unless the capture is usable, so a benchmark cannot
    # silently measure the slow path. See `teacher_lse` for why the mask condition is required.
    capture_lse: str = "never"

    per_layer_losses: dict[int, torch.Tensor] = field(default_factory=dict)
    per_layer_recall: dict[int, float] = field(default_factory=dict)
    backend_used: str | None = field(default=None, init=False)
    # Populated by `hooks` for the duration of one forward pass when capture is active.
    captured_lse: dict[int, torch.Tensor] | None = field(default=None, init=False)
    lse_source: str | None = field(default=None, init=False)
    _warned_lse_fallback: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.stage not in ("dense", "sparse"):
            raise ValueError(f"stage must be 'dense' or 'sparse', got {self.stage!r}")
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        # Checked here rather than only where it is used, so a typo surfaces at construction
        # instead of silently riding along until the run switches to stage 2.
        if self.teacher_mode not in TEACHER_MODES:
            raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}, got {self.teacher_mode!r}")
        if self.force_sink < 0 or self.force_local < 0:
            raise ValueError("force_sink and force_local must be non-negative")
        if self.backend not in ("auto", "torch", "triton"):
            raise ValueError(f"backend must be 'auto', 'torch' or 'triton', got {self.backend!r}")
        if self.block_m & (self.block_m - 1) or self.block_n & (self.block_n - 1):
            raise ValueError(
                f"block_m/block_n must be powers of two, got {self.block_m}, {self.block_n}"
            )
        if self.backend == "triton" and self.stage == "sparse":
            raise NotImplementedError(
                "the Triton path currently covers stage 1 only; stage 2 runs on the torch "
                "backend. Use backend='auto' with stage='sparse'."
            )
        if self.capture_lse not in ("never", "auto", "always"):
            raise ValueError(
                f"capture_lse must be 'never', 'auto' or 'always', got {self.capture_lse!r}"
            )
        if self.capture_lse != "never" and self.stage == "sparse":
            # Stage 2's teacher is restricted to each row's support, so a dense causal lse is
            # the wrong normalizer for it -- 'global' mode needs the full-axis value, which is
            # what teacher_lse_from_qk gives. Refuse rather than quietly ignoring the flag.
            raise NotImplementedError(
                "capture_lse covers stage 1 only; stage 2's teacher_mode='global' needs a "
                "dense logsumexp over the full key axis. Use capture_lse='never'."
            )
        if self.capture_lse == "always" and self.skip_sink_in_loss > 0:
            # Caught at construction because 'always' is a promise the run cannot keep here:
            # every layer would fall back, so a benchmark would report the slow path under a
            # flag that says otherwise.
            raise ValueError(
                "capture_lse='always' is incompatible with skip_sink_in_loss>0: flash's lse "
                "covers causality only, so skipping sink keys would leave the teacher rows "
                "un-normalized. Use capture_lse='auto' to fall back per layer."
            )

    def reset(self) -> None:
        """Drop the losses from the previous forward pass."""
        self.per_layer_losses = {}
        self.per_layer_recall = {}
        self.backend_used = None

    def total_loss(self) -> torch.Tensor:
        """Mean loss over the layers that fired. Raises if none did."""
        if not self.per_layer_losses:
            raise RuntimeError(
                "no layer losses were recorded; register hooks() and run a forward pass first"
            )
        return torch.stack([self.per_layer_losses[k] for k in sorted(self.per_layer_losses)]).mean()

    def mean_recall(self) -> float | None:
        """
        Mean teacher mass the stage-2 support captured, or ``None`` outside stage 2.

        Worth watching: a low value means ``topk`` is too small for how spread out the
        teacher actually is, and the loss alone will not say so -- it can look healthy while
        the objective ignores most of the teacher's mass.
        """
        if not self.per_layer_recall:
            return None
        return sum(self.per_layer_recall.values()) / len(self.per_layer_recall)

    def layer_loss(self, module: nn.Module, kwargs: dict, keys: torch.Tensor) -> torch.Tensor:
        """Tiled loss for one layer, given its cached (post-RoPE) keys."""
        hidden_states = kwargs["hidden_states"]
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError(
                "the attention layer did not receive position_embeddings; the fused trainer "
                "needs them to rebuild the teacher's post-RoPE queries"
            )

        indexer = self.press.get_indexer(module)
        q_len, k_len = hidden_states.shape[1], keys.shape[2]

        # The mask is finalized ONCE, before the logsumexp -- including the sink skip.
        # Masking keys afterwards would leave the teacher rows not summing to one (the
        # masked mass is simply lost), quietly down-weighting the affected rows. That is
        # the trap documented in fused_loss.teacher_probs_from_lse, and applying
        # skip_sink_in_loss after the lse would walk straight back into it.
        mask = build_indexer_mask(
            q_len,
            k_len,
            hidden_states.device,
            attention_mask=kwargs.get("attention_mask"),
            dtype=torch.float32,
        )
        mask = self.apply_sink_skip(mask)
        row_valid = self.row_validity(mask, kwargs)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)

        with torch.no_grad():
            query_states = teacher_query_states(module, hidden_states, position_embeddings)
            key_states = keys
            if self.detach_teacher:
                query_states, key_states = query_states.detach(), key_states.detach()
            group_size = query_states.shape[1] // key_states.shape[1]
            scaling = attention_scaling(module)

        if self.stage == "sparse":
            return self.sparse_layer_loss(
                indexer,
                hidden_states,
                query_states,
                key_states,
                scaling,
                group_size,
                mask=mask,
                row_valid=row_valid,
                cos=cos,
                sin=sin,
                layer_idx=int(module.layer_idx),
            )

        with torch.no_grad():
            lse = self.teacher_lse(
                query_states,
                key_states,
                scaling,
                mask=mask,
                layer_idx=int(module.layer_idx),
                kwargs=kwargs,
            )

        if self.use_triton(indexer, hidden_states, query_states, key_states):
            ok, keep = decompose_mask(
                mask, q_len, k_len, k_len - q_len, bsz=hidden_states.shape[0]
            )
            if ok:
                self.backend_used = "triton"
                return triton_indexer_loss(
                    indexer,
                    hidden_states,
                    query_states,
                    key_states,
                    lse,
                    scaling=scaling,
                    cos=cos,
                    sin=sin,
                    keep=keep,
                    query_offset=k_len - q_len,
                    row_valid=row_valid,
                    block_m=self.block_m,
                    block_n=self.block_n,
                    loss_coeff=self.loss_coeff,
                )
            if self.backend == "triton":
                raise RuntimeError(
                    "backend='triton' was requested but this layer's mask is not "
                    "causal + per-key padding, which the kernels cannot represent. Use "
                    "backend='auto' to fall back, or backend='torch'."
                )
            logger.debug("mask is not kernel-representable; falling back to the torch backend")

        self.backend_used = "torch"
        teacher_alpha = make_recompute_teacher(query_states, key_states, scaling, group_size)

        return fused_indexer_loss(
            indexer,
            hidden_states,
            teacher_alpha,
            lse,
            group_size=group_size,
            cos=cos,
            sin=sin,
            mask=mask,
            row_valid=row_valid,
            key_tile=self.key_tile,
            query_tile=self.query_tile,
            loss_coeff=self.loss_coeff,
        )

    def use_triton(self, indexer, hidden_states, query_states, key_states) -> bool:
        """
        Whether to take the Triton path for this layer.

        ``auto`` requires CUDA and a kernel-supported dtype, and additionally declines under
        ``TRITON_INTERPRET=1``: the interpreter is correct but much slower than the PyTorch
        path, so silently choosing it would make ``auto`` a pessimization. ``triton`` forces
        it (including under the interpreter, which is the point of that mode) and raises if it
        cannot run, rather than quietly measuring the fallback.
        """
        if self.backend == "torch":
            return False

        probe = (
            indexer.w_q.weight,
            hidden_states,
            query_states,
            key_states,
        )
        if self.backend == "triton":
            if not HAS_TRITON:
                raise RuntimeError("backend='triton' was requested but Triton is not installed")
            if not kernels_available(*probe):
                raise RuntimeError(
                    "backend='triton' was requested but the kernels cannot run on these "
                    f"tensors (cuda={all(t.is_cuda for t in probe)}, "
                    f"dtypes={[t.dtype for t in probe]}); float64 in particular has no tl.dot"
                )
            return True

        return kernels_available(*probe) and not triton_interpret_enabled()

    def sparse_layer_loss(
        self,
        indexer,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scaling: float,
        group_size: int,
        *,
        mask: torch.Tensor,
        row_valid: torch.Tensor,
        cos,
        sin,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Stage 2: pick each row's top-k support, then run the KL only there.

        Pass 1 is ``no_grad`` and emits int64 indices, so it contributes no autograd state.
        Pass 2 re-projects q/k *with* gradients -- the projections are recomputed rather than
        reused from pass 1 precisely so pass 1 can stay under ``no_grad``. That costs one
        extra ``hidden -> q/k`` GEMM, which is negligible against the ``L * topk`` term it
        buys.
        """
        k_len = key_states.shape[2]
        topk = resolve_topk(k_len, self.topk, self.keep_ratio)
        self.backend_used = "torch"

        with torch.no_grad():
            support, valid = streaming_topk_support(
                indexer.project_q(hidden_states, cos, sin).detach(),
                indexer.project_k(hidden_states, cos, sin).detach(),
                topk,
                mask=mask,
                force_sink=self.force_sink,
                force_local=self.force_local,
                key_tile=self.key_tile,
                query_tile=self.query_tile,
            )

            teacher_lse = None
            if self.teacher_mode == "global":
                teacher_lse = teacher_lse_from_qk(
                    query_states,
                    key_states,
                    scaling,
                    mask=mask,
                    key_tile=self.key_tile,
                    query_tile=self.query_tile,
                )

        teacher_alpha = make_sparse_recompute_teacher(
            query_states,
            key_states,
            scaling,
            group_size,
            teacher_lse=teacher_lse,
            teacher_mode=self.teacher_mode,
            support=support,
            valid=valid,
            topk_tile=self.topk_tile,
        )

        stats: dict = {}
        loss = fused_sparse_indexer_loss(
            indexer,
            hidden_states,
            support,
            valid,
            teacher_alpha,
            group_size=group_size,
            cos=cos,
            sin=sin,
            row_valid=row_valid & valid.any(dim=-1),
            query_tile=self.query_tile,
            topk_tile=self.topk_tile,
            loss_coeff=self.loss_coeff,
            stats=stats,
        )
        if "recall" in stats:
            rows = row_valid & valid.any(dim=-1)
            weight = rows.to(stats["recall"].dtype).expand_as(stats["recall"])
            self.per_layer_recall[layer_idx] = float(
                (stats["recall"] * weight).sum() / weight.sum().clamp_min(1.0)
            )
        return loss

    def teacher_lse(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scaling: float,
        *,
        mask: torch.Tensor,
        layer_idx: int,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        The teacher's ``(B, H, Sq)`` logsumexp: reuse flash-attention's, or recompute it.

        ``teacher_lse_from_qk`` is exact but streams the whole ``Q @ K^T`` again, on every layer
        of every step, on both backends -- the Triton kernel takes ``lse`` as an *input*, so it
        does not avoid this. flash-attention already computed the same quantity during the
        model's own forward pass and will hand it back for free, which is what
        :func:`~.teacher_lse.capture_teacher_lse` collects into ``self.captured_lse``.

        Using a captured value is only valid when the loss's mask is **pure causal**, because
        flash's ``lse`` covers causality and nothing else. Two things break that, and both are
        checked rather than assumed:

        * padding -- ``exp(alpha - lse)`` then stops summing to one over the kept keys, silently
          under-weighting the rows with the most padding;
        * ``skip_sink_in_loss`` -- extra per-key masking, folded into ``mask`` *before* the
          logsumexp precisely so the rows stay normalized. A causal-only ``lse`` would leave the
          skipped mass in the denominator.

        ``decompose_mask`` already answers "is this mask causal plus per-key padding", so it is
        reused here instead of a second mask analysis: ``keep is None`` is exactly the
        pure-causal case. Anything else falls back to the recompute, which is correct for any
        mask.
        """
        captured = None
        if self.captured_lse is not None:
            captured = self.captured_lse.get(layer_idx)
        if captured is not None:
            ok, keep = decompose_mask(
                mask, query_states.shape[2], key_states.shape[2],
                key_states.shape[2] - query_states.shape[2],
                bsz=query_states.shape[0],
            )
            if ok and keep is None:
                self.lse_source = "captured"
                return captured.to(device=query_states.device, dtype=torch.float32)
            # Say why once per trainer rather than per layer per step: at 36 layers x 1500 steps
            # a per-layer warning is 54000 lines, which buries the one thing worth reading.
            if not self._warned_lse_fallback:
                logger.warning(
                    "a captured teacher lse is available but this layer's mask is not purely "
                    "causal (skip_sink_in_loss=%d, padding=%s), so it cannot be used: flash's "
                    "lse covers causality only, and reusing it here would leave the teacher "
                    "rows un-normalized. Falling back to teacher_lse_from_qk.",
                    self.skip_sink_in_loss,
                    kwargs.get("attention_mask") is not None,
                )
                self._warned_lse_fallback = True

        self.lse_source = "recomputed"
        return teacher_lse_from_qk(
            query_states,
            key_states,
            scaling,
            mask=mask,
            key_tile=self.key_tile,
            query_tile=self.query_tile,
        )

    def apply_sink_skip(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Optionally drop the sink keys from the objective.

        Uses ``MASK_NEG``, the same finite sentinel as :func:`build_indexer_mask`. A larger
        magnitude (say ``finfo.min / 4``) would overflow to ``-inf`` once added to the
        teacher logits, breaking the invariant that a fully-masked row stays finite.

        This can leave the first ``skip_sink_in_loss`` query rows with no valid key at all
        (a causal row only sees keys up to its own position); :meth:`row_validity` then
        drops them from the average.
        """
        if self.skip_sink_in_loss <= 0:
            return mask
        mask = mask.clone()
        mask[..., : self.skip_sink_in_loss] = MASK_NEG
        return mask

    def row_validity(self, mask: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """
        Rows that take part in the average: those with at least one valid key.

        Padded query positions are excluded too, since their loss is meaningless. The
        result is (B, 1, Sq) and broadcasts over the indexer heads.
        """
        row_valid = (mask > MASK_NEG / 2).any(dim=-1)  # (B, 1, Sq)
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() == 2:
            q_len = row_valid.shape[-1]
            row_valid = row_valid & attention_mask[:, -q_len:].bool().unsqueeze(1)
        return row_valid

    def forward_hook(self, module: nn.Module, args, kwargs: dict, output):
        """Compute and stash this layer's loss, leaving the model output untouched."""
        cache = kwargs.get("past_key_values")
        if cache is None:
            raise RuntimeError(
                "the fused trainer needs the KV cache to read the teacher's post-RoPE keys; "
                "run the forward pass with use_cache=True"
            )
        keys, _ = extract_keys_and_values(cache, module.layer_idx)
        self.per_layer_losses[int(module.layer_idx)] = self.layer_loss(module, kwargs, keys)
        return output

    @contextmanager
    def hooks(self, model: nn.Module):
        """
        Register the per-layer hooks for the duration of the block.

        When ``capture_lse`` is on, this also wraps the pass in
        :func:`~.teacher_lse.capture_teacher_lse`, which swaps in an attention implementation
        that asks flash-attention for the ``lse`` it already computed. The capture must be
        entered *outside* the forward pass and the dict it yields is filled as layers run --
        which is why it is bound here rather than inside the loss hook: by the time a layer's
        hook fires, that layer's ``lse`` is already in the dict.
        """
        self.press.post_init_from_model(model)
        self.reset()
        handles = []
        with ExitStack() as stack:
            if self.capture_lse != "never":
                try:
                    self.captured_lse = stack.enter_context(capture_teacher_lse(model))
                except ImportError:
                    if self.capture_lse == "always":
                        raise
                    # 'auto' means "use it if available", so a box without flash-attn simply
                    # recomputes. Logged once, since it is a throughput fact worth knowing.
                    if not self._warned_lse_fallback:
                        logger.warning(
                            "capture_lse='auto' but flash-attn is unavailable; recomputing the "
                            "teacher logsumexp with teacher_lse_from_qk"
                        )
                        self._warned_lse_fallback = True
                    self.captured_lse = None
            try:
                for layer in get_language_model(model).layers:
                    handles.append(
                        layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
                    )
                yield self
            finally:
                for handle in handles:
                    handle.remove()
                self.captured_lse = None


def fused_indexer_training_step(
    model: nn.Module,
    trainer: FusedIndexerTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    One forward pass that produces the indexer loss and nothing else.

    ``use_cache=True`` is required (the teacher's post-RoPE keys are read from the cache);
    ``output_attentions`` is deliberately *not* set, so the base model keeps its fast
    attention kernel.

    Returns
    -------
    loss : torch.Tensor
        Mean over layers, ready for ``backward()``.
    per_layer : dict[int, torch.Tensor]
        Layer index -> scalar loss, for logging.
    """
    with trainer.hooks(model):
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        return trainer.total_loss(), dict(trainer.per_layer_losses)
