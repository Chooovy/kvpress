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
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.gqa_indexer.fused_loss import (
    fused_indexer_loss,
    make_recompute_teacher,
    teacher_lse_from_qk,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG, build_indexer_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
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
    key_tile : int
        Keys per tile in the fused loss.
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
    """

    press: GQAIndexerPress
    key_tile: int = 512
    loss_coeff: float = 1.0
    skip_sink_in_loss: int = 0
    detach_teacher: bool = True

    per_layer_losses: dict[int, torch.Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        """Drop the losses from the previous forward pass."""
        self.per_layer_losses = {}

    def total_loss(self) -> torch.Tensor:
        """Mean loss over the layers that fired. Raises if none did."""
        if not self.per_layer_losses:
            raise RuntimeError(
                "no layer losses were recorded; register hooks() and run a forward pass first"
            )
        return torch.stack([self.per_layer_losses[k] for k in sorted(self.per_layer_losses)]).mean()

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

        with torch.no_grad():
            query_states = teacher_query_states(module, hidden_states, position_embeddings)
            key_states = keys
            if self.detach_teacher:
                query_states, key_states = query_states.detach(), key_states.detach()

            group_size = query_states.shape[1] // key_states.shape[1]
            scaling = attention_scaling(module)
            lse = teacher_lse_from_qk(
                query_states, key_states, scaling, mask=mask, key_tile=self.key_tile
            )

        teacher_alpha = make_recompute_teacher(query_states, key_states, scaling, group_size)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)

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
            loss_coeff=self.loss_coeff,
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
        """Register the per-layer hooks for the duration of the block."""
        self.press.post_init_from_model(model)
        self.reset()
        handles = []
        try:
            for layer in get_language_model(model).layers:
                handles.append(
                    layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()


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
