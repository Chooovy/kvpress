# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage indexer distillation.

Stage 1 (dense warmup)
    Teacher = the frozen model's true attention, grouped per KV head.
    Student = ``softmax(indexer_logits)`` over all valid keys.
    Objective = ``KL(teacher || student)``. Teaches the indexer *where to look*.

Stage 2 (sparse refinement)
    Teacher and student are both restricted to the indexer's own top-k support and
    renormalized there. Sharpens the ranking *within* the kept set, which is what actually
    matters at eviction time. Mirrors DSA's ``sparse_loss=True`` path.

Both stages route through :func:`indexer_layer_loss`, which is deliberately a plain
function over explicit tensors: no hooks, no monkeypatching, no global state. That makes
each piece testable in isolation and keeps the numerics reviewable.

This module optimizes for clarity, not throughput. It materializes the full
``(B, n_kv_heads, Sq, Sk)`` logits and the dense attention target, which is fine for the
sequence lengths used to warm up an indexer but will need a fused/chunked kernel for long
context. The seams for that are :func:`indexer_layer_loss` and the target builders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.loss import (
    build_dense_indexer_target,
    build_sparse_indexer_target,
    indexer_loss_from_target,
    masked_log_softmax,
)
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model

logger = logging.getLogger(__name__)


@dataclass
class IndexerTrainConfig:
    """
    Objective configuration for indexer distillation.

    Attributes
    ----------
    stage : str
        ``dense`` (stage 1) or ``sparse`` (stage 2).
    head_reduce : str
        How attention heads combine within a KV group to form that group's target:
        ``mean`` (average demand) or ``amax`` (union of demand).
    loss_coeff : float
        Scalar multiplier on the loss.
    topk : int, optional
        Support size for ``sparse``. Defaults to ``keep_ratio`` of the sequence.
    keep_ratio : float
        Used when ``topk`` is None: ``topk = max(1, int(k_len * keep_ratio))``. Set this to
        ``1 - compression_ratio`` so training matches the eviction budget used at eval.
    skip_sink_in_loss : int
        Exclude the first N keys from the objective. Those tokens are protected at
        inference regardless, so spending target mass on them teaches nothing.
    """

    stage: str = "dense"
    head_reduce: str = "mean"
    loss_coeff: float = 1.0
    topk: int | None = None
    keep_ratio: float = 0.25
    skip_sink_in_loss: int = 0

    def __post_init__(self):
        if self.stage not in ("dense", "sparse"):
            raise ValueError(f"stage must be 'dense' or 'sparse', got {self.stage!r}")
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")


def build_loss_masks(
    indexer_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    skip_sink: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive which (query, key) pairs and which query rows take part in the objective.

    Validity is read back off the indexer's own masked logits rather than rebuilt, so the
    student and the loss can never disagree about what was masked -- the failure mode that
    silently corrupts this kind of distillation.

    Returns
    -------
    valid_mask : torch.Tensor
        Bool (B, h, Sq, Sk); False entries leave the key sum.
    row_valid : torch.Tensor
        Bool (B, h, Sq); False rows leave the row average. A row is dropped when it has no
        valid key, or when ``labels`` marks its query position as ignored.
    """
    valid_mask = indexer_logits > (MASK_NEG / 2)

    if skip_sink > 0:
        valid_mask = valid_mask.clone()
        valid_mask[..., :skip_sink] = False

    row_valid = valid_mask.any(dim=-1)  # (B, h, Sq)

    if labels is not None:
        # Only supervise positions the LM itself is trained on; -100 marks prompt/pad.
        supervised = (labels != -100).unsqueeze(1)  # (B, 1, Sq)
        row_valid = row_valid & supervised
    if attention_mask is not None and attention_mask.dim() == 2:
        q_len = row_valid.shape[-1]
        keep_q = attention_mask[:, -q_len:].bool().unsqueeze(1)  # (B, 1, Sq)
        row_valid = row_valid & keep_q

    return valid_mask, row_valid


def indexer_layer_loss(
    indexer_logits: torch.Tensor,
    attentions: torch.Tensor,
    config: IndexerTrainConfig,
    *,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Indexer KL loss for one layer.

    Parameters
    ----------
    indexer_logits : torch.Tensor
        Masked student logits, (B, n_kv_heads, Sq, Sk), from
        ``GQAIndexerPress.indexer_logits``.
    attentions : torch.Tensor
        Teacher attention probabilities, (B, n_heads, Sq, Sk).
    config : IndexerTrainConfig
        Objective configuration.
    attention_mask : torch.Tensor, optional
        Padding mask, (B, Sk).
    labels : torch.Tensor, optional
        LM labels, (B, Sq); ``-100`` positions are excluded from the row average.

    Returns
    -------
    torch.Tensor
        Scalar loss for this layer.
    """
    if indexer_logits.dim() != 4:
        raise ValueError(f"indexer_logits must be 4D, got {tuple(indexer_logits.shape)}")
    if attentions.shape[-2:] != indexer_logits.shape[-2:]:
        raise ValueError(
            f"teacher/student shape mismatch: attentions {tuple(attentions.shape)} vs "
            f"indexer_logits {tuple(indexer_logits.shape)}"
        )

    n_kv_heads = indexer_logits.shape[1]
    valid_mask, row_valid = build_loss_masks(
        indexer_logits, attention_mask, labels, config.skip_sink_in_loss
    )

    # The teacher is a frozen reference: no gradient may flow back into the base model.
    attentions = attentions.detach()

    if config.stage == "dense":
        target = build_dense_indexer_target(
            attentions, valid_mask, n_kv_heads=n_kv_heads, head_reduce=config.head_reduce
        )
        log_probs = masked_log_softmax(indexer_logits.float(), valid_mask)
        return indexer_loss_from_target(
            target,
            log_probs,
            loss_coeff=config.loss_coeff,
            valid_mask=valid_mask,
            row_valid=row_valid,
        )

    # sparse: restrict both sides to the indexer's own top-k support.
    k_len = indexer_logits.shape[-1]
    topk = config.topk or max(1, int(k_len * config.keep_ratio))
    topk = min(topk, k_len)

    # Invalid keys must never be selected, so push them below every valid logit.
    selectable = indexer_logits.masked_fill(~valid_mask, float("-inf"))
    topk_logits, topk_indices = selectable.topk(topk, dim=-1)

    # A slot is real only if it landed on a valid key; -1 marks the padding slots that
    # appear when a row has fewer valid keys than topk.
    topk_valid = torch.isfinite(topk_logits)
    topk_indices = topk_indices.masked_fill(~topk_valid, -1)

    target = build_sparse_indexer_target(
        attentions, topk_indices, n_kv_heads=n_kv_heads, head_reduce=config.head_reduce
    )
    log_probs = masked_log_softmax(topk_logits.float(), topk_valid)
    return indexer_loss_from_target(
        target,
        log_probs,
        loss_coeff=config.loss_coeff,
        valid_mask=topk_valid,
        row_valid=row_valid & topk_valid.any(dim=-1),
    )


def compute_indexer_loss(
    press: GQAIndexerPress,
    attn_modules: list[nn.Module],
    hidden_states: tuple[torch.Tensor, ...],
    attentions: tuple[torch.Tensor, ...],
    config: IndexerTrainConfig,
    *,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    position_embeddings: tuple | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Average the per-layer indexer loss over all layers.

    ``hidden_states`` is the tuple from ``output_hidden_states=True``, which has
    ``num_layers + 1`` entries: index 0 is the embedding output and index ``i + 1`` is the
    output of layer ``i``. Layer ``i``'s attention *consumes* ``hidden_states[i]``, so that
    is the tensor the indexer must see -- using ``i + 1`` would feed it the layer's own
    output and quietly hand it information the real forward pass never has.

    Returns
    -------
    loss : torch.Tensor
        Mean loss over layers.
    per_layer : list[torch.Tensor]
        Per-layer losses, for logging.
    """
    if len(hidden_states) < len(attentions):
        raise ValueError(
            f"expected at least {len(attentions)} hidden_states, got {len(hidden_states)}; "
            "pass output_hidden_states=True"
        )

    per_layer = []
    for layer_idx, attn in enumerate(attentions):
        module = attn_modules[layer_idx]
        layer_input = hidden_states[layer_idx]

        kwargs = {"attention_mask": attention_mask}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        logits = press.indexer_logits(module, layer_input, kwargs)
        per_layer.append(
            indexer_layer_loss(
                logits, attn, config, attention_mask=attention_mask, labels=labels
            )
        )

    return torch.stack(per_layer).mean(), per_layer


def freeze_all_but_indexer(model: nn.Module, scorer_attr: str = "indexer") -> list[nn.Parameter]:
    """
    Freeze the base model and return the indexer parameters to optimize.

    Both stages train only the indexer; the backbone stays frozen so the teacher
    distribution is a fixed reference.
    """
    trainable = []
    for name, param in model.named_parameters():
        if f".{scorer_attr}." in name or name.startswith(f"{scorer_attr}."):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    if not trainable:
        raise RuntimeError(
            f"No parameters matched {scorer_attr!r}. Call press.post_init_from_model(model) first."
        )
    return trainable


def get_attention_modules(model: nn.Module) -> list[nn.Module]:
    """Return the attention modules in layer order."""
    return [layer.self_attn for layer in get_language_model(model).layers]


def indexer_state_dict(model: nn.Module, scorer_attr: str = "indexer") -> dict:
    """
    Extract just the indexer weights.

    The indexer is a few MB against a multi-GB backbone, so checkpointing only these keys
    keeps runs cheap. Keys stay fully qualified, so this loads back with
    ``model.load_state_dict(sd, strict=False)`` after ``post_init_from_model``.
    """
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if f".{scorer_attr}." in name or name.startswith(f"{scorer_attr}.")
    }


def load_indexer_state_dict(model: nn.Module, state_dict: dict, scorer_attr: str = "indexer") -> None:
    """
    Load indexer weights, failing loudly if nothing matched.

    ``strict=False`` is required (the dict intentionally omits the backbone) but it also
    silences genuine key mismatches, so the match count is checked explicitly.
    """
    filtered = {
        k: v for k, v in state_dict.items() if f".{scorer_attr}." in k or k.startswith(f"{scorer_attr}.")
    }
    if not filtered:
        raise ValueError(f"state_dict contains no {scorer_attr!r} keys")

    model_keys = set(model.state_dict().keys())
    missing = [k for k in filtered if k not in model_keys]
    if missing:
        raise ValueError(
            f"{len(missing)} indexer keys are absent from the model "
            f"(e.g. {missing[:3]}). Did the indexer geometry change?"
        )
    model.load_state_dict(filtered, strict=False)
    logger.info("Loaded %d %s tensors", len(filtered), scorer_attr)
