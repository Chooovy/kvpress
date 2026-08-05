# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Capture the frozen teacher's per-head logsumexp during its own forward pass.

The teacher distribution is recoverable exactly from ``(logits, lse)``::

    p[i, t, s] = exp(alpha[i, t, s] - lse[i, t])

so distillation never needs the ``(H, L, L)`` attention matrix. flash-attention already
computes ``lse`` internally and will hand it back for free via ``return_attn_probs=True``,
which makes the teacher side of the objective essentially cost-free -- and lets the base
model keep its fast attention kernel instead of being forced onto ``eager`` by
``output_attentions=True``.

Masking caveat
--------------
``lse`` must be computed under the *same* mask the loss applies. flash-attention's ``lse``
covers the causal mask only, so it is valid exactly when no other masking is in play
(no padding, no extra bias). With padding present, ``exp(alpha - lse)`` no longer sums to
one over the kept keys -- the padded mass is silently missing -- which under-weights the
rows with the most padding. :func:`assert_lse_mask_compatible` refuses that combination,
and :func:`~.fused_loss.teacher_lse_from_qk` is the fallback that folds any mask in before
taking the logsumexp.

Verified against the flash-attention source (``flash_attn/flash_attn_interface.py``):

- ``flash_attn_func(..., return_attn_probs=True)`` returns
  ``(out, softmax_lse, S_dmask)`` with ``softmax_lse`` of shape
  ``(batch_size, nheads, seqlen)`` -- the layout :func:`normalize_captured_lse` expects
  first.
- ``causal=True`` aligns the mask to the **bottom-right** corner, which is what
  :func:`~.indexer.build_indexer_mask` produces for its default
  ``query_offset = k_len - q_len``. The two agree for both ``Sq < Sk`` and ``Sq > Sk``.
- GQA maps query head ``i`` to KV head ``i // group_size`` ("head 0, 1, 2 of Q will
  attention to head 0 of K, V"), matching the ``repeat_interleave`` in
  :func:`~.fused_loss.make_recompute_teacher`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG

logger = logging.getLogger(__name__)


def assert_lse_mask_compatible(attention_mask: torch.Tensor | None, source: str) -> None:
    """
    Reject a captured ``lse`` whose mask does not match the loss's mask.

    Raises rather than warns: a mismatch produces an un-normalized teacher, which trains
    quietly and slightly wrongly instead of failing.

    Additive masks are tested against ``MASK_NEG / 2`` rather than ``finfo.min / 2``.
    :func:`~.indexer.build_indexer_mask` deliberately uses a *finite* sentinel (``-1e4``),
    which sits far above ``finfo.min / 2`` -- so the wider threshold classified every
    masked entry as "kept" and the guard never fired.
    """
    if attention_mask is None:
        return
    if attention_mask.dim() == 2:
        fully_attended = bool(attention_mask.all())
    else:
        if attention_mask.dtype == torch.bool:
            keep = attention_mask
        else:
            keep = attention_mask > (MASK_NEG / 2)
        # A pure causal mask keeps exactly the lower triangle; anything sparser is extra.
        q_len, k_len = keep.shape[-2], keep.shape[-1]
        if q_len == k_len:
            causal = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool, device=keep.device))
            fully_attended = bool((keep | ~causal).all())
        else:
            fully_attended = bool(keep.all())
    if not fully_attended:
        raise ValueError(
            f"{source} logsumexp covers only the causal mask, but the batch carries "
            "additional padding/bias masking. Using it would leave the teacher rows "
            "un-normalized. Either drop padding from the batch, or compute the teacher "
            "logsumexp with kvpress.presses.gqa_indexer.fused_loss.teacher_lse_from_qk, "
            "which folds the mask in before the logsumexp."
        )


def normalize_captured_lse(lse: torch.Tensor, bsz: int, n_heads: int, q_len: int) -> torch.Tensor:
    """
    Coerce a flash-attention ``lse`` into ``(B, H, Sq)``.

    ``flash_attn_func``'s documented layout is already ``(batch_size, nheads, seqlen)``, so
    the first branch is the fast path. The alternatives cover other entry points and
    versions -- ``(B, Sq, H)``, or a squeezed 2D form -- rather than being guesses; anything
    unrecognized raises with the shape it received.
    """
    if lse.dim() == 3:
        if lse.shape == (bsz, n_heads, q_len):
            return lse
        if lse.shape == (bsz, q_len, n_heads):
            return lse.transpose(1, 2)
    elif lse.dim() == 2 and bsz == 1:
        if lse.shape == (n_heads, q_len):
            return lse.unsqueeze(0)
        if lse.shape == (q_len, n_heads):
            return lse.t().unsqueeze(0)
    raise ValueError(
        f"cannot interpret logsumexp of shape {tuple(lse.shape)} as (B={bsz}, H={n_heads}, "
        f"Sq={q_len})"
    )


@contextmanager
def capture_teacher_lse(model: nn.Module):
    """
    Capture each attention layer's teacher ``lse`` for the duration of one forward pass.

    Registers a temporary attention implementation that calls flash-attention with
    ``return_attn_probs=True`` and stashes the ``lse`` per layer, then points the model's
    ``config._attn_implementation`` at it. The original implementation is restored on exit,
    including on exception.

    The layer index comes from each module's own ``layer_idx``, so the mapping is correct
    regardless of the order in which layers happen to run.

    Yields
    ------
    dict[int, torch.Tensor]
        Layer index -> ``(B, H, Sq)`` logsumexp, populated as the forward pass runs.

    Examples
    --------
    >>> with capture_teacher_lse(model) as lse_by_layer:
    ...     model(input_ids, use_cache=False)
    >>> lse_by_layer[0].shape
    torch.Size([1, 32, 512])
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:  # pragma: no cover - depends on the runtime env
        raise ImportError(
            "capture_teacher_lse needs flash-attn. Install it, or compute the teacher "
            "logsumexp with fused_loss.teacher_lse_from_qk instead."
        ) from exc

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    captured: dict[int, torch.Tensor] = {}
    impl_name = "kvpress_teacher_lse_capture"

    def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        # flash-attn wants (B, Sq, H, D); the attention layer hands us (B, H, Sq, D).
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        out, lse, _ = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            return_attn_probs=True,
        )
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None:
            captured[int(layer_idx)] = normalize_captured_lse(
                lse.detach(), query.shape[0], query.shape[1], query.shape[2]
            ).float()
        # Back to (B, Sq, H, D) -> (B, Sq, H*D) is the caller's job; match the interface
        # contract of returning (attn_output, attn_weights) with attn_output as (B, Sq, H, D).
        return out, None

    # Configs that carry their own copy of the flag; multimodal models nest one.
    configs = [model.config]
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    previous_impls = [cfg._attn_implementation for cfg in configs]

    # register() writes to the CLASS-level _global_mapping, while __delitem__ (and hence
    # MutableMapping.pop) only touches the instance's _local_mapping. So pop() raises
    # KeyError internally and leaves the entry behind forever -- the cleanup has to go
    # through _global_mapping directly, restoring any pre-existing entry.
    global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    had_previous = impl_name in global_mapping
    previous_fn = global_mapping.get(impl_name)

    ALL_ATTENTION_FUNCTIONS.register(impl_name, capture_attention)
    try:
        for cfg in configs:
            cfg._attn_implementation = impl_name
        yield captured
    finally:
        for cfg, previous in zip(configs, previous_impls):
            cfg._attn_implementation = previous
        if had_previous:
            global_mapping[impl_name] = previous_fn
        else:
            global_mapping.pop(impl_name, None)
