# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress.indexmem.training.losses import distribution_kl, token_cross_entropy


def long_context_weights(long_loss, short_loss, scored, gamma=5.0):
    weights = (short_loss - long_loss).float().exp().clamp(max=gamma)
    return torch.where(scored, weights, 1.0)


def objective_loss(
    student_hidden, input_ids, lm_head, *, objective, teacher_hidden=None, weights=None, ce_weight=0.0, chunk_size=2048
):
    if objective in ("reverse_kl", "forward_kl"):
        loss = distribution_kl(
            student_hidden,
            teacher_hidden,
            lm_head,
            direction="reverse" if objective == "reverse_kl" else "forward",
            chunk_size=chunk_size,
        )
        if ce_weight:
            ce = token_cross_entropy(student_hidden, input_ids, lm_head, chunk_size=chunk_size).mean()
            loss = (1.0 - ce_weight) * loss + ce_weight * ce
        return loss
    tokens = token_cross_entropy(student_hidden, input_ids, lm_head, chunk_size=chunk_size)
    if objective == "longce":
        weights = weights.detach().reshape(-1).to(tokens.dtype)
        return (tokens * weights).sum() / weights.sum()
    return {"ce": tokens.mean()}[objective]
