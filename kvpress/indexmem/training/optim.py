# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def warmup_stable_decay(total_steps, warmup_fraction=0.1, stable_fraction=0.6, final_fraction=0.005):
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    stable_end = max(warmup_steps, int(total_steps * (warmup_fraction + stable_fraction)))

    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step < stable_end:
            return 1.0
        progress = (step - stable_end) / max(1, total_steps - 1 - stable_end)
        return 1.0 + (final_fraction - 1.0) * min(progress, 1.0)

    return multiplier


def build_optimizer(
    parameters,
    *,
    learning_rate,
    total_steps,
    weight_decay=0.0,
    final_fraction=0.005,
    warmup_fraction=0.1,
    stable_fraction=0.6,
):
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_stable_decay(total_steps, warmup_fraction, stable_fraction, final_fraction),
    )
    return optimizer, schedule
