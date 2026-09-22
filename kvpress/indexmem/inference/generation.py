# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch


def sample_token(logits: torch.Tensor, sampling: dict | None) -> torch.Tensor:
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
        drop = cum - sorted_x.softmax(-1) > float(top_p)
        x = x.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return torch.multinomial(x.softmax(-1), 1).squeeze(-1)
