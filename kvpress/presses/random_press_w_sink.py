# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RandomPress_with_sink(ScorerPress):
    """
    Random KV cache compression for baseline comparison.

    Randomly selects which key-value pairs to prune. Useful for establishing baseline
    performance metrics and validating other compression methods.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    seed : int, optional
        Random seed for reproducible compression results.
    """

    compression_ratio: float = 0.0
    seed: Optional[int] = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        n_sink = 4
        
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=keys.device)
            generator.manual_seed(self.seed)
        
        scores = torch.rand(*keys.shape[:-1], generator=generator, 
                        device=keys.device, dtype=keys.dtype)
        
        if scores.shape[-1] > n_sink:
            sink_fill_value = scores[:, :, n_sink:].max() + 1.0
            scores[:, :, :n_sink] = sink_fill_value
        
        return scores
