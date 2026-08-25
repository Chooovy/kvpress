# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for DMA's value-derived, query-independent score."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.gqa_indexer.dma_indexer import DMAIndexer, DMAIndexerConfig
from kvpress.presses.gqa_indexer.press import GQAIndexerPress


def _dma() -> DMAIndexer:
    module = DMAIndexer(DMAIndexerConfig(n_heads=2, head_dim=3))
    with torch.no_grad():
        module.dt_proj.weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
                    [-0.7, 0.8, 0.9, -1.0, 1.1, 1.2],
                ]
            )
        )
        module.A.copy_(torch.tensor([0.25, -0.5]))
    return module


def test_dma_score_matches_the_paper_formula_and_backpropagates():
    """DMA is ``exp(A * softplus(W_dt(concat_heads(V))))``, exactly."""
    module = _dma().to(torch.bfloat16)
    values = torch.tensor(
        [
            [
                [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [-0.6, 0.7, 0.8], [0.9, -1.0, 1.1]],
                [[-0.3, 0.6, 0.1], [0.4, -0.8, 0.7], [1.0, 0.2, -0.5], [-0.4, 0.5, 0.3]],
            ]
        ],
        dtype=torch.bfloat16,
    )

    got = module.score_values(values)
    flattened = values.transpose(1, 2).reshape(1, 4, 6).float()
    projected = F.linear(flattened, module.dt_proj.weight.float())
    expected = torch.exp(F.softplus(projected) * module.A.float()).transpose(1, 2)

    assert module.dt_proj.bias is None
    assert module.is_query_independent is True
    assert got.shape == (1, 2, 4)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected)

    got.sum().backward()
    for gradient in (module.dt_proj.weight.grad, module.A.grad):
        assert gradient is not None
        assert torch.isfinite(gradient.float()).all()
        assert gradient.any()


def test_dma_press_score_depends_on_values_not_hidden_states():
    """The press must not accidentally turn DMA into a hidden-state scorer."""
    module = nn.Module()
    module.indexer = _dma()
    press = GQAIndexerPress(compression_ratio=0.0, scorer="dma", n_sink=0)
    values = torch.randn(1, 2, 5, 3)
    keys = torch.randn_like(values)
    hidden_a = torch.randn(1, 5, 16)
    hidden_b = torch.randn_like(hidden_a)

    from_a = press.score(module, hidden_a, keys, values, torch.empty(0), {})
    from_b = press.score(module, hidden_b, keys, values, torch.empty(0), {})
    torch.testing.assert_close(from_a, from_b)

    selector = module.indexer.project_q(hidden_a.bfloat16())
    assert selector.dtype == torch.float32
    torch.testing.assert_close(selector[0, :, 0], torch.eye(2))

    projected = module.indexer.project_k(hidden_a.bfloat16(), value_states=values)
    assert projected.dtype == torch.float32
    torch.testing.assert_close(projected, from_a.transpose(1, 2))

    changed_values = values.clone()
    changed_values[:, :, 0] += 1.0
    changed = press.score(module, hidden_a, keys, changed_values, torch.empty(0), {})
    assert not torch.equal(from_a, changed)
