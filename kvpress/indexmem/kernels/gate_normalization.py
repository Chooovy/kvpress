# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class _HistoryLogsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, keys, gate_scale, sink_size, window_size, query_offset, tile_size):
        dtype = torch.float64 if queries.dtype == torch.float64 else torch.float32
        shape = queries.shape[:3]
        maximum = torch.full(shape, -float("inf"), device=queries.device, dtype=dtype)
        total = torch.zeros_like(maximum)
        query_positions = torch.arange(shape[2], device=queries.device)[:, None] + query_offset
        for start in range(0, keys.shape[1], tile_size):
            stop = min(start + tile_size, keys.shape[1])
            positions = torch.arange(start, stop, device=queries.device)[None, :]
            history = (positions >= sink_size) & (positions <= query_positions - window_size)
            logits = torch.einsum("bhqd,bkd->bhqk", queries.to(dtype), keys[:, start:stop].to(dtype))
            logits = (logits * gate_scale.to(dtype)).masked_fill(~history, -float("inf"))
            next_maximum = torch.maximum(maximum, logits.amax(-1))
            rescale = torch.where(torch.isfinite(maximum), (maximum - next_maximum).exp(), 0.0)
            finite_maximum = torch.where(torch.isfinite(next_maximum), next_maximum, 0.0)
            total = total * rescale + (logits - finite_maximum[..., None]).exp().sum(-1)
            maximum = next_maximum
        empty = ~torch.isfinite(maximum)
        result = torch.where(empty, 0.0, maximum + total.clamp_min(1e-30).log())
        ctx.save_for_backward(queries, keys, gate_scale, result, empty)
        ctx.geometry = sink_size, window_size, query_offset, tile_size, dtype
        return result

    @staticmethod
    def backward(ctx, grad_output):
        queries, keys, gate_scale, normalizer, empty = ctx.saved_tensors
        sink_size, window_size, query_offset, tile_size, dtype = ctx.geometry
        query_grad = torch.zeros_like(queries, dtype=dtype)
        key_grad = torch.zeros_like(keys, dtype=dtype)
        scale_grad = torch.zeros_like(gate_scale, dtype=dtype)
        query_positions = torch.arange(queries.shape[2], device=queries.device)[:, None] + query_offset
        gradient = (grad_output.to(dtype) * ~empty)[..., None]
        for start in range(0, keys.shape[1], tile_size):
            stop = min(start + tile_size, keys.shape[1])
            positions = torch.arange(start, stop, device=queries.device)[None, :]
            history = (positions >= sink_size) & (positions <= query_positions - window_size)
            key_tile = keys[:, start:stop].to(dtype)
            scores = torch.einsum("bhqd,bkd->bhqk", queries.to(dtype), key_tile)
            logits = (scores * gate_scale.to(dtype)).masked_fill(~history, -float("inf"))
            weights = (logits - normalizer[..., None]).exp() * gradient
            query_grad += torch.einsum("bhqk,bkd->bhqd", weights, key_tile) * gate_scale.to(dtype)
            key_grad[:, start:stop] += torch.einsum("bhqk,bhqd->bkd", weights, queries.to(dtype)) * gate_scale.to(dtype)
            scale_grad += (weights * scores).sum().reshape_as(scale_grad)
        return (
            query_grad.to(queries.dtype),
            key_grad.to(keys.dtype),
            scale_grad.to(gate_scale.dtype),
            None,
            None,
            None,
            None,
        )


def history_logsumexp(queries, keys, gate_scale, *, sink_size, window_size, query_offset=0, tile_size=1024):
    return _HistoryLogsumexp.apply(queries, keys, gate_scale, sink_size, window_size, query_offset, tile_size)
