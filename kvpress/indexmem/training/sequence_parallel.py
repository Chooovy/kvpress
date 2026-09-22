# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist
from torch import nn


def sequence_slice(seq_len: int, rank: int, world_size: int) -> tuple[int, int]:
    (base, extra) = divmod(seq_len, world_size)
    start = rank * base + min(rank, extra)
    return (start, start + base + (1 if rank < extra else 0))


class _AllGatherSequence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, local: torch.Tensor, lengths: tuple[int, ...], group):
        ctx.lengths = lengths
        ctx.rank = dist.get_rank(group)
        width = max(lengths)
        ctx.width = width
        padded = local
        if local.shape[1] < width:
            pad = local.new_zeros((local.shape[0], width - local.shape[1], *local.shape[2:]))
            padded = torch.cat([local, pad], dim=1)
        shards = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(shards, padded.contiguous(), group=group)
        return torch.cat([shard[:, :length] for (shard, length) in zip(shards, lengths)], dim=1)

    @staticmethod
    def backward(ctx, grad_full):
        start = sum(ctx.lengths[: ctx.rank])
        stop = start + ctx.lengths[ctx.rank]
        return (grad_full[:, start:stop].contiguous(), None, None)


def all_gather_sequence(local: torch.Tensor, lengths: tuple[int, ...], group=None):
    return _AllGatherSequence.apply(local, lengths, group)


class _ScatterSequence(torch.autograd.Function):

    @staticmethod
    def forward(ctx, full: torch.Tensor, lengths: tuple[int, ...], group):
        ctx.lengths = lengths
        ctx.group = group
        rank = dist.get_rank(group)
        start = sum(lengths[:rank])
        return full[:, start : start + lengths[rank]].contiguous()

    @staticmethod
    def backward(ctx, grad_local):
        return (all_gather_sequence(grad_local, ctx.lengths, ctx.group), None, None)


def scatter_sequence(full: torch.Tensor, lengths: tuple[int, ...], group=None):
    return _ScatterSequence.apply(full, lengths, group)


class SequenceParallelFFN(nn.Module):

    def __init__(self, inner: nn.Module, group=None):
        super().__init__()
        self.inner = inner
        self.group = group

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        world_size = 1 if self.group is None else dist.get_world_size(self.group)
        if world_size == 1:
            return self.inner(hidden_states)
        seq_len = hidden_states.shape[1]
        lengths = tuple(
            (stop - start for (start, stop) in (sequence_slice(seq_len, r, world_size) for r in range(world_size)))
        )
        local = self.inner(scatter_sequence(hidden_states, lengths, self.group))
        return all_gather_sequence(local, lengths, self.group)

    def extra_repr(self) -> str:
        world_size = 1 if self.group is None else dist.get_world_size(self.group)
        return f"world_size={world_size}"


def wrap_ffn_sequence_parallel(model, group):
    for layer in model.model.layers:
        layer.mlp = SequenceParallelFFN(layer.mlp, group)
