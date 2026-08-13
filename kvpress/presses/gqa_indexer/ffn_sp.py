# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sequence-parallel FFN: each rank keeps only its slice of the MLP activations.

The end-to-end objective retains far more than distillation does, and the reason is structural:
the router sits inside attention, the loss sits on top of the LM head, so the gradient must
traverse **every layer above the router** and each of those layers' activations stays alive until
backward. Distillation computes its loss inside a per-layer hook from detached teacher tensors, so
nothing spans the backbone.

Measured on Qwen3-8B (deduped by storage, liger CE + liger SwiGLU on), the activation split is:

=========================================  =======  =====
term                                        @8K      share
=========================================  =======  =====
FFN                                         18.6     49%
attention (our kernel + the projections)    14.9     39%
rmsnorm                                      4.5     12%
=========================================  =======  =====

Every one of those is ``O(L)`` -- the Triton gated-attention kernel is what made attention linear
rather than quadratic -- so **the shares are constant in sequence length**. FFN is the largest
single term at any length, which is what makes sharding it alone worthwhile.

Why FFN alone is nearly free to shard
-------------------------------------
The MLP is **position-wise**: token ``i``'s output depends on token ``i`` and nothing else. So a
contiguous sequence split needs no cross-rank communication *inside* the FFN -- rank ``r``
computes its own slice and the slices are concatenated afterwards. Verified exact
(``0.00e+00`` on both the forward and ``d/dx``) in ``test_ffn_sp_matches_dense``.

Attention is the opposite: query ``i`` needs keys ``0..i``, which under a sequence split live on
other ranks. Sharding it requires the Ulysses all-to-all that converts a sequence split into a
head split, plus special handling for the indexer's MQA key. None of that is needed here, and
crucially **none of the gate machinery is touched** -- the kernel, the capture hook,
``history_lse`` and ``k_idx`` all keep seeing the full sequence.

What this does and does not buy
------------------------------
Per rank on 8 GPUs, weights + Adam (17.5 GiB) replicated in every column:

===========================  ======  ======  ======
option                        8K      16K     32K
===========================  ======  ======  ======
1 GPU                         55.6    93.7    169.8
**FFN-only SP**               39.3    **61.2**  104.9
full Ulysses                  22.3    27.1    36.7
===========================  ======  ======  ======

So this reaches **16K and not 32K**. It is deliberately the smaller change: the sequence-split
plumbing here is a subset of what full Ulysses needs, so extending it later is additive rather
than a rewrite.

Communication
-------------
One all-gather of ``(L, hidden)`` per layer, against Ulysses' two all-to-alls of
``q/k/v/q_idx``. At ``L=16384, hidden=4096`` in bf16 that is 128 MiB per layer per rank versus
196 MiB, so this is also the *cheaper* option on the wire -- it moves hidden states rather than
projections.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


def sequence_slice(seq_len: int, rank: int, world_size: int) -> tuple[int, int]:
    """
    This rank's ``[start, stop)`` of a contiguous sequence split.

    The remainder goes to the **first** ``seq_len % world_size`` ranks, one token each, so the
    slice lengths differ by at most one. Padding to a multiple of ``world_size`` would be simpler
    but changes what the model sees, and a training script that silently lengthens its input is
    the kind of thing that shows up much later as an off-by-a-few in the loss.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base, extra = divmod(seq_len, world_size)
    start = rank * base + min(rank, extra)
    return start, start + base + (1 if rank < extra else 0)


class _AllGatherSequence(torch.autograd.Function):
    """
    Differentiable all-gather along the sequence axis, tolerant of ragged slices.

    Forward gathers every rank's slice into the full sequence; backward hands each rank back
    exactly the slice it contributed. That asymmetry is the point: the gradient of a gather is a
    scatter, so no rank ever materializes a full-sequence gradient buffer -- which is what would
    undo the saving this module exists for.

    **Padding.** ``all_gather`` requires every rank's tensor to have the *same* shape (gloo raises
    ``invalid tensor size``; nccl would be worse, since mismatched sizes there are undefined rather
    than rejected). When ``L`` is not a multiple of ``world_size`` the slices differ by one, so each
    rank pads its slice up to the widest one and the concatenation trims each piece back to its true
    length.

    The pad is added to the FFN's **output**, never its input: padding the input would run
    gate/up/down over junk rows, which costs compute and -- with a ``silu`` in the path -- could
    put non-finite values into a buffer that then gets gathered. Verified exact reconstruction in
    ``test_ffn_sp_handles_ragged_sequence``.
    """

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
        return torch.cat(
            [shard[:, :length] for shard, length in zip(shards, lengths)], dim=1
        )

    @staticmethod
    def backward(ctx, grad_full):
        start = sum(ctx.lengths[: ctx.rank])
        stop = start + ctx.lengths[ctx.rank]
        # Only this rank's own slice; the others take theirs from their own backward. The pad rows
        # contributed nothing to the output, so they receive no gradient and are simply absent.
        return grad_full[:, start:stop].contiguous(), None, None


def all_gather_sequence(local: torch.Tensor, lengths: tuple[int, ...], group=None):
    """Gather per-rank sequence slices into the full sequence, differentiably."""
    return _AllGatherSequence.apply(local, lengths, group)


class SequenceParallelFFN(nn.Module):
    """
    Wraps a position-wise FFN so each rank computes and retains only its own slice.

    Deliberately a wrapper rather than a monkey-patch of ``forward``: the wrapped module keeps its
    identity, its parameters and its place in the tree, so ``state_dict``, the indexer's own
    plumbing, and liger's SwiGLU rebinding all continue to work unchanged. Liger in particular
    rebinds ``forward`` on the *inner* module, which this calls -- so the two compose, and the
    SwiGLU saving (2 tensors instead of 3) applies to the slice.

    ``world_size == 1`` returns the inner module's output directly, with no slicing and no
    collective, so a single-process run is bit-identical to not using this at all.
    """

    def __init__(self, inner: nn.Module, group=None):
        super().__init__()
        self.inner = inner
        self.group = group

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        world_size = 1 if self.group is None else dist.get_world_size(self.group)
        if world_size == 1:
            return self.inner(hidden_states)

        rank = dist.get_rank(self.group)
        seq_len = hidden_states.shape[1]
        lengths = tuple(
            stop - start
            for start, stop in (sequence_slice(seq_len, r, world_size) for r in range(world_size))
        )
        start, stop = sequence_slice(seq_len, rank, world_size)

        # The slice is a VIEW, so the full hidden_states is not copied -- but note it is also
        # still alive, held by the caller. This shards the FFN's own activations (the (L, inter)
        # tensors, which are 3x larger than hidden), not its input.
        local = self.inner(hidden_states[:, start:stop])
        return all_gather_sequence(local, lengths, self.group)

    def extra_repr(self) -> str:
        world_size = 1 if self.group is None else dist.get_world_size(self.group)
        return f"world_size={world_size}"


def wrap_ffn_sequence_parallel(model: nn.Module, group=None) -> int:
    """
    Replace every decoder layer's ``mlp`` with a :class:`SequenceParallelFFN`.

    Returns the number wrapped. Raises if it finds none, because a flag that silently does
    nothing is the failure mode this codebase has been bitten by twice (gradient checkpointing
    gated on ``module.training``, liger's ``skip_logits`` gated on the same thing) -- and here it
    would surface only as an OOM with no indication why.

    Idempotent: an already-wrapped layer is skipped rather than double-wrapped, which would slice
    a slice and quietly train on ``1/N^2`` of the sequence.
    """
    from kvpress.presses.gqa_indexer.press import get_language_model

    layers = getattr(get_language_model(model), "layers", None)
    if layers is None:
        raise RuntimeError(
            "could not find decoder layers (expected model.model.layers); FFN sequence "
            "parallelism needs them to wrap each layer's mlp"
        )

    wrapped = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or isinstance(mlp, SequenceParallelFFN):
            continue
        layer.mlp = SequenceParallelFFN(mlp, group=group)
        wrapped += 1

    if wrapped == 0:
        raise RuntimeError(
            "no MLP modules were wrapped for sequence parallelism -- every layer either lacks "
            "`mlp` or was already wrapped. This flag would then save nothing while claiming to."
        )
    world_size = 1 if group is None else dist.get_world_size(group)
    logger.info(
        "FFN sequence parallelism: wrapped %d MLP(s) across %d rank(s). Each rank now retains "
        "1/%d of the FFN activations -- the largest single activation term (49%% of the total).",
        wrapped, world_size, world_size,
    )
    return wrapped


def unwrap_ffn_sequence_parallel(model: nn.Module) -> int:
    """Restore the original ``mlp`` modules; returns how many were unwrapped."""
    from kvpress.presses.gqa_indexer.press import get_language_model

    layers = getattr(get_language_model(model), "layers", [])
    count = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, SequenceParallelFFN):
            layer.mlp = mlp.inner
            count += 1
    return count


@contextmanager
def ffn_sequence_parallel(model: nn.Module, group=None):
    """Wrap the FFNs for the duration of the block, restoring them on exit."""
    wrap_ffn_sequence_parallel(model, group=group)
    try:
        yield model
    finally:
        unwrap_ffn_sequence_parallel(model)
