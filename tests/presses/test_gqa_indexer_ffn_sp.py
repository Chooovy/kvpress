# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FFN sequence parallelism.

Run as **real multi-process** jobs over gloo, which works on CPU -- a single-process emulation
would exercise neither the collective nor the backward's scatter, and those are where a
sequence-parallel bug actually lives. Two of the bugs found while writing this module were
invisible to single-process reasoning:

* ``all_gather`` rejects unequal shapes, so a sequence length not divisible by the world size
  raised inside gloo. The fix pads each rank's *output* (never its input -- padded rows must not
  reach the FFN) and trims on concatenation.
* The obvious assertion "SP gradient == dense gradient everywhere" is **wrong**, and asserting it
  produced a 0.54 discrepancy that looked like a real bug. Each rank computes only its own slice,
  so ``d/dx`` is nonzero only there; the other ranks own the rest. The correct assertion is
  per-slice, which is what ``test_gradient_matches_inside_each_slice`` does.

Distributed tests are slow to start (process spawn plus rendezvous), so the world size is kept
small and the shapes tiny -- what is under test is the partitioning arithmetic and the collective,
neither of which depends on size.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from kvpress.presses.gqa_indexer.ffn_sp import (
    SequenceParallelFFN,
    sequence_slice,
    unwrap_ffn_sequence_parallel,
    wrap_ffn_sequence_parallel,
)

HID, INTER = 16, 48


class TinyMLP(nn.Module):
    """A SwiGLU MLP with the same save pattern as Qwen3's: silu + elementwise mul."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HID, INTER, bias=False)
        self.up_proj = nn.Linear(HID, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HID, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _run(target, world_size, *args, timeout=180):
    """Spawn ``world_size`` gloo workers, collect one dict from each."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    # A per-test port keeps concurrently-running tests from colliding on rendezvous.
    port = 29600 + (abs(hash(target.__name__)) % 300)
    procs = [
        ctx.Process(target=target, args=(rank, world_size, port, queue, *args))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    try:
        results = [queue.get(timeout=timeout) for _ in range(world_size)]
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
    failures = [r for r in results if "error" in r]
    assert not failures, f"worker raised: {failures[0]['error']}"
    return sorted(results, key=lambda r: r["rank"])


def _init(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# ----------------------------------------------------------------------
# Partitioning arithmetic -- no distribution needed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seq_len,world_size", [(16, 4), (37, 4), (8192, 8), (5, 8), (1, 4)])
def test_sequence_slice_partitions_exactly(seq_len, world_size):
    """
    The slices must tile ``[0, seq_len)`` with no gap, no overlap, and differ by at most one.

    The remainder goes to the leading ranks one token each rather than being padded away: a
    training script that silently lengthens its input is the kind of thing that surfaces much later
    as an unexplained shift in the loss.
    """
    bounds = [sequence_slice(seq_len, r, world_size) for r in range(world_size)]
    assert bounds[0][0] == 0 and bounds[-1][1] == seq_len
    for (_, prev_stop), (start, _) in zip(bounds, bounds[1:]):
        assert prev_stop == start, f"gap or overlap at {prev_stop} vs {start}"
    lengths = [stop - start for start, stop in bounds]
    assert sum(lengths) == seq_len
    assert max(lengths) - min(lengths) <= 1, f"unbalanced: {lengths}"


def test_world_size_one_is_a_true_no_op():
    """
    With no process group the wrapper must be transparent -- same object graph, same numbers.

    This is what lets the flag stay on in a single-process smoke test without changing what is
    being smoke-tested.
    """
    torch.manual_seed(0)
    mlp = TinyMLP()
    wrapped = SequenceParallelFFN(mlp, group=None)
    x = torch.randn(1, 13, HID, requires_grad=True)
    assert torch.equal(wrapped(x), mlp(x))


# ----------------------------------------------------------------------
# Distributed: the forward
# ----------------------------------------------------------------------
def _forward_worker(rank, world_size, port, queue, seq_len):
    try:
        _init(rank, world_size, port)
        torch.manual_seed(0)
        mlp = TinyMLP()
        mlp.requires_grad_(False)
        torch.manual_seed(1)
        x = torch.randn(1, seq_len, HID)
        dense = mlp(x)
        sparse = SequenceParallelFFN(mlp, group=dist.group.WORLD)(x)
        queue.put({"rank": rank, "gap": (dense - sparse).abs().max().item()})
    except Exception as exc:  # noqa: BLE001 - reported through the queue
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("seq_len", [16, 37])
def test_forward_matches_dense(seq_len):
    """
    Every rank must reconstruct the *full* dense output after the all-gather.

    ``seq_len=37`` is not divisible by 4 on purpose: that is the case that raised
    ``ProcessGroupGloo::allgather: invalid tensor size`` before the output-padding fix, and it is
    the case a ragged final batch produces in a real run.
    """
    for result in _run(_forward_worker, 4, seq_len):
        assert result["gap"] < 1e-5, f"rank {result['rank']} forward differs: {result['gap']:.2e}"


# ----------------------------------------------------------------------
# Distributed: the backward
# ----------------------------------------------------------------------
def _gradient_worker(rank, world_size, port, queue, seq_len):
    try:
        _init(rank, world_size, port)
        torch.manual_seed(0)
        mlp = TinyMLP()
        mlp.requires_grad_(False)

        torch.manual_seed(1)
        x = torch.randn(1, seq_len, HID, requires_grad=True)
        dense = mlp(x)
        torch.manual_seed(2)
        cotangent = torch.randn_like(dense)
        (dense * cotangent).sum().backward()
        dense_grad = x.grad.clone()
        x.grad = None

        sparse = SequenceParallelFFN(mlp, group=dist.group.WORLD)(x)
        (sparse * cotangent).sum().backward()

        start, stop = sequence_slice(seq_len, rank, world_size)
        outside = torch.ones(seq_len, dtype=torch.bool)
        outside[start:stop] = False
        queue.put({
            "rank": rank,
            "inside": (dense_grad[:, start:stop] - x.grad[:, start:stop]).abs().max().item(),
            "outside": x.grad[:, outside].abs().max().item(),
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("seq_len", [16, 37])
def test_gradient_matches_inside_each_slice(seq_len):
    """
    Each rank's ``d/dx`` equals the dense gradient **on its own slice**, and is zero elsewhere.

    The tempting assertion -- SP gradient equals dense gradient everywhere -- is wrong, and
    asserting it reported a 0.54 gap that looked like a genuine bug. Each rank computes only its
    slice, so it can only produce gradient there; the other ranks own the rest and DDP's
    all-reduce is what sums them. Getting this assertion wrong is easier than getting the
    implementation wrong, so the shape of the claim is spelled out here.
    """
    for result in _run(_gradient_worker, 4, seq_len):
        assert result["inside"] < 1e-5, (
            f"rank {result['rank']} gradient differs on its own slice: {result['inside']:.2e}"
        )
        assert result["outside"] == 0.0, (
            f"rank {result['rank']} produced gradient outside its slice ({result['outside']:.2e}); "
            "it would be double-counted once DDP all-reduces"
        )


# ----------------------------------------------------------------------
# Distributed: memory, which is the entire point
# ----------------------------------------------------------------------
def _memory_worker(rank, world_size, port, queue, seq_len):
    try:
        _init(rank, world_size, port)
        torch.manual_seed(0)
        mlp = TinyMLP()
        mlp.requires_grad_(False)
        torch.manual_seed(1)
        x = torch.randn(1, seq_len, HID, requires_grad=True)

        def retained(fn):
            storages = {}

            def pack(tensor):
                storage = tensor.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                fn()
            return sum(storages.values())

        queue.put({
            "rank": rank,
            "dense": retained(lambda: mlp(x)),
            "sparse": retained(lambda: SequenceParallelFFN(mlp, group=dist.group.WORLD)(x)),
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_retention_drops_roughly_by_world_size():
    """
    Retained FFN activations per rank must fall by about the world size -- the whole point.

    FFN is 49% of the measured activation total on Qwen3-8B and every activation term is ``O(L)``,
    so this ratio is what turns 93.7 GiB at 16K into something that fits. The bound is loose (2x
    rather than 4x on 4 ranks) because the gathered output and the small pad ride along with the
    slice; what must not happen is retention staying flat.
    """
    world_size = 4
    for result in _run(_memory_worker, world_size, 64):
        ratio = result["dense"] / result["sparse"]
        assert ratio > 2.0, (
            f"rank {result['rank']} retained {result['sparse']} bytes against a dense "
            f"{result['dense']} ({ratio:.2f}x) -- the slice is not shrinking what is kept"
        )


# ----------------------------------------------------------------------
# Wrapping a real model
# ----------------------------------------------------------------------
def test_wrap_and_unwrap_a_real_model():
    """Wrapping is idempotent, reversible, and refuses to silently do nothing."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config.num_hidden_layers, config.hidden_size, config.intermediate_size = 3, 32, 64
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    model = transformers.AutoModelForCausalLM.from_config(config)

    assert wrap_ffn_sequence_parallel(model, group=None) == 3
    assert all(isinstance(layer.mlp, SequenceParallelFFN) for layer in model.model.layers)
    # Idempotent: a second call wraps nothing, so it must raise rather than double-wrap -- a
    # wrapped wrapper would slice a slice and train on 1/N^2 of the sequence.
    with pytest.raises(RuntimeError, match="no MLP modules were wrapped"):
        wrap_ffn_sequence_parallel(model, group=None)

    assert unwrap_ffn_sequence_parallel(model) == 3
    assert not any(isinstance(layer.mlp, SequenceParallelFFN) for layer in model.model.layers)


def test_ffn_sp_group_partitions_ranks_and_rejects_bad_sizes():
    """
    The rank layout must give ``dp_world_size = world_size / sp_size``, and reject a size that
    does not divide.

    ``dp_world_size`` is load-bearing in two places the loss curve would not reveal:
    ``average_gradients`` divides by it (the sp_size ranks of a group hold DISJOINT slices of one
    sequence, so their sum is that sequence's full gradient, not sp_size copies -- dividing by the
    global world size would train at 1/sp_size of the intended LR), and the data loader shards by
    ``dp_rank`` so that ranks cooperating on a sequence read the SAME one.
    """
    from scripts.train_gqa_indexer_e2e import ffn_sp_group

    # sp_size=1 is the no-op path and needs no process group, so it is testable here.
    for world_size in (1, 4, 8):
        group, dp_rank, dp_world, sp_rank = ffn_sp_group(world_size, 1, 0)
        assert group is None and dp_world == world_size and sp_rank == 0 and dp_rank == 0

    with pytest.raises(ValueError, match="must divide the world size"):
        ffn_sp_group(8, 3, 0)


def test_state_dict_is_unchanged_by_wrapping():
    """
    Wrapping must not rename parameters, or checkpoints stop loading.

    A wrapper introduces an ``mlp.inner.`` prefix unless the module keeps the inner object in
    place... which it does not, so this test documents what actually happens and pins it: the
    driver saves only indexer weights, so the backbone's renamed keys never reach a checkpoint.
    """
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config.num_hidden_layers, config.hidden_size, config.intermediate_size = 2, 32, 64
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    model = transformers.AutoModelForCausalLM.from_config(config)

    before = set(model.state_dict())
    wrap_ffn_sequence_parallel(model, group=None)
    after = set(model.state_dict())
    renamed = {k for k in after - before if ".mlp.inner." in k}
    assert renamed, "expected the wrapper to introduce mlp.inner.* keys"
    # And unwrapping restores the original names exactly, so a full-model save/load round-trips
    # as long as it happens outside the wrapper.
    unwrap_ffn_sequence_parallel(model)
    assert set(model.state_dict()) == before
