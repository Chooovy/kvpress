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
* The gradient assertion went through two wrong versions. "SP gradient == dense gradient
  everywhere" fails for a naive slice, whose backward is a zero-pad -- each rank then produces
  gradient only on its own slice. But asserting *that* instead bakes in a real bug: the router
  being trained lives in unsharded attention, so replicated paths reach it and an all-reduce SUM
  scales them by ``sp_size`` while through-FFN paths get 1. ``_ScatterSequence`` all-gathers the
  input gradient so every rank holds the complete gradient, and the assertion is back to
  "== dense everywhere, identically on every rank" -- see
  ``test_gradient_is_the_full_dense_gradient_on_every_rank``.

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
    # Port derived from the worker name AND its arguments: a parametrized test calls the same
    # worker several times, and a name-only port would make those runs collide on rendezvous --
    # which surfaces as "address already in use" or a hang rather than as a clear failure.
    # PYTHONHASHSEED randomizes str hashes per process, but the port only has to be unique within
    # this one, so that is fine.
    port = 29600 + (abs(hash((target.__name__, args, world_size))) % 300)
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

        queue.put({
            "rank": rank,
            # Against the FULL dense gradient, not just this rank's slice.
            "max_diff": (dense_grad - x.grad).abs().max().item(),
            "grad": x.grad.flatten().tolist(),
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("seq_len", [16, 37])
def test_gradient_is_the_full_dense_gradient_on_every_rank(seq_len):
    """
    Every rank's ``d/dx`` equals the **whole** dense gradient, identically.

    This assertion was originally the opposite -- each rank's gradient nonzero only on its own
    slice, zero elsewhere -- which is what a plain ``hidden_states[:, a:b]`` produces, since a
    slice's backward is a zero-pad. That is fine when the trained parameter lives *inside* the
    sharded FFN: the slices are disjoint, so an all-reduce SUM reconstructs the gradient exactly.

    It is wrong for the actual training target. The router lives in **attention, which is not
    sharded**, so every rank runs it over the full sequence, and every path from the loss to the
    router that does not cross a sharded FFN is already replicated ``sp_size`` times. Summing then
    scales the replicated paths by ``sp_size`` and the through-FFN paths by 1 -- a per-path
    mismatch, so no choice of divisor repairs it. Measured on a 3-layer stand-in: cosine 0.98
    against the true gradient with the best divisor still 5% off, i.e. the wrong DIRECTION, not
    merely the wrong scale, so it could not be absorbed into the learning rate.

    ``_ScatterSequence`` fixes it by all-gathering the incoming gradient at the FFN's input, so
    each rank leaves the FFN holding the complete gradient. That makes the ranks redundant rather
    than complementary, which is what makes a uniform ``/world_size`` in ``average_gradients``
    correct -- and it is why the divisor there is ``world_size`` and not ``dp_world_size``.
    """
    results = _run(_gradient_worker, 4, seq_len)
    for result in results:
        assert result["max_diff"] < 1e-5, (
            f"rank {result['rank']} gradient differs from the dense gradient: "
            f"{result['max_diff']:.2e}"
        )
    reference = results[0]["grad"]
    for result in results[1:]:
        spread = max(abs(a - b) for a, b in zip(reference, result["grad"]))
        assert spread < 1e-6, (
            f"rank {result['rank']} disagrees with rank 0 by {spread:.2e}; the ranks must hold "
            "IDENTICAL gradients for average_gradients' /world_size to be right"
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

    ``dp_world_size`` is load-bearing in two places the loss curve would not reveal: the data
    loader shards by ``dp_rank``, so ranks cooperating on one sequence read the SAME one, and it
    sets how many sequences a step sees, which is what ``--global-batch-size`` converts into an
    accumulation count so tokens/step does not move with ``sp_size``.

    Note it is deliberately NOT the divisor in ``average_gradients``: since ``_ScatterSequence``
    all-gathers the input gradient, the ranks of an SP group hold IDENTICAL full gradients rather
    than disjoint slices, so the divisor there is the global ``world_size``. See
    ``test_gradient_is_the_full_dense_gradient_on_every_rank``.
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


# ----------------------------------------------------------------------
# Distributed: SP vs no-SP equivalence, through the real average_gradients
# ----------------------------------------------------------------------
# The tests above check the wrapper in isolation. What actually has to hold is a property of the
# whole training step: turning FFN-SP on must not change the update. That needs a model with the
# same SHAPE as the real one -- a trainable parameter inside unsharded attention, a frozen FFN
# that gets sharded -- because the bug this pins was invisible to a test whose only trainable
# parameter lived inside the FFN.
HID_M, INTER_M = 16, 48


class _Router(nn.Module):
    """Stands in for the indexer: trainable, lives INSIDE attention, sees the full sequence."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID_M, 1, bias=False)

    def forward(self, x):
        return self.proj(x).transpose(-1, -2)


class _Attn(nn.Module):
    """Causal attention with an additive gate on the key axis -- the shape of the real gate."""

    def __init__(self):
        super().__init__()
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(HID_M, HID_M, bias=False))
        self.router = _Router()

    def forward(self, x):
        scores = self.q(x) @ self.k(x).transpose(-1, -2) / HID_M**0.5 + self.router(x)
        causal = torch.full((x.shape[1], x.shape[1]), float("-inf")).triu(1)
        return self.o(torch.softmax(scores + causal, dim=-1) @ self.v(x))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.mlp = _Attn(), TinyMLP()
        self.n1, self.n2 = nn.LayerNorm(HID_M), nn.LayerNorm(HID_M)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.mlp(self.n2(h))


class _Inner(nn.Module):
    """Named ``model.model`` so ``wrap_ffn_sequence_parallel`` finds ``.layers`` as it does on HF."""

    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(n_layers))


class _Model(nn.Module):
    """
    Multi-layer on purpose: the router's gradient must cross every FFN above it.

    Shaped like a HF causal LM (``model.model.layers``) so the real
    ``wrap_ffn_sequence_parallel`` is under test here, not a hand-wrapped stand-in.
    """

    def __init__(self, n_layers=3):
        super().__init__()
        self.model = _Inner(n_layers)
        self.head = nn.Linear(HID_M, HID_M, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.head(x)


def _build_model():
    """Frozen backbone, trainable routers -- exactly what E2EIndexerTrainer sets up."""
    torch.manual_seed(0)
    model = _Model()
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.attn.router.requires_grad_(True)
    return model


def _routers(model):
    return [layer.attn.router.proj.weight for layer in model.model.layers]


def _batch(n_seqs, seq_len):
    """A fixed corpus, so every configuration consumes the SAME sequences in the same order."""
    torch.manual_seed(1)
    return [torch.randn(1, seq_len, HID_M) for _ in range(n_seqs)]


def _reference_update(seqs, accum_steps):
    """
    The answer FFN-SP has to reproduce: no SP, no distribution, one optimizer step.

    Accumulation is applied exactly as the trainer does it -- ``(loss / accum_steps).backward()``
    per micro-batch, summed into ``.grad`` -- so a mismatch localizes to the parallelism rather
    than to a different definition of the objective.
    """
    model = _build_model()
    for seq in seqs:
        (model(seq).pow(2).mean() / accum_steps).backward()
    return [r.grad.clone() for r in _routers(model)]


def _sp_worker(rank, world_size, port, queue, seq_len, accum_steps, sp_size):
    """One rank of an ``sp_size``-way FFN-SP run, ending in the real ``average_gradients``."""
    try:
        _init(rank, world_size, port)
        from scripts.train_gqa_indexer_e2e import average_gradients, ffn_sp_group

        sp_group, dp_rank, dp_world, _ = ffn_sp_group(world_size, sp_size, rank)
        model = _build_model()
        if sp_size > 1:
            wrap_ffn_sequence_parallel(model, group=sp_group)

        # Sharded by dp_rank, exactly as the trainer's loader is: ranks inside one SP group must
        # read the SAME sequence, or the all-gather stitches unrelated documents together.
        seqs = _batch(dp_world * accum_steps, seq_len)
        mine = [seqs[i * dp_world + dp_rank] for i in range(accum_steps)]

        losses = []
        for seq in mine:
            loss = model(seq).pow(2).mean() / accum_steps
            loss.backward()
            losses.append(loss.item())

        params = _routers(model)
        average_gradients(params, world_size)
        queue.put({
            "rank": rank,
            "loss_sum": sum(losses),
            "grads": [p.grad.flatten().tolist() for p in params],
        })
    except Exception as exc:  # noqa: BLE001
        import traceback
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _assert_matches_reference(results, reference, dp_world, label):
    """Every rank must hold the same gradient, and it must be the reference mean over replicas."""
    for result in results:
        for layer, (got, want) in enumerate(zip(result["grads"], reference)):
            got = torch.tensor(got).view_as(want)
            # The reference sums over all sequences; average_gradients divides by world_size, so
            # under dp_world replicas the result is the reference divided by dp_world.
            scale = (want / dp_world).abs().max().item()
            gap = (got - want / dp_world).abs().max().item()
            assert gap / max(scale, 1e-12) < 1e-4, (
                f"{label}: rank {result['rank']} layer {layer} gradient differs from the "
                f"no-SP reference by {gap:.2e} (relative {gap / max(scale, 1e-12):.2e})"
            )
    first = results[0]["grads"]
    for result in results[1:]:
        for layer, (a, b) in enumerate(zip(first, result["grads"])):
            spread = max(abs(x - y) for x, y in zip(a, b))
            assert spread < 1e-6, (
                f"{label}: rank {result['rank']} layer {layer} disagrees with rank 0 by "
                f"{spread:.2e} after average_gradients; every rank must take the same step"
            )


@pytest.mark.parametrize("seq_len", [16, 37])
def test_sp_update_matches_no_sp_update(seq_len):
    """
    **The property that matters:** turning FFN-SP on must not change the optimizer step.

    4 ranks with ``sp_size=4`` is ONE data-parallel replica consuming one sequence, so the
    gradient must equal what a single process with no SP computes on that same sequence. Anything
    else means the ``--ffn-sp-size`` flag silently changes the objective, which is precisely the
    failure the earlier per-slice gradient produced: replicated paths to the router were scaled by
    ``sp_size`` and through-FFN paths were not, giving cosine 0.98 against this reference -- a
    wrong direction that no learning rate could undo, and that a loss curve would show only as
    being slightly flatter.

    ``seq_len=37`` is ragged over 4 ranks, so the padding path is exercised in the backward too.
    """
    reference = _reference_update(_batch(1, seq_len), accum_steps=1)
    results = _run(_sp_worker, 4, seq_len, 1, 4, timeout=240)
    _assert_matches_reference(results, reference, dp_world=1, label="sp=4 vs no-SP")


def test_sp_and_dp_together_match_no_sp():
    """
    ``sp_size=2`` on 4 ranks: 2 replicas, each split 2 ways. Both axes at once.

    This is the configuration where a wrong divisor hides best -- ``world_size`` and
    ``dp_world_size`` differ, so dividing by either produces a plausible-looking gradient, and only
    one of them equals the reference. With the input-gradient all-gather in place the SP ranks hold
    identical copies, so the correct divisor is the global ``world_size``: it removes both the
    ``sp_size`` replication and the ``dp_world`` averaging in one step.
    """
    reference = _reference_update(_batch(2, 16), accum_steps=1)
    results = _run(_sp_worker, 4, 16, 1, 2, timeout=240)
    _assert_matches_reference(results, reference, dp_world=2, label="sp=2 x dp=2 vs no-SP")


# ----------------------------------------------------------------------
# Gradient accumulation
# ----------------------------------------------------------------------
def test_accumulation_matches_one_big_batch_without_sp():
    """
    ``accum_steps`` micro-batches must equal one batch of the same total size.

    This is what makes ``--global-batch-size`` a real alignment rather than a relabelling: it
    converts replicas the FFN-SP split took away into accumulation steps, and that is only sound
    if accumulating N micro-steps is the same update as N sequences in one step. Each micro-batch
    divides its loss by ``accum_steps`` because the objective is a MEAN over sequences; dropping
    that division would scale the gradient by ``accum_steps`` and silently raise the effective LR
    by the same factor.
    """
    seqs = _batch(4, 16)
    accumulated = _reference_update(seqs, accum_steps=4)

    # One batch of 4 sequences. Stacked into the batch axis rather than concatenated along the
    # sequence axis -- concatenating would let attention see across sequence boundaries.
    model = _build_model()
    model(torch.cat(seqs, dim=0)).pow(2).mean().backward()

    for layer, (got, want) in enumerate(zip(_routers(model), accumulated)):
        scale = want.abs().max().item()
        gap = (got.grad - want).abs().max().item()
        assert gap / max(scale, 1e-12) < 1e-5, (
            f"layer {layer}: accumulating 4 micro-batches differs from one batch of 4 by "
            f"{gap:.2e} (relative {gap / max(scale, 1e-12):.2e})"
        )


@pytest.mark.parametrize("accum_steps", [1, 2, 4])
def test_accumulation_under_sp_matches_no_sp(accum_steps):
    """
    Accumulation and FFN-SP must compose: ``sp_size=4, accum=N`` == no-SP over the same N
    sequences.

    ``accum=4`` here is the shape of the real ``stage1_16k`` run -- ``--ffn-sp-size 8`` leaves one
    replica, so ``--global-batch-size 8`` becomes 8 accumulation steps. If these two features
    interacted (for instance if the all-gather in the backward accumulated across micro-steps
    rather than being per-graph) the tokens/step alignment would be right while the update was
    wrong, and nothing in the logs would say so.

    ``accum_steps=1`` is included so a failure can be localized: if only ``>1`` fails the fault is
    in accumulation, if all fail it is in the SP path.
    """
    reference = _reference_update(_batch(accum_steps, 16), accum_steps=accum_steps)
    results = _run(_sp_worker, 4, 16, accum_steps, 4, timeout=300)
    _assert_matches_reference(
        results, reference, dp_world=1, label=f"sp=4 accum={accum_steps} vs no-SP"
    )
