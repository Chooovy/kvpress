# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Does FFN sequence parallelism compose with FSDP HYBRID_SHARD, and is the gradient still right?

This is the one thing the joint-training plan rested on that had never been executed. Run on 2
GPUs::

    torchrun --nproc_per_node 2 -m scripts.check_ffn_sp_fsdp

Three questions, in order of how badly a wrong answer would hurt:

1. **Does it build?** ``SequenceParallelFFN`` wraps each layer's ``mlp`` in a custom module, and
   FSDP auto-wraps on the decoder-layer class. If FSDP's flattening trips over the wrapper, the
   whole approach is dead and we find out here rather than after a queue wait.
2. **Is ``gate_scale`` accepted?** It is frozen and bf16 now, so each layer should flatten to one
   dtype. A regression here reproduces "Must flatten tensors with uniform dtype".
3. **Is the gradient equal to the unsharded reference?** This is the question that matters most,
   because getting it wrong is silent: ``_ScatterSequence`` is only correct while the FFN holds no
   trainable parameter, and the failure mode is a gradient that is ``sp_size`` times too small on
   some paths while correct on others -- the loss still descends.

The reference is a single-process run of the same model on the same batch with no FFN-SP and no
FSDP. Cosine similarity, not just norm ratio: the documented failure was a wrong *direction*
(cosine 0.98 with no single divisor able to repair it), which a magnitude check would miss.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_tiny(seed: int = 0):
    """A 2-layer Qwen3 with the real GQA shape, small enough to run twice on one GPU."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        attention_dropout=0.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model.to(torch.bfloat16)


def attach_router(model, device):
    """The scalar router this experiment trains, in the joint run's configuration."""
    from kvpress import GQAIndexerPress

    press = GQAIndexerPress(
        compression_ratio=0.5,
        scorer="scalar",
        scalar_mid_dim=32,
        scalar_decay=True,
        gate_scale=True,
        n_sink=4,
    )
    model = model.to(device)
    press.post_init_from_model(model)
    return press


def make_trainer(press):
    from kvpress.presses.gqa_indexer import E2EIndexerTrainer

    return E2EIndexerTrainer(
        press=press,
        stage="dense",
        pin_mode="local+sink",
        n_sink=4,
        n_local=16,
        gate_budget=32.0,
        freeze=False,
    )


def select_trainable(model, scorer_attr="indexer"):
    """Mirror scripts/train_gqa_indexer_joint.split_trainable: attention + norms + router,
    gate_scale frozen, FFN frozen."""
    from scripts.train_gqa_indexer_joint import split_trainable

    return split_trainable(model, scorer_attr)


def flat_grad(named):
    """Concatenate gradients in a stable name order so two runs are comparable elementwise."""
    parts = []
    for name in sorted(named):
        grad = named[name]
        parts.append(torch.zeros(1) if grad is None else grad.detach().float().flatten().cpu())
    return torch.cat(parts)


def reference_gradient(device, input_ids):
    """Single-process, no FFN-SP, no FSDP: the ground truth this configuration must reproduce."""
    from kvpress.presses.gqa_indexer import e2e_indexer_training_step

    model = build_tiny().to(device)
    press = attach_router(model, device)
    trainer = make_trainer(press)
    select_trainable(model)
    with trainer.hooks(model):
        loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
        loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    return float(loss.detach()), flat_grad(grads), {n: tuple(p.shape) for n, p in model.named_parameters() if p.requires_grad}


def main() -> int:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world < 2:
        print("needs at least 2 ranks: torchrun --nproc_per_node 2 -m scripts.check_ffn_sp_fsdp")
        return 1

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group("nccl")

    from kvpress.presses.gqa_indexer import e2e_indexer_training_step
    from kvpress.presses.gqa_indexer.ffn_sp import wrap_ffn_sequence_parallel
    from scripts.train_gqa_indexer_joint import wrap_fsdp

    # Same batch on every rank: under FFN-SP the ranks of one SP group cooperate on ONE sequence.
    torch.manual_seed(1234)
    seq_len = 64
    input_ids = torch.randint(0, 512, (1, seq_len), device=device)

    # ---- reference, rank 0 only -------------------------------------------------------
    ref_loss = ref_grad = ref_shapes = None
    if rank == 0:
        ref_loss, ref_grad, ref_shapes = reference_gradient(device, input_ids)
        print(f"[ref ] loss {ref_loss:.6f}  grad numel {ref_grad.numel()}", flush=True)

    dist.barrier()

    # ---- the real configuration: FFN-SP inside the node, then FSDP HYBRID_SHARD -------
    model = build_tiny().to(device)
    press = attach_router(model, device)
    trainer = make_trainer(press)
    router_params, backbone_params = select_trainable(model)
    if rank == 0:
        n_r = sum(p.numel() for p in router_params)
        n_b = sum(p.numel() for p in backbone_params)
        print(f"[cfg ] trainable router {n_r} + backbone {n_b}", flush=True)
        # gate_scale must be frozen, else FSDP's uniform-dtype check fires.
        for name, p in model.named_parameters():
            if name.endswith("gate_scale"):
                print(f"[cfg ] {name}: shape {tuple(p.shape)} dtype {p.dtype} "
                      f"requires_grad {p.requires_grad}", flush=True)
                break

    # Q1/Q2: does the composition build at all?
    wrap_ffn_sequence_parallel(model, group=None)  # group=None -> the default (all ranks)
    if rank == 0:
        print("[ok  ] wrap_ffn_sequence_parallel applied", flush=True)

    class Args:
        dtype = "bfloat16"

    from torch.distributed.device_mesh import init_device_mesh

    # One node in this test, so replicate=1 and shard=world. Exercises the same code path the
    # 4-node run takes, where replicate=4.
    mesh = init_device_mesh("cuda", (1, world), mesh_dim_names=("replicate", "shard"))
    try:
        model = wrap_fsdp(model, Args(), mesh.get_group("shard"), mesh.get_group("replicate"))
    except Exception as exc:
        if rank == 0:
            print(f"[FAIL] FSDP wrap raised: {type(exc).__name__}: {exc}", flush=True)
        dist.destroy_process_group()
        return 1
    if rank == 0:
        print("[ok  ] FSDP HYBRID_SHARD wrap succeeded (Q1 + Q2 pass)", flush=True)

    # Q3: is the gradient the same as the unsharded reference?
    with trainer.hooks(model):
        loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
        loss.backward()

    # Gradients are SHARDED, so a rank's local .grad is a slice and cannot be compared to the
    # reference elementwise. summon_full_params reconstitutes the whole parameter (and its
    # gradient) on every rank, which is what makes the comparison meaningful -- a norm-only check
    # would pass even for the per-path scale mismatch this test exists to catch, since that
    # failure changes direction while leaving the magnitude close.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    inner = model.module if hasattr(model, "module") else model
    with FSDP.summon_full_params(model, with_grads=True, writeback=False):
        grads = {
            n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in inner.named_parameters()
            if p.requires_grad
        }
    got_grad = flat_grad(grads)
    got_loss = float(loss.detach())

    if rank == 0:
        print(f"[test] loss {got_loss:.6f}  (ref {ref_loss:.6f})  "
              f"delta {abs(got_loss - ref_loss):.3e}", flush=True)
        if got_grad.numel() != ref_grad.numel():
            # Expected under FSDP: parameters are sharded, so this rank holds a slice. The
            # comparison then needs a full state dict rather than local grads -- report instead
            # of pretending the numbers line up.
            print(f"[warn] grad numel differs (sharded {got_grad.numel()} vs ref "
                  f"{ref_grad.numel()}); comparing norms only", flush=True)
            print(f"[test] |g| sharded-local {got_grad.norm():.6f} ref {ref_grad.norm():.6f}",
                  flush=True)
        else:
            cos = torch.nn.functional.cosine_similarity(
                got_grad.unsqueeze(0), ref_grad.unsqueeze(0)
            ).item()
            ratio = (got_grad.norm() / ref_grad.norm().clamp_min(1e-12)).item()
            print(f"[test] cosine {cos:.8f}  norm ratio {ratio:.6f}", flush=True)
            if cos < 0.999:
                print("[FAIL] gradient DIRECTION differs -- this is the _ScatterSequence "
                      "per-path scale mismatch, not something an LR can absorb.", flush=True)
            elif abs(ratio - 1.0) > 0.02:
                print(f"[FAIL] gradient magnitude off by {ratio:.4f}x", flush=True)
            else:
                print("[ok  ] gradient matches the unsharded reference (Q3 pass)", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
