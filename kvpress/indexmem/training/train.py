# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from kvpress.indexmem.checkpoint import attach_scorers, load_scorer_checkpoint, save_scorer_checkpoint
from kvpress.indexmem.training.data import DocumentCache, parse_schedule
from kvpress.indexmem.training.distributed import (
    average_gradients,
    finish_distributed,
    sequence_parallel_group,
    setup_distributed,
)
from kvpress.indexmem.training.loop import train_loop
from kvpress.indexmem.training.optim import build_optimizer
from kvpress.indexmem.training.sequence_parallel import wrap_ffn_sequence_parallel
from kvpress.indexmem.training.trainer import RetentionTrainer


@dataclass
class TrainingConfig:
    model: str
    tokenized: str
    out: str
    subsets: list[str] = field(default_factory=lambda: ["2e15", "2e16", "synth_cwe", "synth_rex"])
    scorer: dict = field(default_factory=lambda: {"kind": "mlp", "mid_dim": 256, "decay": True, "age_scale": 16384.0})
    schedule: str = "8192:300,16384:300"
    objective: str = "reverse_kl"
    ce_weight: float = 0.0
    teacher_cache: str | None = None
    longce_cache: str | None = None
    gate_mass: float = 256.0
    sink_size: int = 4
    window_size: int = 128
    learning_rate: float = 1e-3
    final_lr: float = 5e-6
    weight_decay: float = 0.0
    warmup_fraction: float = 0.1
    stable_fraction: float = 0.6
    grad_clip: float = 1.0
    batch_size: int = 1
    global_batch_size: int = 8
    ffn_sp_size: int = 1
    workers: int = 2
    shuffle_buffer: int = 64
    seed: int = 1000
    model_seed: int = 42
    take_from: str = "head"
    chunk_size: int = 2048
    liger: bool = True
    log_every: int = 10
    save_every: int = 100


def build_model(config, device):
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    if config.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            model=model,
            rope=False,
            rms_norm=False,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )
    return model


def run_training(config, *, init_from=None, resume=None, max_steps=None):
    if (config.teacher_cache or config.longce_cache) and config.take_from != "head":
        raise ValueError("Document caches require take_from=head")
    rank, world_size, device = setup_distributed()
    group, data_rank, data_world_size = sequence_parallel_group(rank, world_size, config.ffn_sp_size)
    torch.manual_seed(config.model_seed)
    model = build_model(config, device)
    checkpoint = None
    if resume or init_from:
        checkpoint = load_scorer_checkpoint(model, resume or init_from)
        scorer_config = checkpoint["scorer_config"]
    else:
        scorer_config = {
            "hidden_size": model.config.hidden_size,
            "n_heads": model.config.num_key_value_heads,
            "gate_scale": True,
            **config.scorer,
        }
        attach_scorers(model, scorer_config)
    trainer = RetentionTrainer(
        model, gate_mass=config.gate_mass, sink_size=config.sink_size, window_size=config.window_size
    )
    parameters = trainer.freeze_backbone()
    if config.ffn_sp_size > 1:
        wrap_ffn_sequence_parallel(model, group)
    optimizer, schedule = build_optimizer(
        parameters,
        learning_rate=config.learning_rate,
        total_steps=sum(count for _, count in parse_schedule(config.schedule)),
        final_fraction=config.final_lr / config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        stable_fraction=config.stable_fraction,
    )
    start_step = 0
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["schedule"])
        start_step = checkpoint["step"]
    teacher = DocumentCache(config.teacher_cache, kind="teacher") if config.teacher_cache else None
    weights = DocumentCache(config.longce_cache, kind="longce") if config.objective == "longce" else None

    def loss_fn(input_ids, doc_ids):
        teacher_hidden = teacher.batch(doc_ids, input_ids.shape[1], device, torch.bfloat16) if teacher else None
        loss_weights = weights.batch(doc_ids, input_ids.shape[1], device, torch.float32) if weights else None
        return trainer.loss(
            input_ids,
            objective=config.objective,
            teacher_hidden=teacher_hidden,
            weights=loss_weights,
            ce_weight=config.ce_weight,
            chunk_size=config.chunk_size,
        )

    def update_fn():
        if world_size > 1:
            average_gradients(parameters, world_size)
        return torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)

    def save_fn(path, step):
        if rank == 0:
            save_scorer_checkpoint(
                path,
                model,
                scorer_config,
                step=step,
                optimizer=optimizer,
                schedule=schedule,
                training_config=asdict(config),
            )

    train_loop(
        config,
        optimizer,
        schedule,
        loss_fn,
        update_fn,
        save_fn,
        device=device,
        rank=rank,
        world_size=world_size,
        data_rank=data_rank,
        data_world_size=data_world_size,
        start_step=start_step,
        max_steps=max_steps,
    )
    finish_distributed(world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-from")
    source.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = TrainingConfig(**json.loads(Path(args.config).read_text()))
    run_training(config, init_from=args.init_from, resume=args.resume, max_steps=args.max_steps)
