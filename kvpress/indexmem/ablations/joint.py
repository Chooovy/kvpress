# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import functools
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from kvpress.indexmem.checkpoint import load_scorer_checkpoint
from kvpress.indexmem.training.data import parse_schedule
from kvpress.indexmem.training.distributed import finish_distributed, sequence_parallel_group, setup_distributed
from kvpress.indexmem.training.loop import train_loop
from kvpress.indexmem.training.optim import build_optimizer
from kvpress.indexmem.training.sequence_parallel import wrap_ffn_sequence_parallel
from kvpress.indexmem.training.train import TrainingConfig, build_model
from kvpress.indexmem.training.trainer import RetentionTrainer


@dataclass
class JointTrainingConfig(TrainingConfig):
    objective: str = "ce"
    subsets: list[str] = field(default_factory=lambda: ["2e16", "2e17"])
    schedule: str = "8192:300"
    backbone_lr: float = 2e-5
    scorer_lr_multiplier: float = 5.0
    final_lr_fraction: float = 0.01


def joint_parameters(model):
    model.requires_grad_(False)
    scorer_parameters = []
    backbone_parameters = []
    for layer in model.model.layers:
        scorer = layer.self_attn.retention_scorer
        scorer.gate_scale.data = scorer.gate_scale.data.to(layer.self_attn.q_proj.weight.dtype)
        for name, parameter in scorer.named_parameters():
            if name != "gate_scale":
                parameter.requires_grad_(True)
                scorer_parameters.append(parameter)
        scorer_ids = {id(parameter) for parameter in scorer.parameters()}
        for parameter in layer.self_attn.parameters():
            if id(parameter) not in scorer_ids:
                parameter.requires_grad_(True)
                backbone_parameters.append(parameter)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            for parameter in norm.parameters():
                parameter.requires_grad_(True)
                backbone_parameters.append(parameter)
    return scorer_parameters, backbone_parameters


def wrap_joint_model(model, world_size):
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    mesh = init_device_mesh(
        "cuda", (world_size // local_world_size, local_world_size), mesh_dim_names=("replicate", "shard")
    )
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={type(model.model.layers[0])},
        ),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        process_group=(mesh.get_group("shard"), mesh.get_group("replicate")),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
        ),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )


def run_joint_training(config, init_from, *, max_steps=None):
    if config.objective != "ce":
        raise ValueError("Joint training supports the CE ablation")
    rank, world_size, device = setup_distributed()
    group, data_rank, data_world_size = sequence_parallel_group(rank, world_size, config.ffn_sp_size)
    torch.manual_seed(config.model_seed)
    model = build_model(config, device)
    checkpoint = load_scorer_checkpoint(model, init_from)
    trainer = RetentionTrainer(
        model, gate_mass=config.gate_mass, sink_size=config.sink_size, window_size=config.window_size
    )
    scorer_parameters, backbone_parameters = joint_parameters(model)
    if config.ffn_sp_size > 1:
        wrap_ffn_sequence_parallel(model, group)
    wrapped = wrap_joint_model(model, world_size)
    optimizer, schedule = build_optimizer(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr},
            {"params": scorer_parameters, "lr": config.backbone_lr * config.scorer_lr_multiplier},
        ],
        learning_rate=config.backbone_lr,
        total_steps=sum(count for _, count in parse_schedule(config.schedule)),
        final_fraction=config.final_lr_fraction,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        stable_fraction=config.stable_fraction,
    )

    def loss_fn(input_ids, doc_ids):
        with trainer.hooks():
            extra = {"skip_logits": True} if config.liger else {}
            return wrapped(input_ids=input_ids, labels=input_ids, use_cache=False, **extra).loss

    def save_fn(path, step):
        with FSDP.state_dict_type(
            wrapped, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            state = wrapped.state_dict()
        if rank == 0:
            state = {key.replace(".mlp.inner.", ".mlp."): value for key, value in state.items()}
            scorers = {}
            for layer in range(model.config.num_hidden_layers):
                prefix = f"model.layers.{layer}.self_attn.retention_scorer."
                scorers.update(
                    {f"{layer}.{key[len(prefix):]}": value for key, value in state.items() if key.startswith(prefix)}
                )
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "scorer_config": checkpoint["scorer_config"],
                    "scorers": scorers,
                    "backbone": state,
                    "step": step,
                    "training_config": asdict(config),
                },
                path,
            )

    train_loop(
        config,
        optimizer,
        schedule,
        loss_fn,
        lambda: wrapped.clip_grad_norm_(config.grad_clip),
        save_fn,
        device=device,
        rank=rank,
        world_size=world_size,
        data_rank=data_rank,
        data_world_size=data_world_size,
        max_steps=max_steps,
    )
    finish_distributed(world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = JointTrainingConfig(**json.loads(Path(args.config).read_text()))
    run_joint_training(config, args.init_from, max_steps=args.max_steps)
