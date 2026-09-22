# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from pathlib import Path

from kvpress.indexmem.training.data import document_loader, next_batch, parse_schedule
from kvpress.indexmem.training.distributed import mean_loss


def train_loop(
    config,
    optimizer,
    schedule,
    loss_fn,
    update_fn,
    save_fn,
    *,
    device,
    rank,
    world_size,
    data_rank,
    data_world_size,
    start_step=0,
    max_steps=None,
):
    stages = parse_schedule(config.schedule)
    total_steps = sum(count for _, count in stages)
    stop_step = min(total_steps, max_steps) if max_steps is not None else total_steps
    accumulation_steps = config.global_batch_size // (data_world_size * config.batch_size)
    started = time.monotonic()
    stage_start = 0
    output = Path(config.out)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    metrics = (output / "metrics.jsonl").open("a") if rank == 0 else None
    step = start_step
    try:
        for sequence_length, stage_steps in stages:
            stage_end = stage_start + stage_steps
            if stage_end <= start_step:
                stage_start = stage_end
                continue
            loader = document_loader(
                config.tokenized,
                sequence_length,
                config.subsets,
                batch_size=config.batch_size,
                workers=config.workers,
                seed=config.seed,
                shuffle_buffer=config.shuffle_buffer,
                rank=data_rank,
                world_size=data_world_size,
                take_from=config.take_from,
            )
            iterator = iter(loader)
            for _ in range(max(0, start_step - stage_start) * accumulation_steps):
                _, iterator = next_batch(loader, iterator)
            for step_index in range(max(stage_start, start_step), min(stage_end, stop_step)):
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0.0
                for _ in range(accumulation_steps):
                    batch, iterator = next_batch(loader, iterator)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    loss = loss_fn(input_ids, batch["doc_ids"])
                    (loss / accumulation_steps).backward()
                    accumulated += loss.detach().item() / accumulation_steps
                grad_norm = update_fn()
                optimizer.step()
                schedule.step()
                step = step_index + 1
                if step % config.log_every == 0 or step == stop_step:
                    loss = mean_loss(accumulated, device, world_size)
                    if metrics:
                        record = {
                            "step": step,
                            "sequence_length": sequence_length,
                            "loss": loss,
                            "grad_norm": float(grad_norm),
                            "learning_rates": schedule.get_last_lr(),
                            "tokens": config.global_batch_size * sequence_length,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        metrics.write(json.dumps(record) + "\n")
                        metrics.flush()
                        print(json.dumps(record), flush=True)
                if config.save_every and step % config.save_every == 0:
                    save_fn(output / f"step{step}.pt", step)
            stage_start = stage_end
            if step >= stop_step:
                break
        save_fn(output / "final.pt", step)
    finally:
        if metrics:
            metrics.close()
    return step
