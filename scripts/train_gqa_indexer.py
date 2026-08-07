# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer by attention distillation on the longmino corpus.

    python -m scripts.train_gqa_indexer \\
        --data-root /path/to/longmino_256k_filtered \\
        --model Qwen/Qwen3-8B --stage dense --schedule 8192:200,32768:800

Only the indexer is trained; the backbone is frozen, which is what makes the teacher a fixed
reference. ``freeze_all_but_indexer`` enforces that and raises if it matches nothing, so a
typo in ``--scorer-attr`` cannot silently train the whole model.

Two things this script does that a generic training loop would not:

**The length curriculum rebuilds the loader.** Stage 1 is ``O(L^2)`` in compute -- measured
3.9x per doubling on an H20 -- so 200 steps at 8K cost roughly what 13 steps at 32K do, while
still teaching the indexer the shape of attention. Changing ``seq_len`` changes which
documents are even eligible, so the dataloader is rebuilt at each stage boundary rather than
reused.

**It checkpoints only the indexer.** ``indexer_state_dict`` filters to the scorer parameters,
a few MB against a 16 GB backbone, and ``load_indexer_state_dict`` refuses a checkpoint whose
keys do not match the current geometry instead of silently loading nothing.

Stage 2 (``--stage sparse``) needs a fixed ``--topk``: deriving it from a keep-ratio makes the
retained support tensor ``O(L^2)``, which would cap length far below what the loss itself
allows. ``FusedIndexerTrainer`` refuses a ratio in that mode for the same reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer import (  # noqa: E402
    FusedIndexerTrainer,
    freeze_all_but_indexer,
    fused_indexer_training_step,
    indexer_state_dict,
    load_indexer_state_dict,
)
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    SUBSETS,
    LengthSchedule,
    LongminoConfig,
    build_dataloader,
    describe_subsets,
    env_rank_and_world_size,
)

logger = logging.getLogger("train_gqa_indexer")


def build_model(name: str, dtype: torch.dtype, attn: str):
    """
    Load the frozen backbone.

    ``eval()`` and ``requires_grad_(False)`` are both applied: eval disables dropout, which
    would otherwise make the teacher stochastic and the distillation target noisy, and the
    grad flag is a second line of defence behind ``freeze_all_but_indexer``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        # `dtype` replaced `torch_dtype` mid-2025; accept either rather than pinning a version.
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = attn

    model = model.to("cuda").eval()
    model.requires_grad_(False)
    return model, tokenizer


def build_optimizer(params, args):
    """AdamW plus a linear-warmup/cosine-decay schedule over the whole curriculum."""
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    total = args.total_steps
    warmup = max(1, int(total * args.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0))
        )

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def loader_for(seq_len: int, args, tokenizer, rank: int, world_size: int):
    """A loader for one stage of the curriculum."""
    config = LongminoConfig(
        root=args.data_root,
        subsets=tuple(args.subsets),
        seq_len=seq_len,
        min_tokens=args.min_tokens or seq_len,
        take_from=args.take_from,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed + seq_len,  # a different stream per stage, still reproducible
    )
    return build_dataloader(
        config,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )


def save(path: Path, model, args, step: int, extra: dict | None = None) -> None:
    """Write the indexer weights plus enough metadata to know what produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "model": args.model,
            "stage": args.stage,
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "topk": args.topk,
            "teacher_mode": args.teacher_mode,
            "lr": args.lr,
            "seed": args.seed,
        },
    }
    if extra:
        payload["metrics"] = extra
    torch.save(payload, path)
    logger.info("saved %s (step %d)", path, step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True, help="longmino_256k_filtered root")
    data.add_argument(
        "--subsets",
        nargs="+",
        default=["2e15", "2e16", "8k_32k", "synth_cwe", "synth_rex"],
        choices=list(SUBSETS),
        help="2e17 is excluded by default: its median is 168K tokens, so at seq_len<=32K most "
        "of it would be read and discarded. Add it for long-context stages.",
    )
    data.add_argument("--take-from", choices=("head", "random"), default="random")
    data.add_argument("--shuffle-buffer", type=int, default=64)
    data.add_argument("--min-tokens", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--batch-size", type=int, default=1)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen3-8B")
    model_group.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16",
        help="fp32 is rejected: flash-attention has no fp32 kernel, and capture_teacher_lse "
        "refuses rather than casting behind your back.",
    )
    model_group.add_argument("--attn", default="sdpa", help="backbone attention kernel")
    model_group.add_argument("--compression-ratio", type=float, default=0.5)
    model_group.add_argument("--rope-dim", type=int, default=None)
    model_group.add_argument("--head-dim", type=int, default=None)
    model_group.add_argument("--n-heads", type=int, default=None)
    model_group.add_argument("--scorer-attr", default="indexer")
    model_group.add_argument("--init-from", default=None, help="resume indexer weights")

    loss = parser.add_argument_group("objective")
    loss.add_argument("--stage", choices=("dense", "sparse"), default="dense")
    loss.add_argument("--topk", type=int, default=0, help="required for --stage sparse")
    loss.add_argument("--teacher-mode", choices=("global", "support"), default="global")
    loss.add_argument("--force-local", type=int, default=0)
    loss.add_argument("--force-sink", type=int, default=0)
    loss.add_argument("--skip-sink-in-loss", type=int, default=0)
    loss.add_argument("--loss-coeff", type=float, default=1.0)
    loss.add_argument("--key-tile", type=int, default=512)
    loss.add_argument("--query-tile", type=int, default=512)
    loss.add_argument("--topk-tile", type=int, default=128)
    loss.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:200,32768:800", help="SEQ_LEN:STEPS,...")
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--min-lr-frac", type=float, default=0.1)
    optim.add_argument("--warmup-frac", type=float, default=0.03)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("io")
    io.add_argument("--out", default="checkpoints/gqa_indexer")
    io.add_argument("--save-every", type=int, default=200)
    io.add_argument("--log-every", type=int, default=10)
    io.add_argument("--metrics-file", default=None, help="append JSONL metrics here")
    io.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.stage == "sparse" and args.topk <= 0:
        parser.error(
            "--stage sparse needs an explicit --topk. Deriving it from a keep-ratio makes the "
            "retained support O(L^2) -- 696 GB across 36 layers at L=32K, ratio 0.25 -- which "
            "would cap sequence length far below what the loss itself allows."
        )
    if not torch.cuda.is_available():
        parser.error("no CUDA device; indexer distillation needs a GPU")

    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size = env_rank_and_world_size()

    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.out)

    logger.info("subsets under %s:\n%s", args.data_root, describe_subsets(args.data_root))
    logger.info(
        "schedule: %s (%d steps total)",
        ", ".join(f"{n}x{s}" for s, n in schedule.stages),
        schedule.total_steps,
    )

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn)

    press_kwargs = {"compression_ratio": args.compression_ratio, "scorer_attr": args.scorer_attr}
    for name in ("rope_dim", "head_dim", "n_heads"):
        value = getattr(args, name)
        if value is not None:
            press_kwargs[name] = value
    press = GQAIndexerPress(**press_kwargs)
    press.post_init_from_model(model)

    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu")
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        logger.info("initialized indexer from %s", args.init_from)

    params = freeze_all_but_indexer(model, args.scorer_attr)
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable %.2fM of %.2fB parameters (%.3f%%)",
        trainable / 1e6,
        total / 1e9,
        100 * trainable / total,
    )

    trainer = FusedIndexerTrainer(
        press=press,
        stage=args.stage,
        key_tile=args.key_tile,
        query_tile=args.query_tile,
        topk_tile=args.topk_tile,
        topk=args.topk or None,
        teacher_mode=args.teacher_mode,
        force_sink=args.force_sink,
        force_local=args.force_local,
        skip_sink_in_loss=args.skip_sink_in_loss,
        loss_coeff=args.loss_coeff,
        backend=args.backend,
    )
    optimizer, lr_schedule = build_optimizer(params, args)

    metrics_handle = open(args.metrics_file, "a") if args.metrics_file else None
    current_len, loader, iterator = None, None, None
    window: list[float] = []
    started = time.time()
    step = 0

    try:
        for step, seq_len in schedule.lengths():
            if seq_len != current_len:
                # A new length means new eligibility, so the loader is rebuilt rather than
                # reused; iterating a stale loader would keep the old seq_len silently.
                logger.info("step %d: switching to seq_len=%d", step, seq_len)
                loader = loader_for(seq_len, args, tokenizer, rank, world_size)
                iterator = iter(loader)
                current_len = seq_len

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(args.accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    # The corpus is finite; restart rather than ending the run early.
                    logger.info("step %d: corpus exhausted at seq_len=%d, restarting", step, seq_len)
                    iterator = iter(loader)
                    batch = next(iterator)

                input_ids = batch["input_ids"].to("cuda", non_blocking=True)
                loss, per_layer = fused_indexer_training_step(
                    model, trainer, input_ids=input_ids
                )
                (loss / args.accum_steps).backward()
                accumulated += float(loss) / args.accum_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            window.append(accumulated)
            if step % args.log_every == 0 or step == args.total_steps - 1:
                recall = trainer.mean_recall()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "step %4d/%d L=%-6d loss %.4f (avg %.4f) |g| %.3f lr %.2e "
                    "peak %.1f GiB %s%.1f s/step",
                    step,
                    args.total_steps,
                    seq_len,
                    accumulated,
                    sum(window) / len(window),
                    float(grad_norm),
                    lr_schedule.get_last_lr()[0],
                    peak,
                    f"recall {recall:.3f} " if recall is not None else "",
                    (time.time() - started) / (step + 1),
                )
                if metrics_handle:
                    metrics_handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "seq_len": seq_len,
                                "loss": accumulated,
                                "grad_norm": float(grad_norm),
                                "lr": lr_schedule.get_last_lr()[0],
                                "recall": recall,
                                "peak_gib": peak,
                                "backend": trainer.backend_used,
                                "per_layer": {str(k): float(v) for k, v in per_layer.items()},
                            }
                        )
                        + "\n"
                    )
                    metrics_handle.flush()
                window = window[-50:]

            if rank == 0 and args.save_every and (step + 1) % args.save_every == 0:
                save(out_dir / f"step{step + 1}.pt", model, args, step + 1,
                     {"loss": accumulated})

            if args.dry_run and step >= 1:
                logger.info("dry run complete")
                break
    finally:
        if metrics_handle:
            metrics_handle.close()

    if rank == 0:
        save(out_dir / "final.pt", model, args, step + 1)
    logger.info("done in %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
