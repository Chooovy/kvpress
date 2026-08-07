# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer by attention distillation on the longmino corpus.

    # once: remove tokenization from the training loop
    python -m scripts.pretokenize_longmino --data-root RAW --out TOK --seq-len 65536

    # stage 1 at 32K, stage 2 at 64K
    python -m scripts.train_gqa_indexer --data-root RAW --tokenized TOK \\
        --stage dense --schedule 32768:1500

Only the indexer is trained; the backbone is frozen, which is what makes the teacher a fixed
reference. ``freeze_all_but_indexer`` enforces that and raises if it matches nothing, so a
typo in ``--scorer-attr`` cannot silently train the whole model.

Three things this script does that a generic training loop would not:

**WSD, not cosine.** Warmup 10% to ``--peak-lr``, hold 60%, then decay linearly to
``--final-lr`` on the last step. The stable phase can be lengthened without changing any other
step's value; with cosine, extending a run rescales the whole curve, so a resumed or extended
run is not comparable with the original.

**The length curriculum rebuilds the loader.** Changing ``seq_len`` changes which documents
are even eligible -- and, with ``--tokenized``, which slice of the stored array is read -- so
the loader is rebuilt at each stage boundary rather than reused. Reusing it would silently
keep the previous length.

**It checkpoints only the indexer.** ``indexer_state_dict`` filters to the scorer parameters,
a few MB against a 16 GB backbone, and ``load_indexer_state_dict`` refuses a checkpoint whose
keys do not match the current geometry instead of silently loading nothing.

Stage 2 (``--stage sparse``) needs a fixed ``--topk``: deriving it from a keep-ratio makes the
retained support tensor ``O(L^2)``, which would cap length far below what the loss itself
allows. ``FusedIndexerTrainer`` refuses a ratio in that mode for the same reason.

At 64K only ``2e16`` and ``2e17`` have documents long enough to qualify (measured: 100% of
each, versus 0% for ``2e15``/``synth_*``, whose medians sit near 43K). So the stage-2 subset
list is not a preference -- it is the only choice that yields any data at that length.
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
import torch.distributed as dist

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
    TokenizedConfig,
    build_dataloader,
    build_tokenized_dataloader,
    describe_subsets,
    env_rank_and_world_size,
    read_index,
    wsd_lr_lambda,
)

logger = logging.getLogger("train_gqa_indexer")


def setup_distributed() -> tuple[int, int, int]:
    """
    Join the process group if ``torchrun`` launched us, and bind this rank to its GPU.

    Returns ``(rank, world_size, local_rank)``. Single-process runs get ``(0, 1, 0)`` and no
    process group, so the same script runs both ways without a flag.

    ``set_device`` must happen before any CUDA allocation: NCCL binds the current device at
    init, and every rank defaulting to ``cuda:0`` is the classic way to get eight processes
    fighting over one GPU while the other seven idle.
    """
    rank, world_size = env_rank_and_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError(f"WORLD_SIZE={world_size} but torch.distributed is unavailable")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # NCCL for gradients; the 680 MB fp32 allreduce per step costs ~3 ms on NVLink,
            # against a step that is seconds long at these sequence lengths.
            dist.init_process_group(backend="nccl")
        logger.info(
            "rank %d/%d on cuda:%d (%s)",
            rank,
            world_size,
            local_rank,
            torch.cuda.get_device_name(local_rank),
        )
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_model(name: str, dtype: torch.dtype, attn: str, device: str):
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

    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def average_gradients(params, world_size: int) -> None:
    """
    All-reduce the indexer gradients to their mean across ranks.

    Done explicitly rather than via ``DistributedDataParallel``: DDP hooks a module's forward
    to know when to reduce, but the indexers are invoked from ``FusedIndexerTrainer``'s loss
    hook, not from the wrapped model's forward -- so DDP's autograd hooks would never fire for
    them. The reduction itself is what DDP would do anyway, and at 170M parameters (680 MB
    fp32, ~3 ms on NVLink) it is under 1% of a multi-second step.

    Gradients are flattened into one buffer for a single collective; eight separate small
    allreduces per layer would be latency-bound rather than bandwidth-bound.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        # Every rank must call this the same number of times or the collective deadlocks, so
        # an empty gradient list is a hard error rather than a silent skip.
        raise RuntimeError("no gradients to average; ranks would desynchronize")
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        grad.copy_(synced)


def all_reduce_mean(value: float, device: str) -> float:
    """Mean of a scalar across ranks, for logging."""
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item()) / dist.get_world_size()


def build_optimizer(params, args):
    """
    AdamW with a Warmup-Stable-Decay schedule.

    The optimizer is constructed at ``--peak-lr`` and ``wsd_lr_lambda`` returns a multiplier,
    so the peak is reached exactly rather than approached. WSD rather than cosine because the
    stable phase can be lengthened without changing any other step's value -- with cosine,
    extending a run rescales the entire curve.
    """
    optimizer = torch.optim.AdamW(
        params, lr=args.peak_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    lr_lambda = wsd_lr_lambda(
        args.total_steps,
        warmup_frac=args.warmup_frac,
        stable_frac=args.stable_frac,
        peak_lr=args.peak_lr,
        final_lr=args.final_lr,
    )
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def loader_for(seq_len: int, args, tokenizer, rank: int, world_size: int):
    """
    A loader for one stage of the curriculum.

    With ``--tokenized`` this reads pre-tokenized ``.npy`` shards, which removes the ~0.45 s
    per 64K sample that tokenization costs on the data path. Otherwise it tokenizes text on
    the fly, which is fine for short runs but throttles the GPU at long sequence lengths.
    """
    if args.tokenized:
        return build_tokenized_dataloader(
            TokenizedConfig(
                root=args.tokenized,
                seq_len=seq_len,
                subsets=tuple(args.subsets) if args.subsets else None,
                take_from=args.take_from,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed + seq_len,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )
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
            "peak_lr": args.peak_lr,
            "final_lr": args.final_lr,
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
        "--tokenized",
        default=None,
        help="read pre-tokenized .npy shards from here (scripts/pretokenize_longmino.py). "
        "Removes tokenization from the training loop entirely.",
    )
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
    optim.add_argument("--schedule", default="32768:1500", help="SEQ_LEN:STEPS,...")
    optim.add_argument("--peak-lr", type=float, default=1e-3, help="WSD plateau")
    optim.add_argument("--final-lr", type=float, default=5e-6, help="WSD floor, hit on the last step")
    optim.add_argument("--warmup-frac", type=float, default=0.10)
    optim.add_argument("--stable-frac", type=float, default=0.60)
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
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"

    # Only rank 0 logs at INFO; the rest would interleave eight copies of every line.
    # Warnings and errors stay visible everywhere, since a fault on rank 5 must not be silent.
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)

    # Distinct seeds so the ranks draw different windows -- an identical seed would make all
    # eight see the same documents, turning an 8x batch into an 8x-redundant one.
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.out)

    if args.tokenized:
        index = read_index(args.tokenized)
        if not index.get("complete", True):
            logger.warning(
                "%s/index.json is marked incomplete: some shards failed to pretokenize, so "
                "this run will see less data than the index claims",
                args.tokenized,
            )
        logger.info(
            "pre-tokenized corpus: %d docs at seq_len<=%d, subsets %s",
            index["total_docs"],
            index["seq_len"],
            index["subsets"],
        )
    else:
        logger.info("subsets under %s:\n%s", args.data_root, describe_subsets(args.data_root))
    total = schedule.total_steps
    # Sum over stages, not stages[0] x total: with a length curriculum the stages have
    # different seq_len, so extrapolating from the first one overstates a short warmup stage
    # and understates the long final one.
    tokens = sum(sl * st for sl, st in schedule.stages) * world_size * args.batch_size
    logger.info(
        "world_size %d x batch_size %d = %d sequences/step; %d optimizer steps over %s "
        "= %.0fM tokens",
        world_size,
        args.batch_size,
        world_size * args.batch_size,
        total,
        " -> ".join(f"{sl // 1024}K" for sl, _ in schedule.stages),
        tokens / 1e6,
    )
    logger.info(
        "schedule: %s (%d steps); WSD warmup %d -> peak %.1e, stable %d, decay %d -> %.1e",
        ", ".join(f"{n}x{s}" for s, n in schedule.stages),
        total,
        int(total * args.warmup_frac),
        args.peak_lr,
        int(total * args.stable_frac),
        total - int(total * (args.warmup_frac + args.stable_frac)),
        args.final_lr,
    )

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn, device)

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

    if world_size > 1:
        # DDP wraps the *indexer modules*, not the backbone: the backbone is frozen, so
        # wrapping the whole model would register 8B parameters with the reducer and allreduce
        # nothing. Each indexer is wrapped separately because they are invoked from the loss
        # hook rather than from a single forward, so there is no one module DDP could hook.
        #
        # A plain allreduce of the gradients after backward is used instead, which is exactly
        # what DDP would do here (170M params, 680 MB fp32, ~3 ms on NVLink) without needing
        # DDP's forward-hook machinery to fire on a module it never sees called.
        logger.info(
            "distributed: averaging %.1fM gradients across %d ranks each step (%.0f MB fp32)",
            trainable / 1e6,
            world_size,
            trainable * 4 / 1e6,
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

    # Only rank 0 writes metrics: eight ranks appending to one file interleave partial lines,
    # and the logged values are already cross-rank means, so the other seven would duplicate
    # them. The parent directory is created here because --metrics-file normally points inside
    # --out, which otherwise does not exist until the first checkpoint is written.
    metrics_handle = None
    if args.metrics_file and rank == 0:
        metrics_path = Path(args.metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_handle = open(metrics_path, "a")
    current_len, loader, iterator = None, None, None
    window: list[float] = []
    started = time.time()
    step = 0

    try:
        for step, seq_len in schedule.lengths():
            if seq_len != current_len:
                # A new length means new eligibility, so the loader is rebuilt rather than
                # reused; iterating a stale loader would keep the old seq_len silently.
                if current_len is not None:
                    # Say this out loud at the boundary: loss(L) ~ log(L) + const, so the
                    # reported loss steps up by log(new/old) purely because the softmax got
                    # wider. Without the warning it reads as a regression at exactly the
                    # moment something changed, which is the worst time to be guessing.
                    logger.info(
                        "step %d: seq_len %d -> %d; expect the loss to rise by ~%.2f "
                        "(log %d/%d) from the wider softmax alone -- watch "
                        "loss_minus_log_seq instead",
                        step,
                        current_len,
                        seq_len,
                        math.log(seq_len / current_len),
                        seq_len,
                        current_len,
                    )
                else:
                    logger.info("step %d: starting at seq_len=%d", step, seq_len)
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

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                loss, per_layer = fused_indexer_training_step(
                    model, trainer, input_ids=input_ids
                )
                (loss / args.accum_steps).backward()
                accumulated += float(loss) / args.accum_steps

            if world_size > 1:
                # Average gradients BEFORE clipping, so every rank clips the same vector and
                # therefore takes an identical step. Clipping first would let each rank scale
                # by its own local norm, and the averaged result would not equal the clipped
                # average -- the ranks would silently diverge.
                average_gradients(params, world_size)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            if world_size > 1 and step % args.log_every == 0:
                # Report the mean across ranks: rank 0's own loss is one sequence out of the
                # eight the step actually consumed, so on its own it is a noisier curve than
                # the thing being optimized.
                accumulated = all_reduce_mean(accumulated, device)

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
                                # loss(L) ~ log(L) + const, measured +0.69 per doubling, so a
                                # length curriculum puts a ~log-2 step in the raw curve at each
                                # boundary. Subtracting log(seq_len) gives a quantity that IS
                                # comparable across stages -- plot this one to see whether the
                                # objective is improving through a length change.
                                "loss_minus_log_seq": accumulated - math.log(seq_len),
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
    if world_size > 1:
        # Barrier before teardown: rank 0 is still writing the checkpoint, and destroying the
        # group underneath it can abort the write.
        dist.barrier()
        dist.destroy_process_group()
    logger.info("done in %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
