#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke-test the cross-replay objective on a real model, and report what the loss curve cannot.

Not a training script. It runs a handful of steps and prints the three readouts that decide whether
this objective is working at all -- each of which can look fine in the loss while being broken:

* **gate participation** -- the effective fraction of keys the gate spreads its mass over. ~1.0 means
  the gate is flat, i.e. no ranking was learned, whatever the loss says. Falling towards 0 is the
  concentration eviction needs.
* **gate_scale** -- per-layer gate strength. A collapse towards 0 says that layer's router is not
  earning its place.
* **the shuffle control** -- replay loss with the learned scores permuted along the key axis. If
  shuffling does not hurt, the scores carry no usable ranking and the objective is measuring nothing.
  This is the one number that separates "trained" from "trained-looking".

Usage::

    python -m scripts.smoke_cross_replay --model /path/to/Qwen3-8B --context-len 4096 --steps 5
"""

from __future__ import annotations

import argparse
import logging

import torch

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer.cross_replay import (
    CrossReplayTrainer,
    cross_replay_training_step,
    gate_participation,
    shuffled_scores,
)

logger = logging.getLogger("smoke_cross_replay")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--query-chunk", type=int, default=1024)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--scalar-mid-dim", type=int, default=0, help="0 = SparseK's linear score")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logit-chunk",
        type=int,
        default=None,
        help="rows of lm_head output to materialize at a time; bounds the logits term "
        "(0.87 GiB per 1024 rows on Qwen3-8B) independently of --query-chunk",
    )
    parser.add_argument(
        "--no-shuffle-control",
        action="store_true",
        help="skip the shuffle control, which costs two extra replay passes",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(args.device)
    model.eval()  # dropout off; only the indexers train

    press = GQAIndexerPress(
        compression_ratio=0.5,
        scorer="scalar",
        scalar_mid_dim=args.scalar_mid_dim,
        gate_scale=True,
        n_sink=args.n_sink,
    )
    trainer = CrossReplayTrainer(press=press, query_chunk=args.query_chunk)

    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, args.context_len), device=args.device)
    logger.warning(
        "using RANDOM token ids: this checks the mechanism and the memory profile, NOT whether the "
        "objective learns anything. Point it at real text before reading the numbers as evidence."
    )

    with trainer.hooks(model):
        params = trainer.indexer_parameters(model)
        logger.info("training %d indexer tensors", len(params))
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        if args.device.startswith("cuda"):
            # Report the weights separately, then reset, so the per-step peak measures the STEP and
            # not the model load. Without the reset the peak is dominated by whatever the largest
            # allocation since process start was, which makes it insensitive to changes in the step
            # -- a real trap: a memory fix that worked would look like it had done nothing.
            logger.info(
                "weights + cache on device: %.1f GiB allocated, %.1f GiB reserved",
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
            )
            torch.cuda.reset_peak_memory_stats()

        for step in range(args.steps):
            if args.device.startswith("cuda"):
                # Per-step, so a step's peak cannot be inherited from an earlier one.
                torch.cuda.reset_peak_memory_stats()
            # backward happens inside, chunk by chunk -- that is what bounds the peak. The returned
            # loss is already detached.
            loss = cross_replay_training_step(
                model, trainer, input_ids=ids, logit_chunk=args.logit_chunk
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            gates = {
                idx: trainer.gate(idx, trainer._scores[idx].shape[1])
                for idx in sorted(trainer._scores)
            }
            pr = [gate_participation(g, min(args.n_sink, g.shape[-1])) for g in gates.values()]
            scales = list(trainer.gate_scales.values())
            peak = (
                torch.cuda.max_memory_allocated() / 2**30
                if args.device.startswith("cuda")
                else float("nan")
            )
            logger.info(
                "step %d | loss %.4f | grad_norm %.3f | participation %.4f | gate_scale %.5f | "
                "peak %.1f GiB",
                step, loss.item(), float(grad_norm),
                sum(pr) / len(pr), sum(scales) / len(scales), peak,
            )

        if args.no_shuffle_control:
            logger.info("shuffle control skipped (--no-shuffle-control)")
        else:
            # The control. Permute each layer's scores along the key axis and re-measure: a score
            # that ranks keys usefully must do worse when its ranking is destroyed. Equal losses
            # mean the gate is flat, or the replay does not depend on the ranking -- either way,
            # nothing learned.
            with torch.no_grad():
                clean = cross_replay_training_step(
                    model, trainer, input_ids=ids, backward=False
                )
                generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
                perm = torch.randperm(args.context_len, generator=generator).to(args.device)
                # Re-run with the SAME scores, permuted. score_context would overwrite them, so
                # shuffled_scores replaces that step for the duration -- and restores it even if the
                # inner call raises, since a leaked patch would train every later step on a
                # permutation.
                with shuffled_scores(trainer, perm):
                    shuffled = cross_replay_training_step(
                        model, trainer, input_ids=ids, backward=False
                    )

            delta = shuffled.item() - clean.item()
            logger.info(
                "shuffle control | learned %.4f | shuffled %.4f | delta %+.4f",
                clean.item(), shuffled.item(), delta,
            )
            if delta <= 0:
                logger.warning(
                    "shuffling the scores did NOT hurt (delta %+.4f). The gate carries no usable "
                    "ranking yet -- expected after only %d steps, but if it persists the objective "
                    "is not training the router.",
                    delta, args.steps,
                )
            else:
                logger.info(
                    "scores carry a ranking: destroying it costs %.4f nats/token after %d steps.",
                    delta, args.steps,
                )

    d_max = 2 * args.context_len - 1
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and d_max <= limit:
        logger.info("positions ok: d_max %d <= max_position_embeddings %d", d_max, limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
