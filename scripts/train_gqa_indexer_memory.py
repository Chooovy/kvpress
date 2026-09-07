# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train the linear **memory** that compensates for what the router evicted.

The third arm alongside :mod:`scripts.train_gqa_indexer` (distillation) and
:mod:`scripts.train_gqa_indexer_e2e` (end-to-end router training). Those two train the router --
*which* keys to keep. This one takes a trained router as given, freezes it, and trains a
constant-size linear state that summarizes the keys it throws away, fused into the same softmax::

    o = fuse( flex_attention(q, k, v, mask=router's support), memory(q) )

    python -m scripts.train_gqa_indexer_memory \\
        --init-router .../stage1_16k_mid256_longce/final.pt \\
        --data-root RAW --schedule 8192:200,16384:400

Everything that is not the objective is deliberately identical to the two sibling scripts -- the
same WSD schedule, curriculum handling, loader, seeding and checkpoint layout, imported from them
rather than copied -- because the memory arm has to be comparable against the router-only arm it is
built on top of.

Why the router is frozen
------------------------
So the question this run answers is single-variable: *can a constant-size memory recover what a
fixed, already-good router discarded?* If the router moved at the same time, the evicted set ``E``
would move with it and the memory's target would be non-stationary -- and a result either way would
be unattributable. Joint training is a later stage, and the interface for it already exists
(``--ingest-scale``, see below).

The headroom is measured, not assumed
-------------------------------------
Under the trained LongCE router at ``keep_ratio=0.25``, the share of each row's softmax mass sitting
on evicted keys is ``rho = 0.213 / 0.219 / 0.234`` at ``L = 4K / 8K / 32K``. Flat in length, and two
orders of magnitude above the point where compensation would be pointless. The distribution is
skewed -- median 0.03-0.17 against p90 0.37-0.92 -- so the gain should come from the tail of rows
that lose nearly everything, not uniformly.

What to watch, in order
-----------------------
1. ``mass_share`` -- ``d/(D_S + d)``, the share of softmax mass the memory actually took. Compare
   against ``rho`` above. Climbing well past it means the layer stopped compensating for eviction and
   started degenerating towards pure linear attention.
2. ``gamma`` -- must climb off ``exp(-18) = 1.5e-8`` within ~100 steps. If it does not, the memory is
   still switched off and the loss curve cannot tell you that. Two things throttle it and both are
   handled here (fp32 scalars, and an AdamW ``eps`` below the gradient's own scale) -- but a
   too-small ``--scalar-lr`` would reintroduce it.
3. ``weight_participation`` under ``--longce-weights`` -- judge the weighting by this, against the
   failed delta arm's 0.13-0.18 and the offline 0.66-0.87.
4. The loss, last.

The prediction this run should be read against
----------------------------------------------
A rank-16 state cannot store "the uuid is at position 41022"; it can restore roughly what the
distant context was about. So expect RULER's niah tasks to barely move while vt/qa/cwe and
perplexity improve. A uniform gain across all 13 tasks would be more suspicious than a selective
one. Supporting measurement: fitting one vector per head to the exact evicted output leaves 78%
relative residual, so the rank is doing real work and ``--memory-rank 0`` should lose.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    DEFAULT_KERNEL_LR,
    DEFAULT_LOG_GAMMA,
    DEFAULT_SCALAR_EPS,
    DEFAULT_SCALAR_LR,
    DEFAULT_TAU,
    MEMORY_SCHEDULES,
    LengthSchedule,
    MemoryTrainer,
    describe_subsets,
    load_indexer_state_dict,
    load_memory_state_dict,
    memory_lm_step,
    memory_longce_step,
    memory_state_dict,
    read_index,
    wsd_lr_lambda,
)
from kvpress.presses.gqa_indexer.train import press_kwargs_from_checkpoint  # noqa: E402

# Imported, not reimplemented: these are exactly the pieces that must not differ between this arm
# and the router-only arm, or the comparison stops being about the memory.
from scripts.train_gqa_indexer import (  # noqa: E402
    all_reduce_mean,
    average_gradients,
    build_model,
    loader_for,
    setup_distributed,
)

logger = logging.getLogger("train_gqa_indexer_memory")


def save(path: Path, model, args, step: int, extra: dict | None = None, optimizer=None) -> None:
    """
    Write the memory weights plus enough metadata to know what produced them.

    Deliberately stores the memory **separately** from the router, under a ``memory`` key rather
    than ``indexer``. The router is an input to this run, not an output, and recording
    ``init_router`` is what makes "which router was this memory trained against" answerable from the
    artifact -- a question that matters because the memory summarizes precisely the keys *that*
    router chose to drop, and is not meaningful against a different one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "memory": memory_state_dict(model, args.memory_attr),
        "step": step,
        "config": {
            "objective": "memory_longce" if args.longce_weights else "memory_lm_loss",
            "model": args.model,
            # The router this memory was trained against. Not recoverable from the weights, and the
            # memory is only meaningful with respect to the support that router produces.
            "init_router": args.init_router,
            "scorer": "scalar",
            "memory_rank": args.memory_rank,
            "memory_mid_dim": args.memory_mid_dim,
            "memory_per_head": args.memory_per_head,
            "eviction_schedule": args.eviction_schedule,
            # The partition the memory was trained under. A memory trained at one (topk, sink,
            # local) and evaluated at another is reading a state built over a different evicted set,
            # which no weight shape would catch.
            "keep_ratio": args.keep_ratio,
            "topk": args.topk,
            "force_sink": args.force_sink,
            "force_local": args.force_local,
            "log_gamma_init": args.log_gamma_init,
            "tau_init": args.tau_init,
            "ingest_scale": args.ingest_scale,
            "kernel_lr": args.kernel_lr,
            "scalar_lr": args.scalar_lr,
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "longce_weights": args.longce_weights,
            "final_lr": args.final_lr,
            "seed": args.seed,
        },
    }
    if extra:
        payload["metrics"] = extra
    if optimizer is not None and args.save_optimizer:
        payload["optim"] = optimizer.state_dict()
    torch.save(payload, path)
    logger.info("saved %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- data / model ------------------------------------------------------------------
    p.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    p.add_argument("--data-root", default="", help="raw longmino shards (tokenized on the fly)")
    p.add_argument("--tokenized", default="", help="pre-tokenized .npy shards; preferred at length")
    p.add_argument("--subsets", nargs="*", default=["2e16", "2e17"])
    p.add_argument("--take-from", default="random", choices=["head", "random"])
    p.add_argument("--shuffle-buffer", type=int, default=8)
    # Same names and defaults as the sibling scripts', because loader_for is imported from them:
    # a divergence here would mean the arms draw different documents at the same --schedule.
    p.add_argument("--min-tokens", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)

    # --- the router this memory compensates for ---------------------------------------
    p.add_argument(
        "--init-router",
        required=True,
        help="trained scalar-indexer checkpoint. Frozen: the memory is trained against the support "
        "THIS router produces, so it is an input to the run rather than something it learns.",
    )
    p.add_argument(
        "--init-memory",
        default="",
        help="resume the memory weights from a previous memory checkpoint",
    )

    # --- the memory itself -------------------------------------------------------------
    p.add_argument(
        "--memory-rank",
        type=int,
        default=16,
        help="state rank R. 0 is the rank-0 ablation (one learned vector per head x the mass), "
        "which measured 78%% relative residual against the exact evicted output. Sweep {0,8,16,64}.",
    )
    p.add_argument("--memory-mid-dim", type=int, default=256, help="kernel trunk width R'")
    p.add_argument(
        "--memory-per-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-KV-head kernel readout (the heads overlap on only 14-17%% of their top-k, so "
        "what they need summarized differs). --no-memory-per-head is the shared ablation.",
    )
    p.add_argument("--memory-attr", default="kv_memory")
    p.add_argument(
        "--log-gamma-init",
        type=float,
        default=DEFAULT_LOG_GAMMA,
        help="initial log(gamma). Carries the OFF state, so the run starts at the eviction "
        "baseline; must be chosen against |E| since d = gamma*|E|*phi.(z/W).",
    )
    p.add_argument(
        "--tau-init",
        type=float,
        default=DEFAULT_TAU,
        help="initial decay constant in tokens. inf disables decay (pure accumulation, as LESS).",
    )
    p.add_argument(
        "--ingest-scale",
        type=float,
        default=0.0,
        help="'a' in w_j = lambda^age * exp(a*s~_j + b): lets the router's score drive how strongly "
        "each key is ingested. 0 (default) leaves it off; it is a live parameter there "
        "(dw/da = w*s~ != 0) so it can switch itself on. This is also the only path by which an "
        "EVICTED key can receive gradient, hence the hook for joint router training later.",
    )
    p.add_argument(
        "--eviction-schedule",
        default="streaming",
        choices=list(MEMORY_SCHEDULES),
        help="streaming gives every query block its own state (L supervised positions, |E| swept "
        "0..L-topk inside one sequence); oneshot matches deployment and is the validation setting.",
    )

    # --- the partition the memory is trained under -------------------------------------
    p.add_argument("--keep-ratio", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=0, help="0 derives topk from --keep-ratio")
    p.add_argument("--force-sink", type=int, default=4)
    p.add_argument(
        "--force-local",
        type=int,
        default=64,
        help="must match evaluation's --force_local, or the memory is read over a different "
        "partition than it was built over",
    )

    # --- objective ---------------------------------------------------------------------
    p.add_argument(
        "--longce-weights",
        default="",
        help="LongCE weight cache. Recommended: the delta arm collapsed RULER 66.24 -> ~35.2 by "
        "concentrating on the high-loss tail, while LongCE's spearman(w, L_long) is -0.001..-0.029. "
        "Empty runs the plain LM loss, which is the ablation baseline.",
    )

    # --- schedule ----------------------------------------------------------------------
    p.add_argument("--schedule", default="8192:200,16384:400")
    p.add_argument("--kernel-lr", type=float, default=DEFAULT_KERNEL_LR)
    p.add_argument(
        "--scalar-lr",
        type=float,
        default=DEFAULT_SCALAR_LR,
        help="learning rate for the fp32 scalars (log_gamma/log_tau/a/b). Two orders above "
        "--kernel-lr on purpose: log_gamma has ~8 units of log space to cross to switch the memory "
        "on, and at 1e-3 that takes ~8000 steps while the loss curve looks fine throughout.",
    )
    p.add_argument("--scalar-eps", type=float, default=DEFAULT_SCALAR_EPS)
    p.add_argument("--final-lr", type=float, default=5e-6)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--stable-frac", type=float, default=0.65)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # --- output ------------------------------------------------------------------------
    p.add_argument("--out", required=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--liger", action="store_true", help="fuse lm_head into the CE loss")

    args = p.parse_args()
    if not args.data_root and not args.tokenized:
        p.error("one of --data-root or --tokenized is required")
    if args.memory_rank < 0:
        p.error("--memory-rank must be non-negative")
    if not 0 < args.keep_ratio <= 1:
        p.error("--keep-ratio must be in (0, 1]")
    if args.longce_weights and args.take_from != "head":
        # Not a preference. The cached weight vector is per-position and truncated per stage, which
        # is only valid because the losses are causal and a shorter stage reads a *prefix* --
        # `--take-from random` draws a different window of the document, so the weights line up with
        # different tokens. The cache's own digest check catches it and raises, so this only turns a
        # mid-run failure into an argument error; the sibling script passes `head` for the same
        # reason.
        p.error(
            "--longce-weights requires --take-from head: the cached weights are per-position and "
            "'random' draws a different window of each document, so they would line up with "
            "different tokens (the cache's token digest catches this and aborts mid-run)."
        )
    if args.force_sink + args.force_local <= 0:
        logger.warning(
            "force_sink = force_local = 0: nothing is protected, so the router's raw top-k decides "
            "everything including the most recent tokens"
        )
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.out)

    if args.tokenized:
        index = read_index(args.tokenized)
        if not index.get("complete", True):
            logger.warning("%s/index.json is marked incomplete", args.tokenized)
    elif rank == 0:
        logger.info("subsets under %s:\n%s", args.data_root, describe_subsets(args.data_root))

    model, tokenizer = build_model(args.model, torch.bfloat16, "sdpa", device)
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(model=model)

    # --- the frozen router ------------------------------------------------------------
    router_ckpt = torch.load(args.init_router, map_location="cpu", weights_only=False)
    router_sd = router_ckpt.get("indexer", router_ckpt)
    router_cfg = router_ckpt.get("config", {})
    scorer, scorer_kwargs = press_kwargs_from_checkpoint(router_sd, router_cfg)
    if scorer != "scalar":
        # Structural, not a missing feature: a constant-size state can only stand for the evicted
        # set if that set depends on the query's position alone.
        raise SystemExit(
            f"--init-router holds a {scorer!r} indexer; the memory arm requires 'scalar'. A "
            "query-dependent router evicts a different set for every query, and no single "
            "accumulated state represents that."
        )
    press = GQAIndexerPress(
        compression_ratio=1.0 - args.keep_ratio,
        scorer="scalar",
        gate_scale=any("gate_scale" in k for k in router_sd),
        n_sink=args.force_sink,
        n_local=args.force_local,
        memory=True,
        memory_rank=args.memory_rank,
        memory_mid_dim=args.memory_mid_dim,
        memory_per_head=args.memory_per_head,
        memory_attr=args.memory_attr,
        **scorer_kwargs,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, router_sd)
    logger.info(
        "router loaded from %s (step %s, objective %s) and FROZEN",
        args.init_router, router_ckpt.get("step"), router_cfg.get("objective"),
    )

    # Non-default memory init lands on the modules the press just built. Done by assignment rather
    # than through MemoryConfig so the press keeps one construction path; these are scalars, not
    # geometry, so nothing about the shapes depends on them.
    for layer in model.model.layers:
        memory = getattr(layer.self_attn, args.memory_attr)
        memory.log_gamma.data.fill_(args.log_gamma_init)
        if args.tau_init == float("inf"):
            memory.log_tau.data.fill_(float("inf"))
            memory.log_tau.requires_grad_(False)
        else:
            memory.log_tau.data.fill_(float(torch.log(torch.tensor(args.tau_init))))
        memory.ingest_scale.data.fill_(args.ingest_scale)

    if args.init_memory:
        payload = torch.load(args.init_memory, map_location="cpu", weights_only=False)
        load_memory_state_dict(model, payload.get("memory", payload), args.memory_attr)
        logger.info("memory resumed from %s (step %s)", args.init_memory, payload.get("step"))

    trainer = MemoryTrainer(
        press=press,
        topk=args.topk or None,
        keep_ratio=args.keep_ratio,
        force_sink=args.force_sink,
        force_local=args.force_local,
        schedule=args.eviction_schedule,
    )
    groups = trainer.parameter_groups(
        model,
        kernel_lr=args.kernel_lr,
        scalar_lr=args.scalar_lr,
        scalar_eps=args.scalar_eps,
    )
    params = [p for g in groups for p in g["params"]]
    logger.info(
        "training %.2fM memory parameters in %d group(s): %s",
        sum(p.numel() for p in params) / 1e6,
        len(groups),
        {g["name"]: f"lr={g['lr']:g}" for g in groups},
    )

    # LambdaLR scales each group by its own base_lr, so the two learning rates keep their ratio
    # through warmup and decay -- which is the point of having two.
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    lr_lambda = wsd_lr_lambda(
        args.total_steps,
        warmup_frac=args.warmup_frac,
        stable_frac=args.stable_frac,
        peak_lr=1.0,
        final_lr=args.final_lr / max(args.kernel_lr, 1e-12),
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metrics_handle = None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_handle = open(out_dir / "metrics.jsonl", "a")

    current_len, loader, iterator = None, None, None
    longce_cache = None
    longce_missing = longce_seen = 0
    started = time.time()
    step = 0

    try:
        for step, seq_len in schedule.lengths():
            if seq_len != current_len:
                logger.info("step %d: seq_len -> %d", step, seq_len)
                loader = loader_for(
                    seq_len, args, tokenizer, rank, world_size, batch_size=args.batch_size
                )
                iterator = iter(loader)
                if args.longce_weights:
                    # Reopened per stage: the digest to verify against is the one taken at THIS
                    # seq_len, so an unusable cache fails at the boundary with a message naming the
                    # fix rather than at the first lookup.
                    from kvpress.presses.gqa_indexer.longce_weights import LongCEWeightCache

                    longce_cache = LongCEWeightCache(args.longce_weights, seq_len=seq_len)
                    longce_missing = longce_seen = 0
                    if rank == 0:
                        logger.info("LongCE weight cache: %s", longce_cache.summary())
                current_len = seq_len

            will_log = step % args.log_every == 0 or step == args.total_steps - 1
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            stats: dict = {}
            for micro in range(args.accum_steps):
                # Only on logged steps and only on the last micro-batch: the diagnostics cost a
                # reduction per layer, and measuring every micro-step pays that accum_steps times to
                # report one number.
                trainer.measure = will_log and micro == args.accum_steps - 1
                try:
                    batch = next(iterator)
                except StopIteration:
                    logger.info("step %d: corpus exhausted at seq_len=%d, restarting", step, seq_len)
                    iterator = iter(loader)
                    batch = next(iterator)
                input_ids = batch["input_ids"].to(device, non_blocking=True)

                if longce_cache is not None:
                    # The lookup verifies a token digest per document and raises on a mismatch --
                    # see LongCEWeightCache.lookup for why that has to be fatal: mismatched weights
                    # optimize the wrong positions while every visible number stays intact.
                    batch_weights = []
                    for row, doc_id in enumerate(batch["doc_ids"]):
                        longce_seen += 1
                        if doc_id in longce_cache:
                            batch_weights.append(
                                torch.from_numpy(longce_cache.lookup(doc_id, input_ids[row]))
                            )
                        else:
                            # 1.0 is this weighting's neutral value, so an uncached document
                            # contributes exactly as it would under the plain mean. Counted rather
                            # than tolerated silently: a high miss rate means the run is mostly the
                            # plain objective wearing the LongCE arm's name.
                            longce_missing += 1
                            batch_weights.append(
                                torch.ones(input_ids.shape[1] - 1, dtype=torch.float32)
                            )
                    loss, stats = memory_longce_step(
                        model,
                        trainer,
                        input_ids=input_ids,
                        weights=torch.stack(batch_weights).to(device, non_blocking=True),
                    )
                else:
                    loss, stats = memory_lm_step(model, trainer, input_ids=input_ids)
                (loss / args.accum_steps).backward()
                accumulated += float(loss.detach()) / args.accum_steps

            if world_size > 1:
                average_gradients(params, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            if will_log:
                mean_loss = all_reduce_mean(accumulated, device) if world_size > 1 else accumulated
                record = {
                    "step": step,
                    "seq_len": seq_len,
                    "loss": mean_loss,
                    "grad_norm": float(grad_norm),
                    "lr": lr_schedule.get_last_lr()[0],
                    "elapsed": time.time() - started,
                    **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
                }
                if longce_cache is not None and longce_seen:
                    record["longce_missing_frac"] = longce_missing / longce_seen
                if rank == 0:
                    logger.info(
                        "step %d/%d L=%d loss %.4f | mass_share %.3e gamma %.3e | grad %.2e",
                        step, args.total_steps, seq_len, mean_loss,
                        record.get("mass_share", float("nan")),
                        record.get("gamma", float("nan")),
                        float(grad_norm),
                    )
                    metrics_handle.write(json.dumps(record) + "\n")
                    metrics_handle.flush()

            if rank == 0 and args.save_every and (step + 1) % args.save_every == 0:
                save(out_dir / f"step{step + 1}.pt", model, args, step + 1, optimizer=optimizer)
            if args.max_steps and step + 1 >= args.max_steps:
                logger.info("stopping at --max-steps %d", args.max_steps)
                break
    finally:
        if rank == 0:
            save(out_dir / "final.pt", model, args, step + 1, optimizer=optimizer)
            if metrics_handle is not None:
                metrics_handle.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
