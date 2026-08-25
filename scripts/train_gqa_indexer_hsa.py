# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer end to end through **two-level (HSA) chunk attention**, on longmino.

The fourth arm, beside :mod:`scripts.train_gqa_indexer` (distillation),
:mod:`scripts.train_gqa_indexer_e2e` (additive gate) and
:mod:`scripts.train_gqa_indexer_exact_k` (exact-K chunk subset). The operator::

    out = sum_c  w_c * softmax_within-chunk-c(q k^T) @ v_c        w = softmax(s)

    python -m scripts.train_gqa_indexer_hsa --data-root RAW --tokenized TOK \\
        --schedule 8192:300,16384:300,32768:900 --max-steps 600

Why this arm, given the other three
-----------------------------------
Because the within-chunk softmax already sums to 1, ``w_c`` **is** chunk ``c``'s share of the
output -- verified to 1.1e-16. Three things follow, and each removes a piece of machinery the other
arms need:

1. **No pinning.** The additive gate can go flat along the key axis and a flat gate is inert
   (softmax is shift-invariant), so the model reverts to the frozen backbone with no ranking
   learned; ``--pin-mode`` exists to forbid that. Here a flat router gives *uniform mixing*, which
   measures 0.44 away from dense attention. There is no zero-cost setting to fall into.

2. **What is trained is what inference ranks on.** ``ROUTER_LEARNABILITY.md`` §6: the additive
   gate's optimum is ``log(mass) - LSE_c``, which for a frozen backbone is a **constant** -- so a
   correction that carries no ranking at all, and inference then top-k's on a quantity the router
   was never asked to order. This arm's score IS the mass.

3. **No candidate pool.** The chunk softmax runs over every chunk, so every chunk receives a
   content-dependent gradient every step. That is the exact-K arm's *measured* bottleneck removed by
   construction: 11-15% of oracle-best chunks never entered its ``M=32`` pool, and a chunk outside
   the pool appears nowhere in the graph, so no backward estimator can reach it. Affordable because
   the score matrix is pooled to chunks -- ``(B, Hkv, Sq, n_chunk)`` is 34 MiB at 8K against the
   token logits' 2 GiB.

Everything that is not the objective is imported from the other scripts rather than reimplemented --
WSD schedule, curriculum, loader, seeding, FFN-SP, gradient averaging, Liger, checkpoint format. A
run of this against a run of those differs in the objective and nothing else.

Flags that are absent on purpose
--------------------------------
* ``--pin-mode`` / ``--gate-budget`` -- nothing to pin (point 1).
* ``--n-candidate`` / ``--explore-frac`` / ``--topk-chunk`` -- no pool, no sampling, no budget at
  train time (point 3). The budget is an *inference* parameter here: training learns a full
  distribution over chunks and ``evaluate_sparse.py --topk`` truncates it. ``mass_topquarter`` in the
  metrics is the training-time estimate of how much that truncation keeps.
* ``--query-block`` -- selection is per query. The exact-K arm shared a subset across a block as a
  memory concession that the GPU measurement later showed was unnecessary; there is nothing to
  concede at 34 MiB.
* ``--hard`` -- the forward is deterministic, so there is no sampling ablation to run.

The diagnostic to watch
-----------------------
``entropy`` (normalized to ``[0, 1]``). This objective's one failure mode is a router that learns to
*use* a near-uniform mixture rather than to *choose*: uniform mixing is a legitimate operator, so the
LM loss can descend that way. Entropy stuck near 1.0 while the loss falls is that failure.

``lse_corr`` is the second and it has no counterpart in any other arm. The optimal score is known in
**closed form** -- the chunk's own log-sum-exp, up to a per-query constant -- so this is a direct
Spearman against the target on real text, with no oracle, no swap experiment and no second forward
pass. Rising means the router is learning the right thing; flat near 0 means it is not, whatever the
loss does.
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
    DEFAULT_POS_SLOPE,
    HSAIndexerTrainer,
    hsa_indexer_training_step,
    indexer_state_dict,
    load_indexer_state_dict,
)
from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    SUBSETS,
    LengthSchedule,
    describe_subsets,
    read_index,
)

# Imported, not reimplemented: these are exactly the pieces that must not differ between the three
# objectives, or the comparison stops being about the objective.
from scripts.train_gqa_indexer import (  # noqa: E402
    all_reduce_mean,
    average_gradients,
    build_model,
    build_optimizer,
    loader_for,
    resume_training_state,
    setup_distributed,
)
from scripts.train_gqa_indexer_e2e import apply_liger_fused_ce, ffn_sp_group  # noqa: E402

logger = logging.getLogger("train_gqa_indexer_hsa")


def save(
    path: Path,
    model,
    args,
    step: int,
    extra: dict | None = None,
    optimizer=None,
    lr_schedule=None,
) -> None:
    """
    Write the indexer weights plus enough metadata to know what produced them.

    Same layout as the other two scripts' ``save`` (an ``indexer`` state dict under a ``config``
    blob), so a checkpoint from any objective loads into the press the same way and
    ``evaluate_sparse.py`` reads all three. ``objective`` records which one produced it, since the
    weights alone do not say.

    ``chunk_size`` is recorded because it is **not** recoverable from the weights -- it is not a
    parameter shape -- and scoring at a different ``chunk_size`` than training used is a silent
    mismatch, exactly the class of bug ``rope_dim`` is guarded against elsewhere in this package.
    ``evaluate_sparse.py --chunk_size -1`` reads it back from here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "objective": "hsa_two_level",
            "model": args.model,
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim if args.scorer == "scalar" else None,
            "scalar_pos_slope": args.scalar_pos_slope if args.scorer == "scalar" else None,
            # Routing geometry: NOT recoverable from the weights (chunk_size is not a parameter
            # shape), and evaluating at a different chunk_size than training used is silent. The
            # eval reads this via `chunk_size=-1`.
            "chunk_size": args.chunk_size,
            "chunk_aggregate": args.chunk_aggregate,
            # The RESOLVED number, not the None that means "derive it". For chunk_aggregate="lse"
            # this is a temperature inside the reduction (logsumexp is not scale-equivariant), so an
            # eval that re-derived it differently would rank on a different functional than training
            # optimized -- silently. args.resolved_score_scale is set once in main().
            "score_scale": getattr(args, "resolved_score_scale", args.score_scale),
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "peak_lr": args.peak_lr,
            "final_lr": args.final_lr,
            "seed": args.seed,
        },
    }
    if extra:
        payload["metrics"] = extra
    if args.save_optimizer and optimizer is not None and lr_schedule is not None:
        payload["optim"] = optimizer.state_dict()
        payload["lr_schedule"] = lr_schedule.state_dict()
    torch.save(payload, path)
    logger.info("saved %s (step %d)", path, step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True, help="longmino_256k_filtered root")
    data.add_argument(
        "--tokenized",
        default=None,
        help="read pre-tokenized .npy shards from here. The same corpus the other two scripts use "
        "-- share it rather than tokenizing three times.",
    )
    data.add_argument(
        "--subsets", nargs="+", default=["2e15", "2e16", "8k_32k", "synth_cwe", "synth_rex"],
        choices=list(SUBSETS),
    )
    data.add_argument("--take-from", choices=("head", "random"), default="random")
    data.add_argument("--shuffle-buffer", type=int, default=64)
    data.add_argument("--min-tokens", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--batch-size", type=int, default=1)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen3-8B")
    model_group.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    model_group.add_argument("--attn", default="sdpa", help="backbone attention kernel")
    model_group.add_argument("--compression-ratio", type=float, default=0.5)
    model_group.add_argument("--rope-dim", type=int, default=None)
    model_group.add_argument("--head-dim", type=int, default=None)
    model_group.add_argument("--n-heads", type=int, default=None)
    model_group.add_argument("--scorer", choices=("pairwise", "scalar"), default="pairwise")
    model_group.add_argument("--scalar-mid-dim", type=int, default=256)
    model_group.add_argument("--scalar-pos-slope", type=float, default=DEFAULT_POS_SLOPE)
    model_group.add_argument("--press-n-sink", type=int, default=4)
    model_group.add_argument("--scorer-attr", default="indexer")
    model_group.add_argument(
        "--ffn-sp-size", type=int, default=1,
        help="shard the FFN activations across this many ranks (sequence parallel). Same mechanism "
        "and the same constraint as the gated script: ranks in one SP group read the SAME sequence, "
        "so the loader shards by data-parallel rank.",
    )
    model_group.add_argument(
        "--liger", action="store_true",
        help="fuse lm_head into the cross-entropy so the (L, vocab) logits are never materialized. "
        "Needed here for the same reason as the gated arm: the router's gradient comes from the LM "
        "loss, so that tensor is retained -- 7.0 GiB at L=8192 on Qwen3-8B, scaling with L.",
    )
    model_group.add_argument(
        "--init-from", default=None,
        help="load indexer WEIGHTS only and start a fresh schedule. Accepts a distillation or a "
        "gated checkpoint: the parameter names are shared, and gate_scale (which this objective "
        "does not read) is simply carried along. Warm-starting from the gated arm's step-600 is the "
        "intended use.",
    )
    model_group.add_argument(
        "--resume-from", default=None,
        help="continue an interrupted run: restore weights, AdamW state and LR-schedule position. "
        "Mutually exclusive with --init-from.",
    )

    route = parser.add_argument_group("routing")
    route.add_argument(
        "--chunk-size", type=int, default=64,
        help="tokens per chunk. Must be > 1: at 1 the within-chunk softmax is softmax(one element) "
        "= 1, so the q.k term drops out and the router would have to learn the whole attention "
        "distribution itself, losing the two-level decomposition AND the frozen backbone's prior "
        "(ROUTER_LEARNABILITY.md section 6 -- structural, not tuning). Selection happens at this "
        "granularity, so it must match what the press uses at eval; recorded in the checkpoint.",
    )
    route.add_argument(
        "--chunk-aggregate", choices=("lse", "mean", "max"), default="lse",
        help="how token scores become a chunk score. lse (DEFAULT, and the principled choice): "
        "ROUTER_LEARNABILITY.md section 6 verifies the true chunk mass is softmax_c(LSE_c) exactly "
        "for a frozen backbone, and this operator makes w = softmax_c(s_chunk) BE the realized mass, "
        "so the target is s_chunk = LSE_c. The indexer's token score imitates the backbone's "
        "attention LOGIT, so the aggregation must be the same functional -- logsumexp -- or the chunk "
        "level cannot match even with a perfect token scorer. Measured Spearman against the true "
        "chunk LSE with an exact token scorer: lse 1.000, mean 0.756, max 0.631; in a needle regime "
        "(one high-logit token per 64-token chunk) needle recall at top-4 is lse 1.000 vs mean 0.533, "
        "because mean dilutes a lone needle ~64x. mean/max are the ablations.",
    )
    route.add_argument(
        "--score-scale", type=float, default=None,
        help="multiplier on the pooled chunk score before the softmax (default: head_dim ** -0.5). "
        "Unlike the exact-K arm's --route-scale this is NOT fixing a saturation hazard -- softmax "
        "has no floor or ceiling and is shift-invariant, so only score DIFFERENCES matter. It fixes "
        "the INITIALIZATION: IndexerNorm leaves the raw qi.ki dot at std ~sqrt(head_dim) = 11.3, and "
        "a softmax over 128 chunks at that scale starts w nearly one-hot on a RANDOMLY chosen chunk "
        "-- an output far from the frozen backbone's, which is the prior this arm exists to keep.",
    )
    route.add_argument(
        "--attach-score-input", action="store_true",
        help="let the score's gradient flow back into hidden_states. OFF by default because it "
        "DIVERGES -- measured on the exact-K arm, whose mechanism is identical: the score is a "
        "function of hidden_states, so this path deposits gradient in the residual stream that every "
        "router below re-amplifies. grad_norm 2.1e3 at 4 layers, 8.6e13 at 12, inf at 24, nan at 36, "
        "against 1.1e6 detached; per-layer amplification 10-50x against 1.1x for the same backbone "
        "with no router. What it costs: a router loses the second-order term 'my routing changed "
        "what a LOWER router sees'. Each router is still trained by the full LM loss through its own "
        "attention. Use this flag to reproduce the divergence.",
    )
    route.add_argument(
        "--no-lse-corr", action="store_true",
        help="skip the Spearman correlation against the chunk log-sum-exp. Leave it on: "
        "ROUTER_LEARNABILITY.md section 6 proves the optimal score for a frozen backbone IS the "
        "chunk LSE up to a per-query constant, so this measures progress against a closed-form "
        "target -- the only arm in this package that can. Costs one reduction over logits the "
        "forward already computed, on a handful of sampled query rows.",
    )
    route.add_argument(
        "--score-tile-bytes", type=int, default=0,
        help="bytes budgeted for one score tile's token logits (0 = 64 MiB). The router's "
        "(B, Hkv, Sq, Sk) logits are 8 GiB in fp32 at 16K while the chunk-pooled result is 64x "
        "smaller, so the wide tensor is built a tile at a time. A BYTE budget rather than a query "
        "count, because the right count depends on Sk and the head count -- a count that fits at 8K "
        "does not at 16K. Memory knob only; the result is tile-invariant.",
    )
    route.add_argument(
        "--no-checkpoint-scores", action="store_true",
        help="retain each score tile's token logits instead of recomputing. 288 GiB across 36 layers "
        "at 16K for a result 64x smaller. Debugging only.",
    )
    route.add_argument(
        "--no-checkpoint-attention", action="store_true",
        help="retain each query tile's attention logits. The dominant term by a wide margin at "
        "length. Debugging only.",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:300,16384:300,32768:300", help="SEQ_LEN:STEPS,...")
    optim.add_argument(
        "--max-steps", type=int, default=0,
        help="stop after this many optimizer steps regardless of the schedule total (0 = the whole "
        "schedule). The LR is still computed from the full --schedule, so a run can be truncated at "
        "a defined point WITHOUT reshaping WSD -- which is what makes a step-600 checkpoint "
        "comparable against the other two arms at the same step.",
    )
    optim.add_argument("--peak-lr", type=float, default=1e-3, help="WSD plateau")
    optim.add_argument("--final-lr", type=float, default=5e-6, help="WSD floor")
    optim.add_argument("--warmup-frac", type=float, default=0.10)
    optim.add_argument("--stable-frac", type=float, default=0.60)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument(
        "--global-batch-size", type=int, default=0,
        help="sequences per OPTIMIZER step, across all ranks. Set this and --accum-steps is derived "
        "so tokens/step matches regardless of --ffn-sp-size, which is the only way this run is "
        "comparable to the others step-for-step.",
    )
    optim.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("io")
    io.add_argument("--out", default="checkpoints/gqa_indexer_hsa")
    io.add_argument("--save-every", type=int, default=200)
    io.add_argument("--log-every", type=int, default=10)
    io.add_argument("--metrics-file", default=None, help="append JSONL metrics here")
    io.add_argument(
        "--save-optimizer", action=argparse.BooleanOptionalAction, default=True,
        help="write AdamW state and the LR-schedule position into each checkpoint so --resume-from "
        "can continue the run.",
    )
    io.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.init_from and args.resume_from:
        parser.error(
            "--init-from and --resume-from are mutually exclusive: the first is a weights-only warm "
            "start that restarts the schedule and Adam, the second continues an interrupted run "
            "with both preserved. Pick one."
        )
    if args.chunk_size <= 1:
        parser.error(
            f"--chunk-size must be > 1, got {args.chunk_size}. At 1 the within-chunk softmax is "
            "softmax(one element) = 1: the q.k term vanishes entirely and the router would have to "
            "learn the whole attention distribution from scratch, discarding the frozen backbone's "
            "prior. See ROUTER_LEARNABILITY.md section 6 -- this is structural, not a tuning choice."
        )
    if not torch.cuda.is_available():
        parser.error("no CUDA device; indexer training needs a GPU")

    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)
    sp_group, dp_rank, dp_world_size, sp_rank = ffn_sp_group(world_size, args.ffn_sp_size, rank)
    if args.ffn_sp_size > 1:
        logger.info(
            "FFN sequence parallel: sp_size=%d -> %d data-parallel replica(s). This rank is "
            "sp_rank=%d of dp_rank=%d. Ranks within one SP group read the SAME sequence.",
            args.ffn_sp_size, dp_world_size, sp_rank, dp_rank,
        )

    # Seeded by DATA-PARALLEL rank, not global rank: ranks cooperating on one sequence must draw the
    # same one, or the FFN all-gather stitches together fragments of unrelated documents.
    torch.manual_seed(args.seed + dp_rank)
    out_dir = Path(args.out)

    if args.tokenized:
        index = read_index(args.tokenized)
        if not index.get("complete", True):
            logger.warning(
                "%s/index.json is marked incomplete: some shards failed to pretokenize",
                args.tokenized,
            )
        logger.info(
            "pre-tokenized corpus: %d docs at seq_len<=%d, subsets %s",
            index["total_docs"], index["seq_len"], index["subsets"],
        )
    else:
        logger.info("subsets under %s:\n%s", args.data_root, describe_subsets(args.data_root))

    total = schedule.total_steps
    if args.global_batch_size:
        per_replica_step = dp_world_size * args.batch_size
        if args.global_batch_size % per_replica_step:
            raise SystemExit(
                f"--global-batch-size {args.global_batch_size} is not divisible by "
                f"{dp_world_size} replica(s) x --batch-size {args.batch_size} = "
                f"{per_replica_step}. Gradient accumulation can only reach multiples of that."
            )
        args.accum_steps = args.global_batch_size // per_replica_step
        logger.info(
            "--global-batch-size %d / (%d replica(s) x batch %d) -> --accum-steps %d. This is what "
            "keeps tokens/step independent of --ffn-sp-size, and so comparable to the other arms at "
            "the same step number.",
            args.global_batch_size, dp_world_size, args.batch_size, args.accum_steps,
        )
    seqs_per_step = dp_world_size * args.batch_size * args.accum_steps
    tokens = sum(sl * st for sl, st in schedule.stages) * seqs_per_step
    logger.info(
        "%d replica(s) x batch_size %d x accum %d = %d sequences/step; %d optimizer steps over %s "
        "= %.0fM tokens",
        dp_world_size, args.batch_size, args.accum_steps, seqs_per_step, total,
        " -> ".join(f"{sl // 1024}K" for sl, _ in schedule.stages), tokens / 1e6,
    )
    logger.info(
        "schedule: %s (%d steps); WSD warmup %d -> peak %.1e, stable %d, decay %d -> %.1e",
        ", ".join(f"{n}x{s}" for s, n in schedule.stages), total,
        int(total * args.warmup_frac), args.peak_lr, int(total * args.stable_frac),
        total - int(total * (args.warmup_frac + args.stable_frac)), args.final_lr,
    )

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn, device)

    if args.liger:
        apply_liger_fused_ce(model, args.model)

    if args.ffn_sp_size > 1:
        # AFTER liger, for the reason the gated script documents: liger rebinds forward on the mlp
        # module itself, so wrapping second means the slice goes through liger's SwiGLU.
        from kvpress.presses.gqa_indexer.ffn_sp import wrap_ffn_sequence_parallel

        wrap_ffn_sequence_parallel(model, group=sp_group)

    # gate_scale=True even though this objective never reads it: it keeps the checkpoint
    # byte-compatible with the gated arm's, so --init-from works in both directions and
    # evaluate_sparse.py's has_gate detection behaves identically for all three arms.
    press_kwargs = {
        "compression_ratio": args.compression_ratio,
        "scorer_attr": args.scorer_attr,
        "gate_scale": True,
        "n_sink": args.press_n_sink,
        "scorer": args.scorer,
        # So the press ranks at the same granularity the router was trained to score.
        "chunk_size": args.chunk_size,
        "chunk_aggregate": args.chunk_aggregate,
    }
    if args.scorer == "scalar":
        press_kwargs["scalar_mid_dim"] = args.scalar_mid_dim
        press_kwargs["scalar_pos_slope"] = args.scalar_pos_slope
    for name in ("rope_dim", "head_dim", "n_heads"):
        value = getattr(args, name)
        if value is not None:
            press_kwargs[name] = value
    press = GQAIndexerPress(**press_kwargs)
    press.post_init_from_model(model)

    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        ckpt_scorer = (payload.get("config") or {}).get("scorer")
        if ckpt_scorer is not None and ckpt_scorer != args.scorer:
            raise SystemExit(
                f"--init-from was trained with scorer={ckpt_scorer!r} but this run is "
                f"scorer={args.scorer!r}. The two have different parameter names, so nothing would "
                f"load and the run would silently start from scratch."
            )
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        logger.info(
            "initialized indexer from %s (objective=%s)",
            args.init_from, (payload.get("config") or {}).get("objective", "unknown"),
        )

    trainer = HSAIndexerTrainer(
        press=press,
        chunk_size=args.chunk_size,
        chunk_aggregate=args.chunk_aggregate,
        detach_score_input=not args.attach_score_input,
        score_scale=args.score_scale,
        measure_lse_corr=not args.no_lse_corr,
        score_tile_bytes=args.score_tile_bytes,
        checkpoint_scores=not args.no_checkpoint_scores,
        checkpoint_attention=not args.no_checkpoint_attention,
    )
    # Record the resolved score_scale for the checkpoint: `None` means "head_dim ** -0.5", and the
    # eval must not have to re-derive it (see save()).
    _first_attn = get_language_model(model).layers[0].self_attn
    args.resolved_score_scale = trainer.resolved_score_scale(press.get_indexer(_first_attn))
    logger.info(
        "score_scale = %.6g (%s). Under chunk_aggregate=%s this multiplies the TOKEN scores inside "
        "the aggregation; for lse that is a temperature, not a cosmetic factor.",
        args.resolved_score_scale,
        "explicit" if args.score_scale is not None else "head_dim ** -0.5",
        args.chunk_aggregate,
    )

    # Called here, not left to hooks(), so the parameter list handed to the optimizer is the same
    # object the run trains.
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    trainable = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable %.2fM of %.2fB parameters (%.3f%%); objective = LM loss through two-level chunk "
        "attention: chunk=%d aggregate=%s over ALL chunks (no candidate pool, no pinning)",
        trainable / 1e6, total_params / 1e9, 100 * trainable / total_params,
        args.chunk_size, args.chunk_aggregate,
    )
    if world_size > 1:
        logger.info(
            "distributed: averaging %.1fM gradients across %d ranks each step (%.0f MB fp32)",
            trainable / 1e6, world_size, trainable * 4 / 1e6,
        )

    optimizer, lr_schedule = build_optimizer(params, args)

    start_step = 0
    if args.resume_from:
        start_step = resume_training_state(
            args.resume_from, model, optimizer, lr_schedule, args.scorer_attr, args.schedule, device
        )

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
            if step < start_step:
                continue
            if seq_len != current_len:
                if current_len is not None:
                    logger.info(
                        "step %d: seq_len %d -> %d; the LM loss has no log(L) term, so any jump "
                        "here is real rather than arithmetic",
                        step, current_len, seq_len,
                    )
                else:
                    logger.info("step %d: starting at seq_len=%d", step, seq_len)
                loader = loader_for(
                    seq_len, args, tokenizer, dp_rank, dp_world_size, batch_size=args.batch_size
                )
                iterator = iter(loader)
                current_len = seq_len
                # Nothing to clear across a curriculum boundary: unlike the exact-K arm this
                # objective keeps no cross-step state (its forward is deterministic, so there is no
                # selection-stability metric that would compare shapes from two different seq_lens).

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            will_log = (
                step % args.log_every == 0 or step == args.total_steps - 1
                or (bool(args.max_steps) and step + 1 >= args.max_steps)
            )
            for micro in range(args.accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    logger.info(
                        "step %d: corpus exhausted at seq_len=%d, restarting", step, seq_len
                    )
                    iterator = iter(loader)
                    batch = next(iterator)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                # skip_logits must be explicit: liger's default gates on self.training, and this
                # backbone stays in eval() to keep dropout off, so the default would silently fall
                # back to materializing the (L, vocab) logits.
                loss = hsa_indexer_training_step(
                    model, trainer, input_ids=input_ids,
                    skip_logits=True if args.liger else None,
                )
                (loss / args.accum_steps).backward()
                accumulated += float(loss) / args.accum_steps

            if world_size > 1:
                # Before clipping, so every rank clips the same vector. Divide by WORLD_SIZE, not
                # dp_world_size -- see the gated script's note on _ScatterSequence.
                average_gradients(params, world_size)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            if world_size > 1 and step % args.log_every == 0:
                accumulated = all_reduce_mean(accumulated, device)

            window.append(accumulated)
            reached_max = bool(args.max_steps) and step + 1 >= args.max_steps
            if will_log:
                entropy = trainer.mean_chunk_entropy()
                top1 = trainer.mean_mass_top1()
                topq = trainer.mean_mass_topquarter()
                lse_corr = trainer.mean_score_lse_corr()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "step %4d/%d L=%-6d lm_loss %.4f (avg %.4f) |g| %.3f lr %.2e "
                    "entropy %s top1 %s top25%% %s lse_corr %s peak %.1f GiB %.1f s/step",
                    step, args.total_steps, seq_len, accumulated,
                    sum(window) / len(window), float(grad_norm),
                    lr_schedule.get_last_lr()[0],
                    # THE diagnostic. Normalized to [0, 1]: 1.0 = the router is still mixing chunks
                    # uniformly and has learned no ranking, 0 = fully committed. A loss that descends
                    # while this sits at 1.0 means the router learned to USE a blunt average rather
                    # than to CHOOSE -- the one failure this objective does not rule out
                    # structurally, and the loss curve cannot distinguish it.
                    f"{entropy:.4f}" if entropy is not None else "n/a",
                    # How concentrated the mass is. Rising means a top-k truncation at inference
                    # loses less.
                    f"{top1:.3f}" if top1 is not None else "n/a",
                    # Mass on the top quarter of chunks -- the closest training-time proxy for what
                    # the eval's --topk budget will actually retain.
                    f"{topq:.3f}" if topq is not None else "n/a",
                    # Spearman against the CLOSED-FORM optimum (the chunk log-sum-exp; see
                    # ROUTER_LEARNABILITY.md section 6). No other arm has a target it can measure
                    # against directly. Rising = learning the right thing; flat near 0 = not,
                    # whatever the loss does.
                    f"{lse_corr:+.3f}" if lse_corr is not None else "n/a",
                    peak, (time.time() - started) / (step - start_step + 1),
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
                                # Normalized chunk-weight entropy in [0, 1]. THE readout on whether
                                # the router COMMITTED. Worth plotting per layer: a layer stuck at
                                # 1.0 is a layer whose router is averaging rather than choosing.
                                "chunk_entropy_mean": entropy,
                                "chunk_entropy": {
                                    str(k): v for k, v in trainer.chunk_entropy.items()
                                },
                                # Mass concentration. mass_topquarter is the training-time estimate
                                # of what the eval's --topk truncation retains.
                                "mass_top1_mean": top1,
                                "mass_top1": {str(k): v for k, v in trainer.mass_top1.items()},
                                "mass_topquarter_mean": topq,
                                "mass_topquarter": {
                                    str(k): v for k, v in trainer.mass_topquarter.items()
                                },
                                # Spearman against the closed-form optimum (chunk LSE). The one
                                # diagnostic in this package that needs no oracle and no second
                                # forward pass.
                                "score_lse_corr_mean": lse_corr,
                                "score_lse_corr": {
                                    str(k): v for k, v in trainer.score_lse_corr.items()
                                },
                                "chunk_size": args.chunk_size,
                                "chunk_aggregate": args.chunk_aggregate,
                                "peak_gib": peak,
                                "batch_size": args.batch_size,
                                "accum_steps": args.accum_steps,
                                "tokens": seqs_per_step * seq_len,
                            }
                        )
                        + "\n"
                    )
                    metrics_handle.flush()
                window = window[-50:]

            if rank == 0 and args.save_every and (step + 1) % args.save_every == 0:
                save(out_dir / f"step{step + 1}.pt", model, args, step + 1, {"loss": accumulated},
                     optimizer=optimizer, lr_schedule=lr_schedule)

            if reached_max:
                logger.info(
                    "reached --max-steps %d of the %d-step schedule; stopping before decay",
                    args.max_steps, args.total_steps,
                )
                break

            if args.dry_run and step >= 1:
                logger.info("dry run complete")
                break
    finally:
        if metrics_handle:
            metrics_handle.close()

    if rank == 0:
        save(out_dir / "final.pt", model, args, step + 1,
             optimizer=optimizer, lr_schedule=lr_schedule)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    logger.info("done in %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
