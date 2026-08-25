# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer end to end through an **exact-K chunk subset**, on longmino.

The third arm, beside :mod:`scripts.train_gqa_indexer` (distillation) and
:mod:`scripts.train_gqa_indexer_e2e` (gated attention). Where the gated arm adds the router's score
inside the softmax::

    out = softmax(scale * q @ k^T + gate_scale * qi @ ki^T) @ v

this one *replaces* attention with a genuine ``K``-of-``M`` chunk subset, sampled in the forward and
differentiated through the exact inclusion marginals::

    g   = (z - mu).detach() + mu           # z ~ exactly-K subset,  mu = P(z_i = 1 | sum z = K)
    out = (g * exp(a)) / sum(g * exp(a)) @ v

    python -m scripts.train_gqa_indexer_exact_k --data-root RAW --tokenized TOK \\
        --schedule 8192:300,16384:300,32768:900 --max-steps 600

Why bother, in one line
-----------------------
An additive gate can go **flat** along the key axis, and a flat gate is inert -- softmax is
shift-invariant -- so the model reverts to the frozen backbone and the LM loss is satisfied with no
ranking learned. The gated arm patches that with ``--pin-mode``. Here the forward commits to exactly
``K`` chunks, so no configuration of the scores reproduces dense attention and there is nothing to
pin. See ``kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md`` §7 and
``HANDOFF_exact_k_subset.md``.

Everything that is not the objective is imported from the gated script or the distillation one
rather than reimplemented -- the WSD schedule, curriculum handling, loader, seeding, FFN-SP,
gradient averaging, Liger patching, checkpoint format. That is the point: a run of this against a
run of those differs in the objective and nothing else.

What differs from the gated arm, and why
----------------------------------------
* **No ``--pin-mode``, no ``--gate-budget``.** Nothing to pin; see above. Passing one would be a
  flag that silently does nothing, which this package treats as an error rather than a convenience.
* **No ``gate_scale``.** The score is a routing logit that goes through ``sigmoid`` into a Bernoulli
  probability, so a positive multiplier is a temperature, not a scale match. The press still builds
  the parameter so a checkpoint stays interchangeable with the gated arm's (``--init-from`` works
  both ways) -- it is simply not read.
* **New geometry flags** ``--chunk-size``, ``--query-block``, ``--topk-chunk``, ``--n-candidate``.
  ``--n-candidate`` (``M``) is the one that sets the step time; see "Cost" below.
* **Different diagnostics.** ``gate_scales`` / ``gate_sparsity`` have no counterpart. In their place:
  ``marginal_entropy`` (has the router *committed*, or is it still uniform at ``K/M``),
  ``effective_topk`` (is the configured budget actually reachable near the diagonal), and
  ``jaccard`` (does the stochastic forward make the selection unstable). ``marginal_entropy`` is the
  one to watch -- a loss that falls while it stays flat means the router learned to *use* a random
  subset rather than to *choose* one.

Cost, measured on an H20 rather than extrapolated
-------------------------------------------------
``HANDOFF_exact_k_subset.md`` §4 concluded from CPU timings that this was "dead on arrival" at
~1690 s per layer per step. That was wrong by about four orders of magnitude, and wrong
structurally -- the DP is **launch-bound** on GPU, so its cost is independent of the row count. See
:mod:`kvpress.presses.gqa_indexer.exact_k_attention` for the numbers. What it costs in practice, at
Qwen3-8B geometry with ``chunk_size=64, query_block=256, M=32, K=8``:

=========  ===================  ==============  =================
seq_len    ms/layer (fwd+bwd)   peak (1 layer)  dense SDPA, same
=========  ===================  ==============  =================
8192       151                  2.8 GiB         34.5 ms
16384      263                  3.5 GiB         87.6 ms
=========  ===================  ==============  =================

So ~3-4x dense attention per layer, which is the price of a differentiable discrete subset. Raising
``--n-candidate`` scales that roughly linearly; raising ``--topk-chunk`` is nearly free.
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
    ExactKIndexerTrainer,
    exact_k_indexer_training_step,
    indexer_state_dict,
    load_indexer_state_dict,
)
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

logger = logging.getLogger("train_gqa_indexer_exact_k")


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

    The routing geometry is recorded too. It is **not** recoverable from the weights -- ``chunk_size``
    and ``query_block`` are not parameter shapes -- and evaluating at a different ``chunk_size`` than
    training used is a silent mismatch, exactly the class of bug ``rope_dim`` is guarded against
    elsewhere in this package.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "objective": "exact_k_subset",
            "model": args.model,
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim if args.scorer == "scalar" else None,
            "scalar_pos_slope": args.scalar_pos_slope if args.scorer == "scalar" else None,
            # Routing geometry: not recoverable from the weights, and a mismatch at eval is silent.
            "chunk_size": args.chunk_size,
            "query_block": args.query_block,
            "topk_chunk": args.topk_chunk,
            "n_candidate": args.n_candidate,
            "explore_frac": args.explore_frac,
            "n_sink_chunk": args.n_sink_chunk,
            "n_local_chunk": args.n_local_chunk,
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
        help="tokens per chunk. Selection happens at chunk granularity, so this should match the "
        "chunk_size the press uses at eval -- it is recorded in the checkpoint for that reason.",
    )
    route.add_argument(
        "--query-block", type=int, default=256,
        help="queries sharing one chunk subset. A real modelling concession (coarser than per-query "
        "selection), though NOT a performance requirement -- the GPU measurement contradicts the "
        "CPU analysis that claimed it was. 1 restores per-query selection and is affordable.",
    )
    route.add_argument(
        "--topk-chunk", type=int, default=8,
        help="K: chunks each query block commits to. 0 derives it from the compression ratio. "
        "Nearly free to raise -- the DP's cost is independent of K.",
    )
    route.add_argument(
        "--n-candidate", type=int, default=32,
        help="M: the candidate pool the subset is drawn from. THE knob that sets step time -- the "
        "DP is O(M) sequential kernel launches and the training-time attention is over "
        "M * chunk_size keys. Budget this first.",
    )
    route.add_argument(
        "--explore-frac", type=float, default=0.10,
        help="fraction of the pool drawn at random rather than by score. Without it a chunk outside "
        "top-M appears nowhere in the graph, so it receives exactly zero gradient and can never be "
        "promoted -- the failure that kept the selected-gate proxy at 0.0%% recall. 0 is the "
        "ablation and warns.",
    )
    route.add_argument("--n-sink-chunk", type=int, default=1, help="pool slots reserved for the leading chunks")
    route.add_argument("--n-local-chunk", type=int, default=1, help="pool slots reserved at the block's diagonal")
    route.add_argument("--chunk-aggregate", choices=("mean", "max"), default="mean")
    route.add_argument("--query-aggregate", choices=("mean", "max"), default="mean")
    route.add_argument(
        "--hard", action="store_true",
        help="take the deterministic top-K instead of sampling. The sampling ablation: a "
        "deterministic selection cannot explore, so a chunk's score is only ever compared against "
        "chunks already selected. Warns.",
    )
    route.add_argument(
        "--attach-score-input", action="store_true",
        help="let the score's gradient flow back into hidden_states. OFF by default because it "
        "DIVERGES: the score is a function of hidden_states, so this path deposits gradient in the "
        "residual stream that every router below then re-amplifies -- measured grad_norm 8.6e13 at "
        "12 layers, inf at 24, nan at 36, against 1.1e6 detached. The per-layer amplification is "
        "10-50x, against 1.1x for the same backbone running dense attention. What it costs: a "
        "router loses the second-order term 'my routing changed what a lower router sees'. Each "
        "router is still trained by the full LM loss through its own attention. Use this flag to "
        "reproduce the divergence.",
    )
    route.add_argument(
        "--route-scale", type=float, default=None,
        help="multiplier on the pooled chunk score before it becomes a Bernoulli logit "
        "(default: head_dim ** -0.5). REQUIRED, not cosmetic: IndexerNorm leaves the raw qi.ki dot "
        "product at std ~sqrt(head_dim) = 11.3, and at that scale 31%% of the exact marginals "
        "saturate to 0 or 1 -- so 31%% of candidates get no usable gradient, losing the "
        "boundary-credit property this method exists for. Measured unscaled: grad_norm 6.5e4 and no "
        "descent. A temperature, so sweep it.",
    )
    route.add_argument(
        "--score-tile-bytes", type=int, default=0,
        help="bytes budgeted for one score tile's token logits (0 = 64 MiB). The router's "
        "(B, Hkv, Sq, Sk) logits are 8 GiB in fp32 at 16K while the pooled result is 0.5 MiB, so "
        "the wide tensor is built a tile at a time. A BYTE budget rather than a query count, "
        "because the right count depends on Sk and the head count -- a count that fits at 8K does "
        "not at 16K. Memory knob only; the result is tile-invariant.",
    )
    route.add_argument(
        "--no-checkpoint-scores", action="store_true",
        help="retain each score tile's token logits instead of recomputing them. 288 GiB across 36 "
        "layers at 16K, for a result that is 0.5 MiB per layer. Debugging only.",
    )
    route.add_argument(
        "--no-checkpoint-dp", action="store_true",
        help="retain the marginals' DP instead of recomputing it in the backward. 11.13 GiB vs 0.28 "
        "across 36 layers, so this is for debugging only.",
    )
    route.add_argument(
        "--no-checkpoint-attention", action="store_true",
        help="retain each query tile's attention. 69 GiB vs 3.3 for ONE layer at 16K. Debugging only.",
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
    io.add_argument("--out", default="checkpoints/gqa_indexer_exact_k")
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
    if args.topk_chunk and args.topk_chunk > args.n_candidate:
        parser.error(
            f"--topk-chunk {args.topk_chunk} exceeds --n-candidate {args.n_candidate}: the subset is "
            "drawn from the candidate pool, so K <= M by construction."
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

    trainer = ExactKIndexerTrainer(
        press=press,
        chunk_size=args.chunk_size,
        query_block=args.query_block,
        topk_chunk=args.topk_chunk,
        n_candidate=args.n_candidate,
        keep_ratio=1.0 - args.compression_ratio,
        explore_frac=args.explore_frac,
        n_sink_chunk=args.n_sink_chunk,
        n_local_chunk=args.n_local_chunk,
        chunk_aggregate=args.chunk_aggregate,
        query_aggregate=args.query_aggregate,
        detach_score_input=not args.attach_score_input,
        route_scale=args.route_scale,
        score_tile_bytes=args.score_tile_bytes,
        checkpoint_scores=not args.no_checkpoint_scores,
        checkpoint=not args.no_checkpoint_dp,
        checkpoint_attention=not args.no_checkpoint_attention,
        hard=args.hard,
    )
    # Called here, not left to hooks(), so the parameter list handed to the optimizer is the same
    # object the run trains.
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    trainable = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable %.2fM of %.2fB parameters (%.3f%%); objective = LM loss through an exact-K chunk "
        "subset: chunk=%d query_block=%d K=%s M=%d explore=%.2f (sink %d, local %d)%s",
        trainable / 1e6, total_params / 1e9, 100 * trainable / total_params,
        args.chunk_size, args.query_block,
        args.topk_chunk or f"{1 - args.compression_ratio:g} * n_chunk", args.n_candidate,
        args.explore_frac, args.n_sink_chunk, args.n_local_chunk,
        " [HARD selection -- sampling ablation]" if args.hard else "",
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
                # The selection is compared against the previous step's, and n_qblock changes with
                # seq_len -- selection_jaccard returns NaN on a shape mismatch rather than raising,
                # but clearing here makes the boundary explicit instead of relying on that.
                trainer._previous_selection.clear()

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
                loss = exact_k_indexer_training_step(
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
                entropy = trainer.mean_marginal_entropy()
                effective = trainer.mean_effective_topk()
                jaccard = trainer.mean_jaccard()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "step %4d/%d L=%-6d lm_loss %.4f (avg %.4f) |g| %.3f lr %.2e "
                    "H(mu) %s eff_K %s jaccard %s peak %.1f GiB %.1f s/step",
                    step, args.total_steps, seq_len, accumulated,
                    sum(window) / len(window), float(grad_norm),
                    lr_schedule.get_last_lr()[0],
                    # THE diagnostic. Max at init (uniform marginals at K/M), falling as the router
                    # commits. A loss that descends while this stays flat means the router learned
                    # to USE a random subset rather than to CHOOSE one -- the one failure mode
                    # exact-K does not rule out structurally.
                    f"{entropy:.4f}" if entropy is not None else "n/a",
                    # Below --topk-chunk means some query blocks cannot see K chunks, so the
                    # realized budget is smaller than the configured one.
                    f"{effective:.2f}" if effective is not None else "n/a",
                    # Selection stability across steps. The forward is stochastic; this says whether
                    # that matters.
                    f"{jaccard:.3f}" if jaccard is not None else "n/a",
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
                                # Mean Bernoulli entropy of the marginals, in nats. The readout on
                                # whether the router COMMITTED; log 2 per item at maximum
                                # indecision, 0 when decided. Worth plotting per layer: a layer
                                # stuck at its init value is a layer whose router is not choosing.
                                "marginal_entropy_mean": entropy,
                                "marginal_entropy": {
                                    str(k): v for k, v in trainer.marginal_entropy.items()
                                },
                                # Realized budget. Below topk_chunk near the diagonal, by design.
                                "effective_topk_mean": effective,
                                "effective_topk": {
                                    str(k): v for k, v in trainer.effective_topk.items()
                                },
                                # Jaccard overlap of the selected sets against the previous step.
                                # null on the first step of each curriculum stage.
                                "jaccard_mean": jaccard,
                                "jaccard": {str(k): v for k, v in trainer.jaccard.items()},
                                "chunk_size": args.chunk_size,
                                "query_block": args.query_block,
                                "topk_chunk": args.topk_chunk,
                                "n_candidate": args.n_candidate,
                                "explore_frac": args.explore_frac,
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
