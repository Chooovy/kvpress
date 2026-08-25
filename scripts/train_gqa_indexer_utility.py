# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer by **utility self-distillation**, on longmino.

The fifth arm, beside :mod:`scripts.train_gqa_indexer` (attention-KL distillation),
:mod:`scripts.train_gqa_indexer_e2e` (additive gate), :mod:`scripts.train_gqa_indexer_exact_k`
(exact-K chunk subset) and :mod:`scripts.train_gqa_indexer_hsa` (two-level chunk attention). The
target, from ``differentiable_topk_for_sparse_attention.md`` §31::

    u_j = -dL/db_j = -alpha_j * <dL/do, v_j - o>

    python -m scripts.train_gqa_indexer_utility --data-root RAW --tokenized TOK \\
        --schedule 8192:300,16384:300,32768:900 --max-steps 600

What is different about this arm
--------------------------------
**The forward pass is unmodified.** No gate, no chunk mixture, no hard subset, no straight-through
estimator -- attention runs exactly as the frozen backbone runs it, and
``test_forward_is_bit_identical_to_the_unhooked_model`` pins that. The router is supervised by a
target read out of the backbone's own *backward*.

The consequence is worth stating rather than discovering: since the router is not on the forward path,
``dL_LM/dtheta_router`` is **None** -- absent, not small (``test_lm_loss_alone_gives_the_router_no_
gradient``). So ``loss = loss_rank`` is the whole objective and this is a **distillation** arm, in the
same class as ``train_gqa_indexer.py``, not an end-to-end one. What it changes is the *teacher*, and
that change is large. Against the true single-key drop effect on real text:

=================================  ==================
teacher                            Spearman vs truth
=================================  ==================
``alpha`` (attention-KL uses this) **+0.037**
``u`` (this arm)                   **+0.991**
=================================  ==================

``alpha`` is nearly *uninformative* about which keys matter, because a key can hold a lot of attention
while its value already sits at the row's output -- and ``u``'s ``v_j - o`` factor is exactly that
correction. This is the mechanism behind SAS's 96.8% attention mass at 79.5% accuracy.

The result to expect, and why the run is still worth doing
---------------------------------------------------------
``u``'s ranking is **dominated by a factor the router cannot observe**. It factors as ``alpha_j``
(a function of ``q . k`` -- reachable) times ``<dL/do, v_j - o>`` (a function of ``v_j`` and of the
loss direction -- a ``q . k`` scorer sees neither). Measured on Qwen3-8B:
``spearman(u, alpha)`` = +0.03 to +0.32, ``spearman(u, value term)`` = +0.752, and the best
construction that additionally uses ``v`` magnitude reaches only +0.24. So there is a **ceiling** on
``lm_loss``-free ranking well below 1, and it is a property of the hypothesis class rather than of
this loss.

``score_corr`` in the metrics is measured against that ceiling every logged step. **That is the
number this run answers.** A plateau near +0.3 means the router has converged to the limit of what
``qi . ki`` can represent, and the next move is architectural (let the indexer see values) rather than
another loss. A correlation that keeps climbing past it means the probes were measured on too shallow
a truncation -- which has already inverted one conclusion in this investigation, so it is a live
possibility rather than a hedge.

One caveat on the target that looks like a result
-------------------------------------------------
``u`` contains ``dL/do``, computed from the **label**. Selecting top-K by ``u`` beats *dense
attention itself* -- 15.3 against 18.66 row loss at K=32 of 511 keys. That is target leakage, not a
better operator. It is legitimate for a teacher, but it means ``u``'s absolute quality is not an
achievable bound and any part of its ranking that exists only because it knows the answer is
unlearnable in principle.

Flags that are absent on purpose
--------------------------------
* ``--stage`` / ``--pin-mode`` / ``--gate-budget`` -- the forward is dense and ungated, so there is
  no scope to choose and no no-op hole to pin against. The gate cannot flatten because there is no
  gate.
* ``--n-candidate`` / ``--explore-frac`` -- **no candidate pool, by construction**. One backward
  assigns a utility to *every* key of a sampled row, which is precisely the exact-K arm's measured
  dead end removed: 11-15% of oracle-best chunks never entered its ``M=32`` pool, and a chunk outside
  the pool appears nowhere in the graph, so no backward estimator can reach it.
* ``--chunk-size`` -- selection is per token. Chunk aggregation is a separate post-processing step on
  token scores (:mod:`~kvpress.presses.gqa_indexer.aggregate`), and the train-consistent chunk-wise
  eval measured *worse* (17.2 RULER) than token-wise for the exact-K arm, so it is not adopted here.
* ``--liger`` is accepted but does nothing useful -- see the flag's own help.

Everything that is not the objective is imported from the other scripts rather than reimplemented --
WSD schedule, curriculum, loader, seeding, FFN-SP, gradient averaging, checkpoint format. A run of
this against a run of those differs in the objective and nothing else.
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
    UtilityIndexerTrainer,
    indexer_state_dict,
    load_indexer_state_dict,
    utility_indexer_training_step,
)
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    SUBSETS,
    LengthSchedule,
    describe_subsets,
    read_index,
)
from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402

# Imported, not reimplemented: these are exactly the pieces that must not differ between the arms,
# or the comparison stops being about the objective.
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

logger = logging.getLogger("train_gqa_indexer_utility")


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

    Same layout as the other arms' ``save`` (an ``indexer`` state dict under a ``config`` blob), so a
    checkpoint from any objective loads into the press the same way and ``evaluate_sparse.py`` reads
    all of them. ``objective`` records which one produced it, since the weights alone do not say.

    ``score_scale`` is recorded for the same reason ``chunk_size`` is in the HSA arm: it is **not**
    recoverable from the weights. Here it does not change the *ranking* the router learned (only score
    differences reach the loss, and a positive multiplier changes no order), so unlike the HSA arm's it
    is metadata rather than a correctness requirement -- but a reader comparing two runs needs to know
    which margin scale each was trained at.

    ``gate_scale`` rides along untrained. This objective never gives it a gradient -- a global
    multiplier is unidentifiable from a ranking loss -- and it is kept only so the checkpoint stays
    byte-compatible with the gated arm's, so ``--init-from`` works in both directions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "objective": "utility_self_distillation",
            "model": args.model,
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim if args.scorer == "scalar" else None,
            "scalar_pos_slope": args.scalar_pos_slope if args.scorer == "scalar" else None,
            # The RESOLVED number, not the None that means "derive it".
            "score_scale": getattr(args, "resolved_score_scale", args.score_scale),
            # What the supervision actually saw. Not recoverable from the weights, and the whole
            # comparison against the other arms is at matched steps -- so a reader has to be able to
            # tell how many rankings per step produced these weights.
            "n_rows": args.n_rows,
            "n_pairs": args.n_pairs,
            "band": args.band,
            "budget": args.budget,
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
        help="read pre-tokenized .npy shards from here. The same corpus the other arms use -- share "
        "it rather than tokenizing five times.",
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
        "and constraint as the other arms: ranks in one SP group read the SAME sequence, so the "
        "loader shards by data-parallel rank.",
    )
    model_group.add_argument(
        "--liger", action="store_true",
        help="fuse lm_head into the cross-entropy. Accepted for parity with the other arms, but it "
        "saves LESS here than there and is not needed: this arm's target is dL/do at each attention "
        "output, which the backward produces either way, and the (L, vocab) logits are freed as soon "
        "as the backward passes them -- they are not RETAINED for a router that sits in the forward. "
        "Harmless to enable.",
    )
    model_group.add_argument(
        "--init-from", default=None,
        help="load indexer WEIGHTS only and start a fresh schedule. Accepts a checkpoint from any "
        "arm: the parameter names are shared, and gate_scale (which this objective never trains) is "
        "carried along untouched.",
    )
    model_group.add_argument(
        "--resume-from", default=None,
        help="continue an interrupted run: restore weights, AdamW state and LR-schedule position. "
        "Mutually exclusive with --init-from.",
    )

    target = parser.add_argument_group("target and sampling")
    target.add_argument(
        "--n-rows", type=int, default=16,
        help="query rows supervised per layer per step. NOT a candidate pool: the utility is exact "
        "for EVERY key of a sampled row, which is the property section 31 exists for. Only the set of "
        "queries supervising the router is thinned, and the router's parameters are shared across all "
        "of them -- 16 rows x 36 layers x 8 KV heads is ~4600 full rankings per step. Raising it "
        "costs one (B, Hkv, r, Sk) score matrix per layer, a few MiB.",
    )
    target.add_argument(
        "--n-pairs", type=int, default=64,
        help="boundary pairs drawn per row. A row of 8192 keys has 34M pairs, so the full pairwise "
        "loss is out of reach; see --band for why a small sample near the boundary is not a "
        "compromise.",
    )
    target.add_argument(
        "--band", type=int, default=32,
        help="rank half-width of the sampling window around the budget. SMALL ON PURPOSE (section "
        "23.3): top-k depends only on the order ACROSS the K-th boundary, so a pair at ranks 3 and "
        "7000 is already ordered right by any usable router and its gradient is wasted. The band is "
        "taken around the ROUTER's own current ranking, which makes the sampler self-correcting -- it "
        "tracks where the router is still uncertain. A band as wide as the row degenerates to uniform "
        "sampling.",
    )
    target.add_argument(
        "--budget", type=int, default=None,
        help="rank the top-k boundary sits at (default: each row's midpoint). The default is right "
        "when the eval budget is a compression RATIO rather than a fixed count, which is how "
        "--compression-ratio 0.5 evaluates.",
    )
    target.add_argument(
        "--score-scale", type=float, default=None,
        help="multiplier on the router's raw qi.ki before the loss (default: head_dim ** -0.5). It "
        "cannot change what the router is asked to learn -- the loss reads only score ORDER. It "
        "matters for softplus: IndexerNorm leaves the raw dot at std ~sqrt(head_dim) = 11.3, where "
        "softplus is effectively a hinge, either saturated flat (no gradient on badly-ordered pairs) "
        "or linear -- losing the soft margin near the boundary that --band exists to exploit.",
    )
    target.add_argument(
        "--attach-score-input", action="store_true",
        help="let the ranking loss's gradient flow back into hidden_states. OFF by default, for a "
        "simpler reason than in the other arms: there the score feeds the forward and attaching it "
        "created a feedback loop that drove grad_norm to nan at 36 layers. Here the score feeds only "
        "the loss, so there is no loop -- but the ranking loss would then deposit gradient into the "
        "residual stream WHILE the backbone's own backward is still running, making the result depend "
        "on hook ordering. Detached keeps the LM backward exactly the frozen model's.",
    )
    target.add_argument(
        "--no-normalize-weights", action="store_true",
        help="do NOT rescale each row's pair weights to mean 1. The ablation, and it DOES NOT TRAIN: "
        "u is proportional to alpha_j (~1/Sq) times dL/do (which carries the LM loss's 1/(B*Sq) mean), "
        "so |u| ~ 1/Sq^2 -- measured mean |u| falls 4x per doubling of the sequence, reaching 3.5e-10 "
        "at 8K on Qwen3-8B with a router gradient norm of ~3e-8. There AdamW's eps=1e-8 dominates its "
        "denominator and the optimizer stops being scale-invariant (realized step is 42.9%% of ideal at "
        "gradient 1e-8, 8.8%% at 1e-9, 1.0%% at 1e-10), while --grad-clip, an absolute threshold, never "
        "fires. Both scale with Sq, so the EFFECTIVE LEARNING RATE becomes a function of the curriculum "
        "stage -- 16x between 8K and 32K, with nothing in the loss curve to show it. Use this flag only "
        "to reproduce that.",
    )
    target.add_argument(
        "--no-recall", action="store_true",
        help="skip the top-k recall diagnostic. Cheap; leave it on -- it is closer to what the eval "
        "does than the rank correlation, since inference takes a top-k.",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:300,16384:300,32768:300", help="SEQ_LEN:STEPS,...")
    optim.add_argument(
        "--max-steps", type=int, default=0,
        help="stop after this many optimizer steps regardless of the schedule total (0 = the whole "
        "schedule). The LR is still computed from the full --schedule, so a run can be truncated at a "
        "defined point WITHOUT reshaping WSD -- which is what makes a step-600 checkpoint comparable "
        "against the other arms at the same step.",
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
    io.add_argument("--out", default="checkpoints/gqa_indexer_utility")
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
            "start that restarts the schedule and Adam, the second continues an interrupted run with "
            "both preserved. Pick one."
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
                f"{dp_world_size} replica(s) x --batch-size {args.batch_size} = {per_replica_step}. "
                "Gradient accumulation can only reach multiples of that."
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

    # gate_scale=True even though this objective never trains it: a global multiplier is
    # unidentifiable from a ranking loss. It keeps the checkpoint byte-compatible with the gated arm's
    # so --init-from works in both directions and evaluate_sparse.py's has_gate detection behaves
    # identically for all arms.
    press_kwargs = {
        "compression_ratio": args.compression_ratio,
        "scorer_attr": args.scorer_attr,
        "gate_scale": True,
        "n_sink": args.press_n_sink,
        "scorer": args.scorer,
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
                "load and the run would silently start from scratch."
            )
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        logger.info(
            "initialized indexer from %s (objective=%s)",
            args.init_from, (payload.get("config") or {}).get("objective", "unknown"),
        )

    trainer = UtilityIndexerTrainer(
        press=press,
        n_rows=args.n_rows,
        n_pairs=args.n_pairs,
        band=args.band,
        budget=args.budget,
        score_scale=args.score_scale,
        normalize_weights=not args.no_normalize_weights,
        detach_score_input=not args.attach_score_input,
        measure_recall=not args.no_recall,
        # THE accumulation subtlety. The ranking loss is backwarded from inside the LM loss's
        # backward, so this driver's own scaling cannot reach it -- without this factor a run with
        # --accum-steps 8 would apply 8x the intended router gradient, which changes only the
        # effective learning rate and so no diagnostic would reveal it.
        loss_scale=1.0 / args.accum_steps,
    )
    _first_attn = get_language_model(model).layers[0].self_attn
    args.resolved_score_scale = trainer.resolved_score_scale(press.get_indexer(_first_attn))
    logger.info(
        "score_scale = %.6g (%s); loss_scale = %.6g (= 1 / accum_steps %d)",
        args.resolved_score_scale,
        "explicit" if args.score_scale is not None else "head_dim ** -0.5",
        trainer.loss_scale, args.accum_steps,
    )

    # Called here, not left to hooks(), so the parameter list handed to the optimizer is the same
    # object the run trains.
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    trainable = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable %.2fM of %.2fB parameters (%.3f%%); objective = pairwise ranking against "
        "u_j = -alpha_j <dL/do, v_j - o>, %d rows x %d pairs per layer, band +-%d. The FORWARD IS "
        "UNMODIFIED dense attention -- the router is not on it, so the LM loss gives it no gradient "
        "and loss = loss_rank is the whole objective.",
        trainable / 1e6, total_params / 1e9, 100 * trainable / total_params,
        args.n_rows, args.n_pairs, args.band,
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
                    logger.info("step %d: seq_len %d -> %d", step, current_len, seq_len)
                else:
                    logger.info("step %d: starting at seq_len=%d", step, seq_len)
                loader = loader_for(
                    seq_len, args, tokenizer, dp_rank, dp_world_size, batch_size=args.batch_size
                )
                iterator = iter(loader)
                current_len = seq_len

            optimizer.zero_grad(set_to_none=True)
            rank_loss_acc, lm_loss_acc = 0.0, 0.0
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
                # NO .backward() on the return value. The router's entire gradient was already
                # deposited by the reentrant backwards inside this call's LM backward; calling
                # backward() here would re-run the whole backbone backward AND every ranking loss,
                # doubling the router's gradient. The LM loss is returned detached, for logging.
                lm_loss = utility_indexer_training_step(
                    model, trainer, input_ids=input_ids,
                    skip_logits=True if args.liger else None,
                    seed=args.seed + step * args.accum_steps + micro,
                )
                lm_loss_acc += float(lm_loss) / args.accum_steps
                rank_loss = trainer.mean_rank_loss()
                if rank_loss is not None:
                    rank_loss_acc += rank_loss / args.accum_steps

            if world_size > 1:
                # Before clipping, so every rank clips the same vector. Divide by WORLD_SIZE, not
                # dp_world_size -- see the gated script's note on _ScatterSequence.
                average_gradients(params, world_size)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            if world_size > 1 and step % args.log_every == 0:
                rank_loss_acc = all_reduce_mean(rank_loss_acc, device)
                lm_loss_acc = all_reduce_mean(lm_loss_acc, device)

            window.append(rank_loss_acc)
            reached_max = bool(args.max_steps) and step + 1 >= args.max_steps
            if will_log:
                corr = trainer.mean_score_corr()
                recall = trainer.mean_utility_recall()
                u_scale = trainer.mean_utility_scale()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "step %4d/%d L=%-6d rank_loss %.4f (avg %.4f) lm_loss %.4f |g| %.3f lr %.2e "
                    "score_corr %s recall %s |u| %s peak %.1f GiB %.1f s/step",
                    step, args.total_steps, seq_len, rank_loss_acc,
                    sum(window) / len(window), lm_loss_acc, float(grad_norm),
                    lr_schedule.get_last_lr()[0],
                    # THE diagnostic. The rank_loss above is weighted by |u_i - u_j|, which scales
                    # with ||dL/do||, so it falls when the batch gets EASIER -- it cannot say whether
                    # the router is learning. This is against a fixed quantity. Read it against the
                    # +0.03 to +0.32 representability ceiling the probes measured: a plateau there is
                    # the hypothesis class (change the architecture), not the optimizer (keep tuning).
                    f"{corr:+.3f}" if corr is not None else "n/a",
                    # Closer to what the eval does, since inference takes a top-k.
                    f"{recall:.3f}" if recall is not None else "n/a",
                    # Teacher scale. If this collapses, every |u_i - u_j| weight goes to 0 and the
                    # loss falls to 0 while the router learns nothing -- which reads as convergence.
                    f"{u_scale:.2e}" if u_scale is not None else "n/a",
                    peak, (time.time() - started) / (step - start_step + 1),
                )
                if metrics_handle:
                    metrics_handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "seq_len": seq_len,
                                # The objective's own value. Weighted by the teacher's utility gaps,
                                # so NOT comparable across steps on its own -- see score_corr.
                                "rank_loss": rank_loss_acc,
                                # Dense LM loss. Constant-in-expectation here (the forward is the
                                # frozen model's), so it is a data-difficulty readout rather than a
                                # training curve -- useful precisely for normalizing rank_loss.
                                "lm_loss": lm_loss_acc,
                                "grad_norm": float(grad_norm),
                                "lr": lr_schedule.get_last_lr()[0],
                                # THE readout. Worth plotting per layer: the representability ceiling
                                # need not be uniform in depth, and if the deep layers plateau while
                                # early ones keep climbing that is a different conclusion than a
                                # uniform plateau.
                                "score_corr_mean": corr,
                                "score_corr": {str(k): v for k, v in trainer.score_corr.items()},
                                "utility_recall_mean": recall,
                                "utility_recall": {
                                    str(k): v for k, v in trainer.utility_recall.items()
                                },
                                "utility_scale_mean": u_scale,
                                "n_rows": args.n_rows,
                                "n_pairs": args.n_pairs,
                                "band": args.band,
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
                save(out_dir / f"step{step + 1}.pt", model, args, step + 1,
                     {"rank_loss": rank_loss_acc, "score_corr": trainer.mean_score_corr()},
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
