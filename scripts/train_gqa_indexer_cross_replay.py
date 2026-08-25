# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a query-independent GQA indexer from the **cross-replay LM loss**, on the longmino corpus.

The third arm, alongside :mod:`scripts.train_gqa_indexer` (distillation) and
:mod:`scripts.train_gqa_indexer_e2e` (end-to-end gated attention). All three train the same object --
a per-key importance score the press uses to evict -- and differ only in where the supervision comes
from:

* **distillation**: match the frozen model's own attention weights. The score never touches the
  forward pass.
* **end-to-end**: add the score inside the attention softmax on the ordinary causal LM loss. Every
  key is judged by the queries that *follow* it -- a triangle.
* **cross-replay** (this script): prefill ``C`` densely, then replay the same tokens as ``C'`` against
  ``KV(C)`` alone, so **every** replay query sees **every** context key -- a full rectangle. One score
  is forced to serve many queries at once, which is the query-agnostic reuse value eviction actually
  needs. See ``cross_replay_e2e.md`` §4, and §7.2 for why KVzip's block-diagonal supervision is not
  this.

Everything that is not the objective is deliberately identical to the other two -- same WSD schedule,
curriculum handling, loader, seeding, gradient averaging, checkpoint format -- and is *imported* from
them rather than copied, so the three cannot drift apart. Run at the same ``--schedule`` and
``--max-steps`` and the checkpoints are comparable step for step.

    python -m scripts.train_gqa_indexer_cross_replay --data-root RAW --tokenized TOK \\
        --schedule 8192:300,16384:300,32768:900 --max-steps 600

Read the diagnostics, not the loss
----------------------------------
This objective's characteristic failure is a **clean loss curve with an untrained router**: a gate that
is flat along the key axis adds a per-row constant, which cancels in the softmax, so the model reverts
to the frozen dense backbone -- already strong -- and the LM loss is satisfied with no ranking learned.
``cross_replay_e2e.md`` §9 records four separate bugs in this work whose only symptom was
"correct-looking output, wrong quantity", every one of which passed the exactness tests. So two
readouts are logged as first-class metrics rather than as an afterthought:

* **participation** (:func:`~.cross_replay.gate_participation`) -- the effective fraction of keys the
  gate spreads over. ``~1.0`` is a flat gate and no ranking, whatever the loss says; falling towards 0
  is the concentration eviction needs.
* **the shuffle control** (``--shuffle-control-every``) -- replay loss with the learned scores permuted
  along the key axis. If destroying the ranking does not hurt, there is no ranking. This is the single
  number that separates "trained" from "trained-looking", and it is worth its two extra passes.

What this script refuses to accept, and why
-------------------------------------------
Three flags that the e2e script offers are **rejected here rather than ignored**, because each would be
a silent no-op and this package has shipped that exact bug twice already (Liger's ``skip_logits`` was
threaded through this objective for a full revision while doing nothing; gradient checkpointing was
gated on ``module.training`` and looked like it worked):

* ``--liger`` -- would fuse ``lm_head`` into ``*ForCausalLM.forward``, but this objective calls the
  **base** model (it must, to control the cache and the mask) and computes the loss itself. Use
  ``--logit-chunk``, which bounds the same tensor and is verified exact.
* ``--ffn-sp-size > 1`` -- unnecessary (16K peaks at 33.5 GiB measured, against ~93.7 for e2e) and
  unsound: FFN sequence parallelism slices the sequence per rank, while this objective runs two passes
  over the same tokens and asserts the key axis stays exactly ``|C|``.
* ``--pin-mode`` other than ``sink`` -- rejected by :class:`~.cross_replay.CrossReplayTrainer` itself.
  Under ``[C ; C']`` a ``self`` pin pins zero *visible* keys, so the normalizer goes inert and the
  flat-gate no-op reopens (``cross_replay_e2e.md`` §3).

The last one has a consequence worth stating: the e2e script's ``ablate`` mode (``--pin-mode none``,
the honest "does pinning matter" baseline) has **no counterpart here**. It is not an oversight.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    CrossReplayTrainer,
    cross_replay_training_step,
    gate_participation,
    indexer_state_dict,
    load_indexer_state_dict,
    shuffled_scores,
)
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
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

logger = logging.getLogger("train_gqa_indexer_cross_replay")


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
    blob), so ``evaluation/evaluate_indexer_press.py`` and ``--init-from`` accept a checkpoint from
    any arm. ``objective`` records which one produced it, since the weights alone do not say --
    and here that matters more than usual, because a cross-replay checkpoint and an e2e one have
    *identical* parameter names and shapes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "objective": "cross_replay_lm_loss",
            "model": args.model,
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim,
            "scalar_pos_slope": args.scalar_pos_slope,
            # The prefix branch's geometry, for the same reason as `budget` below: head_dim and
            # value_dim are parameter shapes and would fail loudly, but prefix_zero_init is not,
            # and a run whose flag differed would be a different experiment with every tensor
            # still loading cleanly.
            "prefix_head_dim": args.prefix_head_dim if args.scorer == "prefix" else None,
            "prefix_value_dim": args.prefix_value_dim if args.scorer == "prefix" else None,
            "prefix_zero_init": args.prefix_zero_init if args.scorer == "prefix" else None,
            "pin_mode": args.pin_mode,
            "n_sink": args.n_sink,
            # Recorded because it decides the gate's converged concentration, so a checkpoint whose
            # budget is unknown cannot be compared against another. This is exactly how the
            # scalar-vs-cross-replay A/B turned out to be confounded: `scalar_mid_dim` differed
            # 256-vs-0 and only the checkpoint's own config revealed it.
            "budget": args.budget,
            # Same reason as `budget`: these change what the router is trained on, so a checkpoint
            # that does not record them cannot be compared against another (§13, §16).
            "lookahead": args.lookahead,
            "demand_reduce": args.demand_reduce,
            "cross_doc_replay": args.cross_doc_replay,
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "query_chunk": args.query_chunk,
            "logit_chunk": args.logit_chunk,
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


def measure_participation(trainer: CrossReplayTrainer, n_sink: int) -> dict[int, float]:
    """
    Per-layer gate participation, from the scores the last step already computed.

    Free, unlike the e2e trainer's equivalent: a query-independent gate needs no streaming pass over
    an ``(Sq, Sk)`` history to get at it, so this can run on every logged step rather than being
    opt-in. That is the practical dividend of ``cross_replay_e2e.md`` §1.1.
    """
    out = {}
    for layer_idx in sorted(trainer._scores):
        gate = trainer.gate(layer_idx, trainer._scores[layer_idx].shape[1])
        out[layer_idx] = gate_participation(gate, min(n_sink, gate.shape[-1]))
    return out


def shuffle_control(model, trainer, input_ids, args, generator: torch.Generator) -> float:
    """
    Replay loss with the learned scores permuted, minus the unpermuted loss.

    Positive means the score carries a usable ranking: destroying the ordering costs the model
    something. ``<= 0`` means it does not, which is the "trained-looking" outcome no loss curve would
    reveal -- see the module docstring.

    Costs two extra replay passes (~27 s at 16K on an H20), hence ``--shuffle-control-every`` rather
    than every step. Runs entirely under ``no_grad`` with ``backward=False``: this measures, it must
    not contribute gradient. Chunked evaluation is legal in that mode precisely because there is no
    graph to hold.
    """
    with torch.no_grad():
        clean = cross_replay_training_step(
            model, trainer, input_ids=input_ids, backward=False, logit_chunk=args.logit_chunk
        )
        perm = torch.randperm(input_ids.shape[1], generator=generator).to(input_ids.device)
        with shuffled_scores(trainer, perm):
            shuffled = cross_replay_training_step(
                model, trainer, input_ids=input_ids, backward=False, logit_chunk=args.logit_chunk
            )
    return float(shuffled) - float(clean)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so the validation below can be tested without launching a run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True, help="longmino_256k_filtered root")
    data.add_argument("--tokenized", default=None, help="pre-tokenized corpus root")
    data.add_argument("--subsets", nargs="+", default=["2e16", "2e17"])
    data.add_argument("--take-from", choices=("head", "random"), default="random")
    data.add_argument("--shuffle-buffer", type=int, default=64)
    data.add_argument("--min-tokens", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="must be 1: cross_replay_training_step indexes pass 1's cache and the per-key scores "
        "per layer with no batch axis to reconcile, and the memory profile is set by the sequence "
        "length rather than the batch. Use --global-batch-size for the effective batch.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen3-8B")
    model_group.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    model_group.add_argument("--attn", default="sdpa", help="backbone attention kernel")
    model_group.add_argument("--compression-ratio", type=float, default=0.5)
    model_group.add_argument("--scorer-attr", default="indexer")
    model_group.add_argument(
        "--scorer",
        choices=("scalar", "prefix"),
        default="scalar",
        help="which QUERY-INDEPENDENT router to train. 'pairwise' is not offered: it would need an "
        "(Sq, Sk) gate, which is the cost the whole design avoids (cross_replay_e2e.md §1.1), and "
        "a pairwise press is rejected in score_context anyway. 'scalar' scores each key from its "
        "own hidden state; 'prefix' scores it from its whole prefix through the indexer's own "
        "causal attention -- still query-independent, so this objective's rectangle applies "
        "unchanged, and a strict superset of 'scalar' so the A/B is single-variable.",
    )
    model_group.add_argument("--scalar-mid-dim", type=int, default=0, help="0 = a linear score")
    model_group.add_argument("--scalar-pos-slope", type=float, default=DEFAULT_POS_SLOPE)
    model_group.add_argument(
        "--prefix-head-dim",
        type=int,
        default=128,
        help="--scorer prefix only: q/k width of the indexer's own prefix attention.",
    )
    model_group.add_argument(
        "--prefix-value-dim",
        type=int,
        default=128,
        help="--scorer prefix only: width of the prefix attention's v readout.",
    )
    model_group.add_argument(
        "--no-prefix-zero-init",
        dest="prefix_zero_init",
        action="store_false",
        help="--scorer prefix only: start the prefix branch at random init rather than zero. By "
        "default it is zeroed, so training starts bit-identical to --scorer scalar and 'reads the "
        "prefix' is the only variable. See the e2e script's copy of this flag for why the zero is "
        "an escapable saddle rather than a dead start.",
    )
    model_group.set_defaults(prefix_zero_init=True)
    model_group.add_argument("--press-n-sink", type=int, default=4)
    model_group.add_argument(
        "--init-from",
        default=None,
        help="load indexer WEIGHTS only and start a fresh schedule. Accepts a distillation or e2e "
        "checkpoint (same parameter names), which is the intended way to warm-start.",
    )
    model_group.add_argument(
        "--resume-from",
        default=None,
        help="continue an interrupted run: weights, AdamW state and LR position, skipping the steps "
        "already done. Requires the same --schedule. Mutually exclusive with --init-from.",
    )
    # Accepted only so they can be REJECTED with a reason. A flag that silently does nothing is the
    # failure mode this whole file is organised against; see the module docstring.
    model_group.add_argument("--liger", action="store_true", help=argparse.SUPPRESS)
    model_group.add_argument("--ffn-sp-size", type=int, default=1, help=argparse.SUPPRESS)

    gate = parser.add_argument_group("gate")
    gate.add_argument(
        "--pin-mode",
        default="sink",
        help="only 'sink' is permitted. Under [C ; C'] a 'self' pin's diagonal key lies in the C' "
        "block, which this objective masks out, so it pins nothing visible, the normalizer goes "
        "inert and the flat-gate no-op reopens. Rejected in CrossReplayTrainer.__post_init__.",
    )
    gate.add_argument("--n-sink", type=int, default=4, help="leading keys exempt from the gate")
    gate.add_argument(
        "--budget",
        type=int,
        default=None,
        help="B in the gate's `s - LSE(s) + log B`, applied to the gated keys only. LEAVE AT 1. "
        "The identity `sum_gated exp(g) = B` makes B the number of pinned-sink equivalents the whole "
        "history is worth, so it decides how concentrated the gate becomes at convergence. RETRACTED: "
        "this flag used to say 'set this to the inference top-k', argued from representability; the "
        "2x2 grid measured B=1 at 48.20 and B=2048 at 20.43 on RULER 8K, i.e. B=topk costs 27.8 "
        "points, against an 18.0-point objective gap. B=n_gated remains the flat-gate no-op; unset "
        "resolves there and warns. See cross_replay_e2e.md §15.3.",
    )

    shape = parser.add_argument_group("supervision shape (cross_replay_e2e.md §16)")
    shape.add_argument(
        "--lookahead",
        type=int,
        default=None,
        help="bound on how far past its own position a replay row may see, in keys. Unset = the "
        "unbounded rectangle, which every measured arm trained on. 0 reproduces the causal triangle "
        "the e2e LM loss uses; a small positive value keeps a growing candidate set (hence the "
        "difficulty ramp the rectangle lacks). §16.3 argues the rectangle's N-way averaging is what "
        "over-concentrates every cross-replay arm. NOTE: a bounded horizon is a real mask, so it "
        "disqualifies the flex path and lands on SDPA MATH (46.7 GiB) -- lower --query-chunk.",
    )
    shape.add_argument(
        "--demand-reduce",
        choices=("sum", "max", "mean"),
        default="sum",
        help="how replay queries' demands on one key combine into dL/ds. 'sum' is plain autograd "
        "accumulation and what every measured arm used. 'max' keeps each key's strongest demand per "
        "query CHUNK instead of averaging over all N queries, which is what KVzip's max-attention "
        "label does for free because it never differentiates (§16.4); it needs >= 2 chunks, so set "
        "--query-chunk accordingly. 'mean' is the null control (same direction, 1/n_chunks scale). "
        "MEASURED: 'max' reaches the flat-gate degeneracy (six flat shuffle controls, participation "
        "rising to 0.96) and was killed -- see §17.4.1 before using it.",
    )
    shape.add_argument(
        "--cross-doc-replay",
        action="store_true",
        help="replay an UNRELATED document against KV(C) instead of C itself -- the cross-document "
        "control, promoted from an eval-time null to a TRAINING condition. An unrelated document's "
        "next token cannot be predicted from C's keys, so if the reconstruction relation is what "
        "teaches the router, this must collapse. If it instead trains to a comparable score AND lands "
        "in the same selection cluster as the rectangle arms (§18.3), reconstruction was never the "
        "teacher and the router is learning a document-independent salience -- which would explain "
        "why 26x capacity, 2048x budget and the causal mask shape all failed to move it. The donor is "
        "the next batch from the same loader, so it is real corpus text at identical length and "
        "distribution. See cross_replay_e2e.md §19.",
    )

    memory = parser.add_argument_group("memory")
    memory.add_argument(
        "--query-chunk",
        type=int,
        default=1024,
        help="replay queries per forward, each still attending to the WHOLE key axis so the "
        "rectangle is preserved exactly (unlike KVzip's chunking, §7.2). The main memory knob, and "
        "exact: measured peak at 8K is 21.9/24.4/27.8/34.6 GiB for 128/512/1024/2048 and 61.3 "
        "unchunked. Default 1024 -> 27.8 GiB at 8K, 33.5 at 16K.",
    )
    memory.add_argument(
        "--logit-chunk",
        type=int,
        default=None,
        help="rows of lm_head output to materialize at a time. Bounds the (rows, vocab) logits plus "
        "the fp32 copy cross-entropy needs -- 0.87 GiB per 1024 rows on Qwen3-8B -- independently of "
        "--query-chunk. This is the replacement for --liger, which cannot work here. At "
        "--query-chunk 1024 it buys 27.8 -> 26.0 GiB at 8K for logit-chunk 128.",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:300,16384:300", help="SEQ_LEN:STEPS,...")
    optim.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps regardless of the schedule total. Pair with a "
        "matching --schedule to truncate WITHOUT reshaping the WSD curve, which is how a run stays "
        "comparable to the other objectives at the same step number.",
    )
    optim.add_argument("--global-batch-size", type=int, default=0, help="sequences per optimizer step")
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--peak-lr", type=float, default=1e-3)
    optim.add_argument("--final-lr", type=float, default=5e-6)
    optim.add_argument("--warmup-frac", type=float, default=0.10)
    optim.add_argument("--stable-frac", type=float, default=0.60)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("io")
    io.add_argument("--out", default="checkpoints/gqa_indexer_cross_replay")
    io.add_argument("--save-every", type=int, default=200)
    io.add_argument("--log-every", type=int, default=10)
    io.add_argument("--metrics-file", default=None, help="append JSONL metrics here")
    io.add_argument(
        "--shuffle-control-every",
        type=int,
        default=0,
        help="optimizer steps between shuffle controls; 0 disables. THE readout that separates a "
        "trained router from a trained-looking one: replay loss with the learned scores permuted "
        "along the key axis, minus the unpermuted loss. <= 0 means the score carries no usable "
        "ranking, which the loss curve cannot tell you. Costs two extra replay passes (~27 s at "
        "16K), hence periodic rather than every step. 100 is a reasonable setting.",
    )
    io.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write AdamW state and LR position so --resume-from can continue the run",
    )
    io.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps")
    return parser


def validate(args, parser: argparse.ArgumentParser) -> None:
    """
    Reject the configurations that would run but not mean what they say.

    Separated from :func:`main` so it is testable on CPU without a model, a corpus or a GPU. Every
    check here corresponds to a silent failure documented in ``cross_replay_e2e.md``; none of them is
    a matter of taste.
    """
    if args.liger:
        parser.error(
            "--liger cannot work with this objective and would silently do nothing. Liger fuses "
            "lm_head into *ForCausalLM.forward and needs `labels` passed to it, but cross-replay "
            "calls the BASE model (it must, to control the cache and the rectangle mask) and computes "
            "the loss itself. This exact flag was threaded through this objective for a full revision "
            "before that was noticed (cross_replay_e2e.md §6.1). Use --logit-chunk instead: it bounds "
            "the same (rows, vocab) tensor, and it is verified exact."
        )
    if args.ffn_sp_size != 1:
        parser.error(
            f"--ffn-sp-size {args.ffn_sp_size} is not supported here, and is not needed: cross-replay "
            "peaks at a measured 33.5 GiB at 16K with --query-chunk 1024 (the e2e objective needs "
            "8-way FFN-SP only because its 16K is ~93.7 GiB). It is also unsound as written -- FFN-SP "
            "slices the sequence across ranks, while this objective runs two passes over the same "
            "tokens against a ReadOnlyCache whose key axis must stay exactly |C|, which _attention "
            "asserts. Use --query-chunk to control memory."
        )
    if args.pin_mode != "sink":
        parser.error(
            f"--pin-mode {args.pin_mode!r} is rejected: only 'sink' pins anything under [C ; C']. "
            "Query j's diagonal key lies in the C' block, which this objective masks out, so a "
            "'self' pin pins zero visible keys, the gate's normalizer becomes a per-row constant "
            "that cancels in the softmax, and the flat-gate no-op reopens -- training runs cleanly "
            "and the router learns no ranking (cross_replay_e2e.md §3). This also means the e2e "
            "script's 'ablate' mode has no counterpart here."
        )
    if args.n_sink <= 0:
        parser.error(
            f"--n-sink {args.n_sink} pins nothing, which makes the gate's normalizer inert and "
            "reopens the no-op the pin exists to close."
        )
    if args.batch_size != 1:
        parser.error(
            f"--batch-size {args.batch_size} is not supported: cross_replay_training_step requires "
            "(1, N) input. Use --global-batch-size, which reaches the same effective batch through "
            "gradient accumulation."
        )
    if args.init_from and args.resume_from:
        parser.error(
            "--init-from and --resume-from are mutually exclusive: the first is a weights-only warm "
            "start that restarts the schedule and Adam, the second continues an interrupted run "
            "with both preserved."
        )
    if not torch.cuda.is_available():
        parser.error("no CUDA device; indexer training needs a GPU")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    validate(args, parser)

    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)
    # Per-rank seed so the ranks draw different windows. No FFN-SP here, so unlike the e2e script
    # every rank is its own data-parallel replica and the global rank is the right seed offset.
    torch.manual_seed(args.seed + rank)
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

    if args.global_batch_size:
        per_step = world_size * args.batch_size
        if args.global_batch_size % per_step:
            raise SystemExit(
                f"--global-batch-size {args.global_batch_size} is not divisible by {world_size} "
                f"rank(s) x --batch-size {args.batch_size} = {per_step}; gradient accumulation can "
                "only reach multiples of that."
            )
        args.accum_steps = args.global_batch_size // per_step
        logger.info(
            "--global-batch-size %d / (%d rank(s) x batch %d) -> --accum-steps %d",
            args.global_batch_size, world_size, args.batch_size, args.accum_steps,
        )
    seqs_per_step = world_size * args.batch_size * args.accum_steps
    tokens = sum(sl * st for sl, st in schedule.stages) * seqs_per_step
    logger.info(
        "%d rank(s) x batch %d x accum %d = %d sequences/step; %d optimizer steps over %s = %.0fM "
        "tokens",
        world_size, args.batch_size, args.accum_steps, seqs_per_step, args.total_steps,
        " -> ".join(f"{sl // 1024}K" for sl, _ in schedule.stages), tokens / 1e6,
    )

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn, device)

    press_kwargs = dict(
        compression_ratio=args.compression_ratio,
        scorer_attr=args.scorer_attr,
        gate_scale=True,
        n_sink=args.press_n_sink,
        scorer=args.scorer,
        scalar_mid_dim=args.scalar_mid_dim,
        scalar_pos_slope=args.scalar_pos_slope,
    )
    if args.scorer == "prefix":
        press_kwargs.update(
            prefix_head_dim=args.prefix_head_dim,
            prefix_value_dim=args.prefix_value_dim,
            prefix_zero_init=args.prefix_zero_init,
        )
    press = GQAIndexerPress(**press_kwargs)
    press.post_init_from_model(model)

    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        ckpt_scorer = (payload.get("config") or {}).get("scorer")
        if ckpt_scorer is not None and ckpt_scorer != args.scorer:
            raise SystemExit(
                f"--init-from was trained with scorer={ckpt_scorer!r} but this run is "
                f"scorer={args.scorer!r}. The parameter names differ, so nothing would load and the "
                "run would silently start from scratch."
            )
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        logger.info("initialized indexer from %s", args.init_from)

    trainer = CrossReplayTrainer(
        press=press,
        pin_mode=args.pin_mode,
        n_sink=args.n_sink,
        query_chunk=args.query_chunk,
        # None stays None so the trainer's own warning about resolving to the no-op point fires,
        # rather than being silently turned into some default here.
        log_budget=None if args.budget is None else math.log(args.budget),
        lookahead=args.lookahead,
        demand_reduce=args.demand_reduce,
    )
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    trainable = sum(p.numel() for p in params)
    logger.info(
        "trainable %.2fM of %.2fB parameters; objective = cross-replay LM loss, pin=%s n_sink=%d, "
        "budget=%s query_chunk=%s logit_chunk=%s",
        trainable / 1e6, sum(p.numel() for p in model.parameters()) / 1e9,
        trainer.pin_mode, trainer.sink_count,
        "n_gated (NO-OP)" if args.budget is None else args.budget,
        args.query_chunk, args.logit_chunk,
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

    # Seeded once and advanced per control, so successive shuffle controls use DIFFERENT permutations
    # -- a single fixed permutation could be accidentally easy and would read as a stable delta.
    shuffle_generator = torch.Generator(device="cpu").manual_seed(args.seed + 12345)
    current_len, loader, iterator = None, None, None
    window: list[float] = []
    started = time.time()
    step = 0

    try:
        with trainer.hooks(model):
            for step, seq_len in schedule.lengths():
                if step < start_step:
                    continue
                if seq_len != current_len:
                    if current_len is not None:
                        logger.info(
                            "step %d: seq_len %d -> %d; an LM loss has no log(L) term, so any jump "
                            "here is real rather than arithmetic",
                            step, current_len, seq_len,
                        )
                    else:
                        logger.info("step %d: starting at seq_len=%d", step, seq_len)
                    loader = loader_for(
                        seq_len, args, tokenizer, rank, world_size, batch_size=args.batch_size
                    )
                    iterator = iter(loader)
                    current_len = seq_len

                optimizer.zero_grad(set_to_none=True)
                accumulated = 0.0
                will_log = (
                    step % args.log_every == 0 or step == args.total_steps - 1
                    or (bool(args.max_steps) and step + 1 >= args.max_steps)
                )
                last_ids = None
                for _ in range(args.accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        logger.info(
                            "step %d: corpus exhausted at seq_len=%d, restarting", step, seq_len
                        )
                        iterator = iter(loader)
                        batch = next(iterator)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    last_ids = input_ids
                    # The cross-document control (--cross-doc-replay): replay an UNRELATED document
                    # against KV(C) instead of C itself. The next token of an unrelated document
                    # cannot be predicted from C's keys, so if "reconstruction" is what teaches the
                    # router anything, this must collapse. If instead it trains to the same score --
                    # and lands in the same selection cluster as the rectangle arms (§18.3) -- then
                    # the reconstruction relation was never the teacher and the router is learning a
                    # document-independent salience. See cross_replay_e2e.md §19.
                    #
                    # Drawn from the SAME loader as C, one batch later, so it is real corpus text at
                    # the identical length and from the identical distribution: the only thing that
                    # changes is whether C' is related to C. Using noise or a shuffle instead would
                    # confound "unrelated" with "not natural text".
                    replay_ids = None
                    if args.cross_doc_replay:
                        try:
                            donor = next(iterator)
                        except StopIteration:
                            iterator = iter(loader)
                            donor = next(iterator)
                        replay_ids = donor["input_ids"].to(device, non_blocking=True)
                        if replay_ids.shape != input_ids.shape:
                            # The trainer requires equal shapes; the loader emits fixed-length
                            # windows, so a mismatch means a ragged tail. Skip rather than pad,
                            # since padding would put PAD tokens in the loss.
                            replay_ids = replay_ids[:, : input_ids.shape[1]]
                            if replay_ids.shape[1] < input_ids.shape[1]:
                                replay_ids = input_ids  # degenerate tail: fall back to self-replay
                    # loss_scale, NOT (loss / accum_steps).backward(): this function differentiates
                    # internally and returns a DETACHED scalar, so scaling the return value would
                    # divide a number with no graph attached and leave the gradients accum_steps
                    # times too large -- with a perfectly normal-looking loss. See its docstring.
                    loss = cross_replay_training_step(
                        model,
                        trainer,
                        input_ids=input_ids,
                        replay_ids=replay_ids,
                        logit_chunk=args.logit_chunk,
                        loss_scale=1.0 / args.accum_steps,
                    )
                    accumulated += float(loss)

                if world_size > 1:
                    # Before clipping, so every rank clips the same vector and takes an identical
                    # step. No FFN-SP here, so world_size is the plain data-parallel divisor.
                    average_gradients(params, world_size)

                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                lr_schedule.step()

                shuffle_delta = None
                if (
                    args.shuffle_control_every
                    and last_ids is not None
                    and step % args.shuffle_control_every == 0
                ):
                    shuffle_delta = shuffle_control(
                        model, trainer, last_ids, args, shuffle_generator
                    )
                    if shuffle_delta <= 0:
                        logger.warning(
                            "step %d: shuffling the scores did NOT hurt (delta %+.4f). The gate "
                            "carries no usable ranking -- expected very early, but if it persists "
                            "the objective is training nothing, whatever the loss says.",
                            step, shuffle_delta,
                        )

                if world_size > 1 and will_log:
                    accumulated = all_reduce_mean(accumulated, device)
                window.append(accumulated)
                reached_max = bool(args.max_steps) and step + 1 >= args.max_steps

                if will_log:
                    per_layer = measure_participation(trainer, args.n_sink)
                    participation = sum(per_layer.values()) / max(len(per_layer), 1)
                    scales = list(trainer.gate_scales.values())
                    gate_scale = sum(scales) / len(scales) if scales else float("nan")
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(
                        "step %4d/%d L=%-6d loss %.4f (avg %.4f) |g| %.3f lr %.2e gate %.4f "
                        "participation %.4f shuffle %s peak %.1f GiB %.1f s/step",
                        step, args.total_steps, seq_len, accumulated,
                        sum(window) / len(window), float(grad_norm),
                        lr_schedule.get_last_lr()[0], gate_scale, participation,
                        f"{shuffle_delta:+.4f}" if shuffle_delta is not None else "-",
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
                                    "gate_scale_mean": gate_scale,
                                    "gate_scales": {
                                        str(k): v for k, v in trainer.gate_scales.items()
                                    },
                                    # ~1.0 = flat gate, nothing learned; -> 0 = concentrating.
                                    # Free here (no query axis), so logged every time.
                                    "participation_mean": participation,
                                    "participation": {str(k): v for k, v in per_layer.items()},
                                    # null unless this step ran a control. Positive = the score
                                    # carries a ranking; <= 0 = it does not.
                                    "shuffle_delta": shuffle_delta,
                                    "peak_gib": peak,
                                    "budget": args.budget,
            # Same reason as `budget`: these change what the router is trained on, so a checkpoint
            # that does not record them cannot be compared against another (§13, §16).
            "lookahead": args.lookahead,
            "demand_reduce": args.demand_reduce,
            "cross_doc_replay": args.cross_doc_replay,
                                    "query_chunk": args.query_chunk,
                                    "logit_chunk": args.logit_chunk,
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
                         {"loss": accumulated}, optimizer=optimizer, lr_schedule=lr_schedule)

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
