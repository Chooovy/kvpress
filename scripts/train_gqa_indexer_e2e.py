# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train a GQA lightning indexer **end to end from the LM loss**, on the longmino corpus.

The counterpart of :mod:`scripts.train_gqa_indexer`, which distills. There the indexer is
supervised to match the frozen model's own attention weights and its score never touches the
forward pass. Here the score is added inside the attention softmax::

    out = softmax(scale * q @ k^T + gate_scale * qi @ ki^T) @ v

so the router sits on the forward path and its gradient comes from the model's own next-token
loss. No teacher, no KL, no second forward pass.

    python -m scripts.train_gqa_indexer_e2e --data-root RAW --tokenized TOK \\
        --schedule 8192:300,16384:300,32768:300

Everything that is not the objective is deliberately **identical** to the distillation script --
the same WSD schedule, curriculum handling, loader, seeding, gradient averaging, checkpoint
format and metrics fields -- because the point of this script is to be comparable with that one.
The shared pieces are imported from it rather than copied, so they cannot drift apart.

Pinning is not optional
-----------------------
Adding the same number to every key of a row cancels in the softmax, so a gate that is flat
along the key axis is a **no-op**: the model falls back to the frozen dense backbone, which is
already strong, and the LM loss is satisfied with no ranking learned. The router reaches that
point at zero cost. Under a positive fixed or ratio budget, ``--pin-mode`` exempts some keys from
the gate's normalizer, which makes a flat gate arithmetically impossible. ``--gate-budget 0`` is
the raw-score ablation that restores this escape route. ``--pin-mode none`` remains the unpinned
baseline and warns.

``sink`` is the default rather than ``self``: it is query-independent, so it folds into the
concatenated QK and stays a single SDPA call at any length, whereas ``self`` needs an
``O(Sq * Sk)`` second attention path until a fused kernel exists. See
:mod:`kvpress.presses.gqa_indexer.gate_pin`.

What differs from distillation, and why
--------------------------------------
* **No teacher tensors.** The distillation script rebuilds the teacher's post-RoPE queries and
  streams a tiled KL; none of that exists here, so ``--key-tile``/``--capture-lse``/``--backend``
  have no counterpart. The one tile knob that remains is ``--key-tile`` for the gate's history
  ``logsumexp``.
* **``use_cache=False``.** Distillation reads the teacher's keys out of the KV cache; here
  nothing does, so building one only costs memory.
* **Autotune is not wired.** It profiles the *distillation* loss kernels, and their memory
  profile is not this one's. ``--batch-size`` is explicit instead.
* **The reported loss is comparable across lengths.** Distillation's loss grows like
  ``log(L)`` because the softmax it normalizes over gets wider, which puts a ``log 2`` step in
  the curve at every curriculum boundary. An LM loss has no such term, so the raw number is
  already comparable and ``loss_minus_log_seq`` is not emitted.
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
    PIN_MODES,
    E2EIndexerTrainer,
    e2e_indexer_training_step,
    indexer_state_dict,
    load_indexer_state_dict,
)
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    SUBSETS,
    LengthSchedule,
    describe_subsets,
    read_index,
)

# Imported, not reimplemented: these are exactly the pieces that must not differ between the two
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

logger = logging.getLogger("train_gqa_indexer_e2e")


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

    Same layout as the distillation script's ``save`` (an ``indexer`` state dict under a
    ``config`` blob), so a checkpoint from either objective loads into the press the same way and
    ``--init-from`` accepts both. ``objective`` records which one produced it, since the weights
    alone do not say. With ``--save-optimizer`` (on by default) the AdamW state and LR-schedule
    position ride along too, which is what ``--resume-from`` restores to continue an interrupted
    run; a weights-only reader ignores those keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexer": indexer_state_dict(model, args.scorer_attr),
        "step": step,
        "config": {
            "objective": "e2e_lm_loss",
            "model": args.model,
            # Which router produced these weights. Without it the two arms' checkpoints are
            # indistinguishable, and --init-from would load one into the other -- the parameter
            # names differ, so load_indexer_state_dict (strict=False) would drop everything and
            # silently start from init.
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim if args.scorer in ("scalar", "prefix") else None,
            "scalar_pos_slope": (
                args.scalar_pos_slope if args.scorer in ("scalar", "prefix") else None
            ),
            # The prefix branch's geometry. Recorded for the same reason as `scorer`: head_dim and
            # value_dim ARE parameter shapes, so a mismatch would fail to load loudly -- but
            # prefix_zero_init is not, and a run resumed with the flag flipped would train a
            # different experiment (one that no longer nests inside the scalar arm) while every
            # tensor still loaded cleanly.
            "prefix_head_dim": args.prefix_head_dim if args.scorer == "prefix" else None,
            "prefix_value_dim": args.prefix_value_dim if args.scorer == "prefix" else None,
            "prefix_zero_init": args.prefix_zero_init if args.scorer == "prefix" else None,
            "stage": args.stage,
            "pin_mode": args.pin_mode,
            "n_sink": args.n_sink,
            "gate_budget": args.gate_budget,
            "gate_budget_ratio": args.gate_budget_ratio,
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "topk": args.topk,
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


def ffn_sp_group(world_size: int, sp_size: int, rank: int):
    """
    Build the FFN sequence-parallel group, and the data-parallel group that pairs with it.

    Returns ``(sp_group, dp_rank, dp_world_size, sp_rank)``.

    The ranks are laid out so that consecutive ranks form one SP group: with ``world_size=8`` and
    ``sp_size=4`` the groups are ``[0,1,2,3]`` and ``[4,5,6,7]``, giving 2 data-parallel replicas.
    Consecutive rather than strided because ranks close together are usually the ones sharing a
    node's fast interconnect, and this group carries an all-gather per layer.

    **Every rank in an SP group must read the SAME sequence** -- they cooperate on one, they do not
    each hold their own. So the loader is sharded by ``dp_rank``, not by the global rank. Getting
    that wrong is silent: each rank would slice a *different* sequence, the all-gather would stitch
    together fragments of unrelated documents, and the loss would still look plausible.
    """
    if sp_size <= 1:
        return None, rank, world_size, 0
    if world_size % sp_size:
        raise ValueError(
            f"--ffn-sp-size {sp_size} must divide the world size {world_size}; otherwise some "
            "sequence-parallel group would be short a rank and its slices would not tile the "
            "sequence"
        )

    n_groups = world_size // sp_size
    group = None
    for index in range(n_groups):
        ranks = list(range(index * sp_size, (index + 1) * sp_size))
        # Every rank must call new_group for every group, in the same order -- it is collective.
        candidate = dist.new_group(ranks)
        if rank in ranks:
            group = candidate
    return group, rank // sp_size, n_groups, rank % sp_size


def apply_liger_fused_ce(model, model_name: str) -> bool:
    """
    Patch **only** the fused linear+cross-entropy, and verify it did not change the loss.

    Why only that one. ``apply_liger_kernel_to_*`` also replaces RMSNorm, SwiGLU and
    ``apply_rotary_pos_emb`` by default. The whole memory saving comes from the fused CE, and the
    RoPE swap in particular touches something the indexer depends on: the press narrows the
    layer's ``position_embeddings`` for its own scoring, so a different cos/sin convention would
    train the router against a positional signal it never sees at inference. That is the silent
    train/inference mismatch this package warns about everywhere else, traded for a speedup nobody
    measured. So the other three are explicitly disabled.

    What the fused kernel buys. ``lm_head`` and the cross-entropy are computed in row chunks and
    each chunk's ``(chunk, vocab)`` logits are freed immediately, so no ``(L, vocab)`` tensor is
    ever retained. On Qwen3-8B at ``L=8192`` that is 2.32 GiB of bf16 logits plus 4.64 GiB of
    fp32 ``log_softmax`` -- **7.0 GiB, scaling linearly with L** (13.8 at 16K, 27.6 at 32K).
    ``grad_weight`` is only allocated when ``lm_head.weight.requires_grad``, which it does not
    here, so its ``(vocab, hidden)`` buffer is skipped too.

    Returns whether the patch was applied. Raises if Liger is missing or has no patch for this
    architecture -- a flag that silently does nothing is the failure mode this function exists to
    avoid, and it has already bitten this script twice (gradient checkpointing gated on
    ``module.training``, and Liger's own ``skip_logits`` default gated on the same thing).
    """
    try:
        from liger_kernel.transformers import monkey_patch
    except ImportError as exc:
        raise RuntimeError(
            "--liger needs liger-kernel installed (pip install liger-kernel)"
        ) from exc

    architecture = type(model).__name__.replace("ForCausalLM", "").lower()
    patch = getattr(monkey_patch, f"apply_liger_kernel_to_{architecture}", None)
    if patch is None:
        available = sorted(
            name.removeprefix("apply_liger_kernel_to_")
            for name in dir(monkey_patch)
            if name.startswith("apply_liger_kernel_to_")
        )
        raise RuntimeError(
            f"liger has no patch for {type(model).__name__} (looked for "
            f"apply_liger_kernel_to_{architecture}). Available: {', '.join(available)}. "
            "Drop --liger."
        )

    patch(
        # RoPE stays OFF: the press narrows the layer's position_embeddings for the indexer, so a
        # different cos/sin convention would train the router against a positional signal it never
        # sees at inference -- the silent train/inference mismatch this package warns about.
        rope=False,
        # RMSNorm stays OFF: it replaces the Qwen3RMSNorm *class*, and the indexer carries its own
        # IndexerNorm whose fp32-statistics behaviour is load-bearing (see IndexerNorm's docstring).
        # Not verified safe, and the saving is smaller than SwiGLU's.
        rms_norm=False,
        # SwiGLU ON. It touches only the MLP -- no attention, no position_embeddings, none of the
        # trainer's hooks -- and it is the second largest retained term after the loss head.
        # Standard SwiGLU keeps 3 x (L, inter): silu saves its input, and the elementwise mul saves
        # BOTH operands. Liger saves only (gate, up) and recomputes silu in the backward, measured
        # at 2.75 x against 3.75 x -- a 27% cut of the MLP, which is ~6.8 GiB at L=8192 on Qwen3-8B
        # and ~13.5 at 16K. Freezing the backbone does NOT avoid this: a frozen MLP still retains
        # its operands whenever the gradient has to pass through it, which it does here because the
        # router sits below every MLP above it.
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        model=model,
    )
    logger.info(
        "liger fused linear+CE and SwiGLU on for %s (rope/rms_norm deliberately left alone). "
        "Expect peak memory to drop by ~14 GiB at L=8192 (7.0 loss head + 6.8 MLP) and "
        "proportionally more at longer lengths; if it does not, the patch is not taking effect -- "
        "see --liger in --help.",
        type(model).__name__,
    )
    return True


def check_liger_loss_unchanged(model, trainer, input_ids, tol: float = 2e-2) -> None:
    """
    Assert the fused loss head gives the same loss as the unfused one, on one batch.

    **Scope, stated plainly:** this toggles ``skip_logits``, so it compares fused-CE against
    unfused-CE *with SwiGLU already patched in both runs*. It therefore catches a broken loss head
    but **cannot** catch a broken SwiGLU -- both sides would be wrong identically. The SwiGLU
    replacement is covered instead by ``tests/presses/test_gqa_indexer_liger.py``, which compares
    against an unpatched model.

    Worth checking at all because a mismatch here means ``skip_logits`` is routing to a different
    objective, and the run would still descend and still look healthy.

    The tolerance is loose because the two paths accumulate the same sum in a different order --
    chunked fp32 against one large fp32 reduction -- so they agree to roughly bf16 precision on the
    logits, not to fp32 exactness. Anything beyond that is a real difference, not rounding.

    Runs under ``no_grad``: this is a check, and building the graph twice would double the peak the
    check exists to protect.
    """
    with torch.no_grad():
        fused = e2e_indexer_training_step(
            model, trainer, input_ids=input_ids, skip_logits=True
        )
        unfused = e2e_indexer_training_step(
            model, trainer, input_ids=input_ids, skip_logits=False
        )
    gap = abs(float(fused) - float(unfused))
    if gap > tol:
        raise RuntimeError(
            f"liger fused CE changed the loss by {gap:.4f} (fused {float(fused):.4f} vs unfused "
            f"{float(unfused):.4f}, tolerance {tol}). The fused kernel should be numerically "
            "equivalent, so this means something else was patched -- a different RoPE convention "
            "would train the router against a positional signal it never sees at inference. Drop "
            "--liger."
        )
    logger.info(
        "liger check: fused loss %.4f vs unfused %.4f (gap %.2e) -- equivalent",
        float(fused), float(unfused), gap,
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True, help="longmino_256k_filtered root")
    data.add_argument(
        "--tokenized",
        default=None,
        help="read pre-tokenized .npy shards from here (scripts/pretokenize_longmino.py). "
        "The same corpus the distillation script uses -- share it rather than tokenizing twice.",
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
    model_group.add_argument(
        "--scorer",
        choices=("pairwise", "scalar", "prefix"),
        default="pairwise",
        help="which router to train. 'pairwise' scores every (query, key) pair -- query-aware, "
        "and O(t) per decode step, which at 128K makes the router 32x the cost of the sparse "
        "attention it feeds. 'scalar' scores each key once from its own hidden state: O(1) per "
        "decode step and one score per token of cache instead of head_dim, at the cost of "
        "query-awareness (a needle matters only to the query that asks for it, so a frozen "
        "per-key score must keep it always or lose it). 'prefix' scores each key once from its "
        "whole *prefix*, through the indexer's own causal attention -- still query-independent, "
        "so still able to evict, but its view of a key is the prefix rather than the single "
        "vector h_j. It is a strict superset of 'scalar': with --prefix-zero-init (the default) "
        "the score starts bit-identical, so a prefix-vs-scalar A/B has exactly one variable. "
        "All three share this script, the loss, the schedule and the checkpoint format, which is "
        "what makes them comparable; --head-dim and --rope-dim apply only to 'pairwise' and are "
        "rejected with the other two.",
    )
    model_group.add_argument(
        "--scalar-mid-dim",
        type=int,
        default=256,
        help="MLP width for --scorer scalar and --scorer prefix; 0 is SparseK's plain linear "
        "score. This is the arm's "
        "capacity knob, not just a parameter-matching one: in the probe study a nonlinear readout "
        "of the hidden state beat a linear one by +0.12 held-out Spearman on total attention mass, "
        "larger than anything a recurrent state added. At 256 the router is ~1.06M params/layer "
        "against the pairwise arm's 4.72M; 1152 matches it exactly, which is the only setting that "
        "isolates query-dependence from capacity.",
    )
    model_group.add_argument(
        "--scalar-pos-slope",
        type=float,
        default=DEFAULT_POS_SLOPE,
        help="recency tilt (t * eps) for --scorer scalar and --scorer prefix. Keeps top-k "
        "irreversible, which is what "
        "makes a dropped key safe to free: verified 0 re-entries over 1500 steps with an absolute "
        "tilt against 27 when normalised by sequence length. Also carries the recency duty so the "
        "learned part is not pushed to predict ever-larger values (SparseK Sec. 3.2). 0 ablates it.",
    )
    model_group.add_argument(
        "--prefix-head-dim",
        type=int,
        default=128,
        help="--scorer prefix only: q/k width of the indexer's own prefix attention, which also "
        "sets its 1/sqrt(d) softmax scale. Distinct from --head-dim, which is the pairwise "
        "router's geometry and is rejected with this scorer.",
    )
    model_group.add_argument(
        "--prefix-value-dim",
        type=int,
        default=128,
        help="--scorer prefix only: width of the prefix attention's v, i.e. of the readout that "
        "feeds the score. This is what a decode-time indexer cache would cost per token per layer "
        "(alongside --prefix-head-dim for the keys), against n_heads scalars for --scorer scalar. "
        "Training and prefill-time eviction do not pay it -- the readout is consumed immediately.",
    )
    model_group.add_argument(
        "--no-prefix-zero-init",
        dest="prefix_zero_init",
        action="store_false",
        help="--scorer prefix only: do NOT zero-initialize the prefix branch's output projection. "
        "By default it is zeroed, so training starts bit-identical to --scorer scalar and 'reads "
        "the prefix' is the single variable in the comparison. The zero is an escapable saddle, "
        "not a dead start: w_a receives gradient at step one (dL/dW_a = dL/dz (x) norm(a), and "
        "norm(a) is not zero) even though the branch's own projections do not, so they are live "
        "from step two. Pass this to start the branch at random init instead, which is a different "
        "experiment -- the arm no longer nests inside the scalar one.",
    )
    model_group.set_defaults(prefix_zero_init=True)
    model_group.add_argument(
        "--press-n-sink", type=int, default=4,
        help="keys the press protects from eviction. Also the default for --n-sink, so the keys "
        "exempted from the gate during training are the keys kept at inference.",
    )
    model_group.add_argument("--scorer-attr", default="indexer")
    model_group.add_argument(
        "--ffn-sp-size",
        type=int,
        default=1,
        help="shard the FFN activations across this many ranks (sequence parallel). The MLP is "
        "position-wise, so each rank computes its own slice of the sequence with NO communication "
        "inside the FFN -- one all-gather per layer afterwards. FFN is the largest activation term "
        "(49%% of the measured total on Qwen3-8B), so this is what makes 16K fit: ~93.7 GiB on one "
        "GPU against ~61 with 8-way FFN-SP. Attention is NOT sharded (query i needs keys 0..i, "
        "which would require the Ulysses all-to-all), so this reaches 16K and not 32K. Ranks in one "
        "SP group read the SAME sequence and the data loader shards by data-parallel rank instead.",
    )
    model_group.add_argument(
        "--liger",
        action="store_true",
        help="fuse lm_head into the cross-entropy with liger-kernel, so the (L, vocab) logits are "
        "never materialized. Saves ~7.0 GiB at L=8192 on Qwen3-8B and scales with L (13.8 at 16K, "
        "27.6 at 32K) -- the single largest retained tensor on this path, because the router's "
        "gradient comes from the LM loss. ONLY the fused CE is patched: liger's RMSNorm/SwiGLU/RoPE "
        "replacements are left off, since a different RoPE convention would train the router "
        "against a positional signal it never sees at inference. Verified on the first batch by "
        "comparing the fused and unfused loss.",
    )
    model_group.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="recompute backbone activations in the backward pass instead of keeping them. "
        "END-TO-END NEEDS THIS AT LONG CONTEXT and distillation does not: the router's gradient "
        "travels from the LM head back through every layer, so the whole model's activation graph "
        "stays alive (~37 of the 77.6 GiB measured at L=8192 on Qwen3-8B). Distillation computes "
        "its loss inside a per-layer hook from detached teacher tensors, so no graph spans the "
        "backbone at all. Costs roughly one extra forward pass (~30%% step time).",
    )
    model_group.add_argument(
        "--init-from",
        default=None,
        help="load indexer WEIGHTS only and start a fresh schedule (Adam and the LR restart from "
        "zero). Accepts a distillation checkpoint too -- it has no gate_scale, so that scalar "
        "keeps its init, which is the intended way to start end-to-end from a distilled indexer. "
        "To CONTINUE an interrupted run with the optimizer and LR intact, use --resume-from.",
    )
    model_group.add_argument(
        "--resume-from",
        default=None,
        help="continue an interrupted run: restore indexer weights, AdamW state and LR-schedule "
        "position from a checkpoint saved with --save-optimizer, and skip the steps already done. "
        "Requires the same --schedule the checkpoint was trained with. Mutually exclusive with "
        "--init-from.",
    )

    gate = parser.add_argument_group("gate")
    gate.add_argument(
        "--stage", choices=("dense", "sparse"), default="dense",
        help="dense gates every key (full scope); sparse gates only each row's own top-k. Same "
        "names as the distillation script so the two are compared stage for stage.",
    )
    gate.add_argument(
        "--pin-mode", choices=list(PIN_MODES), default="sink",
        help="keys exempt from a positive-budget gate, so a flat gate cannot become a no-op. "
        "'sink' exempts the leading keys; 'self' exempts each query's own token, matching SAS's "
        "current block; 'none' is the un-pinned ablation and warns. All use O(L) memory.",
    )
    gate.add_argument(
        "--n-sink", type=int, default=None,
        help="leading keys to pin (default: --press-n-sink)",
    )
    budget = gate.add_mutually_exclusive_group()
    budget.add_argument(
        "--gate-budget", type=float, default=1.0,
        help="full-scope gate mode: 0 uses raw history scores; a positive value fixes the "
        "history multiplier budget. The 1.0 default preserves old experiment semantics only",
    )
    budget.add_argument(
        "--gate-budget-ratio", type=float, default=None,
        help="full-scope row-wise budget mode: for each query set B_q to this ratio times the "
        "number of visible, non-pinned history keys",
    )
    gate.add_argument(
        "--key-tile", type=int, default=1024,
        help="key tile for the gate's streaming history logsumexp; a memory knob only, the "
        "result is tile-invariant",
    )
    gate.add_argument(
        "--topk", type=int, default=0,
        help="support size for --stage sparse. Fixed rather than a ratio, for the reason the "
        "distillation script documents: a ratio makes the retained support O(L^2).",
    )
    gate.add_argument("--force-local", type=int, default=0, help="sparse-stage reserved slots")
    gate.add_argument("--force-sink", type=int, default=0, help="sparse-stage reserved slots")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:300,16384:300,32768:300", help="SEQ_LEN:STEPS,...")
    optim.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps regardless of the schedule total (0 = run "
        "the whole schedule). The LR is still computed from the full --schedule, so a run can be "
        "truncated at a defined point WITHOUT reshaping WSD. Use it to match the distillation "
        "script step-for-step: run the SAME 1500-step 8192:300,16384:300,32768:900 schedule with "
        "--max-steps 600, and the LR over 0..600 is identical to that run (warmup 150, still at "
        "peak, no decay), so a step-600 checkpoint is comparable objective-for-objective.",
    )
    optim.add_argument("--peak-lr", type=float, default=1e-3, help="WSD plateau")
    optim.add_argument("--final-lr", type=float, default=5e-6, help="WSD floor")
    optim.add_argument("--warmup-frac", type=float, default=0.10)
    optim.add_argument("--stable-frac", type=float, default=0.60)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument(
        "--global-batch-size",
        type=int,
        default=0,
        help="sequences per OPTIMIZER step, across all ranks (0 = whatever --batch-size x "
        "--accum-steps x replicas happens to give). Set this and --accum-steps is derived so the "
        "tokens/step matches regardless of --ffn-sp-size, which is the only way this run is "
        "comparable to the distillation script. FFN-SP spends ranks on ONE sequence instead of on "
        "sp_size of them, so --ffn-sp-size 8 on 8 GPUs leaves a single replica: without this flag "
        "a step sees 1 sequence where the distillation run's 8-way DDP sees 8, i.e. 1/8 the "
        "tokens at the same step number, and a step-600 checkpoint is NOT comparable. Must be "
        "divisible by replicas x --batch-size.",
    )
    optim.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("io")
    io.add_argument("--out", default="checkpoints/gqa_indexer_e2e")
    io.add_argument("--save-every", type=int, default=200)
    io.add_argument("--log-every", type=int, default=10)
    io.add_argument("--metrics-file", default=None, help="append JSONL metrics here")
    io.add_argument(
        "--gate-sparsity",
        action="store_true",
        help="log the gate's participation ratio -- whether the router became SELECTIVE, which "
        "gate_scale cannot answer (a layer can lean hard on a router that still spreads its mass "
        "over every key). Reported as a fraction of history length: ~1.0 means a flat gate and no "
        "sparsity learned, falling towards 0 means the router is concentrating, which is what "
        "eviction needs. Measured only on logged steps, and only on the last micro-batch of an "
        "accumulation group, but it does cost a second streaming pass over the keys plus an "
        "(Sq, Sk) history mask (256 MiB at 16K) -- hence opt-in rather than always on.",
    )
    io.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write AdamW state and the LR-schedule position into each checkpoint so "
        "--resume-from can continue the run. On by default; --no-save-optimizer keeps "
        "checkpoints weights-only when resume is not needed.",
    )
    io.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.stage == "sparse" and args.topk <= 0:
        parser.error("--stage sparse needs an explicit --topk (a ratio makes the support O(L^2))")
    if args.init_from and args.resume_from:
        parser.error(
            "--init-from and --resume-from are mutually exclusive: the first is a weights-only "
            "warm start that restarts the schedule and Adam, the second continues an interrupted "
            "run with both preserved. Pick one."
        )
    if not torch.cuda.is_available():
        parser.error("no CUDA device; indexer training needs a GPU")

    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)
    # Distinct seeds so the ranks draw different windows, matching the distillation script.
    # The FFN sequence-parallel group, and the data-parallel identity that pairs with it. With
    # --ffn-sp-size 1 this is a no-op and (dp_rank, dp_world) == (rank, world_size).
    sp_group, dp_rank, dp_world_size, sp_rank = ffn_sp_group(world_size, args.ffn_sp_size, rank)
    if args.ffn_sp_size > 1:
        logger.info(
            "FFN sequence parallel: sp_size=%d -> %d data-parallel replica(s). This rank is "
            "sp_rank=%d of dp_rank=%d. Ranks within one SP group read the SAME sequence.",
            args.ffn_sp_size, dp_world_size, sp_rank, dp_rank,
        )

    # Seeded by DATA-PARALLEL rank, not global rank: ranks cooperating on one sequence must draw
    # the same one. A per-global-rank seed would give each of them a different document, and the
    # all-gather would silently stitch together fragments of unrelated text.
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
    # Sequences per optimizer step. dp_world_size, not world_size: under FFN-SP the sp_size ranks
    # of a group process ONE sequence together, so they contribute one sequence, not sp_size.
    if args.global_batch_size:
        per_replica_step = dp_world_size * args.batch_size
        if args.global_batch_size % per_replica_step:
            raise SystemExit(
                f"--global-batch-size {args.global_batch_size} is not divisible by "
                f"{dp_world_size} replica(s) x --batch-size {args.batch_size} = "
                f"{per_replica_step}. Gradient accumulation can only reach multiples of that, so "
                f"the requested global batch is unreachable at --ffn-sp-size {args.ffn_sp_size}."
            )
        args.accum_steps = args.global_batch_size // per_replica_step
        logger.info(
            "--global-batch-size %d / (%d replica(s) x batch %d) -> --accum-steps %d. This is "
            "what keeps tokens/step independent of --ffn-sp-size, and so comparable to the "
            "distillation run at the same step number.",
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
        # AFTER liger: liger rebinds forward on the mlp module itself, and the wrapper calls that
        # module -- so wrapping second means the slice goes through liger's SwiGLU and both
        # savings compose. Wrapping first would leave liger patching the wrapper's `inner`
        # attribute path, which it does not look at.
        from kvpress.presses.gqa_indexer.ffn_sp import wrap_ffn_sequence_parallel

        wrap_ffn_sequence_parallel(model, group=sp_group)

    if args.grad_checkpoint:
        # HF gates checkpointing on ``module.training`` (see GradientCheckpointingLayer.__call__:
        # ``if self.gradient_checkpointing and self.training``), so train(True) is required.
        # Setting the flag while leaving the model in eval() is a silent no-op -- measured:
        # retention unchanged, which is how the first attempt at this appeared to work.
        #
        # eval() is what build_model uses to keep dropout off, so re-enabling train mode has to be
        # paired with a check that dropout really is disabled; a stochastic backbone would make the
        # router's gradient noisy against a frozen reference.
        for config_obj in (model.config, getattr(model.config, "text_config", None)):
            if config_obj is None:
                continue
            for name in ("attention_dropout", "hidden_dropout", "dropout"):
                value = getattr(config_obj, name, 0.0) or 0.0
                if value:
                    parser.error(
                        f"--grad-checkpoint needs train(True), which re-enables {name}={value} "
                        "and would make the router's gradient stochastic. Set it to 0 in the model "
                        "config, or drop --grad-checkpoint."
                    )
        # use_reentrant=False is required, not preferred: the reentrant implementation decides
        # whether to build a graph from the *inputs'* requires_grad, and this backbone is entirely
        # frozen -- so it would skip the graph and the router would receive no gradient at all.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(True)
        logger.warning(
            "--grad-checkpoint is EXPERIMENTAL and has no passing test. On a toy model it raises "
            "CheckpointError ('a different number of tensors was saved during the original forward "
            "and recomputation'). Narrowed so far: the gate path checkpoints cleanly on its own, "
            "and so does a pre-hook that adds a differentiable term, so the divergence is in their "
            "composition and is not yet understood. If it raises, drop the flag and lower "
            "--batch-size or the schedule's sequence length instead."
        )
    # gate_scale=True is what makes the indexer usable as a gate at all; the press builds the
    # parameter only when asked, so a distillation checkpoint stays free of it.
    press_kwargs = {
        "compression_ratio": args.compression_ratio,
        "scorer_attr": args.scorer_attr,
        "gate_scale": True,
        "n_sink": args.press_n_sink,
        "scorer": args.scorer,
    }
    if args.scorer in ("scalar", "prefix"):
        press_kwargs["scalar_mid_dim"] = args.scalar_mid_dim
        press_kwargs["scalar_pos_slope"] = args.scalar_pos_slope
    if args.scorer == "prefix":
        press_kwargs["prefix_head_dim"] = args.prefix_head_dim
        press_kwargs["prefix_value_dim"] = args.prefix_value_dim
        press_kwargs["prefix_zero_init"] = args.prefix_zero_init
    for name in ("rope_dim", "head_dim", "n_heads"):
        value = getattr(args, name)
        if value is not None:
            press_kwargs[name] = value
    press = GQAIndexerPress(**press_kwargs)
    press.post_init_from_model(model)

    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu")
        # Refuse a checkpoint from the other arm. The two routers share no parameter names, so
        # load_indexer_state_dict (strict=False, to accept a distillation checkpoint's missing
        # gate_scale) would drop every tensor and start from init while logging success.
        ckpt_scorer = (payload.get("config") or {}).get("scorer")
        if ckpt_scorer is not None and ckpt_scorer != args.scorer:
            raise SystemExit(
                f"--init-from was trained with scorer={ckpt_scorer!r} but this run is "
                f"scorer={args.scorer!r}. The two have different parameter names, so nothing "
                f"would load and the run would silently start from scratch."
            )
        # strict=False inside load_indexer_state_dict, so a distillation checkpoint (which has no
        # gate_scale) loads fine and leaves the scalar at its init. That is the intended way to
        # start end-to-end training from a distilled indexer.
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        logger.info("initialized indexer from %s", args.init_from)

    trainer = E2EIndexerTrainer(
        press=press,
        stage=args.stage,
        pin_mode=args.pin_mode,
        n_sink=args.n_sink,
        gate_budget=args.gate_budget,
        gate_budget_ratio=args.gate_budget_ratio,
        key_tile=args.key_tile,
        topk=args.topk or None,
        force_sink=args.force_sink,
        force_local=args.force_local,
    )
    # freeze_backbone is what the trainer's own hooks() would do; called here so the parameter
    # list handed to the optimizer is the same object the run trains.
    trainer.freeze_backbone(model)
    params = trainer.indexer_parameters(model)
    trainable = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    budget_label = (
        f"ratio {trainer.gate_budget_ratio:g} * N_H(q)"
        if trainer.gate_budget_ratio is not None
        else "raw" if trainer.gate_budget == 0
        else f"fixed {trainer.gate_budget:g}"
    )
    logger.info(
        "trainable %.2fM of %.2fB parameters (%.3f%%); objective = LM loss, gate pin=%s "
        "n_sink=%d budget=%s",
        trainable / 1e6, total_params / 1e9, 100 * trainable / total_params,
        trainer.gate_pin_mode, trainer.sink_count, budget_label,
    )
    if world_size > 1:
        logger.info(
            "distributed: averaging %.1fM gradients across %d ranks each step (%.0f MB fp32)",
            trainable / 1e6, world_size, trainable * 4 / 1e6,
        )

    optimizer, lr_schedule = build_optimizer(params, args)

    # 0 for a fresh run, the completed-step count for a resumed one. The loop skips every step
    # below it, so a resumed run neither re-runs the curriculum it saw nor re-warms its LR.
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

    # Checked once, on the first real batch (see check_liger_loss_unchanged).
    liger_check_pending = bool(args.liger)
    current_len, loader, iterator = None, None, None
    window: list[float] = []
    started = time.time()
    step = 0

    try:
        for step, seq_len in schedule.lengths():
            if step < start_step:
                # Already completed in the run we resumed; skip with no work so the LR schedule
                # and curriculum position stay aligned with where it stopped. The loader is not
                # built until the first non-skipped step.
                continue
            if seq_len != current_len:
                # Rebuilt rather than reused: a new length changes which documents are eligible
                # and which slice of a stored array is read.
                if current_len is not None:
                    # Unlike the distillation loss, an LM loss has no log(L) term, so no step in
                    # the curve is expected here -- which makes it worth saying, because the
                    # distillation script warns about exactly the opposite.
                    logger.info(
                        "step %d: seq_len %d -> %d; the LM loss has no log(L) term, so any jump "
                        "here is real rather than arithmetic",
                        step, current_len, seq_len,
                    )
                else:
                    logger.info("step %d: starting at seq_len=%d", step, seq_len)
                loader = loader_for(
                    seq_len, args, tokenizer, dp_rank, dp_world_size,
                    batch_size=args.batch_size,
                )
                iterator = iter(loader)
                current_len = seq_len

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            # Decided BEFORE the forward, because the diagnostic is computed inside it. Only on
            # steps that get logged, and only on the LAST micro-batch of an accumulation group:
            # it costs a second streaming pass over the keys plus an (Sq, Sk) history mask, and
            # measuring every micro-step would pay that accum_steps times to report one number.
            will_log = (
                step % args.log_every == 0 or step == args.total_steps - 1
                or (bool(args.max_steps) and step + 1 >= args.max_steps)
            )
            for micro in range(args.accum_steps):
                trainer.measure_sparsity = (
                    args.gate_sparsity and will_log and micro == args.accum_steps - 1
                )
                try:
                    batch = next(iterator)
                except StopIteration:
                    logger.info(
                        "step %d: corpus exhausted at seq_len=%d, restarting", step, seq_len
                    )
                    iterator = iter(loader)
                    batch = next(iterator)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                if liger_check_pending:
                    # On the first real batch, before any optimizer step: prove the fused kernel
                    # is numerically equivalent. Done here rather than on random ids so the check
                    # runs at the shape and dtype the run actually uses.
                    check_liger_loss_unchanged(model, trainer, input_ids)
                    liger_check_pending = False
                # skip_logits must be explicit: liger's default gates on self.training, and
                # this backbone stays in eval() to keep dropout off, so the default would
                # silently fall back to materializing the logits.
                loss = e2e_indexer_training_step(
                    model, trainer, input_ids=input_ids,
                    skip_logits=True if args.liger else None,
                )
                (loss / args.accum_steps).backward()
                accumulated += float(loss) / args.accum_steps

            if world_size > 1:
                # Before clipping, so every rank clips the same vector and takes an identical
                # step -- clipping first would let each rank scale by its own local norm.
                #
                # Divide by WORLD_SIZE, not dp_world_size. SequenceParallelFFN all-gathers the
                # gradient on the way back into the FFN, so the sp_size ranks of one group each
                # end up with the SAME complete gradient for that sequence -- sp_size copies of
                # it, not disjoint slices. A SUM over the group is therefore sp_size x too large,
                # and world_size = dp_world_size * sp_size divides both the replication and the
                # data-parallel averaging out in one step. (Before that gather the ranks held
                # neither copies nor clean slices: paths reaching the router without crossing a
                # sharded FFN were replicated while through-FFN paths were split, so no single
                # divisor was right and the gradient DIRECTION was off -- cosine 0.98, not
                # fixable with the LR. See _ScatterSequence.)
                average_gradients(params, world_size)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            lr_schedule.step()

            if world_size > 1 and step % args.log_every == 0:
                accumulated = all_reduce_mean(accumulated, device)

            window.append(accumulated)
            reached_max = bool(args.max_steps) and step + 1 >= args.max_steps
            if will_log:
                gate_scale = trainer.mean_gate_scale()
                gate_sparsity = trainer.mean_gate_sparsity()
                history_attention_mass = trainer.mean_history_attention_mass()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "step %4d/%d L=%-6d lm_loss %.4f (avg %.4f) |g| %.3f lr %.2e "
                    "gate %.4f sparsity %s history_mass %s peak %.1f GiB %.1f s/step",
                    step, args.total_steps, seq_len, accumulated,
                    sum(window) / len(window), float(grad_norm),
                    lr_schedule.get_last_lr()[0],
                    gate_scale if gate_scale is not None else float("nan"),
                    # Fraction of each row's history the gate effectively spreads over. 1.00 is a
                    # flat gate (no selectivity learned, whatever gate_scale says); falling
                    # towards 0 is the router concentrating, which is what eviction needs.
                    f"{gate_sparsity:.3f}" if gate_sparsity is not None else "off",
                    f"{history_attention_mass:.3f}"
                    if history_attention_mass is not None else "off",
                    peak, (time.time() - started) / (step - start_step + 1),
                )
                if metrics_handle:
                    metrics_handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "seq_len": seq_len,
                                # The model's own next-token loss. Directly comparable across
                                # curriculum stages -- there is no log(L) term to subtract, which
                                # is why loss_minus_log_seq has no counterpart here.
                                "loss": accumulated,
                                "grad_norm": float(grad_norm),
                                "lr": lr_schedule.get_last_lr()[0],
                                # Per-layer gate strength. Worth plotting: a layer whose gate
                                # collapses toward 0 is one whose router is not earning its
                                # place, and the loss curve alone would not reveal it.
                                "gate_scale_mean": gate_scale,
                                "gate_scales": {
                                    str(k): v for k, v in trainer.gate_scales.items()
                                },
                                # Gate participation ratio as a fraction of history length: the
                                # readout on whether the router became SELECTIVE, which
                                # gate_scale cannot answer -- a layer can lean hard on a router
                                # that still spreads its mass over every key. ~1.0 = flat,
                                # -> 0 = peaked. null when --gate-sparsity is off.
                                "gate_sparsity_mean": gate_sparsity,
                                "gate_sparsity": {
                                    str(k): v for k, v in trainer.gate_sparsity.items()
                                },
                                # Actual attention probability on non-pinned history. This is
                                # distinct from the gate's multiplier budget and catches a raw
                                # gate that restores dense attention without becoming selective.
                                "history_attention_mass_mean": history_attention_mass,
                                "history_attention_mass": {
                                    str(k): v
                                    for k, v in trainer.history_attention_mass.items()
                                },
                                "pin_mode": trainer.gate_pin_mode,
                                "gate_budget": trainer.gate_budget,
                                "gate_budget_ratio": trainer.gate_budget_ratio,
                                "peak_gib": peak,
                                "batch_size": args.batch_size,
                                "accum_steps": args.accum_steps,
                                # Sequences x seq_len actually consumed by this optimizer step.
                                # Includes accum: the whole point of --global-batch-size is that
                                # this figure does not move with --ffn-sp-size.
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
                    args.max_steps,
                    args.total_steps,
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
        # Barrier before teardown: rank 0 may still be writing the checkpoint.
        dist.barrier()
        dist.destroy_process_group()
    logger.info("done in %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
