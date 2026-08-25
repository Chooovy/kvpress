# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-flight checks for the prefix indexer: does it still nest inside the scalar arm, and does its
score keep its spread across position?

Two questions, both cheap, both of a kind that is far more expensive to answer *after* a 600-step
run than before one. Neither trains anything.

1. ``--identity`` (default) -- **the nesting check.**
   With the prefix branch zero-initialized, :class:`~.prefix_indexer.PrefixIndexer`'s score must be
   bit-identical to :class:`~.scalar_indexer.ScalarIndexer`'s. That is what makes a prefix-vs-scalar
   A/B single-variable: "reads the prefix" is the only thing that differs. It is unit-tested at toy
   scale in fp64 (``test_gqa_indexer_prefix.py::test_zero_init_is_bit_identical_to_scalar``); this
   re-checks it on the **real model, in the training dtype, at a real sequence length**, which is
   where a norm's fp32 upcast or a bf16 rounding could break the nesting without breaking the test.
   Also prints both arms' parameter counts, so the capacity difference is on the record before any
   loss is compared -- the confound that made an earlier A/B in this project uninterpretable.

2. ``--variance-only`` -- **the variance-collapse check.**
   ``softmax(q_j K^T) V`` is a convex combination of ``{v_i}_{i<=j}``, so ``a_j`` lies in their
   convex hull: as the attention spreads, ``a_j -> mean(v)`` and the spread of the readout *across
   keys* shrinks with position. Top-k compares across positions, so that surfaces as a systematic
   position bias -- and the end-of-training symptom is "the router just learned recency", which is
   many confused steps away from the cause.

   Measured at toy scale on random input: the raw readout's across-``j`` spread falls **4.7x** from
   the first 512 positions to the last (0.00403 -> 0.00086, ``||a||`` 0.584 -> 0.153), while the
   final score's per-bin variance stays flat (ratio **0.970**) because the ``W_in norm(h_j)``
   residual holds it up. The residual is the mitigation, and this mode checks it still holds on
   **real hidden states**, whose ``||v||`` drifts across depth in a way random input does not
   reproduce.

   Reported per layer, because the drift is a per-layer property: a monotone decay in the bin
   variances is the failure mode, roughly flat is healthy.

Usage
-----
::

    python -m scripts.prefix_indexer_identity_check --model $MODEL --mid-dim 256
    python -m scripts.prefix_indexer_identity_check --variance-only --model $MODEL \\
        --tokenized $TOK --seq-len 8192

Both are driven by ``scripts/train_gqa_indexer_prefix_gy.sh {identity,variance}``. Single-process
and single-GPU on purpose: these are measurements, not runs.

With no corpus the variance mode falls back to random tokens and says so. That still exercises the
convex-hull geometry, but **not** the depth-dependent ``||v||`` drift that is the reason to run it
on real data, so the fallback is a smoke test rather than the check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    TokenizedConfig,
    build_tokenized_dataloader,
)
from kvpress.presses.gqa_indexer.prefix_indexer import score_variance_profile  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress  # noqa: E402
from scripts.train_gqa_indexer import build_model  # noqa: E402

logger = logging.getLogger("prefix_check")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 17, 35],
        help="layers to report. Both endpoints plus a middle one: hidden-state norms vary by two "
        "orders of magnitude across depth, so a single layer would not show the drift.",
    )
    parser.add_argument("--mid-dim", type=int, default=256, help="readout MLP width")
    parser.add_argument("--prefix-head-dim", type=int, default=128)
    parser.add_argument("--prefix-value-dim", type=int, default=128)
    parser.add_argument("--pos-slope", type=float, default=1e-6)
    parser.add_argument("--n-bins", type=int, default=8, help="position bins for the variance profile")
    parser.add_argument(
        "--variance-only",
        action="store_true",
        help="skip the nesting check and only report the variance profile",
    )
    data = parser.add_argument_group("data (variance mode)")
    data.add_argument("--data-root", default=None)
    data.add_argument("--tokenized", default=None, help="pre-tokenized corpus root")
    data.add_argument("--subsets", nargs="+", default=["2e16", "2e17"])
    return parser


def attach(model, scorer: str, args) -> GQAIndexerPress:
    """Attach a fresh indexer of the requested kind, with this script's geometry."""
    kwargs = dict(
        compression_ratio=0.5,
        scorer=scorer,
        scalar_mid_dim=args.mid_dim,
        scalar_pos_slope=args.pos_slope,
        gate_scale=True,
    )
    if scorer == "prefix":
        kwargs.update(
            prefix_head_dim=args.prefix_head_dim,
            prefix_value_dim=args.prefix_value_dim,
            prefix_zero_init=True,
        )
    press = GQAIndexerPress(**kwargs)
    # force_reinit: this script attaches both scorers to the same loaded backbone in turn, and the
    # press refuses to silently replace an indexer whose geometry differs from the one it would
    # build. Here the replacement is the point.
    press.post_init_from_model(model, force_reinit=True)
    return press


def layer_hidden_states(model, input_ids: torch.Tensor, layers: list[int]) -> dict:
    """
    The **input** hidden states of each requested layer -- what the indexer actually scores.

    Taken from ``output_hidden_states``, where entry ``i`` is the input to layer ``i`` (entry 0 is
    the embedding output). That is the tensor the press hands the indexer, so scoring anything else
    here would measure a different function than training does.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return {idx: out.hidden_states[idx] for idx in layers}


def report_identity_shared_weights(model, input_ids: torch.Tensor, args) -> bool:
    """
    The nesting check: the shared parameters held **equal** between the two arms.

    Comparing two independently initialized modules would only show that both produce *a* score.
    Nesting is a statement about the same weights: load the prefix arm's shared parameters into a
    scalar arm and the two scores must agree bit for bit.
    """
    layers = [i for i in args.layers if i < model.config.num_hidden_layers]
    hidden = layer_hidden_states(model, input_ids, layers)

    press_prefix = attach(model, "prefix", args)
    captured, prefix_scores = {}, {}
    prefix_params = 0
    for idx in layers:
        indexer = press_prefix.get_indexer(model.model.layers[idx].self_attn)
        prefix_params = sum(p.numel() for p in indexer.parameters())
        captured[idx] = {k: v.clone() for k, v in indexer.state_dict().items()}
        with torch.no_grad():
            prefix_scores[idx] = indexer.score_keys(hidden[idx]).clone()

    press_scalar = attach(model, "scalar", args)
    ok, scalar_params = True, 0
    for idx in layers:
        scalar = press_scalar.get_indexer(model.model.layers[idx].self_attn)
        scalar_params = sum(p.numel() for p in scalar.parameters())
        # Only the shared names: w_pq/w_pk/w_pv/w_a/a_norm exist on the prefix arm alone.
        scalar.load_state_dict({k: captured[idx][k] for k in scalar.state_dict()}, strict=True)
        with torch.no_grad():
            scalar_scores = scalar.score_keys(hidden[idx])
        same = torch.equal(prefix_scores[idx], scalar_scores)
        maxdiff = (prefix_scores[idx] - scalar_scores).abs().max().item()
        logger.info(
            "layer %-2d  identical=%-5s  maxdiff=%.3e  score std=%.4f  |h| mean=%.2f",
            idx, same, maxdiff, prefix_scores[idx].std().item(),
            hidden[idx].float().norm(dim=-1).mean().item(),
        )
        ok &= same

    logger.info(
        "params/layer: prefix %s vs scalar %s (+%.1f%%)  [prefix branch = %s]",
        f"{prefix_params:,}", f"{scalar_params:,}",
        100.0 * (prefix_params - scalar_params) / max(scalar_params, 1),
        f"{prefix_params - scalar_params:,}",
    )
    return ok


def report_variance(model, input_ids: torch.Tensor, args) -> None:
    """Score with a **non-zero** prefix branch and report the position-binned variance."""
    layers = [i for i in args.layers if i < model.config.num_hidden_layers]
    hidden = layer_hidden_states(model, input_ids, layers)
    press = attach(model, "prefix", args)

    logger.info("variance profile over %d position bins (detrended; flat is healthy)", args.n_bins)
    for idx in layers:
        indexer = press.get_indexer(model.model.layers[idx].self_attn)
        # Zero-init would make this measure the SCALAR arm's variance, which is not the question.
        # A unit-ish w_a exercises the branch at the scale training would reach it at.
        with torch.no_grad():
            indexer.w_a.weight.normal_(0, 0.5 / max(args.prefix_value_dim, 1) ** 0.5)
            scores = indexer.score_keys(hidden[idx])
            readout = indexer.prefix_readout(indexer.in_norm(hidden[idx].to(indexer.weight_dtype)))
        centers, variance = score_variance_profile(scores, n_bins=args.n_bins)

        ratio = (variance[-1] / variance[0]).item() if variance[0] != 0 else float("nan")
        logger.info("layer %-2d  bin var: %s", idx, " ".join(f"{v:.4f}" for v in variance.tolist()))
        # The raw readout is where the collapse lives; the score is what the residual protects.
        edge = max(readout.shape[1] // 8, 1)
        first, last = readout[:, :edge].float(), readout[:, -edge:].float()
        logger.info(
            "          last/first score var = %.3f  |  raw a_j across-j spread %.5f -> %.5f "
            "(%.1fx), ||a|| %.4f -> %.4f",
            ratio,
            first.mean(-1).std().item(), last.mean(-1).std().item(),
            (first.mean(-1).std() / last.mean(-1).std().clamp_min(1e-12)).item(),
            first.norm(dim=-1).mean().item(), last.norm(dim=-1).mean().item(),
        )
        if ratio < 0.3:
            logger.warning(
                "layer %d: score variance fell %.1fx across position -- this is the collapse the "
                "residual is supposed to prevent. Shrink --prefix-value-dim or revisit the readout "
                "before starting a training run.", idx, 1.0 / max(ratio, 1e-9),
            )
    logger.info("centers: %s", " ".join(f"{c:.3f}" for c in centers.tolist()))


def get_input_ids(args, tokenizer, device) -> torch.Tensor:
    """One real document at ``--seq-len``, or random tokens when no corpus is configured."""
    if args.tokenized and Path(args.tokenized, "index.json").is_file():
        loader = build_tokenized_dataloader(
            TokenizedConfig(
                root=args.tokenized,
                seq_len=args.seq_len,
                subsets=tuple(args.subsets) if args.subsets else None,
                take_from="head",
            ),
            batch_size=1,
            num_workers=0,
        )
        batch = next(iter(loader))
        logger.info("using one real document from %s", args.tokenized)
        return batch["input_ids"][:, : args.seq_len].to(device)

    logger.warning(
        "no --tokenized corpus: falling back to RANDOM tokens. That exercises the convex-hull "
        "geometry but NOT the depth-dependent ||v|| drift, which is the reason to run this on real "
        "data. Treat the result as a smoke test."
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    return torch.randint(0, vocab, (1, args.seq_len), device=device)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn, device)
    input_ids = get_input_ids(args, tokenizer, device)
    logger.info(
        "%s, dtype=%s, seq_len=%d, mid_dim=%d, prefix head/value = %d/%d",
        args.model, args.dtype, input_ids.shape[1], args.mid_dim,
        args.prefix_head_dim, args.prefix_value_dim,
    )

    if args.variance_only:
        report_variance(model, input_ids, args)
        return 0

    ok = report_identity_shared_weights(model, input_ids, args)
    if not ok:
        logger.error(
            "NESTING BROKEN: the zero-initialized prefix arm does not reproduce the scalar arm's "
            "score on this model. Every prefix-vs-scalar comparison would be confounded by "
            "whatever else differs, and the confound would be invisible -- both arms still train "
            "and both still report a plausible loss. Do not start a run."
        )
        return 1
    logger.info("nesting holds: the A/B is single-variable. Run `variance` next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
