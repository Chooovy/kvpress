# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Is LongCE's key-token weight a different quantity from the loss, or just another proxy for it?

Why this exists
---------------
The previous attempt weighted the router's LM loss by ``w_t = clamp(L_t^dense - L_t^sparse, 0) +
lambda``. It trained 500 steps and took RULER from 66.24 to ~35.2. The diagnosis is settled:
``weight_participation`` sat at 0.13-0.18 and ``delta_positive_frac`` at 9-22%, so the objective
only ever trained the ~15% highest-loss positions, and the collapse was *directed* --
``niah_single_2`` 100 -> 19.7, ``niah_multiquery`` 94.3 -> 3.95. Retrieval tokens are not hard under
a dense run, so they were never in that top 15%, and ``lambda=0.1`` drowned them.

The root cause was an **unverified inference**: that high-entropy tokens would have ``delta ~ 0`` and
so be down-weighted automatically. For dense-vs-sparse that is false -- ``delta`` correlates
strongly *positively* with ``L_sparse``, so the objective degenerated into a power mean and promoted
irreducible entropy hardest of all.

LongCE (Fang et al., ICLR 2025) compares **short context against long context**, and carries one
condition the delta version has no analogue for: a key token must *already be predicted well under
the long context* (``L_long < -beta``, default ``< 2``). That condition excludes irreducible entropy
**by construction** -- a token nobody can predict has high ``L_long`` and is rejected however large
its discrepancy. That is the structural difference, and this probe's only job is to check that the
difference is real on *our* corpus and *our* model before any GPU goes into training on it.

What it measures
----------------
Per ``(document, trunc_len)`` unit, over the positions that actually have a short-context
counterfactual:

* ``spearman_w_vs_long`` -- **the gate.** The delta objective failed precisely because its weight
  was rank-correlated with the loss it multiplied. Near zero or negative means LongCE is a genuinely
  different quantity and is worth training; strongly positive means it is the same mistake wearing a
  different name.
* ``weight_participation`` = ``(sum w)^2 / (n sum w^2)`` -- the *same* statistic the trainer logs, so
  the number here is directly comparable to the failed run's 0.13-0.18. ``~1.0`` means the weighting
  does nothing at all.
* ``key_rate`` -- fraction of scored positions passing the binary criterion, which sets the training
  configuration.
* ``key_loss_mean`` vs ``all_loss_mean`` -- mean ``L_long`` over key tokens against all scored ones.
  The ``beta`` condition pins the former below ``-beta``; **``key_L`` clearly under ``all_L`` is the
  signature that says the criterion is doing what it claims**, and is the one thing the delta
  weighting could never produce.
* ``weight_at_ceiling_frac`` -- share saturated by ``clamp(max=thre)``. High means ``thre`` is the
  binding constraint rather than the data.

Semantics come from the two reference implementations, not from invention:

* ``LongPPL/finetune/finetune.py:21`` ``loss_weight()`` -- ``clamp(exp(L_short - L_long), max=thre)``
* ``LongPPL/longppl/longppl.py:22`` ``find_key_token()`` -- ``(L_short - L_long) > alpha AND
  L_long < -beta``, defaults ``alpha=2, beta=-2``

One deliberate deviation from the reference
-------------------------------------------
``loss_weight`` initializes ``loss_discrepancy = torch.ones(...)``, leaving the first ``trunc_len``
positions at weight **1**. But those positions have no short-context counterfactual -- there is no
shorter context to compare against -- so ``1`` there is a fabricated number, not a measurement. This
probe tracks a ``scored`` boolean mask and excludes them from every statistic instead.

That is not a detail at our lengths: at the 8K stage, ``trunc_len=4096`` would silently pin **half
the sequence** at the baseline weight, and averaging that in would drag every metric here towards
"the weighting does nothing" for reasons that have nothing to do with the data.

Alignment
---------
``per_token_ce`` returns **next-token-indexed** losses (``ce[i]`` predicts ``ids[i+1]``), so a
window's tail is ``chunk_losses[trunc_len-1 : trunc_len-1+span]`` and it is written back at
``short_loss[start+trunc_len-1 : ...]``. An off-by-one here is invisible in every aggregate, so it is
checked structurally rather than by inspection:

    the ``start=0`` window sees the *true* prefix, so its short loss must equal the long loss
    bit-for-bit; every later window is truncated, so its must differ.

Measured at ``L=64, trunc_len=16, window=8`` on ``MaxJeblick/llama2-0b-unit-test``: ``start=0`` gives
``max|short - long| = 0.0`` and later windows ``0.0075``. Both directions are asserted at runtime and
reported in the JSON as ``alignment``.

Usage::

    python -m evaluation.probe_longce_key_tokens \\
        --model /path/Qwen3-8B \\
        --tokenized /path/longmino_tokenized_64k \\
        --subsets 2e16 2e17 --seq-len 8192 --n-docs 8 \\
        --trunc-lens 1024 2048 4096 --logit-chunk 4096 \\
        --out evaluation/longce_probe.json

Needs one GPU and no training. ``trunc_len=4096`` is LongCE's default, but it was tuned for 32K+
finetuning; at 8K it leaves only the back half scorable and halves the contrast, which is why the
shorter values are swept alongside it.
"""

from __future__ import annotations

import argparse
import json
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
from kvpress.presses.gqa_indexer.delta_loss import per_token_ce  # noqa: E402

# Imported rather than reimplemented so the losses here come from the *same* code path the trainer
# uses. A local `model(...).logits` would differ in whether the wrapper's fused loss is on the path
# and would materialize the (L, vocab) logits this deliberately avoids.
from kvpress.presses.gqa_indexer.e2e_trainer import _final_hidden_states  # noqa: E402
from scripts.train_gqa_indexer import build_model  # noqa: E402

logger = logging.getLogger("probe_longce")

#: LongCE's ``thre``: the ceiling on ``exp(L_short - L_long)``.
DEFAULT_THRESHOLD = 5.0

#: LongCE's ``internal``: the stride of the sliding short-context window.
DEFAULT_WINDOW = 1024


# --------------------------------------------------------------------------------------------
# rank statistics
# --------------------------------------------------------------------------------------------


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """
    Ranks of ``values`` in ``1..n``, ties sharing their mean rank, ``(n,)`` float64.

    Tie handling is not cosmetic here: ``clamp(max=thre)`` puts a large block of weights at exactly
    the ceiling, and ranking those arbitrarily would inflate ``|spearman|`` by manufacturing an
    ordering the data does not contain. Vectorized over tie groups rather than looped, since that
    block can hold thousands of positions.
    """
    x = values.detach().to(torch.float64).flatten().cpu()
    n = x.numel()
    if n == 0:
        return x
    order = torch.argsort(x)
    ordered = x[order]
    starts_group = torch.ones(n, dtype=torch.bool)
    starts_group[1:] = ordered[1:] != ordered[:-1]
    group = torch.cumsum(starts_group.to(torch.long), 0) - 1
    n_groups = int(group[-1]) + 1
    positions = torch.arange(1, n + 1, dtype=torch.float64)
    sums = torch.zeros(n_groups, dtype=torch.float64).index_add_(0, group, positions)
    counts = torch.zeros(n_groups, dtype=torch.float64).index_add_(
        0, group, torch.ones(n, dtype=torch.float64)
    )
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = (sums / counts)[group]
    return ranks


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two ``(n,)`` tensors; ``0.0`` when either is constant."""
    x = a.detach().to(torch.float64).flatten().cpu()
    y = b.detach().to(torch.float64).flatten().cpu()
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(x.norm() * y.norm())
    if denom == 0.0:  # a constant vector has no ordering to correlate with
        return 0.0
    return float((x * y).sum() / denom)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation: Pearson on tie-averaged ranks."""
    return pearson(average_ranks(a), average_ranks(b))


# --------------------------------------------------------------------------------------------
# the LongCE measurement
# --------------------------------------------------------------------------------------------


def long_context_losses(model, input_ids: torch.Tensor, *, logit_chunk: int) -> torch.Tensor:
    """
    ``L_long``: per-token CE over the full sequence, ``(L-1,)`` fp32, next-token-indexed.

    ``long_loss[i]`` is the loss of predicting ``input_ids[0, i+1]`` from the whole prefix.
    """
    with torch.no_grad():
        hidden = _final_hidden_states(model, input_ids=input_ids, attention_mask=None)
        losses = per_token_ce(
            model.get_output_embeddings(), hidden, input_ids, chunk_size=logit_chunk
        )
        del hidden
    return losses


def short_context_losses(
    model,
    input_ids: torch.Tensor,
    long_loss: torch.Tensor,
    *,
    trunc_len: int,
    window: int,
    logit_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """
    ``L_short`` on the same next-token index, plus the ``scored`` mask and per-window alignment.

    Follows ``loss_weight``'s loop: for each window, re-run the model on only the last
    ``trunc_len + span`` tokens and keep the last ``span`` losses. Because ``per_token_ce`` is
    next-token-indexed, the tail of a chunk is ``chunk[trunc_len-1 : trunc_len-1+span]`` and it lands
    at ``short_loss[start+trunc_len-1 : ...]`` -- the same index ``long_loss`` uses, so the two are
    directly subtractable.

    Returns
    -------
    (short_loss, scored, windows)
        ``short_loss`` is ``(L-1,)`` with unscored entries left at ``0``; read it only through
        ``scored``. ``windows`` carries ``max|short - long|`` per window, which is the alignment
        evidence: the ``start=0`` entry must be ~0 and the rest must not be.
    """
    length = input_ids.shape[-1]
    if trunc_len >= length:
        raise ValueError(f"trunc_len={trunc_len} needs a sequence longer than {length}")

    short_loss = torch.zeros_like(long_loss)
    scored = torch.zeros(long_loss.shape, dtype=torch.bool, device=long_loss.device)
    windows: list[dict] = []

    with torch.no_grad():
        for start in range(0, length - trunc_len, window):
            # The reference shrinks its stride on the last window so the loop ends exactly at the
            # sequence end. Written as a local `span` rather than by rebinding `window`, which in the
            # original also shortens every subsequent iteration.
            span = min(window, length - start - trunc_len)
            if span <= 0:
                break
            chunk_ids = input_ids[:, start : start + trunc_len + span]
            hidden = _final_hidden_states(model, input_ids=chunk_ids, attention_mask=None)
            chunk_losses = per_token_ce(
                model.get_output_embeddings(), hidden, chunk_ids, chunk_size=logit_chunk
            )
            del hidden
            tail = chunk_losses[trunc_len - 1 : trunc_len - 1 + span]
            lo = start + trunc_len - 1
            short_loss[lo : lo + span] = tail
            scored[lo : lo + span] = True
            windows.append(
                {
                    "start": start,
                    "span": span,
                    "max_abs_diff_vs_long": float(
                        (tail - long_loss[lo : lo + span]).abs().max()
                    ),
                }
            )
    return short_loss, scored, windows


def check_alignment(windows: list[dict], *, tol: float) -> dict:
    """
    Assert the one structural property that catches an off-by-one, and return the evidence.

    The ``start=0`` window's short context *is* the true prefix, so its losses must reproduce the
    long-context losses. Any later window is genuinely truncated, so its must not. A shift of one
    position breaks the first check; a wrong write offset breaks the second. Neither is visible in any
    aggregate this script reports, which is why it is an assertion and not a note.
    """
    if not windows:
        raise RuntimeError("no windows were scored, so alignment cannot be checked")
    first = windows[0]
    if first["start"] != 0:
        raise RuntimeError(f"expected the first window at start=0, got {first['start']}")
    first_diff = first["max_abs_diff_vs_long"]
    later = [w["max_abs_diff_vs_long"] for w in windows[1:]]

    if first_diff > tol:
        raise AssertionError(
            f"the start=0 window sees the true prefix, so its short loss must equal the long loss, "
            f"but max|short - long| = {first_diff:.3e} > tol {tol:.3e}. This is an off-by-one in the "
            "window slicing, and every statistic below would be computed against the wrong tokens."
        )
    if later and max(later) <= tol:
        raise AssertionError(
            f"every truncated window reproduced the long-context loss (max diff "
            f"{max(later):.3e} <= tol {tol:.3e}). The short context is not actually being truncated, "
            "so `L_short - L_long` is identically 0 and the weighting would be a no-op."
        )
    return {
        "start0_max_abs_diff": first_diff,
        "truncated_max_abs_diff": max(later) if later else None,
        "truncated_min_abs_diff": min(later) if later else None,
        "n_windows": len(windows),
        "tol": tol,
    }


def unit_metrics(
    long_loss: torch.Tensor,
    short_loss: torch.Tensor,
    scored: torch.Tensor,
    *,
    threshold: float,
    criteria: list[tuple[float, float]],
) -> dict:
    """
    Every statistic for one ``(document, trunc_len)`` unit, over the scored positions only.

    ``criteria`` is a list of ``(alpha, beta)``; each is evaluated post-hoc on the already-computed
    losses, so sweeping them costs nothing beyond a comparison.
    """
    long_s = long_loss[scored].to(torch.float32)
    short_s = short_loss[scored].to(torch.float32)
    n = int(long_s.numel())
    if n == 0:
        raise RuntimeError("no scored positions in this unit")

    discrepancy = short_s - long_s
    weights = torch.exp(discrepancy).clamp(max=threshold)

    total = float(weights.sum())
    sum_sq = float((weights * weights).sum())
    participation = total**2 / (n * sum_sq) if sum_sq > 0 else 0.0

    metrics = {
        "n_scored": n,
        "n_total": int(long_loss.numel()),
        "scored_frac": n / int(long_loss.numel()),
        # THE gate: the delta objective failed because its weight ranked with the loss.
        "spearman_w_vs_long": spearman(weights, long_s),
        # For contrast. The raw discrepancy is the closest analogue of `delta`; if `w` decorrelates
        # from the loss while this does not, the exp+clamp is what is doing the work.
        "spearman_discrepancy_vs_long": spearman(discrepancy, long_s),
        "spearman_w_vs_short": spearman(weights, short_s),
        "weight_participation": participation,
        "weight_mean": float(weights.mean()),
        "weight_median": float(weights.median()),
        "weight_at_ceiling_frac": float((weights >= threshold - 1e-6).float().mean()),
        "all_loss_mean": float(long_s.mean()),
        "short_loss_mean": float(short_s.mean()),
        "discrepancy_mean": float(discrepancy.mean()),
        "discrepancy_positive_frac": float((discrepancy > 0).float().mean()),
        "criteria": {},
    }

    for alpha, beta in criteria:
        # `find_key_token`: (L_short - L_long) > alpha AND L_long < -beta. beta is stored negative in
        # the reference (default -2) and negated at the comparison, so -beta is the loss ceiling.
        is_key = (discrepancy > alpha) & (long_s < -beta)
        n_key = int(is_key.sum())
        n_alpha_pass = int((discrepancy > alpha).sum())
        metrics["criteria"][f"alpha={alpha},beta={beta}"] = {
            "alpha": alpha,
            "beta": beta,
            "key_rate": n_key / n,
            "n_key": n_key,
            "key_loss_mean": float(long_s[is_key].mean()) if n_key else None,
            "key_short_loss_mean": float(short_s[is_key].mean()) if n_key else None,
            "key_discrepancy_mean": float(discrepancy[is_key].mean()) if n_key else None,
            # The signature to look for: key tokens should be *easier* than average under the long
            # context. The beta condition forces key_loss_mean < -beta; this reports the margin.
            "key_minus_all_loss": (
                float(long_s[is_key].mean()) - float(long_s.mean()) if n_key else None
            ),
            # What share of the discrepancy-passing tokens the beta condition rejects. This is the
            # part LongCE has and the delta weighting does not; near 0 means beta is inert here and
            # the criterion is only a discrepancy threshold after all.
            "beta_rejection_frac": (
                (n_alpha_pass - n_key) / n_alpha_pass if n_alpha_pass else None
            ),
        }
    return metrics


# --------------------------------------------------------------------------------------------
# aggregation & reporting
# --------------------------------------------------------------------------------------------


def _summarize(values: list) -> dict | None:
    """mean/std/min/max over the per-document values, or ``None`` if there are none."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    tensor = torch.tensor(present, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "n": len(present),
    }


SCALAR_KEYS = (
    "spearman_w_vs_long",
    "spearman_discrepancy_vs_long",
    "spearman_w_vs_short",
    "weight_participation",
    "weight_mean",
    "weight_median",
    "weight_at_ceiling_frac",
    "all_loss_mean",
    "short_loss_mean",
    "discrepancy_mean",
    "discrepancy_positive_frac",
    "scored_frac",
)

CRITERION_KEYS = (
    "key_rate",
    "key_loss_mean",
    "key_discrepancy_mean",
    "key_minus_all_loss",
    "beta_rejection_frac",
)


def aggregate(units: list[dict]) -> dict:
    """Per-``trunc_len`` aggregates across documents."""
    by_trunc: dict[int, list[dict]] = {}
    for unit in units:
        by_trunc.setdefault(unit["trunc_len"], []).append(unit["metrics"])

    out = {}
    for trunc_len, rows in sorted(by_trunc.items()):
        summary = {key: _summarize([r[key] for r in rows]) for key in SCALAR_KEYS}
        criteria = {}
        for name in rows[0]["criteria"]:
            criteria[name] = {
                key: _summarize([r["criteria"][name][key] for r in rows])
                for key in CRITERION_KEYS
            }
        out[str(trunc_len)] = {"n_docs": len(rows), "scalars": summary, "criteria": criteria}
    return out


def verdict(aggregates: dict, *, gate: float) -> dict:
    """
    Apply the step-1 gate and pick a ``trunc_len``, so the decision lives in the artifact.

    ``spearman(w, L_long)`` strongly positive means LongCE is a proxy for loss magnitude on this
    corpus too, and training on it would repeat the delta collapse. Reporting that is the useful
    outcome, not a reason to retune.
    """
    rows = []
    for trunc_len, agg in aggregates.items():
        rows.append(
            {
                "trunc_len": int(trunc_len),
                "spearman_w_vs_long": agg["scalars"]["spearman_w_vs_long"]["mean"],
                "weight_participation": agg["scalars"]["weight_participation"]["mean"],
                "scored_frac": agg["scalars"]["scored_frac"]["mean"],
                "key_rates": {
                    name: c["key_rate"]["mean"] for name, c in agg["criteria"].items()
                },
            }
        )
    worst = max(r["spearman_w_vs_long"] for r in rows)

    # A "reasonable" key_rate is single digits to ~20%: below that the objective sees too few tokens
    # to be a signal, above it the weighting stops discriminating. Rank by the gate first, then by
    # how close the best criterion's rate sits to the middle of that band.
    def _score(row):
        return (row["spearman_w_vs_long"], min(abs(r - 0.10) for r in row["key_rates"].values()))

    recommended = min(rows, key=_score)
    return {
        "gate_threshold": gate,
        "max_spearman_w_vs_long": worst,
        "passes": worst <= gate,
        "rows": rows,
        "recommended_trunc_len": recommended["trunc_len"],
        "note": (
            "PASS: spearman(w, L_long) is not strongly positive, so LongCE's weight is a different "
            "quantity from the loss it multiplies -- proceed to step 2."
            if worst <= gate
            else "STOP: spearman(w, L_long) is strongly positive, so LongCE is a proxy for loss "
            "magnitude on this corpus as well. Training on it would repeat the delta collapse."
        ),
    }


def report(aggregates: dict, decision: dict) -> None:
    """Print the numbers the gate is decided on, so the log alone is enough to read the run."""
    for trunc_len, agg in aggregates.items():
        s = agg["scalars"]
        logger.info("=" * 78)
        logger.info("trunc_len=%s  (%d docs)", trunc_len, agg["n_docs"])
        logger.info(
            "  scored_frac               %.3f          (positions with a short-context counterfactual)",
            s["scored_frac"]["mean"],
        )
        logger.info(
            "  spearman(w, L_long)      %+.3f +- %.3f  <-- THE GATE (delta's failure mode)",
            s["spearman_w_vs_long"]["mean"],
            s["spearman_w_vs_long"]["std"],
        )
        logger.info(
            "  spearman(discrep, L_long)%+.3f +- %.3f  (the delta analogue, for contrast)",
            s["spearman_discrepancy_vs_long"]["mean"],
            s["spearman_discrepancy_vs_long"]["std"],
        )
        logger.info(
            "  weight_participation      %.3f +- %.3f  (failed delta run: 0.13-0.18; 1.0 = no-op)",
            s["weight_participation"]["mean"],
            s["weight_participation"]["std"],
        )
        logger.info(
            "  weight_at_ceiling_frac    %.3f          (saturated by clamp)",
            s["weight_at_ceiling_frac"]["mean"],
        )
        logger.info("  all_loss_mean             %.3f", s["all_loss_mean"]["mean"])
        for name, c in agg["criteria"].items():
            key_loss = c["key_loss_mean"]
            gap = c["key_minus_all_loss"]
            rejects = c["beta_rejection_frac"]
            logger.info(
                "    %-18s key_rate %.4f  key_L %s  key_L-all_L %s  beta_rejects %s",
                name,
                c["key_rate"]["mean"],
                f"{key_loss['mean']:.3f}" if key_loss else "n/a  ",
                f"{gap['mean']:+.3f}" if gap else "n/a   ",
                f"{rejects['mean']:.3f}" if rejects else "n/a",
            )
    logger.info("=" * 78)
    logger.info("%s", decision["note"])
    logger.info(
        "max spearman(w, L_long) = %+.3f against gate %.2f; recommended trunc_len = %d",
        decision["max_spearman_w_vs_long"],
        decision["gate_threshold"],
        decision["recommended_trunc_len"],
    )


# --------------------------------------------------------------------------------------------


def parse_criterion(text: str) -> tuple[float, float]:
    """``"2,-2"`` -> ``(alpha=2.0, beta=-2.0)``, matching ``find_key_token``'s signature."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'alpha,beta', got {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse {text!r}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenized", required=True, help="pretokenized corpus root")
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--n-docs", type=int, default=8)
    parser.add_argument("--trunc-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument(
        "--criteria",
        type=parse_criterion,
        nargs="+",
        default=[(2.0, -2.0), (1.0, -2.0), (2.0, -4.0)],
        help="alpha,beta pairs for the binary key-token criterion",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="LongCE's `internal`")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="LongCE's `thre`"
    )
    parser.add_argument("--logit-chunk", type=int, default=4096)
    parser.add_argument(
        "--align-tol",
        type=float,
        default=1e-3,
        help="tolerance for the start=0 window's exactness. Not 0 for a real model: bf16 "
        "attention kernels are not bitwise invariant to sequence length",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=0.4,
        help="spearman(w, L_long) above this fails step 1's gate",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="evaluation/longce_probe.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a GPU: the backbone must run to produce the losses")

    config = TokenizedConfig(
        root=args.tokenized,
        seq_len=args.seq_len,
        subsets=tuple(args.subsets) if args.subsets else None,
        # "head" so a document's window is reproducible and matches what step 2's precompute would
        # cache; "random" would make the probe unrepeatable for no benefit here.
        take_from="head",
        shuffle_buffer=1,
        seed=args.seed,
        max_documents=args.n_docs,
    )
    loader = build_tokenized_dataloader(config, batch_size=1, num_workers=0)

    model, _ = build_model(args.model, getattr(torch, args.dtype), args.attn, args.device)

    units: list[dict] = []
    alignment: list[dict] = []
    for doc_index, batch in enumerate(loader):
        if doc_index >= args.n_docs:
            break
        input_ids = batch["input_ids"].to(args.device)
        doc_id = batch["doc_ids"][0]
        long_loss = long_context_losses(model, input_ids, logit_chunk=args.logit_chunk)
        logger.info(
            "doc %d/%d %s: L=%d  mean L_long=%.3f",
            doc_index + 1,
            args.n_docs,
            doc_id,
            input_ids.shape[-1],
            float(long_loss.mean()),
        )

        for trunc_len in args.trunc_lens:
            short_loss, scored, windows = short_context_losses(
                model,
                input_ids,
                long_loss,
                trunc_len=trunc_len,
                window=args.window,
                logit_chunk=args.logit_chunk,
            )
            check = check_alignment(windows, tol=args.align_tol)
            alignment.append({"doc_id": doc_id, "trunc_len": trunc_len, **check})
            logger.info(
                "  trunc_len=%d: alignment start0=%.3e truncated=%.3e (%d windows)",
                trunc_len,
                check["start0_max_abs_diff"],
                check["truncated_max_abs_diff"] or 0.0,
                check["n_windows"],
            )
            metrics = unit_metrics(
                long_loss,
                short_loss,
                scored,
                threshold=args.threshold,
                criteria=list(args.criteria),
            )
            units.append({"doc_id": doc_id, "trunc_len": trunc_len, "metrics": metrics})
            first_name, first_criterion = next(iter(metrics["criteria"].items()))
            logger.info(
                "    spearman(w,L_long)=%+.3f participation=%.3f key_rate(%s)=%.4f",
                metrics["spearman_w_vs_long"],
                metrics["weight_participation"],
                first_name,
                first_criterion["key_rate"],
            )

    if not units:
        raise SystemExit("no documents were scored; check --tokenized/--subsets/--seq-len")

    aggregates = aggregate(units)
    decision = verdict(aggregates, gate=args.gate)
    report(aggregates, decision)

    payload = {
        "config": {
            "model": args.model,
            "tokenized": args.tokenized,
            "subsets": args.subsets,
            "seq_len": args.seq_len,
            "n_docs": len({u["doc_id"] for u in units}),
            "trunc_lens": args.trunc_lens,
            "criteria": [list(c) for c in args.criteria],
            "window": args.window,
            "threshold": args.threshold,
            "logit_chunk": args.logit_chunk,
            "dtype": args.dtype,
            "attn": args.attn,
            "seed": args.seed,
        },
        "alignment": alignment,
        "aggregates": aggregates,
        "verdict": decision,
        "units": units,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("wrote %s", out)

    if not decision["passes"]:
        raise SystemExit(
            f"step-1 gate failed: max spearman(w, L_long) = "
            f"{decision['max_spearman_w_vs_long']:+.3f} > {args.gate}. Do not train on this."
        )


if __name__ == "__main__":
    main()
