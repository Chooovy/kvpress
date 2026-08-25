# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dissect the *score* of three trained ScalarIndexer arms on real RULER text.

Why this exists: arm B (cross-replay, ``mid_dim=256``, ``budget=2048``) scores 20.43 on RULER 8K
where arm A (cross-replay, linear, ``B=1``) scores 44.75 and arm C (plain e2e LM loss,
``mid_dim=256``) scores 66.24 -- yet every prior expected B to beat A. The collapse is
*indiscriminate* (``vt`` 49.76 -> 0.00, ``niah_single_2`` 59.09 -> 7.58), where a pure
concentration mismatch would be *selective*. So the hypothesis under test is that B's score works
on the replay objective but degenerates under a hard top-k -- e.g. the selected set is nearly
content-independent, or one component dominates it.

The load-bearing measurement is **top-k set stability across documents** (:func:`jaccard_across_docs`).
At ``topk=2048, N=8192`` a content-driven score overlaps at about ``topk/N = 0.25`` between two
unrelated documents; an overlap near 1.0 means the score ignores its input and always picks the same
positions, which would explain an indiscriminate collapse directly.

One backbone pass, three arms
-----------------------------
The indexer is a *read-only* side-car during a dense forward: it never feeds back into the
backbone (see ``cross_replay_e2e.md`` §1 / §6 -- pass 1 is dense and ungated). The three arms also
share one frozen backbone. So the layer-input hidden states are *identical* across arms, and this
script captures them once and scores them with all three checkpoints. Loading the 8B model three
times would produce bit-identical hidden states for 3x the cost.

Real text, not random hidden states: a random ``h`` has none of the outlier-direction structure
that ``in_norm`` and ``w_in`` were trained against, so a score that is degenerate on real text can
look healthy on noise, and vice versa.

Scores come from :meth:`ScalarIndexer.score_keys`, the ``O(L)`` per-key entry point that the press
and the sparse-inference path both use -- not from ``forward``, whose query axis is a broadcast view
of exactly these values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.scalar_indexer import (  # noqa: E402
    ScalarIndexer,
    ScalarIndexerConfig,
)

ARMS = {
    "A": "Qwen-3-8B-gqa_indexer_cross_replay/stage1/final.pt",
    "B": "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256/final.pt",
    "C": "Qwen-3-8B-gqa_indexer_scalar/stage1/final.pt",
}


# ----------------------------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------------------------
def capture_hidden_states(model, input_ids: torch.Tensor, n_layers: int) -> dict[int, torch.Tensor]:
    """Layer-input hidden states per layer, ``{layer_idx: (1, S, hidden)}`` on CPU.

    Hooked on ``self_attn`` and reading ``hidden_states`` exactly as
    :meth:`SparseAttentionContext._capture_hook` does, so these are the same tensors the indexer
    is fed at inference -- the post-input-layernorm attention input, not the residual stream.
    Kept on CPU because 36 x 16K x 4096 bf16 is ~4.8 GB and the GPUs are shared with the eval.
    """
    captured: dict[int, torch.Tensor] = {}

    def hook(module, args, kwargs):
        idx = getattr(module, "layer_idx", None)
        if idx is None:
            return None
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        captured[int(idx)] = h.detach().to("cpu")
        return None

    handles = [
        layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        for layer in model.model.layers
    ]
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    if len(captured) != n_layers:
        raise RuntimeError(f"captured {len(captured)} layers, expected {n_layers}")
    return captured


def build_indexers(ckpt_path: str, hidden_size: int, n_kv_heads: int, device) -> dict[int, ScalarIndexer]:
    """One :class:`ScalarIndexer` per layer, weights from the checkpoint.

    ``mid_dim`` is read from ``w_in``'s shape and ``pos_slope`` from the recorded config -- the
    latter is *not* a parameter (it is added inside ``score_keys``), so a wrong value would
    mis-score with every weight loading cleanly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("indexer", ckpt)
    cfg_rec = ckpt.get("config") or {}
    pos_slope = float(cfg_rec.get("scalar_pos_slope", 1e-6))

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in sd.items():
        parts = key.split(".")
        li = int(parts[parts.index("layers") + 1])
        by_layer.setdefault(li, {})[".".join(parts[parts.index("indexer") + 1 :])] = tensor

    indexers = {}
    for li, weights in by_layer.items():
        mid_dim = weights["w_in.weight"].shape[0] if "w_in.weight" in weights else 0
        cfg = ScalarIndexerConfig(
            hidden_size=hidden_size,
            n_heads=n_kv_heads,
            mid_dim=mid_dim,
            pos_slope=pos_slope,
            gate_scale="gate_scale" in weights,
        )
        idx = ScalarIndexer(cfg)
        missing, unexpected = idx.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"layer {li}: missing={missing} unexpected={unexpected}")
        indexers[li] = idx.to(device).eval()
    return indexers, cfg_rec, pos_slope


# ----------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------
def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rho between two 1-D score vectors, via Pearson on ranks."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def singular_spectrum(w: torch.Tensor) -> dict:
    """Fraction of squared Frobenius norm in the leading 1/4/16 singular values.

    A near-rank-1 ``w_out`` means the KV heads score almost identically, so per-head top-k
    collapses onto one shared set -- exactly the degeneracy that would make selection
    content-blind at the head level.
    """
    s = torch.linalg.svdvals(w.float())
    total = float((s**2).sum())
    out = {"n_sv": int(s.numel()), "sv_max": float(s[0])}
    for k in (1, 4, 16):
        if k <= s.numel():
            out[f"energy_top{k}"] = float((s[:k] ** 2).sum() / total)
    # Participation ratio of the spectrum: 1 = rank-1, n = flat.
    out["eff_rank"] = float((s**2).sum() ** 2 / (s**4).sum())
    return out


def topk_set(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-k key indices per head, ``(n_heads, topk)``. ``scores`` is ``(n_heads, N)``."""
    return scores.topk(topk, dim=-1).indices


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap of two index sets of equal size."""
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def analyse_arm(
    name: str,
    ckpt: str,
    hs_docs: list[dict[int, torch.Tensor]],
    hidden_size: int,
    n_kv_heads: int,
    topk: int,
    device,
) -> dict:
    """Every per-arm measurement, averaged over layers (and over documents where applicable)."""
    indexers, cfg_rec, pos_slope = build_indexers(ckpt, hidden_size, n_kv_heads, device)
    n_layers = len(indexers)
    n_docs = len(hs_docs)
    n = hs_docs[0][0].shape[1]

    per_layer = []
    for li in sorted(indexers):
        idx = indexers[li]
        # (n_docs, n_heads, N) fp32 -- score_keys is the O(L) per-key entry point.
        scores = []
        with torch.inference_mode():
            for doc in hs_docs:
                h = doc[li].to(device)
                scores.append(idx.score_keys(h)[0].float().cpu())
        scores = torch.stack(scores)

        # --- (0) top-k set stability ACROSS DOCUMENTS -- the load-bearing measurement.
        sets = [topk_set(s, topk) for s in scores]
        cross_doc, within_doc_heads = [], []
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                # Same head, different document.
                cross_doc += [jaccard(sets[i][hh], sets[j][hh]) for hh in range(n_kv_heads)]
        for i in range(n_docs):
            # Different head, same document -- how much do the 8 KV heads even differ?
            within_doc_heads += [
                jaccard(sets[i][hh], sets[i][gg])
                for hh in range(n_kv_heads)
                for gg in range(hh + 1, n_kv_heads)
            ]

        # --- (2) per-head Spearman, within a document, averaged over head pairs and docs.
        head_rho = [
            spearman(scores[d, hh], scores[d, gg])
            for d in range(n_docs)
            for hh in range(n_kv_heads)
            for gg in range(hh + 1, n_kv_heads)
        ]

        # --- (3) score distribution.
        flat = scores.reshape(-1)
        mean, std = float(flat.mean()), float(flat.std())
        # Per-head constant offset vs within-head spread: a score dominated by a per-head bias
        # ranks by whatever noise is left, and the bias itself cancels inside a per-head top-k.
        head_means = scores.mean(dim=-1)  # (n_docs, n_heads)
        within_head_std = float(scores.std(dim=-1).mean())

        # --- (4) position dependence. pos_slope adds a deliberate recency tilt; report how much
        # of the score's variation it (or anything else positional) actually explains.
        pos = torch.arange(n, dtype=torch.float32)
        pos_c = pos - pos.mean()
        pos_r = []
        for d in range(n_docs):
            for hh in range(n_kv_heads):
                s = scores[d, hh]
                sc = s - s.mean()
                denom = sc.norm() * pos_c.norm()
                pos_r.append(float((sc @ pos_c) / denom) if denom > 0 else float("nan"))
        # Mean score in 10 equal position buckets, averaged over docs/heads.
        buckets = scores.reshape(-1, n)[:, : (n // 10) * 10].reshape(-1, 10, n // 10).mean(dim=-1)
        # Absolute tilt the fixed slope contributes across the whole context, against the score's
        # own std: this is how to tell "positional because pos_slope" from "positional because the
        # learned part is".
        tilt = pos_slope * (n - 1)

        # --- (1) weight spectra.
        spec = {"w_out": singular_spectrum(indexers[li].w_out.weight.data)}
        if indexers[li].w_in is not None:
            spec["w_in"] = singular_spectrum(indexers[li].w_in.weight.data)

        per_layer.append(
            {
                "layer": li,
                "jaccard_cross_doc": float(np.mean(cross_doc)),
                "jaccard_cross_head": float(np.mean(within_doc_heads)),
                "head_spearman": float(np.nanmean(head_rho)),
                "score_mean": mean,
                "score_std": std,
                "score_min": float(flat.min()),
                "score_max": float(flat.max()),
                "std_over_absmean": std / abs(mean) if mean != 0 else float("inf"),
                "head_bias_spread": float(head_means.std()),
                "within_head_std": within_head_std,
                "pos_pearson": float(np.nanmean(pos_r)),
                "pos_buckets": buckets.mean(dim=0).tolist(),
                "pos_tilt_abs": tilt,
                "pos_tilt_over_std": tilt / std if std > 0 else float("inf"),
                "spectra": spec,
            }
        )

    def avg(key):
        return float(np.mean([p[key] for p in per_layer]))

    summary = {
        "arm": name,
        "ckpt": ckpt,
        "ckpt_config": cfg_rec,
        "n_layers": n_layers,
        "n_docs": n_docs,
        "N": n,
        "topk": topk,
        "topk_over_N": topk / n,
        "jaccard_cross_doc": avg("jaccard_cross_doc"),
        "jaccard_cross_head": avg("jaccard_cross_head"),
        "head_spearman": avg("head_spearman"),
        "score_mean": avg("score_mean"),
        "score_std": avg("score_std"),
        "std_over_absmean": avg("std_over_absmean"),
        "head_bias_spread": avg("head_bias_spread"),
        "within_head_std": avg("within_head_std"),
        "pos_pearson": avg("pos_pearson"),
        "pos_tilt_over_std": avg("pos_tilt_over_std"),
        "w_out_energy_top1": float(np.mean([p["spectra"]["w_out"]["energy_top1"] for p in per_layer])),
        "w_out_eff_rank": float(np.mean([p["spectra"]["w_out"]["eff_rank"] for p in per_layer])),
        "per_layer": per_layer,
    }
    if "w_in" in per_layer[0]["spectra"]:
        summary["w_in_energy_top1"] = float(
            np.mean([p["spectra"]["w_in"]["energy_top1"] for p in per_layer])
        )
        summary["w_in_energy_top16"] = float(
            np.mean([p["spectra"]["w_in"]["energy_top16"] for p in per_layer])
        )
        summary["w_in_eff_rank"] = float(np.mean([p["spectra"]["w_in"]["eff_rank"] for p in per_layer]))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/score_dissection.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    # dtype/torch_dtype: same fallback evaluate_sparse.py carries, for the older signature.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    hidden_size = model.config.hidden_size
    print(f"model: {n_layers} layers, {n_kv_heads} KV heads, hidden {hidden_size}", flush=True)

    # Real RULER contexts, from distinct tasks so the documents are genuinely unrelated -- two
    # samples of the same synthetic template would share boilerplate and inflate the overlap.
    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    picked, seen_tasks = [], set()
    for _, row in df.iterrows():
        if row["task"] in seen_tasks:
            continue
        ids = tok(row["context"], return_tensors="pt").input_ids
        if ids.shape[1] < args.n_tokens:
            continue
        seen_tasks.add(row["task"])
        picked.append((row["task"], ids[:, : args.n_tokens]))
        if len(picked) >= args.n_docs:
            break
    print("documents:", [t for t, _ in picked], flush=True)

    hs_docs = []
    for task, ids in picked:
        hs_docs.append(capture_hidden_states(model, ids.to(device), n_layers))
        print(f"  captured {task}", flush=True)

    # The backbone is no longer needed: the arms differ only in indexer weights, and the hidden
    # states above are identical for all three (the indexer never feeds back into a dense pass).
    del model
    torch.cuda.empty_cache()

    results = {}
    for name, rel in ARMS.items():
        print(f"=== arm {name}", flush=True)
        results[name] = analyse_arm(
            name,
            args.ckpt_root + rel,
            hs_docs,
            hidden_size,
            n_kv_heads,
            args.topk,
            device,
        )

    results["_meta"] = {
        "documents": [t for t, _ in picked],
        "n_tokens": args.n_tokens,
        "topk": args.topk,
        "model": args.model,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}", flush=True)

    # ------------------------------------------------------------------ report
    keys = [
        ("jaccard_cross_doc", "topk Jaccard, 2 docs", ".4f"),
        ("jaccard_cross_head", "topk Jaccard, 2 heads", ".4f"),
        ("head_spearman", "per-head Spearman", ".4f"),
        ("score_mean", "score mean", ".4f"),
        ("score_std", "score std", ".4f"),
        ("std_over_absmean", "std / |mean|", ".3f"),
        ("head_bias_spread", "per-head bias spread", ".4f"),
        ("within_head_std", "within-head std", ".4f"),
        ("pos_pearson", "corr(score, position)", ".4f"),
        ("pos_tilt_over_std", "pos_slope tilt / std", ".2e"),
        ("w_out_energy_top1", "w_out top-1 SV energy", ".4f"),
        ("w_out_eff_rank", "w_out effective rank", ".2f"),
        ("w_in_energy_top1", "w_in top-1 SV energy", ".4f"),
        ("w_in_eff_rank", "w_in effective rank", ".2f"),
    ]
    print(f"\n{'measurement':<26s}" + "".join(f"{a:>14s}" for a in ARMS))
    print("-" * (26 + 14 * len(ARMS)))
    for key, label, fmt in keys:
        cells = ""
        for a in ARMS:
            v = results[a].get(key)
            cells += f"{'--':>14s}" if v is None else f"{format(v, fmt):>14s}"
        print(f"{label:<26s}{cells}")
    print(f"\nchance Jaccard at topk/N = {args.topk}/{args.n_tokens}: "
          f"{args.topk / (2 * args.n_tokens - args.topk):.4f}")


if __name__ == "__main__":
    main()
