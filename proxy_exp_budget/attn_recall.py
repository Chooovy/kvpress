# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functional test of the three arms' selection: how much real attention mass does the top-k keep?

:mod:`dissect_scores` establishes that arm B's top-k is *content-driven* (cross-document Jaccard
0.149 against a 0.143 chance floor, statistically the same as arms A and C). Content-driven is not
the same as correct: a score can depend on its input and still rank the wrong keys first. This
script asks the question that actually predicts RULER -- **is the needle inside the support?** --
by computing the dense attention distribution and measuring the fraction of its mass that lands on
the keys the indexer selected.

Faithful to inference in three ways that matter
-----------------------------------------------
1. **Selection runs through the real path**: ``project_q`` / ``project_k`` fed to
   :func:`streaming_topk_support` with the eval's ``force_sink=4, force_local=64``, i.e. the exact
   call :meth:`SparseAttentionContext._attend` makes. Not ``score_keys`` in fp32 -- ``project_k``
   casts to ``hidden_states.dtype``, so selection is really **bf16**, and the ScalarIndexer
   docstring warns that a bf16 score resolves only ~200 distinct values over 8192 keys. Any tie
   structure that causes is therefore included here rather than assumed away.
2. **Real q/k, post-RoPE**, captured from inside the attention call, so the reference distribution
   is the model's own and not a re-derivation of it.
3. **Real RULER text**, and the recall is reported against the dense attention the sparse run is
   trying to approximate.

``mass_recall`` is the headline: 1.0 means the support carries all the attention, and the sparse
run should behave like the dense one. It is reported alongside ``oracle_recall`` (the best any
``topk``-sized set could do, i.e. the top-k of the *true* attention weights) so a low number can be
attributed to the scorer rather than to the budget being too small for the task.
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

from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support  # noqa: E402

from dissect_scores import ARMS, build_indexers  # noqa: E402


def capture_qkh(model, input_ids: torch.Tensor, n_layers: int):
    """Capture per-layer ``(hidden_states, q, k)`` for one forward pass, on CPU.

    ``q``/``k`` are taken from inside the attention function, so they are post-RoPE and
    post-cache-update -- the same tensors :func:`sparse_gqa_attention` consumes at inference.
    ``hidden_states`` comes from the pre-hook, matching
    :meth:`SparseAttentionContext._capture_hook`.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    hs: dict[int, torch.Tensor] = {}
    qk: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def pre_hook(module, args, kwargs):
        idx = getattr(module, "layer_idx", None)
        if idx is None:
            return None
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        hs[int(idx)] = h.detach().to("cpu")
        return None

    impl_name = "_capture_qk_probe"
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capturing_impl(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        qk[int(module.layer_idx)] = (query.detach().to("cpu"), key.detach().to("cpu"))
        return sdpa(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)

    global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    had_previous, previous_fn = impl_name in global_mapping, global_mapping.get(impl_name)
    ALL_ATTENTION_FUNCTIONS.register(impl_name, capturing_impl)
    handles = [
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        for layer in model.model.layers
    ]
    previous_impl = model.config._attn_implementation
    model.config._attn_implementation = impl_name
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
        model.config._attn_implementation = previous_impl
        if had_previous:
            global_mapping[impl_name] = previous_fn
        else:
            global_mapping.pop(impl_name, None)

    if len(hs) != n_layers or len(qk) != n_layers:
        raise RuntimeError(f"captured {len(hs)} hidden / {len(qk)} qk, expected {n_layers}")
    return hs, qk


def recall_for_layer(
    indexer,
    h: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_rows: torch.Tensor,
    topk: int,
    force_sink: int,
    force_local: int,
    scaling: float,
) -> dict:
    """Attention-mass recall of the indexer's support, for a sample of query rows.

    ``q_rows`` indexes the sampled query positions. Only those rows' dense attention is formed, so
    this is ``O(len(q_rows) * k_len)`` rather than the full quadratic map.
    """
    # --- selection, exactly as SparseAttentionContext._attend builds it (bf16 in, via project_*).
    q_idx = indexer.project_q(h)  # (1, n_kv, Sq, Di)
    k_idx = indexer.project_k(h)  # (1, Sq, Di)
    support, valid = streaming_topk_support(
        q_idx, k_idx, topk, mask=None, force_sink=force_sink, force_local=force_local
    )  # (1, n_kv, Sq, topk)

    n_q_heads, k_len = q.shape[1], k.shape[2]
    n_kv = k.shape[1]
    group = n_q_heads // n_kv
    device = support.device

    q_rows = q_rows.to(device)
    sel = support[0, :, q_rows, :]  # (n_kv, R, topk)
    sel_valid = valid[0, :, q_rows, :]
    qs = q[0, :, q_rows.cpu(), :].to(device).float()  # (n_q_heads, R, D)
    ks = k[0].to(device).float()  # (n_kv, k_len, D)
    key_pos = torch.arange(k_len, device=device).view(1, -1)

    mass, oracle, hit_frac = [], [], []
    for kv in range(n_kv):
        idx = sel[kv].clamp_min(0).long()  # (R, topk); -1 slots masked by sel_valid below
        ok = sel_valid[kv]
        for g in range(group):
            qh = qs[kv * group + g]  # (R, D)
            logits = (qh @ ks[kv].T) * scaling  # (R, k_len)
            # Causal: a sampled query at absolute position p may only see keys <= p.
            logits = logits.masked_fill(key_pos > q_rows.view(-1, 1), float("-inf"))
            w = torch.softmax(logits, dim=-1)  # (R, k_len)
            got = (w.gather(1, idx) * ok).sum(dim=-1)  # (R,)
            mass.append(got.mean().item())
            oracle.append(w.topk(min(topk, k_len), dim=-1).values.sum(dim=-1).mean().item())
            # Fraction of the true top-64 keys that the support contains -- the "is the needle in"
            # question, which mass alone can hide when a few sink keys carry most of the mass.
            n_true = min(64, k_len)
            true_top = w.topk(n_true, dim=-1).indices  # (R, n_true)
            # Membership without Python sets: mark selected keys in a bool row, then look up.
            marked = torch.zeros(idx.shape[0], k_len, dtype=torch.bool, device=device)
            marked.scatter_(1, idx, ok)
            hit_frac.append(marked.gather(1, true_top).float().mean().item())
    return {
        "mass_recall": float(np.mean(mass)),
        "oracle_recall": float(np.mean(oracle)),
        "top64_hit": float(np.mean(hit_frac)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--n-rows", type=int, default=16, help="sampled query positions per layer")
    ap.add_argument("--task", default=None, help="RULER task to use (default: first long enough)")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/attn_recall.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
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
    head_dim = getattr(model.config, "head_dim", hidden_size // model.config.num_attention_heads)
    scaling = head_dim**-0.5

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    row = None
    for _, r in df.iterrows():
        if args.task and r["task"] != args.task:
            continue
        ids = tok(r["context"], return_tensors="pt").input_ids
        if ids.shape[1] >= args.n_tokens:
            row, input_ids = r, ids[:, : args.n_tokens]
            break
    if row is None:
        raise SystemExit("no context long enough")
    print(f"document: task={row['task']} tokens={args.n_tokens}", flush=True)

    hs, qk = capture_qkh(model, input_ids.to(device), n_layers)
    del model
    torch.cuda.empty_cache()

    # Sample query rows late in the sequence: that is where a long-context answer is produced, and
    # where the support has the most keys to choose among.
    n = args.n_tokens
    q_rows = torch.linspace(n // 2, n - 1, args.n_rows).long()

    results = {}
    for name, rel in ARMS.items():
        indexers, cfg_rec, _ = build_indexers(
            args.ckpt_root + rel, hidden_size, n_kv_heads, device
        )
        per_layer = []
        for li in sorted(indexers):
            q, k = qk[li]
            per_layer.append(
                {
                    "layer": li,
                    **recall_for_layer(
                        indexers[li],
                        hs[li].to(device),
                        q,
                        k,
                        q_rows,
                        args.topk,
                        args.force_sink,
                        args.force_local,
                        scaling,
                    ),
                }
            )
            del indexers[li]
        results[name] = {
            "arm": name,
            "ckpt_config": cfg_rec,
            "mass_recall": float(np.mean([p["mass_recall"] for p in per_layer])),
            "oracle_recall": float(np.mean([p["oracle_recall"] for p in per_layer])),
            "top64_hit": float(np.mean([p["top64_hit"] for p in per_layer])),
            "per_layer": per_layer,
        }
        r = results[name]
        print(
            f"arm {name}: mass_recall={r['mass_recall']:.4f} "
            f"top64_hit={r['top64_hit']:.4f} oracle={r['oracle_recall']:.4f}",
            flush=True,
        )

    results["_meta"] = {
        "task": row["task"],
        "n_tokens": n,
        "topk": args.topk,
        "force_sink": args.force_sink,
        "force_local": args.force_local,
        "q_rows": q_rows.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(f"\n{'arm':<6s}{'mass_recall':>14s}{'top64_hit':>12s}{'oracle':>10s}")
    print("-" * 42)
    for name in ARMS:
        r = results[name]
        print(
            f"{name:<6s}{r['mass_recall']:>14.4f}{r['top64_hit']:>12.4f}{r['oracle_recall']:>10.4f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
