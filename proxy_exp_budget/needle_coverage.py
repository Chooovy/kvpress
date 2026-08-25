# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Is the **needle** inside the selected support? Per arm, per layer, on real RULER rows.

This is the measurement the earlier ones kept missing, and the reason they missed it:

* :mod:`attn_recall` / :mod:`layer_divergence` average attention mass over the support. All three
  arms score ~0.72-0.79 because most of a long context's mass sits on the attention sink and the
  local window, which ``force_sink=4`` / ``force_local=64`` pin *unconditionally* and for free. The
  needle is a handful of tokens carrying a tiny slice of mass, so dropping it changes the average by
  about a percent while changing the RULER answer completely.
* :mod:`logit_divergence` samples mid-context positions, which the local window already predicts, so
  every arm matches dense there (B's ``vt`` top-1 agreement is 1.000).

So instead of a proxy, this locates the answer string's own token positions in the context and asks
the direct question: **does the top-k support contain them?** That is exactly what the model needs in
order to answer, and it is the quantity a 62 -> 7 collapse on ``niah_single_2`` has to be explained
by.

``needle_covered`` is the fraction of the answer's token positions that appear in the support,
averaged over KV heads, query rows and rows. Reported per layer, because an answer only has to be
dropped in the layers that do the retrieval for the answer to be lost -- and per arm, so a
difference can be attributed.

The support is built through the real path (``project_q`` / ``project_k`` /
:func:`streaming_topk_support` with the eval's ``force_sink``/``force_local``), from the *dense*
hidden states. Using dense states is deliberate here: it isolates the scorer's ranking from the
sparse run's own drift, which :mod:`layer_divergence` already showed to be comparable across arms.
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

from dissect_scores import ARMS, build_indexers, capture_hidden_states  # noqa: E402


def find_needle_positions(tok, context: str, answer: str) -> list[int]:
    """Token positions in ``context`` that spell ``answer``.

    Located by character offset and mapped to tokens, rather than by matching token ids: the
    answer's standalone tokenisation differs from its in-context one (leading space, digit
    grouping), so an id-level search silently finds nothing for exactly the rows that matter.
    """
    char_at = context.find(str(answer))
    if char_at < 0:
        return []
    enc = tok(context, return_offsets_mapping=True, add_special_tokens=False)
    end = char_at + len(str(answer))
    return [
        i
        for i, (a, b) in enumerate(enc["offset_mapping"])
        if a < end and b > char_at  # any overlap with the answer's character span
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3,vt")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--rows-per-task", type=int, default=4)
    ap.add_argument("--n-q-rows", type=int, default=8, help="query positions sampled at the tail")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/needle_coverage.json"))
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
    n_kv = model.config.num_key_value_heads
    hidden = model.config.hidden_size

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    rows = []
    for task in tasks:
        kept = 0
        for _, r in df[df["task"] == task].iterrows():
            if kept >= args.rows_per_task:
                break
            gold = r["answer"]
            gold = gold[0] if hasattr(gold, "__len__") and not isinstance(gold, str) else gold
            needle = find_needle_positions(tok, r["context"], gold)
            if not needle:
                continue
            ids = tok(r["context"], return_tensors="pt", add_special_tokens=False).input_ids
            rows.append((task, ids, needle, str(gold)))
            kept += 1
    print(f"{len(rows)} rows with a locatable needle, over {len(tasks)} tasks", flush=True)
    for task, ids, needle, gold in rows[:6]:
        print(
            f"  {task:<16s} ctx={ids.shape[1]:>5d} needle tokens={len(needle)} "
            f"at {needle[0]}..{needle[-1]} (depth {needle[0] / ids.shape[1]:.2f}) gold={gold[:24]!r}"
        )

    # Hidden states are arm-independent (the indexer never feeds back into a dense pass), so capture
    # once per row and score with all three arms.
    per_row_hs = []
    for task, ids, needle, gold in rows:
        per_row_hs.append(capture_hidden_states(model, ids.to(device), n_layers))
    del model
    torch.cuda.empty_cache()

    results = {}
    for name, rel in ARMS.items():
        indexers, cfg_rec, _ = build_indexers(args.ckpt_root + rel, hidden, n_kv, device)
        # cov[layer] accumulates over rows; by_task keeps the per-task breakdown.
        cov = {li: [] for li in range(n_layers)}
        by_task: dict[str, list] = {t: [] for t in tasks}
        for (task, ids, needle, gold), hs in zip(rows, per_row_hs):
            n = ids.shape[1]
            q_rows = torch.linspace(max(needle[-1] + 1, n - 512), n - 1, args.n_q_rows).long()
            needle_t = torch.tensor(needle, device=device)
            for li in range(n_layers):
                h = hs[li].to(device)
                with torch.no_grad():
                    q_idx = indexers[li].project_q(h)
                    k_idx = indexers[li].project_k(h)
                    support, valid = streaming_topk_support(
                        q_idx, k_idx, args.topk, mask=None,
                        force_sink=args.force_sink, force_local=args.force_local,
                    )
                    sel = support[0][:, q_rows.to(device), :]  # (n_kv, R, topk)
                    ok = valid[0][:, q_rows.to(device), :]
                    marked = torch.zeros(n_kv, sel.shape[1], n, dtype=torch.bool, device=device)
                    marked.scatter_(2, sel.clamp_min(0).long(), ok)
                    frac = marked[:, :, needle_t].float().mean().item()
                cov[li].append(frac)
                by_task[task].append((li, frac))
        results[name] = {
            "arm": name,
            "ckpt_config": cfg_rec,
            "needle_covered_mean": float(np.mean([np.mean(v) for v in cov.values()])),
            "per_layer": [{"layer": li, "needle_covered": float(np.mean(cov[li]))} for li in cov],
            "by_task": {
                t: float(np.mean([f for _, f in v])) for t, v in by_task.items() if v
            },
        }
        print(
            f"arm {name}: needle_covered={results[name]['needle_covered_mean']:.4f} "
            f"by_task={ {k: round(v, 3) for k, v in results[name]['by_task'].items()} }",
            flush=True,
        )
        del indexers
        torch.cuda.empty_cache()

    results["_meta"] = {
        "tasks": tasks,
        "topk": args.topk,
        "n_rows": len(rows),
        "force_sink": args.force_sink,
        "force_local": args.force_local,
        "chance": args.topk / float(np.mean([r[1].shape[1] for r in rows])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(f"\nneedle coverage (fraction of the answer's key positions inside the top-k support)")
    print(f"{'arm':<5s}{'mean':>10s}" + "".join(f"{t[:14]:>16s}" for t in tasks))
    print("-" * (15 + 16 * len(tasks)))
    for name in ARMS:
        r = results[name]
        cells = "".join(f"{r['by_task'].get(t, float('nan')):>16.4f}" for t in tasks)
        print(f"{name:<5s}{r['needle_covered_mean']:>10.4f}{cells}")
    print(f"\nchance level (topk/N) = {results['_meta']['chance']:.4f}")

    print(f"\nper layer:\n{'layer':>5s}" + "".join(f"{a:>10s}" for a in ARMS))
    for li in range(0, n_layers, 3):
        print(
            f"{li:>5d}"
            + "".join(f"{results[a]['per_layer'][li]['needle_covered']:>10.4f}" for a in ARMS)
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
