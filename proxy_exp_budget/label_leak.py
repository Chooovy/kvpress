# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Is the cross-replay target token visible to the query that predicts it? (Label leak.)

The structural observation, from ``cross_replay_training_step``: replay query ``j`` attends to the
**full rectangle** over ``C`` (``rectangle_mask``, every key visible, deliberately not causal —
§5), and its target is ``replay_ids[j+1]``. With ``C' == C``, key ``j+1`` of ``C`` *is* the target
token. So unlike the e2e LM loss — where query ``t`` sees ``C[0..t]`` and predicts ``C[t+1]``, which
is strictly not visible — the cross-replay objective can read its own label out of the key set.

Why this matters more than it sounds. §14 established that this arm's loss *anti*-correlates with
RULER (arm B: loss 1.18 and RULER 20.43; arm A: loss 2.70 and 44.75), and §15.3 showed the deficit
is the objective itself, not capacity or budget. A label leak would explain both at once: the loss
is partly measuring "can you find token j+1 in the cache", which is a copy/induction task rather
than a retrieval-under-budget task, and the gate can serve it by concentrating on a *single* key —
exactly the over-concentration signature §15.3 found in every cross-replay arm (they win
``niah_single_3``, the fewest-keys task, at every budget and capacity, and lose every many-keys task).

This script measures how much of the loss the leak accounts for, on the frozen model with **no
gate**, so it is a property of the geometry rather than of any checkpoint. Four masks over ``C``:

* ``rect``        — the full rectangle. What cross-replay trains on today.
* ``rect_noself`` — the rectangle with key ``j+1`` (the target) removed for row ``j``.
* ``rect_causal`` — the rectangle truncated to keys ``<= j``, i.e. no key at or beyond the target.
* ``causal``      — keys ``<= j`` only, which is what the e2e loss sees. Reference point.

``leak_share = (L_noself - L_rect) / L_noself`` is the fraction of the loss removed by making the
single target key visible. Near 0 ⇒ the leak is negligible and this hypothesis is dead. Large ⇒ the
objective is substantially a copy task, and the fix is a mask change (cheap) rather than a new loss.

Also reported: ``argmax_on_target``, the fraction of rows whose **most-attended key** is exactly the
target position ``j+1``, at a representative layer. That is the direct fingerprint of an induction
head exploiting the leak, and it is what would drive the gate to concentrate on one key.
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

NEG = -1e30
MODES = ("rect", "rect_noself", "rect_causal", "causal")


def build_mask(mode: str, n: int, device, dtype) -> torch.Tensor:
    """Additive ``(1, 1, N, N)`` mask over ``C`` for replay row ``j`` (absolute position N+j)."""
    j = torch.arange(n, device=device).view(n, 1)
    k = torch.arange(n, device=device).view(1, n)
    if mode == "rect":
        keep = torch.ones((n, n), dtype=torch.bool, device=device)
    elif mode == "rect_noself":
        # Drop ONLY the target key j+1. Everything else, including keys beyond it, stays visible --
        # so this isolates the leak from the broader "sees the future" question that rect_causal asks.
        keep = k != (j + 1)
    elif mode == "rect_causal":
        keep = k <= j
    elif mode == "causal":
        keep = k <= j
    else:
        raise ValueError(mode)
    return torch.where(keep, 0.0, NEG).to(dtype).view(1, 1, n, n)


@torch.inference_mode()
def replay_loss(model, ids: torch.Tensor, mode: str, n: int) -> float:
    """Next-token loss on ``C'`` under one mask, with ``KV(C)`` prefilled and read-only."""
    from transformers import DynamicCache

    sys.path.insert(0, str(REPO_ROOT))
    from kvpress.presses.gqa_indexer.cross_replay import ReadOnlyCache

    cache = DynamicCache()
    model.model(input_ids=ids)  # warm nothing; prefill below is the one that counts
    cache = DynamicCache()
    model.model(input_ids=ids, past_key_values=cache)

    dtype = next(model.parameters()).dtype
    mask = build_mask(mode, n, model.device, dtype)
    # Replay positions N..2N-1, matching cross_replay_training_step.
    pos = torch.arange(n, 2 * n, device=model.device).unsqueeze(0)
    hidden = model.model(
        input_ids=ids,
        past_key_values=ReadOnlyCache(cache),
        position_ids=pos,
        attention_mask=mask,
        use_cache=True,
    ).last_hidden_state

    total, count = 0.0, 0
    for start in range(0, n - 1, 512):
        stop = min(start + 512, n - 1)
        logits = model.lm_head(hidden[:, start:stop]).float()
        target = ids[:, start + 1 : stop + 1]
        total += torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
        ).item()
        count += target.numel()
    return total / count


@torch.inference_mode()
def argmax_on_target(model, ids: torch.Tensor, n: int, layer_idx: int) -> float:
    """Fraction of replay rows whose most-attended key is exactly the target position ``j+1``."""
    from transformers import DynamicCache

    from kvpress.presses.gqa_indexer.cross_replay import ReadOnlyCache

    cache = DynamicCache()
    model.model(input_ids=ids, past_key_values=cache)

    captured = {}

    def hook(module, args, kwargs):
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        captured["h"] = h
        return None

    attn = model.model.layers[layer_idx].self_attn
    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    dtype = next(model.parameters()).dtype
    try:
        model.model(
            input_ids=ids,
            past_key_values=ReadOnlyCache(cache),
            position_ids=torch.arange(n, 2 * n, device=model.device).unsqueeze(0),
            attention_mask=build_mask("rect", n, model.device, dtype),
            use_cache=True,
        )
    finally:
        handle.remove()

    h = captured["h"]
    n_q = model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", model.config.hidden_size // n_q)
    q = attn.q_proj(h).view(1, n, n_q, hd).transpose(1, 2)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    k = cache.layers[layer_idx].keys  # (1, Hkv, N, D) post-RoPE
    group = n_q // k.shape[1]
    kk = k.repeat_interleave(group, dim=1).float()

    # Sample rows; RoPE on q is omitted (q here is pre-RoPE), so treat this as indicative.
    rows = torch.linspace(n // 8, n - 2, 64).long()
    logits = torch.einsum("bhqd,bhkd->bhqk", q[:, :, rows].float(), kk) * hd**-0.5
    top = logits.argmax(-1)  # (1, H, R)
    target = (rows + 1).to(top.device).view(1, 1, -1)
    return float((top == target).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--tasks", default="qa_1,vt,niah_single_2")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/label_leak.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(args.device).eval()
    mid = model.config.num_hidden_layers // 2

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    rows = []
    for task in [t.strip() for t in args.tasks.split(",")]:
        r = df[df["task"] == task].iloc[0]
        ids = tok(r["context"], return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] >= args.n:
            rows.append((task, ids[:, : args.n].to(args.device)))
    print(f"{len(rows)} documents at N={args.n}", flush=True)

    results = []
    for task, ids in rows:
        entry = {"task": task}
        for mode in MODES:
            entry[mode] = replay_loss(model, ids, mode, args.n)
            print(f"  {task:<16s} {mode:<13s} loss={entry[mode]:.4f}", flush=True)
        entry["argmax_on_target"] = argmax_on_target(model, ids, args.n, mid)
        entry["leak_share"] = (entry["rect_noself"] - entry["rect"]) / max(entry["rect_noself"], 1e-9)
        print(
            f"  {task:<16s} leak_share={entry['leak_share']:.3f} "
            f"argmax_on_target={entry['argmax_on_target']:.3f}",
            flush=True,
        )
        results.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n": args.n, "rows": results}, indent=1))

    hdr = f"{'task':<16s}" + "".join(f"{m:>14s}" for m in MODES) + f"{'leak':>8s}{'argmax@tgt':>12s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for e in results:
        print(
            f"{e['task']:<16s}"
            + "".join(f"{e[m]:>14.4f}" for m in MODES)
            + f"{e['leak_share']:>8.3f}{e['argmax_on_target']:>12.3f}"
        )
    print(
        f"\nmean leak_share = {np.mean([e['leak_share'] for e in results]):.3f}"
        "   (fraction of the replay loss removed by making the target key visible)"
    )
    print(
        f"mean argmax_on_target = {np.mean([e['argmax_on_target'] for e in results]):.3f}"
        "   (rows whose top key IS the target -- induction fingerprint)"
    )


if __name__ == "__main__":
    main()
