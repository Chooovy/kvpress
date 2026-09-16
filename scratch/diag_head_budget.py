# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Do KV heads want DIFFERENT numbers of keys? The measurement that decides whether a non-uniform
per-head budget has any headroom at all.

Why this question is not "just try AdaKV"
-----------------------------------------
The gate this arm trains is ``score - lse + log B`` on history and ``0`` on pinned keys
(``gate_pin.gate_from_score`` + ``gated_attention._gate_lse``). Adding a constant ``c`` to a whole
(layer, head)'s score vector leaves ``lse`` shifted by the same ``c``, so ``score - lse`` is
unchanged and the pinned term never saw the score -- **the forward is bitwise invariant to a
per-(layer, head) additive constant.** Eval's top-k runs per (batch, kv_head, row), so its ranking
is invariant too. Nothing in training or inference ever pins that constant down.

That is a gauge freedom, and it is exactly the quantity a cross-head pool compares. AdaKV's
``scores.reshape(bsz, -1)`` + one ``topk`` ranks keys from different heads against each other, so
its allocation is a function of an unidentifiable parameter. ``facility_location_press.py:217``
warns about the same thing in this repo's own words. (``w_out`` even has ``bias=False``, so the
model has no parameter that could express the constant.)

**Retained softmax mass is gauge-invariant**, which is what makes it the honest currency:
``rho_h = sum_{j kept} p_j`` where ``p`` is the true attention softmax. It is also the currency
training already speaks -- ``gate_budget`` constrains ``sum exp(gate) = B``, a mass, while eval
constrains a count. So mass-matched allocation closes the train/eval mismatch that
``hard_evict.py:6-14`` measured, rather than adding a second uncalibrated comparison.

What this reports, per layer and per KV head
--------------------------------------------
1. ``rho_uniform`` -- retained attention mass at today's uniform ``take = topk - sink - local``.
   **If this is flat across heads, uniform allocation is already mass-matched and the whole line
   is dead.** That is the kill-switch this script exists to throw.
2. ``k_at_rho*`` -- keys each head needs to reach a common target ``rho*``, with ``rho*`` chosen by
   bisection so that ``sum_h k_h == H * take``. The budget is conserved exactly; only its split
   moves. The spread of ``k_h`` is the size of the prize.
3. ``rho_matched`` -- the mass each head ends at under that allocation, and the worst head's gain
   against ``rho_uniform``. Redistribution helps by lifting the floor, so ``min_h rho_h`` is the
   number to watch, not the mean.

Ordering is by the **router's own score** (restricted to the evictable pool), so the cumulative
curve follows the keys the press would actually keep -- this measures the deployed policy, not an
oracle. The reference row is the last query row: that is where RULER's question sits and where
retrieval has to work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument(
        "--ckpt",
        default=f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/fwkl_ce01_8k_local128_b256_decay/final.pt",
        help="the best arm (RULER 8K 82.18); the allocation question is about THIS router",
    )
    p.add_argument("--tokenized", default=f"{MODELS}/../datasets/longmino_tokenized_64k")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-local", type=int, default=128)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="scratch/diag_head_budget.json")
    return p.parse_args()


@torch.no_grad()
def head_demand(q, k, score, *, ref_row, scale, group, n_sink, n_local, take):
    """
    Per-KV-head cumulative retained mass along the router's ranking, at one query row.

    Returns ``(cum, pinned_mass, n_pool)``: ``cum[h, r]`` is the attention mass captured by the
    top-``r+1`` evictable keys of head ``h`` **plus** its pinned mass, so ``cum[h, take-1]`` is
    exactly what today's uniform budget retains and ``cum[h, r]`` is what a budget of ``r+1``
    would retain. Averaged over the query heads sharing each KV head, since the budget is per KV
    head but the attention that has to survive is per query head.
    """
    n_q_heads, k_len = q.shape[1], k.shape[2]
    n_kv = k.shape[1]
    dev = q.device

    # (Hq, Sk) logits for the reference row against every causal key.
    qr = q[0, :, ref_row, :].float()                       # (Hq, D)
    kf = k[0].float().repeat_interleave(group, 0)          # (Hq, Sk, D)
    logits = torch.einsum("hd,hsd->hs", qr, kf) * scale
    key_idx = torch.arange(k_len, device=dev)
    causal = key_idx <= ref_row
    logits = logits.masked_fill(~causal, torch.finfo(torch.float32).min)
    p = torch.softmax(logits, dim=-1)                      # (Hq, Sk), sums to 1

    # Fold query heads into their KV head: the budget is per KV head, so its demand is the
    # average demand of the group it serves.
    p_kv = p.view(n_kv, group, k_len).mean(1)              # (Hkv, Sk)

    # Pin geometry, matching pinned_mask / the press's force_sink+force_local.
    sink = key_idx < n_sink
    local = (key_idx > ref_row - n_local) & (key_idx <= ref_row) & ~sink
    pinned = sink | local
    pool = causal & ~pinned                                # evictable at this row

    pinned_mass = p_kv[:, pinned].sum(-1)                  # (Hkv,)

    # Order the POOL by the router's score, descending. This is the policy's own ranking, so the
    # cumulative curve is what the press would actually retain -- not an oracle ordering.
    neg = torch.finfo(torch.float32).min
    pooled_score = torch.where(pool.unsqueeze(0), score.float(), torch.tensor(neg, device=dev))
    order = torch.argsort(pooled_score, dim=-1, descending=True, stable=True)  # (Hkv, Sk)
    n_pool = int(pool.sum())

    mass_ranked = p_kv.gather(-1, order)[:, :n_pool]       # (Hkv, n_pool)
    cum = mass_ranked.cumsum(-1) + pinned_mass.unsqueeze(-1)
    return cum, pinned_mass, n_pool


def allocate_matched(cum, total_budget, *, lo=0.0, hi=1.0, iters=60):
    """
    Per-head budgets that all reach a common mass target, with the TOTAL conserved exactly.

    Bisects ``rho*`` on the monotone map ``rho* -> sum_h k_h(rho*)``, then fixes the residual by
    handing the leftover slots to the heads whose next key is worth most (and reclaiming from the
    heads whose last key is worth least). Conservation is exact by construction, so the arm is a
    pure reallocation of today's budget rather than a quiet budget increase -- the failure mode
    that would make any A/B meaningless.
    """
    n_heads, n_pool = cum.shape

    def k_for(target):
        # first index reaching the target; n_pool when the head can never reach it
        reached = cum >= target
        any_r = reached.any(-1)
        first = reached.to(torch.uint8).argmax(-1) + 1
        return torch.where(any_r, first, torch.full_like(first, n_pool))

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if int(k_for(mid).sum()) > total_budget:
            hi = mid
        else:
            lo = mid
    k = k_for(lo).clamp(min=1, max=n_pool)

    # Exact conservation: spend or reclaim the residual by marginal value of the next/last key.
    residual = total_budget - int(k.sum())
    while residual != 0:
        if residual > 0:
            gain = torch.where(
                k < n_pool,
                cum.gather(-1, k.clamp(max=n_pool - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), -1.0, device=cum.device),
            )
            if float(gain.max()) < 0:
                break
            k[int(gain.argmax())] += 1
            residual -= 1
        else:
            loss = torch.where(
                k > 1,
                cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 2).clamp(min=0).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), float("inf"), device=cum.device),
            )
            k[int(loss.argmin())] -= 1
            residual += 1
    return k, lo


def main():
    args = parse_args()
    dev = args.device
    torch.set_grad_enabled(False)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("indexer", ckpt)
    config = ckpt.get("config") or {}
    scorer, kwargs = press_kwargs_from_checkpoint(state, config)
    press = GQAIndexerPress(
        compression_ratio=0.0,
        gate_scale=any(str(x).endswith("gate_scale") for x in state),
        scorer_attr="indexer",
        scorer=scorer,
        **kwargs,
    )
    press.post_init_from_model(model, force_reinit=True)
    load_indexer_state_dict(model, state, "indexer")
    print(f"loaded {args.ckpt}\n  scorer={scorer} {kwargs}", flush=True)

    layers = get_language_model(model).layers
    n_layers = len(layers)
    cfg = model.config
    n_q_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    group = n_q_heads // n_kv
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q_heads)
    scale = head_dim ** -0.5

    # One forward, every layer captured: 36 x (q,k,v) at 8K bf16 is ~3.6 GiB, which fits and
    # saves 36 forwards.
    grabbed: dict[int, tuple] = {}
    hidden: dict[int, torch.Tensor] = {}

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import torch.nn.functional as F

    def impl(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        grabbed[int(module.layer_idx)] = (query.detach(), key.detach(), value.detach())
        out = F.scaled_dot_product_attention(
            query, key.repeat_interleave(group, 1), value.repeat_interleave(group, 1),
            is_causal=True, scale=scaling,
        )
        return out.transpose(1, 2).contiguous(), None

    name = "diag_head_budget_capture"
    gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    ALL_ATTENTION_FUNCTIONS.register(name, impl)

    def pre_hook(module, a, kwargs_):
        hs = kwargs_.get("hidden_states")
        if hs is None and a:
            hs = a[0]
        hidden[int(module.layer_idx)] = hs.detach()
        return None

    handles = [
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True) for layer in layers
    ]
    configs = [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else [])
    prev = [c._attn_implementation for c in configs]
    for c in configs:
        c._attn_implementation = name

    from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader

    loader = build_tokenized_dataloader(
        TokenizedConfig(root=args.tokenized, seq_len=args.seq_len, take_from="head"),
        batch_size=1, num_workers=0,
    )
    batches = []
    for b in loader:
        batches.append(b["input_ids"][:, : args.seq_len].to(dev))
        if len(batches) >= args.samples:
            break

    take = args.topk - args.n_sink - args.n_local
    print(
        f"L={args.seq_len} topk={args.topk} sink={args.n_sink} local={args.n_local} "
        f"-> take={take}/head, {n_kv} kv heads, {len(batches)} sample(s)\n",
        flush=True,
    )

    rows = []
    for sample_idx, input_ids in enumerate(batches):
        grabbed.clear()
        hidden.clear()
        model(input_ids=input_ids, use_cache=False)
        ref_row = args.seq_len - 1

        for layer_idx in range(n_layers):
            q, k, _v = grabbed[layer_idx]
            h = hidden[layer_idx]
            indexer = press.get_indexer(layers[layer_idx].self_attn)
            if getattr(indexer, "decay", False):
                score = indexer.score_at(h, float(ref_row))[0]
            else:
                score = indexer.score_keys(h)[0]

            cum, pinned_mass, n_pool = head_demand(
                q, k, score, ref_row=ref_row, scale=scale, group=group,
                n_sink=args.n_sink, n_local=args.n_local, take=take,
            )
            rho_uniform = cum[:, min(take, n_pool) - 1]
            k_matched, rho_star = allocate_matched(cum, take * n_kv)
            rho_matched = cum.gather(-1, (k_matched - 1).unsqueeze(-1)).squeeze(-1)

            rows.append({
                "sample": sample_idx,
                "layer": layer_idx,
                "n_pool": n_pool,
                "pinned_mass": [round(float(x), 5) for x in pinned_mass],
                "rho_uniform": [round(float(x), 5) for x in rho_uniform],
                "k_matched": [int(x) for x in k_matched],
                "rho_matched": [round(float(x), 5) for x in rho_matched],
                "rho_star": round(float(rho_star), 5),
            })
            del q, k, _v, cum
        torch.cuda.empty_cache()
        print(f"sample {sample_idx} done", flush=True)

    for handle in handles:
        handle.remove()
    for c, p in zip(configs, prev):
        c._attn_implementation = p
    gm.pop(name, None)

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)

    # ---- report -------------------------------------------------------------------------
    print("\n==== per-layer, averaged over samples ====")
    print(f"{'layer':>5} {'rho_unif mean':>13} {'rho_unif min':>12} {'rho_unif spread':>15} "
          f"{'k range':>15} {'k max/min':>10} {'rho_match min':>13} {'floor gain':>11}")
    agg = {}
    for r in rows:
        agg.setdefault(r["layer"], []).append(r)

    all_ratio, all_gain, all_spread = [], [], []
    for layer_idx in sorted(agg):
        rs = agg[layer_idx]
        ru = torch.tensor([r["rho_uniform"] for r in rs]).mean(0)
        km = torch.tensor([r["k_matched"] for r in rs], dtype=torch.float).mean(0)
        rm = torch.tensor([r["rho_matched"] for r in rs]).mean(0)
        ratio = float(km.max() / km.min().clamp(min=1))
        gain = float(rm.min() - ru.min())
        spread = float(ru.max() - ru.min())
        all_ratio.append(ratio)
        all_gain.append(gain)
        all_spread.append(spread)
        print(f"{layer_idx:>5} {float(ru.mean()):>13.4f} {float(ru.min()):>12.4f} "
              f"{spread:>15.4f} {int(km.min()):>6}-{int(km.max()):<8} {ratio:>10.2f} "
              f"{float(rm.min()):>13.4f} {gain:>+11.4f}")

    print("\n==== VERDICT ====")
    print(f"rho_uniform spread within a layer : mean {sum(all_spread)/len(all_spread):.4f}, "
          f"max {max(all_spread):.4f}")
    print(f"k_matched max/min ratio           : mean {sum(all_ratio)/len(all_ratio):.2f}, "
          f"max {max(all_ratio):.2f}")
    print(f"worst-head mass gain              : mean {sum(all_gain)/len(all_gain):+.4f}, "
          f"max {max(all_gain):+.4f}")
    print(
        "\nA ratio near 1.0 and a gain near 0 mean uniform IS already mass-matched -- "
        "non-uniform allocation has nothing to win and the line should be dropped."
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
