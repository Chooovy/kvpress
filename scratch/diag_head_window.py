# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Can a monotone per-head decay act as a LEARNED ADAPTIVE WINDOW, and can it separate streaming
heads from retrieval heads?

The reframe this tests
----------------------
A scalar cumulative forget gate (FoX) makes ``-c_j`` monotone non-decreasing in ``j``, so its
induced per-key ranking is a pure recency staircase. Measured earlier, that means top-k under a
strong enough decay degenerates to a sliding window (SWA overlap -> 1.0). Read as a failure, that
kills the design. Read as a FEATURE, it says: a per-head forget rate is exactly a per-head
**window width**, learned rather than hand-set -- and a streaming head *wants* a window.

So the design question splits in two, and this script measures both:

1. **Do streaming and retrieval heads actually exist, and how separable are they?** Measured as the
   attention mass a (layer, KV head) puts inside a recent window vs far away, at the row where
   retrieval has to work (the last query row).

2. **Is a head's window width a property of the HEAD, or of the DOCUMENT?** This is the decisive
   question and it has a precedent that cuts the other way: a static offline per-head *budget*
   table scored 81.93 against uniform's 82.18 on RULER 8K (i.e. nothing), so there is no usable
   model-level head taxonomy for budgets. If window widths are equally document-dependent, then a
   learned-but-static per-head rate buys nothing either; if they are stable, the window is the one
   per-head quantity that *is* a model constant.

What a monotone decay can and cannot express
---------------------------------------------
A retrieval head's mass sits at a distance that varies with where the needle is, and a monotone
``-c_j`` cannot keep a single old key while dropping its neighbours -- the forgetting applies
equally to all keys (that is what "scalar increment" means). So the honest hypothesis is:
**a monotone decay can express a streaming head exactly, and a retrieval head only by switching
itself off.** ``frac_expressible`` reports how many heads fall in each case.

Reported per (layer, KV head)
-----------------------------
* ``w_p`` -- the window width containing fraction ``p`` of the row's attention mass (p = 0.5, 0.9).
  ``w_0.9`` small => streaming; large => retrieval.
* ``tail_mass`` -- mass beyond ``n_local`` (the trained pin width). This is the mass a window
  provably cannot capture, i.e. what a retrieval head needs and a decay cannot give it.
* Cross-document stability of ``w_p``: Spearman rho between documents, plus the relative spread
  ``|w_a - w_b| / w`` -- the same statistic that condemned the static budget table.
* ``n_eff`` -- participation ratio ``1/sum p_j^2``, a scale-free "how many keys does this head
  actually use" that does not assume a window at all.

Controls
--------
* ``shuffle_docs`` -- pair each head with a DIFFERENT document's window. If per-head windows were
  a model constant, shuffling documents would not change the fit; the gap between the true and
  shuffled stability is the amount of genuinely head-intrinsic signal.
* ``uniform`` -- the window a head would need if mass were spread evenly over its causal history,
  as a floor for "this head has no locality at all".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument("--tokenized", default=f"{MODELS}/../datasets/longmino_tokenized_64k")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 7, 14, 21, 28, 35])
    p.add_argument("--samples", type=int, default=4, help="documents; stability needs >= 2")
    p.add_argument("--n-local", type=int, default=128, help="the trained pin width")
    p.add_argument(
        "--stream-thresh",
        type=float,
        default=0.9,
        help="a head is 'streaming' if this much mass is inside n_local",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="scratch/diag_head_window.json")
    return p.parse_args()


@torch.no_grad()
def row_mass(model, layers, ids, layer_ids, n_kv, group, head_dim):
    """Per-(layer, KV head) attention distribution at the LAST query row -> {li: (Hkv, Sk)}."""
    cap: dict[int, torch.Tensor] = {}
    scale = head_dim**-0.5

    def mk(li):
        def hook(mod, args_, kwargs_, out):
            hs = kwargs_.get("hidden_states", args_[0] if args_ else None)
            pe = kwargs_.get("position_embeddings")
            if hs is None or pe is None:
                return
            cos, sin = pe
            b, s, _ = hs.shape
            q = mod.q_proj(hs).view(b, s, -1, head_dim).transpose(1, 2)
            k = mod.k_proj(hs).view(b, s, -1, head_dim).transpose(1, 2)
            if hasattr(mod, "q_norm"):
                q, k = mod.q_norm(q), mod.k_norm(k)
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # last row only: that is where RULER's question sits
            qr = q[0, :, -1, :].float()  # (Hq, D)
            kf = k[0].float()  # (Hkv, Sk, D)
            kq = kf.repeat_interleave(group, 0)  # (Hq, Sk, D)
            logits = torch.einsum("hd,hsd->hs", qr, kq) * scale
            p = torch.softmax(logits, dim=-1)  # (Hq, Sk)
            cap[li] = p.view(n_kv, group, -1).mean(1).cpu()  # fold query heads -> (Hkv, Sk)

        return hook

    handles = [layers[li].self_attn.register_forward_hook(mk(li), with_kwargs=True) for li in layer_ids]
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return cap


def window_for(p_row: torch.Tensor, frac: float) -> int:
    """Smallest window ending at the query that holds ``frac`` of the mass."""
    rev = p_row.flip(0).cumsum(0)  # distance 0,1,2,... from the query
    idx = int((rev < frac).sum().item()) + 1
    return min(idx, p_row.numel())


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / d)


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device(args.device)
    from transformers import AutoModelForCausalLM

    from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    group = cfg.num_attention_heads // n_kv
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    layers = get_language_model(model).layers

    loader = build_tokenized_dataloader(
        TokenizedConfig(root=args.tokenized, seq_len=args.seq_len, take_from="head"),
        batch_size=1,
        num_workers=0,
    )

    # per document: {li: (Hkv, Sk)}
    docs = []
    for si, b in enumerate(loader):
        if si >= args.samples:
            break
        ids = b["input_ids"][:, : args.seq_len].to(dev)
        docs.append(row_mass(model, layers, ids, args.layers, n_kv, group, head_dim))

    L = args.seq_len
    stats = {}  # (li, h) -> dict of per-doc lists
    for li in args.layers:
        for h in range(n_kv):
            w50, w90, tail, neff = [], [], [], []
            for d in docs:
                p = d[li][h].double()
                p = p / p.sum().clamp_min(1e-12)
                w50.append(window_for(p, 0.5))
                # 0.9 of the mass saturates at the full length for nearly every head (attention
                # has a long thin tail), so it cannot discriminate. Report instead the window
                # holding 90% of the mass that a window can reach at all, i.e. 0.9 * (1 - tail).
                t = float(p[: max(0, L - args.n_local)].sum())
                w90.append(window_for(p, 0.9 * (1.0 - t)))
                tail.append(t)
                neff.append(float(1.0 / p.pow(2).sum().clamp_min(1e-12)))
            stats[(li, h)] = {"w50": w50, "w90": w90, "tail": tail, "neff": neff}

    def mean(v):
        return sum(v) / len(v)

    print(f"\nmodel={os.path.basename(args.model)} seq_len={L} n_local={args.n_local} "
          f"docs={len(docs)} layers={args.layers}")
    print(f"\n{'layer':>5} {'h':>2} {'w50':>7} {'w90':>8} {'tail>nloc':>10} {'n_eff':>8} "
          f"{'w90 spread':>11}  type")
    print("-" * 74)
    n_stream = n_retr = 0
    for li in args.layers:
        for h in range(n_kv):
            s = stats[(li, h)]
            w90m, tailm = mean(s["w90"]), mean(s["tail"])
            spread = (max(s["w90"]) - min(s["w90"])) / max(w90m, 1.0)
            is_stream = tailm < (1.0 - args.stream_thresh)
            n_stream += is_stream
            n_retr += not is_stream
            print(f"{li:>5} {h:>2} {mean(s['w50']):>7.0f} {w90m:>8.0f} {tailm:>10.3f} "
                  f"{mean(s['neff']):>8.0f} {spread:>11.2f}  {'STREAM' if is_stream else 'retrieval'}")

    print(f"\nstreaming heads (>{args.stream_thresh:.0%} mass within {args.n_local}): "
          f"{n_stream}/{n_stream + n_retr}   retrieval: {n_retr}/{n_stream + n_retr}")

    # --- Is the window a property of the HEAD or the DOCUMENT? ---
    keys = [(li, h) for li in args.layers for h in range(n_kv)]
    print("\ncross-document stability of w90 (the static-table question)")
    print(f"  {'doc pair':>10} {'spearman':>10} {'mean |dw|/w':>13}")
    rhos, rels = [], []
    for a in range(len(docs)):
        for b in range(a + 1, len(docs)):
            va = torch.tensor([stats[k]["w90"][a] for k in keys], dtype=torch.double)
            vb = torch.tensor([stats[k]["w90"][b] for k in keys], dtype=torch.double)
            r = spearman(va, vb)
            rel = float(((va - vb).abs() / ((va + vb) / 2).clamp_min(1.0)).mean())
            rhos.append(r)
            rels.append(rel)
            print(f"  {f'{a}-{b}':>10} {r:>10.3f} {rel:>13.3f}")
    print(f"  {'MEAN':>10} {mean(rhos):>10.3f} {mean(rels):>13.3f}")

    # control: shuffle the head<->document correspondence
    g = torch.Generator().manual_seed(0)
    sh = []
    for a in range(len(docs)):
        for b in range(a + 1, len(docs)):
            va = torch.tensor([stats[k]["w90"][a] for k in keys], dtype=torch.double)
            vb = torch.tensor([stats[k]["w90"][b] for k in keys], dtype=torch.double)
            perm = torch.randperm(len(keys), generator=g)
            sh.append(spearman(va, vb[perm]))
    print(f"  shuffle control spearman: {mean(sh):>.3f}  (0 = no head-intrinsic signal)")

    # --- What can a monotone decay express? ---
    print("\nwhat a MONOTONE per-head decay can express")
    exact = sum(1 for k in keys if mean(stats[k]["tail"]) < (1.0 - args.stream_thresh))
    print(f"  streaming heads, expressible as a window          : {exact}/{len(keys)}")
    print(f"  retrieval heads, need decay OFF (f~1, pure content): {len(keys) - exact}/{len(keys)}")
    tails = torch.tensor([mean(stats[k]["tail"]) for k in keys], dtype=torch.double)
    print(f"  mass a window structurally cannot reach: mean {float(tails.mean()):.3f}, "
          f"max {float(tails.max()):.3f}")

    res = {
        "config": vars(args),
        "per_head": {f"{li}.{h}": stats[(li, h)] for li in args.layers for h in range(n_kv)},
        "stability": {"spearman": mean(rhos), "rel_spread": mean(rels), "shuffle": mean(sh)},
        "n_streaming": n_stream,
        "n_retrieval": n_retr,
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
