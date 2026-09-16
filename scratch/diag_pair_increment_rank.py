# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Is the pair-indexed forget increment LOW RANK on real text? The measurement that decides whether
the "later tokens decide what to forget" family can be made O(1).

The question
------------
Selective Attention (Leviathan et al. 2024) subtracts ``F_ij = sum_{j<k<i} ReLU(q_k . k_j / sqrt d)``
from the logits; DCP (Anagnostidis et al. 2023) multiplies ``prod_{j<n<=i} sigma_alpha(...)``. Both
express "a later token k decides how fast an earlier key j is forgotten", which is strictly more
expressive than FoX's scalar increment ``log f_l`` (that one forgets every key at the same rate, so
no key can ever be immune -- verified: it degenerates to SWA once the accumulated forgetting exceeds
range(s)).

Both pay for it: the increment matrix ``S_{k,j}`` is elementwise-nonlinear in a ``(q_k, k_j)`` dot
product, so it does not factorize and they materialize ``O(n^2)``. But *if* ``S`` is effectively
rank ``R`` with ``R`` small, then ``S_{k,j} ~ u_k . r_j`` and the interval sum telescopes::

    F_ij = sum_{j<k<i} u_k . r_j = (C_i - C_j) . r_j        C_i = sum_{k<i} u_k

which folds into an additive bilinear gate at width ``R+1`` (verified exact to 5.7e-14 on synthetic
data) with ``ki(j) = [r_j, -C_j . r_j]`` cacheable per key. Frozen at prefill it is ``O(1)`` per
decode step -- the only variant that is.

So the entire family's viability reduces to one number: **the effective rank of S on real text.**
If it is ~16, the fold is nearly free. If it is ~200, the width kills it and this line is dead.
That is what this script measures, and it is deliberately a 20-minute measurement that bounds the
whole family before any training is attempted -- the same role ``probe_meanpool_kvzip.py``'s winner
concentration played for the proxy-query family.

What is reported
----------------
Per (layer, KV head), the rank needed to capture 90/99% of the squared-singular-value energy of
``S``, for four increment forms:

* ``linear``   -- ``q_k . k_j / sqrt d``, the trivially rank-``head_dim`` control (upper bound
  ``head_dim``; anything above that is a bug).
* ``relu``     -- Selective Attention's actual form. **This is the number that decides the question.**
* ``logsigmoid`` -- DCP's log-domain factor, ``log sigma(.)``.
* ``softplus`` -- a monotone smooth surrogate that keeps ``S >= 0`` (hence keeps the monotonicity
  that makes eviction legal) without ReLU's hard zero.

Energy fraction, not a hard threshold, because the singular spectrum of a rectified bilinear form
decays smoothly; the 99% column is the one to read for "can I replace this by a rank-R factor".

Two controls that make the number trustworthy
---------------------------------------------
* ``linear`` must come out at exactly ``min(head_dim, block)`` -- it is a dot product of
  ``head_dim``-dim vectors, so a larger value means the SVD or the causal masking is wrong.
* ``shuffle`` -- the same ReLU'd matrix with its rows independently permuted, which destroys any
  (k, j) coupling while preserving the marginal value distribution. If ``relu``'s rank is not well
  below ``shuffle``'s, the structure being claimed is not there.

Causality: ``S_{k,j}`` is only ever consumed for ``j < k``, so the strictly-lower-triangular block
is the object whose rank matters. The upper triangle is zeroed before the SVD rather than left as
raw logits, which would inflate the rank with values the mechanism never reads.
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
    p.add_argument(
        "--block",
        type=int,
        default=1024,
        help="SVD is on a (block, block) causal window of S. 1024 keeps the SVD cheap while "
        "being 8x wider than any plausible R, so a low-rank verdict is not a small-matrix artifact.",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=4096,
        help="where the block starts; mid-sequence so the window has real long-range structure "
        "rather than the degenerate first-tokens regime",
    )
    p.add_argument("--layers", type=int, nargs="+", default=[0, 7, 14, 21, 28, 35])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="scratch/diag_pair_increment_rank.json")
    return p.parse_args()


def eff_rank(M: torch.Tensor, fracs=(0.9, 0.99)) -> dict:
    """Rank needed to capture each energy fraction of ``M``, via squared singular values."""
    sv = torch.linalg.svdvals(M.double())
    en = sv.pow(2)
    tot = en.sum()
    if tot <= 0:
        return {f"r{int(f * 100)}": 0 for f in fracs} | {"n": int(min(M.shape))}
    c = en.cumsum(0) / tot
    out = {}
    for f in fracs:
        out[f"r{int(f * 100)}"] = int((c < f).sum().item()) + 1
    out["n"] = int(min(M.shape))
    return out


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device(args.device)

    # Reuse the repo's loader so RoPE/dtype/attn-impl match every other diagnostic.
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
    scale = head_dim**-0.5

    loader = build_tokenized_dataloader(
        TokenizedConfig(root=args.tokenized, seq_len=args.seq_len, take_from="head"),
        batch_size=1,
        num_workers=0,
    )

    # Capture post-RoPE q/k per layer: the increment is a q.k interaction, so it must be measured
    # in the space the model actually computes it in.
    cap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    layers = get_language_model(model).layers

    def mk_hook(li):
        def hook(mod, args_, kwargs_, out):
            # Qwen3 attention returns (attn_out, attn_weights); q/k are not exposed, so recompute
            # from the captured hidden states the same way the layer does.
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
            cap[li] = (q.float().cpu(), k.float().cpu())

        return hook

    handles = [
        layers[li].self_attn.register_forward_hook(mk_hook(li), with_kwargs=True)
        for li in args.layers
    ]

    forms = ("linear", "relu", "logsigmoid", "softplus", "shuffle")
    acc: dict[str, dict[str, list]] = {f: {"r90": [], "r99": []} for f in forms}
    per_layer: dict[int, dict[str, dict[str, list]]] = {
        li: {f: {"r90": [], "r99": []} for f in forms} for li in args.layers
    }

    o, blk = args.offset, args.block
    tri = None
    for si, batch in enumerate(loader):
        if si >= args.samples:
            break
        ids = batch["input_ids"][:, : args.seq_len].to(dev)
        cap.clear()
        model(input_ids=ids, use_cache=False)

        for li in args.layers:
            q, k = cap[li]
            # Fold query heads into their KV head: the forget decision is per KV head (that is the
            # granularity a cache can evict at), so the emitter side is the group mean.
            qg = q[0, :, o : o + blk, :].view(n_kv, group, blk, head_dim).mean(1)  # (Hkv, blk, D)
            kg = k[0, :, o : o + blk, :]  # (Hkv, blk, D)
            S = torch.einsum("hkd,hjd->hkj", qg, kg) * scale  # (Hkv, blk, blk)

            if tri is None:
                idx = torch.arange(blk)
                tri = (idx.view(-1, 1) > idx.view(1, -1)).float()  # strictly lower: j < k

            for h in range(n_kv):
                base = S[h]
                variants = {
                    "linear": base,
                    "relu": torch.relu(base),
                    "logsigmoid": torch.nn.functional.logsigmoid(base),
                    "softplus": torch.nn.functional.softplus(base),
                }
                g = torch.Generator().manual_seed(1234 + h)
                perm = torch.stack([torch.randperm(blk, generator=g) for _ in range(blk)])
                variants["shuffle"] = torch.relu(base).gather(1, perm)

                for name, M in variants.items():
                    r = eff_rank(M * tri)
                    acc[name]["r90"].append(r["r90"])
                    acc[name]["r99"].append(r["r99"])
                    per_layer[li][name]["r90"].append(r["r90"])
                    per_layer[li][name]["r99"].append(r["r99"])

    for hh in handles:
        hh.remove()

    def mean(v):
        return sum(v) / len(v) if v else 0.0

    print(f"\nmodel={os.path.basename(args.model)}  block={blk}  offset={o}  "
          f"samples={args.samples}  head_dim={head_dim}  n_kv={n_kv}")
    print(f"cells = {args.samples} samples x {len(args.layers)} layers x {n_kv} kv heads "
          f"= {args.samples * len(args.layers) * n_kv}\n")
    print(f"{'form':<12} {'rank@90%':>10} {'rank@99%':>10}   (of {blk})")
    print("-" * 48)
    for f in forms:
        print(f"{f:<12} {mean(acc[f]['r90']):>10.1f} {mean(acc[f]['r99']):>10.1f}")

    print(f"\nper-layer rank@99%")
    print(f"{'layer':<8} " + " ".join(f"{f:>11}" for f in forms))
    for li in args.layers:
        print(f"{li:<8} " + " ".join(f"{mean(per_layer[li][f]['r99']):>11.1f}" for f in forms))

    print("\ninterpretation:")
    print(f"  linear should be ~{min(head_dim, blk)} (= head_dim); a larger value means a bug")
    print("  relu is Selective Attention's real form -- if rank@99% << 100 the O(1) fold is viable")
    print("  relu must be well below shuffle, else there is no (k,j) structure to exploit")

    res = {
        "config": vars(args),
        "head_dim": head_dim,
        "n_kv_heads": n_kv,
        "overall": {f: {m: mean(v) for m, v in d.items()} for f, d in acc.items()},
        "per_layer": {
            str(li): {f: {m: mean(v) for m, v in d.items()} for f, d in per_layer[li].items()}
            for li in args.layers
        },
    }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
