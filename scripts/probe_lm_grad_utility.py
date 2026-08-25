# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Is the LM-gradient utility a *learnable* ranking target? Measured on the real model.

``differentiable_topk_for_sparse_attention.md`` §31 proposes training the indexer to rank keys by

    u_j = -dL/db_j = -alpha_j * <dL/do, v_j - o>

where ``b_j`` is an infinitesimal additive bias on key ``j``'s logit. The identity is exact
(verified to 1.4e-17 in fp64) and it is appealing because **one backward gives every key a utility**
-- no key has to be selected first, which is precisely the dead end the exact-K arm hit with its
candidate pool.

Before building a training arm on it, one question decides whether it can work at all, and it is
cheap to answer:

    the router scores keys with ``qi . ki``, a function of the QUERY and the KEY.
    ``u_j`` factors as ``alpha_j`` (reachable -- it is a function of q.k) times
    ``<dL/do, v_j - o>`` (a function of the VALUE and of the loss direction, neither of which a
    q.k scorer can see).

If ``u``'s ranking is dominated by the second factor, the target is **not representable** by this
router, and a ranking loss against it would ask the indexer to predict something it structurally
cannot. That would show up as a low correlation ceiling, not as a bug.

**On synthetic Gaussian q/k/v this looks fatal**: spearman(u, alpha) = -0.015, while
spearman(u, -value_term) = +0.912. But synthetic tensors have no q/k-to-v correlation by
construction, and that is exactly the kind of setup that has misled this investigation before (the
first oracle-recall measurement looked publishable until baselines showed random scoring 0.85). In a
real trained transformer ``v_j`` is a learned function of the same token that produced ``k_j``, so
the two factors need not be independent.

So this script measures, on the real model and real text:

1. ``spearman(u, alpha)`` -- how much of the target the router can even in principle reach;
2. ``spearman(u, qk)`` -- the same against the raw attention logit, which is what the indexer
   imitates;
3. the **selection** consequence: the LM loss after keeping top-k by ``u`` against top-k by
   ``alpha``, by ``qk``, and at random. A target that ranks better but is unreachable is worth
   knowing about; a target that is reachable *and* ranks better is worth training on.
4. ``spearman(u, chunk LSE)`` at chunk granularity, since that is the quantity the HSA arm's router
   provably wants to learn -- so this says whether the two proposals agree or compete.

Run it on one GPU; it needs no training and no checkpoint.

    python -m scripts.probe_lm_grad_utility --model /path/Qwen3-8B --layers 0,8,17,26,35
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank correlation along the last axis, averaged over leading axes."""
    rx = x.argsort(-1).argsort(-1).double()
    ry = y.argsort(-1).argsort(-1).double()
    rx = rx - rx.mean(-1, keepdim=True)
    ry = ry - ry.mean(-1, keepdim=True)
    denom = (rx.norm(dim=-1) * ry.norm(dim=-1)).clamp_min(1e-12)
    return float(((rx * ry).sum(-1) / denom).mean())


def real_tokens(tokenizer, seq_len: int, device: str) -> torch.Tensor:
    """
    Real text, not random ids.

    Random ids put the loss at ``log(vocab)`` for any method, so ``dL/do`` points in an arbitrary
    direction and the value term -- the thing being measured -- becomes noise. The repo's own docs
    are the most convenient real text guaranteed present on any checkout.
    """
    docs = [
        REPO_ROOT / "kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md",
        REPO_ROOT / "differentiable_topk_for_sparse_attention.md",
        REPO_ROOT / "kvpress/presses/gqa_indexer/README.md",
        REPO_ROOT / "README.md",
    ]
    text = "\n\n".join(p.read_text() for p in docs if p.exists())
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :seq_len]
    if ids.shape[1] < seq_len:
        reps = -(-seq_len // ids.shape[1])
        ids = ids.repeat(1, reps)[:, :seq_len]
    return ids.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--layers", default="0,8,17,26,35")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=32, help="query rows sampled per layer")
    ap.add_argument("--topk", type=int, default=256, help="budget for the selection test")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument(
        "--truncate", type=int, default=0,
        help="keep only the first N layers, to fit beside another job. 0 keeps the whole model. "
        "CAVEAT WORTH TAKING SERIOUSLY: a truncated model's dL/do is not the trained model's, and "
        "in this very investigation a truncated-model recall probe INVERTED the ranking that the "
        "full-model eval produced (it said no router beat recency; the real eval said +49). So use "
        "this to check the machinery and to get a preliminary signal, and re-run at full depth "
        "before concluding anything.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.truncate:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[: args.truncate]))
        model.config.num_hidden_layers = args.truncate
        print(
            f"--truncate {args.truncate}: NOT the trained model's loss geometry. Preliminary only "
            f"-- see --truncate in --help for why this caveat is not boilerplate here."
        )
    model = model.to(device).eval()
    probe = [int(x) for x in args.layers.split(",") if int(x) < model.config.num_hidden_layers]
    if not probe:
        raise SystemExit(
            f"--layers {args.layers} has nothing below num_hidden_layers="
            f"{model.config.num_hidden_layers}"
        )
    ids = real_tokens(tokenizer, args.seq_len, device)

    # Capture q/k/v per probed layer, and dL/do at the attention OUTPUT (before o_proj) -- which is
    # what the identity is written against.
    captured: dict[int, dict] = {}
    handles = []

    def make_hook(li):
        def hook(module, args_in, kwargs_in):
            hs = kwargs_in.get("hidden_states")
            if hs is None and args_in:
                hs = args_in[0]
            captured[li] = {"hidden": hs}
            return None
        return hook

    layers = model.model.layers
    for li in probe:
        handles.append(layers[li].self_attn.register_forward_pre_hook(make_hook(li), with_kwargs=True))

    # One forward+backward to get dL/d(attn_out). Grab it by hooking o_proj's input.
    grads: dict[int, torch.Tensor] = {}
    o_inputs: dict[int, torch.Tensor] = {}

    def make_oproj_hook(li):
        def hook(module, inp, out):
            x = inp[0]
            x.retain_grad()
            o_inputs[li] = x
            return None
        return hook

    for li in probe:
        handles.append(layers[li].self_attn.o_proj.register_forward_hook(make_oproj_hook(li)))

    out = model(input_ids=ids, labels=ids, use_cache=False)
    print(f"LM loss {float(out.loss):.4f} on {ids.shape[1]} real tokens")
    out.loss.backward()
    for h in handles:
        h.remove()

    results = {}
    for li in probe:
        attn = layers[li].self_attn
        hs = captured[li]["hidden"]
        x = o_inputs[li]
        if x.grad is None:
            print(f"layer {li}: no gradient at o_proj input, skipping")
            continue

        b, q_len, _ = hs.shape
        n_heads = model.config.num_attention_heads
        n_kv = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // n_heads
        group = n_heads // n_kv

        with torch.no_grad():
            # Recompute q/k/v exactly as the layer does, including its norms and RoPE.
            q = attn.q_proj(hs).view(b, q_len, n_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hs).view(b, q_len, n_kv, head_dim).transpose(1, 2)
            v = attn.v_proj(hs).view(b, q_len, n_kv, head_dim).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            pos = torch.arange(q_len, device=device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(v, pos)
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # dL/d(attn_out), reshaped to per-head. o_proj's input is (B, Sq, Hq*Dv).
            g_out = x.grad.view(b, q_len, n_heads, head_dim).transpose(1, 2).float()

            # Sample query rows from the second half so a row has keys to rank.
            rows = torch.linspace(q_len // 2, q_len - 1, args.rows, device=device).long().unique()
            scale = head_dim**-0.5
            kk = k.repeat_interleave(group, dim=1).float()
            vv = v.repeat_interleave(group, dim=1).float()
            qq = q[:, :, rows].float()
            logits = torch.einsum("bhrd,bhsd->bhrs", qq, kk) * scale
            causal = torch.arange(q_len, device=device).view(1, q_len) <= rows.view(-1, 1)
            logits = logits.masked_fill(~causal.view(1, 1, len(rows), q_len), -float("inf"))
            alpha = torch.softmax(logits, -1)
            o = torch.einsum("bhrs,bhsd->bhrd", alpha, vv)
            g = g_out[:, :, rows]

            # u_j = -alpha_j * <dL/do, v_j - o>
            proj = torch.einsum("bhrd,bhsd->bhrs", g, vv) - (g * o).sum(-1, keepdim=True)
            u = -(alpha * proj)

            # Restrict every correlation to VISIBLE keys, per row, and require enough of them.
            per_row = {"u_alpha": [], "u_qk": [], "u_proj": []}
            for ri in range(len(rows)):
                m = causal[ri]
                n_vis = int(m.sum())
                if n_vis < 8 * args.topk:
                    continue
                sel = m.nonzero(as_tuple=True)[0]
                a_ = alpha[:, :, ri][..., sel]
                l_ = logits[:, :, ri][..., sel]
                u_ = u[:, :, ri][..., sel]
                p_ = proj[:, :, ri][..., sel]
                per_row["u_alpha"].append(spearman(u_, a_))
                per_row["u_qk"].append(spearman(u_, l_))
                per_row["u_proj"].append(spearman(u_, -p_))

            # The selection test: LM-relevant proxy = how much of the dense output is preserved.
            # Measured as ||o_selected - o_dense||, since the true LM loss would need a full replay.
            def keep(score):
                idx = score.topk(args.topk, dim=-1).indices
                lg = logits.gather(-1, idx)
                p = torch.softmax(lg, -1)
                vsel = vv.unsqueeze(2).expand(-1, -1, len(rows), -1, -1).gather(
                    3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
                )
                return float(
                    (torch.einsum("bhrk,bhrkd->bhrd", p, vsel) - o).norm(dim=-1).mean()
                )

            rand = torch.rand_like(alpha).masked_fill(
                ~causal.view(1, 1, len(rows), q_len), -1.0
            )
            err = {
                "by_u": keep(u.masked_fill(~causal.view(1, 1, len(rows), q_len), -float("inf"))),
                "by_alpha": keep(alpha),
                "by_qk": keep(logits),
                "random": keep(rand),
            }

        row = {
            k2: (statistics.mean(v2) if v2 else float("nan")) for k2, v2 in per_row.items()
        }
        row["n_rows"] = len(per_row["u_alpha"])
        row["output_error"] = err
        results[li] = row
        print(
            f"layer {li:2d} (n={row['n_rows']:2d} rows): "
            f"spearman(u,alpha) {row['u_alpha']:+.3f}  spearman(u,qk) {row['u_qk']:+.3f}  "
            f"spearman(u,-value_term) {row['u_proj']:+.3f}\n"
            f"           ||o_sel - o_dense|| at topk={args.topk}: "
            f"u {err['by_u']:.4f}  alpha {err['by_alpha']:.4f}  qk {err['by_qk']:.4f}  "
            f"random {err['random']:.4f}"
        )

    if results:
        print("\n=== summary over probed layers ===")
        for key, label in (
            ("u_alpha", "spearman(u, alpha)      [what a q.k router CAN reach]"),
            ("u_qk", "spearman(u, q.k logit)  [what the indexer imitates]"),
            ("u_proj", "spearman(u, -value term)[what it CANNOT reach]"),
        ):
            vals = [r[key] for r in results.values() if r[key] == r[key]]
            if vals:
                print(f"{label:52s} {statistics.mean(vals):+.3f}")
        print(
            "\nRead it this way: if spearman(u, alpha) is near 0 while spearman(u, -value term) is "
            "high, the utility is dominated by a factor the router cannot observe, and a ranking "
            "loss against u asks for something unrepresentable. If alpha-selection is already close "
            "to u-selection on output error, the extra target buys little regardless."
        )
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "layers": results}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
