#!/usr/bin/env python
"""
Test the proxy assumption behind a stateful per-key indexer score.

The state diagnostics (``diag_rnn_state_design.py``) establish that a recurrent state ``z``
can detect *redundancy* -- "have I seen something like this already?" -- far better than
chance. But redundancy is not what an indexer is for. A per-key score decides whether a key
is worth keeping, and the ground truth for that is **how much attention the key actually
receives from future queries**. Redundancy has only ever been a proxy for it.

The two can come apart, and there is a documented precedent for a plausible proxy being
systematically wrong: SeerAttention-R optimises attention-mass coverage, reaches 96.8% of it,
and still loses downstream to SAS at 79.5% (SAS paper, Figure 4). So the proxy is measured
rather than assumed:

  * redundancy is backward-looking  ("do I resemble the past?")
  * key utility is forward-looking  ("will the future ask for me?")

What is computed
----------------
For every key position t, from the frozen model's own attention:

  utility(t) = how much attention mass future queries put on key t

Several definitions are reported, because the choice is itself under test:

  ``sum``     total mass over all future queries. Favours keys many queries like a bit.
  ``max``     the single largest attention any future query gives it. Favours keys that
              matter enormously to one query -- the needle-in-a-haystack case, which
              ``sum`` washes out.
  ``topq``    mean over the queries that want it most (the top 1% of its column).
  ``late``    ``sum`` restricted to queries in the last quarter, i.e. utility to the
              queries that a compressed cache actually has to serve.

Then, against each candidate score (the state designs, plus controls):

  * Spearman correlation with each utility definition -- does the score rank keys the way
    real attention does?
  * Recall at a keep budget -- of the keys that truly carry the most mass, how many does
    a top-k on this score retain? This is the quantity that matters operationally, and a
    score can correlate decently while still missing the important tail.
  * An oracle row (utility itself) and a random row bracket every number.

How to read the outcome
-----------------------
Strong correlation and recall: the proxy holds, and the state-design conclusions transfer.
Weak: the state work is optimising the wrong target -- keep the architecture, retrain it
against utility directly, and do not trust redundancy as a training signal.
Also compare against ``no state`` and ``recency``: if a trivial score already predicts
utility this well, the state is not buying anything regardless of its redundancy skill.

Run:
    python scripts/diag_utility_proxy.py --layers 8,18,28 --seq-len 8192

Attention is quadratic, so it is never materialised whole: q/k are recomputed from the one
layer under analysis and the four reductions are accumulated over query tiles, keeping peak
cost at ``(H, tile, L)``. Asking the model for ``output_attentions=True`` instead would
retain every layer's ``(H, Sq, Sk)`` -- 144 GB at 8192 tokens on a 36-layer model, to serve
one layer. ``--fake`` exercises the harness with no model.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from diag_rnn_state_design import (  # reuse the verified state implementations
    load_text,
    novelty_ema,
    novelty_full_attention,
    novelty_matrix_states,
    resolve_device,
)


# ----------------------------------------------------------------------
# Utility targets, from real attention
# ----------------------------------------------------------------------
def key_utility(stats: dict, mode: str, *, group_reduce: str = "mean") -> torch.Tensor:
    """Reduce a per-head utility statistic ``(H, Sk)`` to one value per key ``(Sk,)``.

    Heads are combined the way the press does it (mean over the KV group by default), so a
    key's utility here is the same quantity the indexer is scored against downstream.
    ``amax`` answers "does *any* head want this key", which is the right question when heads
    specialise and a single head's need is enough to justify keeping the key.
    """
    a = stats[mode].float()
    if group_reduce == "mean":
        return a.mean(dim=0)
    if group_reduce == "amax":
        return a.amax(dim=0)
    raise ValueError(f"unknown group_reduce {group_reduce!r}")


def attention_probs_from_module(mod, hidden_states, position_embeddings, tile: int = 1024):
    """Per-key utility statistics for one attention module, without an ``(H, L, L)`` tensor.

    Recomputes q/k from the module's own projections (including the q/k norms and RoPE that
    Qwen3 applies), then walks query tiles and accumulates the four utility reductions on
    the fly. Only ``(H, tile, L)`` is live at once, so an 8192-token layer costs ~1 GB
    instead of 4, and the peak does not grow with the number of layers analysed.

    Returns a dict of ``(H, Sk)`` tensors -- the reductions, not the probabilities, since
    nothing downstream needs the full matrix.
    """
    import torch.nn.functional as F

    hs = hidden_states
    B, L, _ = hs.shape
    hd = mod.head_dim
    q = mod.q_proj(hs).view(B, L, -1, hd)
    k = mod.k_proj(hs).view(B, L, -1, hd)
    # Qwen3 / Llama-3 style per-head RMSNorm on q and k, when present.
    if getattr(mod, "q_norm", None) is not None:
        q = mod.q_norm(q)
    if getattr(mod, "k_norm", None) is not None:
        k = mod.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, L, hd)
    if position_embeddings is not None:
        cos, sin = position_embeddings

        def rope(x):
            c, s = cos.unsqueeze(1).to(x.dtype), sin.unsqueeze(1).to(x.dtype)
            half = x.shape[-1] // 2
            rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
            return x * c + rot * s

        q, k = rope(q), rope(k)
    # GQA: expand kv heads to query heads so every query head gets its own row.
    n_q, n_kv = q.shape[1], k.shape[1]
    if n_q != n_kv:
        k = k.repeat_interleave(n_q // n_kv, dim=1)
    q, k = q[0], k[0]  # (H, L, hd)
    scale = getattr(mod, "scaling", None) or hd**-0.5

    H = q.shape[0]
    topq_k = max(1, int(0.01 * L))
    # Each tile keeps its own top-k, then the top-k of those is taken. That equals a global
    # top-k only while every tile retains at least k candidates -- otherwise a tile holding
    # many of the true best is truncated and the estimate is silently low.
    tile = max(tile, topq_k)
    acc = {
        "sum": torch.zeros(H, L, dtype=torch.float32, device=q.device),
        "max": torch.zeros(H, L, dtype=torch.float32, device=q.device),
        "late": torch.zeros(H, L, dtype=torch.float32, device=q.device),
    }
    topq_buf = []
    late_start = 3 * L // 4
    for s in range(0, L, tile):
        e = min(s + tile, L)
        logits = (q[:, s:e] @ k.transpose(-1, -2)) * scale  # (H, tile, L)
        qpos = torch.arange(s, e, device=q.device).unsqueeze(-1)
        kpos = torch.arange(L, device=q.device).unsqueeze(0)
        logits = logits.float().masked_fill(kpos > qpos, -float("inf"))
        p = F.softmax(logits, dim=-1)
        acc["sum"] += p.sum(dim=1)
        acc["max"] = torch.maximum(acc["max"], p.amax(dim=1))
        if e > late_start:
            acc["late"] += p[:, max(late_start - s, 0) :].sum(dim=1)
        topq_buf.append(p.topk(min(topq_k, e - s), dim=1).values)
        del logits, p
    # topq: mean over the queries that want each key most, taken across tiles.
    cat = torch.cat(topq_buf, dim=1)
    acc["topq"] = cat.topk(min(topq_k, cat.shape[1]), dim=1).values.mean(dim=1)
    return acc


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank correlation, tie-aware, on whatever subset the caller passes."""
    if x.numel() < 3:
        return float("nan")

    def rank(v: torch.Tensor) -> torch.Tensor:
        v = v.double()
        order = v.argsort()
        r = torch.empty_like(v)
        r[order] = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        uniq, inv, cnt = torch.unique(v, return_inverse=True, return_counts=True)
        return (torch.zeros_like(uniq).index_add_(0, inv, r) / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float((rx @ ry) / denom) if denom > 0 else float("nan")


def utility_stats_from_probs(attn: torch.Tensor) -> dict:
    """The same four reductions computed from a full ``(H, Sq, Sk)`` probability tensor.

    Reference path: used for the synthetic stream, and for verifying that the tiled
    :func:`attention_probs_from_module` agrees with the obvious implementation.
    """
    a = attn.float()
    _, Sq, Sk = a.shape
    k = max(1, int(0.01 * Sk))
    return {
        "sum": a.sum(dim=1),
        "max": a.amax(dim=1),
        "topq": a.topk(min(k, Sq), dim=1).values.mean(dim=1),
        "late": a[:, 3 * Sq // 4 :].sum(dim=1),
    }


def recall_at_budget(score: torch.Tensor, utility: torch.Tensor, keep_frac: float) -> float:
    """Fraction of the true top-``keep_frac`` utility MASS retained by a top-k on ``score``.

    Mass rather than set overlap: keeping many low-mass keys and missing one dominant key
    is a failure that set overlap would score generously. Random selection scores about
    ``keep_frac`` on this metric, which is the baseline to beat.
    """
    n = score.numel()
    k = max(1, int(round(keep_frac * n)))
    kept = score.topk(k).indices
    total = utility.sum()
    return float(utility[kept].sum() / total) if total > 0 else float("nan")


def build_scores(h: torch.Tensor, state_dim: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Candidate per-key scores: the state designs, plus the controls that bracket them.

    Uses a random projection to ``state_dim`` in place of a trained ``w_k``, as in the state
    diagnostic, so these are lower bounds on what a trained scorer reaches.
    """
    d_state = min(state_dim, h.shape[-1])
    if d_state < h.shape[-1]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        proj = (torch.randn(h.shape[-1], d_state, generator=g) / math.sqrt(d_state)).to(h.device)
        hp = h @ proj
    else:
        hp = h
    hp = torch.nn.functional.normalize(hp - hp.mean(0, keepdim=True), dim=-1)

    L = hp.shape[0]
    out = {
        "novelty: full attn (redundancy ceiling)": novelty_full_attention(hp),
        "novelty: EMA hl=512": novelty_ema(hp, 512),
        "no state: ||h_t||": hp.norm(dim=-1),
        "control: recency (+t)": torch.arange(L, dtype=hp.dtype, device=hp.device),
        "control: random": torch.randn(L, generator=torch.Generator().manual_seed(seed)).to(
            device=hp.device, dtype=hp.dtype
        ),
    }
    out.update(
        novelty_matrix_states(
            hp,
            [
                ("novelty: delta b=0.5 dec=0.99", 0.5, 0.99),
                ("novelty: delta b=1 dec=0.99", 1.0, 0.99),
            ],
        )
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--layers", default="8,18,28")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-frac", type=float, default=0.1, help="cache budget for recall")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="utility_proxy.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)
    layers = [int(x) for x in args.layers.split(",")]
    util_modes = ("sum", "max", "topq", "late")
    results = {}

    for layer in layers:
        h, stats = get_layer(args, device, layer)
        L = h.shape[0]
        print(f"\n{'=' * 88}\nlayer {layer}: {L} tokens, {stats['sum'].shape[0]} heads"
              f"\n{'=' * 88}", flush=True)

        utils = {m: key_utility(stats, m) for m in util_modes}
        del stats
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # How much do the utility definitions even agree with each other? If they disagree,
        # "predicting utility" is not one target and no single score can serve all of them.
        print("0. do the utility definitions agree? (spearman between them)")
        for i, a in enumerate(util_modes):
            row = "   ".join(
                f"{b}={spearman(utils[a], utils[b]):+.2f}" for b in util_modes[i + 1 :]
            )
            if row:
                print(f"   {a:>5} vs  {row}")

        scores = build_scores(h, args.state_dim)
        # Skip the warmup region: the first keys are seen by nearly every query, so their
        # utility is inflated by position alone and would flatter any score correlated with t.
        warm = max(32, L // 20)
        sel = torch.zeros(L, dtype=torch.bool, device=h.device)
        sel[warm:] = True

        print(f"\n1. spearman(score, utility)   [keys {warm}..{L}]")
        print(f"   {'score':<40} " + " ".join(f"{m:>7}" for m in util_modes))
        corr = {}
        for name, s in scores.items():
            cs = {m: spearman(s[sel], utils[m][sel]) for m in util_modes}
            corr[name] = cs
            print(f"   {name:<40} " + " ".join(f"{cs[m]:+7.3f}" for m in util_modes))
        print(f"   {'ORACLE (utility itself)':<40} " + " ".join(f"{1.0:+7.3f}" for _ in util_modes))

        print(f"\n2. recall of true utility MASS at keep={args.keep_frac:.0%}"
              f"   (random ~= {args.keep_frac:.2f}, oracle = the ceiling)")
        orac = {
            m: recall_at_budget(utils[m][sel], utils[m][sel], args.keep_frac) for m in util_modes
        }
        # Without the oracle the recall numbers are unreadable: if utility is spread evenly
        # then even a perfect score only recovers keep_frac of the mass, and 0.11 would be
        # near-optimal rather than near-useless. 'norm' rescales each score onto
        # [random, oracle] so the columns are comparable across utility definitions.
        print(f"   {'concentration: oracle recall':<40} "
              + " ".join(f"{orac[m]:7.3f}" for m in util_modes))
        print(f"   {'score':<40} " + " ".join(f"{m:>7}" for m in util_modes)
              + "   | normalised to [random, oracle]")
        rec = {}
        for name, s in scores.items():
            rs = {m: recall_at_budget(s[sel], utils[m][sel], args.keep_frac) for m in util_modes}
            rec[name] = rs
            norm = {
                m: (rs[m] - args.keep_frac) / max(orac[m] - args.keep_frac, 1e-9)
                for m in util_modes
            }
            print(f"   {name:<40} " + " ".join(f"{rs[m]:7.3f}" for m in util_modes)
                  + "   | " + " ".join(f"{norm[m]:+6.2f}" for m in util_modes))
        print(f"   {'ORACLE (utility itself)':<40} " + " ".join(f"{orac[m]:7.3f}" for m in util_modes)
              + "   | " + " ".join(f"{1.0:+6.2f}" for _ in util_modes))

        # The headline number: does redundancy predict utility better than a trivial score?
        print("\n3. verdict")
        for m in ("sum", "max"):
            best_state = max(
                (abs(corr[n][m]), n) for n in corr if n.startswith("novelty:") and "ceiling" not in n
            )
            trivial = max((abs(corr[n][m]), n) for n in corr if n.startswith(("no state", "control")))
            print(f"   utility={m:<5} best state |rho|={best_state[0]:.3f} ({best_state[1]})")
            print(f"   {'':13} best trivial |rho|={trivial[0]:.3f} ({trivial[1]})")
            gain = best_state[0] - trivial[0]
            verdict = ("state beats trivial" if gain > 0.05 else
                       "NO GAIN over a trivial score" if gain <= 0.0 else "marginal")
            print(f"   {'':13} -> {verdict} (gap {gain:+.3f})")

        results[f"layer{layer}"] = {"spearman": corr, "recall": rec, "oracle_recall": orac,
                                    "n_keys": int(sel.sum())}
        del h, scores, utils
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHow to read this:")
    print("  Section 0: if the utility definitions disagree with each other, there is no single")
    print("    'usefulness' to predict, and sum-vs-max is a design choice you have to make.")
    print("  Section 1/2: a novelty score must beat BOTH 'no state' and 'control: recency'.")
    print("    Recency is the real bar -- StreamingLLM works, so recency already predicts")
    print("    utility well; the state has to add something on top of it.")
    print("  Section 3: 'NO GAIN' means the redundancy signal, however strong on its own")
    print("    task, does not transfer to key utility. Retarget the training objective")
    print("    rather than tuning the state.")


def get_layer(args, device: torch.device, layer: int):
    """``(hidden_states, utility_stats)`` for one layer: ``(L, d)`` and dict of ``(H, L)``."""
    if args.fake:
        g = torch.Generator().manual_seed(0)
        L, d, H = args.seq_len, 256, 8
        basis = torch.randn(16, d, generator=g)
        idx = torch.randint(16, (L,), generator=g)
        h = basis[idx] + 0.3 * torch.randn(L, d, generator=g)
        h[:, 0] += 8.0
        # Attention that is partly recency-driven and partly content-driven, so the
        # controls and the state designs both have something to find.
        logits = (h @ h.T) / math.sqrt(d)
        logits = logits - 0.01 * (torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(-1)).abs()
        logits = logits.masked_fill(
            torch.arange(L).unsqueeze(0) > torch.arange(L).unsqueeze(-1), -float("inf")
        )
        attn = torch.softmax(logits, dim=-1).unsqueeze(0).expand(H, L, L)
        return h.to(device), utility_stats_from_probs(attn.to(device))

    from transformers import AutoModel, AutoTokenizer

    print(f"loading {args.model} (layer {layer}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # sdpa (the default) is fine and preferred: the attention statistics are computed here
    # from the layer's own q/k, so the model never has to build a probability matrix. Forcing
    # eager -- which an earlier version of this script did, to make output_attentions work --
    # would only make the forward slower and more memory-hungry for no benefit.
    kw = {"low_cpu_mem_usage": True}
    try:
        model = AutoModel.from_pretrained(args.model, dtype=dtype, **kw)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, **kw)
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    if not 0 <= layer < n_layers:
        raise SystemExit(f"--layers has {layer}, model has decoder layers 0..{n_layers - 1}")

    grabbed = {}

    def hook_hidden(_m, _i, out):
        grabbed["h"] = (out[0] if isinstance(out, tuple) else out).detach()[0].float()

    # Attention for ONE layer, recomputed from that layer's own q/k rather than requested
    # from the model. output_attentions=True retains every layer's (H, Sq, Sk) tensor --
    # 36 layers x 32 heads at 8192 tokens is 144 GB, an instant OOM to serve one layer.
    # Hooking the attention module and redoing the small qk here costs one (H, L, L)
    # tensor, tiled over queries so even that is never fully resident in fp32.
    def hook_attn(mod, args_, kwargs_):
        hs = kwargs_.get("hidden_states", args_[0] if args_ else None)
        pe = kwargs_.get("position_embeddings")
        grabbed["attn"] = attention_probs_from_module(mod, hs, pe)

    attn_mod = model.layers[layer].self_attn
    handles = [
        model.layers[layer].register_forward_hook(hook_hidden),
        attn_mod.register_forward_pre_hook(hook_attn, with_kwargs=True),
    ]
    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.seq_len].to(device)
    print(f"  forward on {ids.shape[1]} tokens (layer {layer} attention only) ...", flush=True)
    t0 = time.time()
    try:
        with torch.no_grad():
            model(ids)
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            f"CUDA OOM at seq_len={args.seq_len}. One layer's attention is "
            f"{model.config.num_attention_heads * args.seq_len ** 2 * 4 / 2**30:.1f} GB in fp32.\n"
            f"  {exc}\nTry --seq-len {args.seq_len // 2}."
        ) from exc
    for hd in handles:
        hd.remove()
    if "attn" not in grabbed:
        raise SystemExit(
            f"the attention pre-hook on layer {layer} never fired; the module tree is not the "
            f"expected model.layers[i].self_attn shape."
        )
    stats, h = grabbed["attn"], grabbed["h"]
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  done in {time.time() - t0:.1f}s, {stats['sum'].shape[0]} heads, "
              f"peak {peak:.1f} GB", flush=True)
    else:
        print(f"  done in {time.time() - t0:.1f}s, {stats['sum'].shape[0]} heads", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return h, stats


if __name__ == "__main__":
    main()
