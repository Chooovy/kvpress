#!/usr/bin/env python
"""
Does feeding a recurrent state ``z`` into the indexer make its score better?

This is the question the earlier proxy diagnostic got wrong. That script scored a
hand-picked readout of the state -- the residual *norm* ``||h - Sh||`` -- directly against
key utility, and found it weak (rho ~ 0.19). But that is not the proposed design. The design
is ``score_t = f(h_t, z_{t-1})`` with ``f`` **learned**, so the state contributes a vector
that ``f`` decides how to use, not a scalar somebody chose in advance. The right question is
therefore incremental:

    does adding z-derived features to a learned scorer beat the same scorer without them?

Measured as nested models, each fit to predict real per-key attention utility:

    M1  h_t                      what the indexer already sees
    M2  h_t + position           plus the free positional signal
    M3  h_t + position + z       the proposal

The number that matters is **M3 - M2**. M2 is the honest baseline because position is free:
recency alone reaches rho ~ 0.77 on late-query utility (StreamingLLM works for a reason), so
a state that merely rediscovers position is worth nothing.

Three controls, because a probe with hundreds of features will fit noise:

  * held-out documents -- fit on some, score on others. In-sample R^2 is meaningless here.
  * **shuffled z** -- the same z features permuted across positions. Same dimensionality,
    same marginal distribution, no real information. If M3 beats M2 by as much with shuffled
    z as with real z, the gain was capacity, not signal. This is the decisive control.
  * per-KV-head fitting, with sink and local keys excluded -- the press protects those by
    rule, so including them lets a probe take credit for work it does not do (``||h_t||``
    alone scores as well as any state on mass recall, purely by finding attention sinks).

z features are vectors, not norms: ``u = h - S h`` (delta-rule residual), ``S h`` (what the
state already predicts about this key), and ``h - z`` (EMA deviation). Those are already
interactions between the token and its history, so a linear probe on them can express the
bilinear ``h' M' z`` term the design is really about.

Run:
    python scripts/diag_state_probe.py --layers 18 --seq-len 8192 --n-docs 6

Reports, per utility definition, the held-out Spearman of each nested model plus the
real-vs-shuffled gap. Read the gap, not the absolute numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from diag_rnn_state_design import load_text, resolve_device
from diag_utility_proxy import attention_probs_from_module, key_utility, spearman

# The nested models. M2 is the honest baseline: position is free to compute, and reaches
# rho ~ 0.77 on late-query utility by itself, so a state that merely rediscovers position is
# worth nothing. M3' is the decisive control -- same feature count and marginals as M3, no
# real information -- which separates "z carries signal" from "the probe got more capacity".
COMBOS = {
    "M1 h": ("h",),
    "M2 h+pos": ("h", "pos"),
    "M3 h+pos+z": ("h", "pos", "z"),
    "M3' h+pos+SHUFFLED z": ("h", "pos", "zs"),
}


def state_features(h: torch.Tensor, beta: float, decay: float, ema_hl: float) -> torch.Tensor:
    """Per-token z-derived features, ``(L, 3*d)``.

    Three blocks, all vectors so a linear probe can form the ``h' M' z`` interaction the
    design needs (a norm would collapse exactly that):

      ``u = h_t - S_{t-1} h_t``   delta-rule residual: the part of this key the state
                                  cannot already reconstruct
      ``S_{t-1} h_t``             what the state does predict -- the complement of u, and
                                  not recoverable from u alone once h is also given
      ``h_t - z_{t-1}``           EMA deviation, the cheap vector-state alternative

    ``S`` and ``z`` are always the state *before* absorbing ``h_t``, so no feature sees its
    own token; using the post-update state would leak the label through the token itself.
    """
    L, d = h.shape
    hn = torch.nn.functional.normalize(h, dim=-1)
    S = torch.zeros(d, d, dtype=h.dtype, device=h.device)
    z = torch.zeros(d, dtype=h.dtype, device=h.device)
    lam = 0.5 ** (1.0 / ema_hl)
    feats = torch.empty(L, 3 * d, dtype=h.dtype, device=h.device)
    for t in range(L):
        x = hn[t]
        Sx = S @ x
        u = x - Sx
        feats[t, :d] = u
        feats[t, d : 2 * d] = Sx
        feats[t, 2 * d :] = x - z
        S = decay * S + beta * torch.outer(u, x)
        z = lam * z + (1 - lam) * x
    return feats


def position_features(L: int, device, dtype) -> torch.Tensor:
    """Free positional signal: what a rule-based scorer gets for nothing.

    Several shapes because "position matters" is not one feature -- utility depends on
    absolute index, on remaining distance to the end, and on both logarithmically (attention
    decays roughly geometrically with distance). Giving M2 a generous positional basis is
    what makes the M3 - M2 gap an honest measure of the state's contribution.
    """
    t = torch.arange(L, device=device, dtype=dtype)
    return torch.stack(
        [
            t / L,
            (L - 1 - t) / L,
            torch.log1p(t) / math.log(L),
            torch.log1p(L - 1 - t) / math.log(L),
        ],
        dim=-1,
    )


def ridge_fit_predict(
    xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Ridge regression in fp64, standardised on the training split only.

    Standardising with test statistics would leak; fitting in fp32 makes the normal
    equations ill-conditioned once features are correlated, which these heavily are (``u``
    and ``S h`` sum to ``h``).
    """
    xtr, ytr, xte = xtr.double(), ytr.double(), xte.double()
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-8)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ytr_c = ytr - ytr.mean()
    n_feat = xtr.shape[1]
    gram = xtr.T @ xtr + alpha * n_feat * torch.eye(n_feat, dtype=xtr.dtype, device=xtr.device)
    w = torch.linalg.solve(gram, xtr.T @ ytr_c)
    return xte @ w


def mlp_fit_predict(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    *,
    hidden: int = 256,
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
) -> torch.Tensor:
    """Two-layer MLP probe, early-stopped on a validation slice of the training split.

    Exists to close the one real gap in the ridge result. A linear probe on the z features
    can only express ``a' S h`` -- ``d`` free parameters -- whereas the design's ``h' M' z``
    interaction has ``d^2``. So "linear use of z adds nothing" does not settle "z adds
    nothing"; only a probe that can actually form the products does.

    The comparison stays fair because the *same* probe is fitted to every nested model: if
    the extra capacity finds structure, it finds it for h and position too, and the M3 - M2
    gap is still attributable to z. The shuffled-z control is what guards against the
    capacity itself producing a gap.

    Targets are standardised, and the validation split is carved out of training data only
    -- the test documents are never touched during fitting.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    xtr, ytr, xte = xtr.float(), ytr.float(), xte.float()
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-8)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ym, ys = ytr.mean(), ytr.std().clamp_min(1e-8)
    ytr = (ytr - ym) / ys

    n = xtr.shape[0]
    perm = torch.randperm(n, generator=g).to(xtr.device)
    n_val = max(1, int(val_frac * n))
    vi, ti = perm[:n_val], perm[n_val:]
    xt, yt, xv, yv = xtr[ti], ytr[ti], xtr[vi], ytr[vi]

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden // 2),
        torch.nn.GELU(),
        torch.nn.Linear(hidden // 2, 1),
    ).to(xtr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(xt).squeeze(-1), yt).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(xv).squeeze(-1), yv).item()
        if v < best - 1e-5:
            best, patience = v, 0
            best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 40:  # converged or overfitting; keep the best weights
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(xte).squeeze(-1).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--layers", default="18")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--ema-half-life", type=float, default=512.0)
    ap.add_argument("--n-docs", type=int, default=6, help="documents; split into train/test")
    ap.add_argument("--n-sink", type=int, default=4, help="leading keys the press protects")
    ap.add_argument("--n-local", type=int, default=128, help="trailing keys the press protects")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    ap.add_argument(
        "--probe",
        default="both",
        choices=("ridge", "mlp", "both"),
        help="readout to fit. 'both' runs each and prints them side by side, which is the "
        "informative setting: ridge bounds the linear use of z, mlp the nonlinear.",
    )
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--mlp-epochs", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default="state_probe.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)
    util_modes = ("sum", "max", "topq", "late")
    results = {}

    for layer in [int(x) for x in args.layers.split(",")]:
        docs = collect_docs(args, device, layer)
        n_tr = max(1, len(docs) * 2 // 3)
        print(f"\n{'=' * 86}\nlayer {layer}: {len(docs)} docs "
              f"({n_tr} train / {len(docs) - n_tr} test)\n{'=' * 86}", flush=True)

        n_kv = docs[0]["n_kv"]
        probes = ("ridge", "mlp") if args.probe == "both" else (args.probe,)
        layer_res = {p: {} for p in probes}
        for kv in range(n_kv):
            for probe in probes:
                per_mode = {m: {} for m in util_modes}
                for mode in util_modes:
                    tr_x, tr_y, te_x, te_y = [], [], [], []
                    for i, doc in enumerate(docs):
                        keep = doc["keep"]
                        y = key_utility_kv(doc["stats"], mode, kv, doc["group"])[keep]
                        parts = {
                            "h": doc["h"][keep],
                            "pos": doc["pos"][keep],
                            "z": doc["z"][keep],
                            "zs": doc["z_shuf"][keep],
                        }
                        (tr_x if i < n_tr else te_x).append(parts)
                        (tr_y if i < n_tr else te_y).append(y)
                    ytr, yte = torch.cat(tr_y), torch.cat(te_y)
                    for name, keys in COMBOS.items():
                        xtr = torch.cat([torch.cat([p[k] for k in keys], dim=-1) for p in tr_x])
                        xte = torch.cat([torch.cat([p[k] for k in keys], dim=-1) for p in te_x])
                        if probe == "ridge":
                            pred = ridge_fit_predict(xtr, ytr, xte, args.alpha)
                        else:
                            pred = mlp_fit_predict(
                                xtr, ytr, xte, hidden=args.mlp_hidden, epochs=args.mlp_epochs
                            )
                        per_mode[mode][name] = spearman(pred, yte.double())
                layer_res[probe][f"kv{kv}"] = per_mode
            if kv == 0:
                print(f"   {'probe / model':<30} " + " ".join(f"{m:>8}" for m in util_modes)
                      + "     (held-out spearman, KV head 0)")
                if len(probes) > 1:
                    # If the MLP scores far below ridge on the SAME features it is
                    # undertrained, and any M3 - M2 gap it reports is meaningless. This is
                    # the check that the two probes are comparable at all.
                    r = layer_res["ridge"]["kv0"]
                    mm = layer_res["mlp"]["kv0"]
                    worst = min(mm[m]["M2 h+pos"] - r[m]["M2 h+pos"] for m in util_modes)
                    if worst < -0.05:
                        print(f"   WARNING: mlp trails ridge by {abs(worst):.3f} on M2 -- "
                              f"undertrained. Raise --mlp-epochs before reading the gaps.")
            if kv < 2:
                for probe in probes:
                    for name in COMBOS:
                        r = layer_res[probe][f"kv{kv}"]
                        print(f"   kv{kv} {probe:<5} {name:<19} "
                              + " ".join(f"{r[m][name]:+8.3f}" for m in util_modes))

        gaps = {}
        for probe in probes:
            print(f"\n   [{probe}] gaps averaged over {n_kv} KV heads"
                  + " " * max(1, 26 - len(probe)) + " ".join(f"{m:>8}" for m in util_modes))
            gaps[probe] = {}
            for label, a, b in (
                ("z over h+pos (REAL)", "M3 h+pos+z", "M2 h+pos"),
                ("z over h+pos (SHUFFLED)", "M3' h+pos+SHUFFLED z", "M2 h+pos"),
                ("pos over h", "M2 h+pos", "M1 h"),
            ):
                row = {}
                for m in util_modes:
                    vals = [
                        layer_res[probe][f"kv{k}"][m][a] - layer_res[probe][f"kv{k}"][m][b]
                        for k in range(n_kv)
                    ]
                    row[m] = sum(vals) / len(vals)
                gaps[probe][label] = row
                print(f"   {label:<30} " + " ".join(f"{row[m]:+8.3f}" for m in util_modes))
            # How many heads individually improve: an average of +0.01 built from 5 up and 3
            # down is noise, the same average from 8/8 up is a small real effect.
            wins = {}
            for m in util_modes:
                w = sum(
                    layer_res[probe][f"kv{k}"][m]["M3 h+pos+z"]
                    > layer_res[probe][f"kv{k}"][m]["M2 h+pos"]
                    for k in range(n_kv)
                )
                wins[m] = w
            gaps[probe]["heads_improved"] = wins
            print(f"   {'heads where z helps (of ' + str(n_kv) + ')':<30} "
                  + " ".join(f"{wins[m]:>8}" for m in util_modes))

            print(f"\n   [{probe}] verdict")
            for m in util_modes:
                real = gaps[probe]["z over h+pos (REAL)"][m]
                shuf = gaps[probe]["z over h+pos (SHUFFLED)"][m]
                # REAL is the quantity that matters: it is what z actually buys over a
                # baseline that already has h and position. SHUFFLED only certifies that a
                # positive REAL is information rather than capacity -- a large REAL-minus-
                # SHUFFLED gap driven by a negative SHUFFLED means the probe dislikes noise,
                # not that z helps, which is why the two are reported separately.
                v = (
                    "z ADDS real signal" if real > 0.02 and real > shuf
                    else "z HURTS" if real < -0.02
                    else "z adds ~nothing"
                )
                print(f"   {m:>5}: real {real:+.3f}  (shuffled {shuf:+.3f}, "
                      f"{wins[m]}/{n_kv} heads up)  -> {v}")

        results[f"layer{layer}"] = {"per_kv": layer_res, "gaps": gaps}
        del docs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHow to read this:")
    print("  Read the REAL row: that is what z buys over a baseline which already has h and")
    print("  position. SHUFFLED is a certificate, not a score -- it confirms a positive REAL")
    print("  is information rather than capacity. A big REAL-minus-SHUFFLED gap driven by a")
    print("  negative SHUFFLED just means the probe dislikes noise.")
    print("  'heads where z helps' guards against averages: +0.01 from 5 heads up and 3 down")
    print("  is noise; the same +0.01 from 8/8 up is a small real effect.")
    print("  ridge bounds the LINEAR use of z (d free parameters on the S h interaction);")
    print("  mlp can form the full bilinear h' M' z (d^2). If mlp does not beat ridge on the")
    print("  REAL row, nonlinearity is not the missing ingredient.")
    print("  'pos over h' is the competitor to beat, and it costs nothing to compute.")


def key_utility_kv(stats: dict, mode: str, kv: int, group: int) -> torch.Tensor:
    """Utility for one KV head: mean over the query heads that read it.

    Per-KV-head rather than over all heads, because that is the granularity the press evicts
    at -- averaging all 32 query heads washes out exactly the per-head sparsity the indexer
    exploits (it also flattens the utility distribution, which is why the earlier diagnostic
    saw an oracle recall of only 0.27).
    """
    return stats[mode][kv * group : (kv + 1) * group].float().mean(dim=0)


def collect_docs(args, device, layer):
    """One entry per document: hidden states, features, utility stats, and the key mask."""
    if args.fake:
        docs = []
        for di in range(args.n_docs):
            g = torch.Generator().manual_seed(di)
            L, d, H, n_kv = args.seq_len, 128, 8, 2
            basis = torch.randn(16, d, generator=g)
            h = basis[torch.randint(16, (L,), generator=g)] + 0.3 * torch.randn(
                L, d, generator=g
            )
            h[:, 0] += 8.0
            logits = (h @ h.T) / math.sqrt(d)
            logits -= 0.01 * (torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(-1)).abs()
            logits = logits.masked_fill(
                torch.arange(L).unsqueeze(0) > torch.arange(L).unsqueeze(-1), -float("inf")
            )
            p = torch.softmax(logits, dim=-1).unsqueeze(0).expand(H, L, L)
            k = max(1, int(0.01 * L))
            stats = {
                "sum": p.sum(1), "max": p.amax(1),
                "topq": p.topk(min(k, L), dim=1).values.mean(1),
                "late": p[:, 3 * L // 4 :].sum(1),
            }
            docs.append(make_doc(h.to(device), stats, H // n_kv, n_kv, args))
        return docs

    from transformers import AutoModel, AutoTokenizer

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModel.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(device).eval()
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads

    grabbed = {}
    handles = [
        model.layers[layer].register_forward_hook(
            lambda m, i, o: grabbed.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach()[0].float()
            )
        ),
        model.layers[layer].self_attn.register_forward_pre_hook(
            lambda m, a, kw: grabbed.__setitem__(
                "stats",
                attention_probs_from_module(
                    m, kw.get("hidden_states", a[0] if a else None), kw.get("position_embeddings")
                ),
            ),
            with_kwargs=True,
        ),
    ]

    text = load_text(args)
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    per = args.seq_len
    docs = []
    for di in range(args.n_docs):
        chunk = ids_all[di * per : (di + 1) * per]
        if chunk.numel() < per // 2:
            print(f"  only {di} documents of {per} tokens available in the text; "
                  f"raise --n-docs text or lower --seq-len", flush=True)
            break
        ids = chunk.unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            model(ids)
        docs.append(make_doc(grabbed["h"], grabbed["stats"], n_q // n_kv, n_kv, args))
        print(f"  doc {di}: {ids.shape[1]} tokens, {time.time() - t0:.1f}s", flush=True)
    for hd in handles:
        hd.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not docs:
        raise SystemExit("no documents collected; check --seq-len against the text length")
    return docs


def make_doc(h_full, stats, group, n_kv, args):
    """Project, build features, and mark which keys the probe is allowed to see."""
    d_state = min(args.state_dim, h_full.shape[-1])
    if d_state < h_full.shape[-1]:
        g = torch.Generator(device="cpu").manual_seed(0)
        proj = (torch.randn(h_full.shape[-1], d_state, generator=g) / math.sqrt(d_state)).to(
            h_full.device
        )
        h = h_full @ proj
    else:
        h = h_full
    h = torch.nn.functional.normalize(h - h.mean(0, keepdim=True), dim=-1)
    L = h.shape[0]

    z = state_features(h, args.beta, args.decay, args.ema_half_life)
    # Shuffled control: identical features, positions permuted. Same count, same marginals,
    # no alignment with the token -- so any gain it produces is the probe's capacity, not z.
    perm = torch.randperm(L, generator=torch.Generator().manual_seed(1)).to(h.device)
    keep = torch.ones(L, dtype=torch.bool, device=h.device)
    keep[: args.n_sink] = False  # sinks: kept by rule, and ||h|| already finds them
    keep[L - args.n_local :] = False  # local window: kept by rule
    return {
        "h": h,
        "pos": position_features(L, h.device, h.dtype),
        "z": z,
        "z_shuf": z[perm],
        "stats": stats,
        "group": group,
        "n_kv": n_kv,
        "keep": keep,
    }


if __name__ == "__main__":
    main()
