# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Probe: what is the *reachable* bound for one query-independent CMP token per chunk?

The linear-memory arm died of **addressability**, not capacity: ridge regression from ``q_t`` to the
exact evicted output ``oE*`` had held-out relative error 0.86 against 0.89 for a per-head constant --
a 3% win, and ridge(q) upper bounds the whole family because it is strictly more expressive than
``phi(q)`` reading a rank-R state. The chunk-CMP design replaces that learned soft addressing with
attention's own hard addressing: ``q_t . k_c + b_c``, one pseudo-KV per chunk of the key axis.

**Why this probe is tight rather than merely an upper bound.** The CMP logit is *exactly linear in*
``q_t``, so ridge on ``q_t`` with targets ``log D_c(t)`` recovers the optimal ``(k_c, b_c)`` for the
hypothesis class itself (up to the ridge penalty) -- not an optimistic relaxation of it. Contrast the
k-means/SVD oracles in `diag_convex_bound.py`, which picked per-row coefficients using the true
``oE*`` and therefore bounded *coverage*; that reasoning error is recorded in
[[memory-direction-not-learnable-from-q]]. Here, if the probe says no, it is no.

One caveat on tightness, stated up front: ``k_c`` is fitted **free in post-RoPE space**. Placing the
CMP at its chunk's landmark position and applying RoPE is a *restriction* of that class. So arm 5 is
exact for "arbitrary ``k_c``" and an upper bound for the RoPE-placed variant. That is the right order
-- if the free version loses, the restricted one cannot win.

Geometry: the C1/C2 split, not a mid-prefill row
------------------------------------------------
Keys ``[0, L1)`` are C1 (compressible); query rows ``[L1, L)`` are C2. Every C2 row sits after every
C1 key, so:

* the evicted set over C1 is **row-independent** -- one ``(k_c, v_c, b_c)`` per chunk really does
  serve every reader, which is what deployment requires;
* no C1 key is rescued by a local window, so the eviction decision is the router's alone
  (:mod:`~kvpress.presses.gqa_indexer.split_loss` makes the same argument for the same reason);
* C2's own keys stay dense, which is the training-time image of ``force_local``.

Held out on **query rows**, not documents
-----------------------------------------
``(k_c, v_c, b_c)`` is derived from its own chunk's KV, so it is document-specific by construction
and a cross-document split would be meaningless. The real leak risk is the query axis: at deployment
the CMP is built before any future query is seen. So the fit uses the first half of C2 and the report
uses the second half. ``analytic`` never touches ``q`` at all, hence is automatically honest -- which
is its own reason to exist, beyond being simple.

Budget-neutral by default
-------------------------
``n_chunks`` CMP slots are paid for out of the exact budget: ``budget - n_chunks`` retained keys plus
``n_chunks`` CMP, against ``budget`` retained keys for the eviction baseline. Comparing at unmatched
budget would be measuring the budget, not the idea.

The arms, and the two controls that must return zero
----------------------------------------------------
=====================  ===========================================================  =============
 arm                    ``(k_c, v_c, b_c)``                                          expectation
=====================  ===========================================================  =============
 ``exact``              true ``(lse_c(t), vbar_c(t))`` per row -- not a CMP at all    rel err = 0
 ``analytic`` @chunk=1  one residual token per chunk: ``k_j, v_j, 0``                 rel err = 0
 ``none``               no CMP                                                        0.11-0.20
 ``analytic``           ``kbar, vbar, log|residual_c|``, zero parameters              ?
 ``ridge``              ridge-optimal ``(k_c, b_c)``; mass-weighted ``v_c``           tight bound
 ``ridge_vstar``        ridge ``(k_c, b_c)``; per-row optimal ``v_c``                 COVERAGE only
=====================  ===========================================================  =============

``ridge_vstar`` uses the true per-row direction and is therefore exactly the kind of quantity that
misled the earlier analysis. It is reported *only* to separate "the direction is unreachable" from
"the mass is unreachable", and must never be quoted as achievable.

Mass and direction are reported separately, always. That is the ``v_norm`` lesson: a healthy ``cos``
with a broken magnitude means a scale bug, not a training failure, and a single fused number cannot
tell the two apart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import (  # noqa: E402
    GQAIndexerPress,
    get_language_model,
)
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument(
        "--router",
        default=f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce/final.pt",
        help="the NON-decay router on purpose: decay makes deadlines() approximate (1.64%% mask "
        "error), and this probe is a go/no-go on CMP, not on two things at once",
    )
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--split", type=int, default=4096, help="L1: keys < L1 are C1, rows >= L1 are C2")
    p.add_argument("--keep-ratio", type=float, default=0.25, help="exact budget as a fraction of C1")
    p.add_argument("--chunks", default="1,64,128,256", help="chunk sizes; 1 is the identity control")
    p.add_argument("--layers", default="0,9,18,27,30,32,33,34,35")
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--row-stride", type=int, default=4)
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge penalty, relative to tr(X'X)/p")
    p.add_argument("--docs", type=int, default=3)
    p.add_argument("--out", default="scratch/probe_chunk_cmp.json")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# per-chunk exact quantities
# ----------------------------------------------------------------------------------------------
def chunk_targets(logits, vf, resid_mask, chunk_size, n_c1):
    """
    ``(lse_c, vbar_c)`` per (query head, row, chunk): the exact contribution of each chunk's
    residual set.

    ``lse_c[h, t, c] = logsumexp_{j in residual_c} logit[h, t, j]`` and
    ``vbar_c[h, t, c] = sum_j softmax_within_c(logit) v_j``.

    Together ``(lse_c, vbar_c)`` is what a *perfect* CMP would have to emit, so feeding them back in
    place of the CMP reconstitutes ``o_dense`` exactly -- that is the ``exact`` control. Kept in log
    space throughout: chunk logits reach ~30 here and a direct ``sum exp`` would overflow fp32 on the
    high-mass chunks, which are precisely the ones that matter.
    """
    H, T, _ = logits.shape
    n_ch = (n_c1 + chunk_size - 1) // chunk_size
    neg = torch.finfo(torch.float32).min
    lg = logits[:, :, :n_c1].masked_fill(~resid_mask.view(1, 1, -1), neg)

    pad = n_ch * chunk_size - n_c1
    if pad:
        lg = torch.nn.functional.pad(lg, (0, pad), value=neg)
    lg = lg.view(H, T, n_ch, chunk_size)

    lse_c = torch.logsumexp(lg, dim=-1)  # (H, T, n_ch); -inf where the chunk kept nothing
    w = torch.softmax(lg, dim=-1)  # rows that are all -neg give a uniform w, killed by empty_c
    vv = vf[:n_c1]
    if pad:
        vv = torch.nn.functional.pad(vv, (0, 0, 0, pad))
    vbar_c = torch.einsum("htcs,csd->htcd", w, vv.view(n_ch, chunk_size, -1))
    return lse_c, vbar_c, n_ch


def fuse(logits_keep, vf_keep, cmp_logit, cmp_v, empty_c):
    """
    One softmax over ``[retained keys ; CMP slots]`` -- the whole point of the design.

    Empty chunks are masked to ``-inf`` rather than dropped so every arm sees an identical column
    layout; a chunk with no residual token must contribute nothing, and letting ``log 0`` flow into
    the softmax would make it contribute ``exp(-inf) = 0`` only by luck of the fp path.
    """
    neg = torch.finfo(torch.float32).min
    cl = cmp_logit.masked_fill(empty_c.view(1, 1, -1), neg)
    all_logits = torch.cat([logits_keep, cl], dim=-1)
    p = torch.softmax(all_logits, dim=-1)
    n_keep = logits_keep.shape[-1]
    o = torch.einsum("hts,sd->htd", p[..., :n_keep], vf_keep)
    o = o + torch.einsum("htc,htcd->htd", p[..., n_keep:], cmp_v)
    return o


def ridge_fit(X, y, lam):
    """
    Batched ridge: ``X (G, N, p)``, ``y (G, N)`` -> ``w (G, p)``.

    ``G`` is (KV head x chunk), and the ``N`` axis stacks **every query head in the GQA group**. That
    stacking is the load-bearing detail: a CMP token is a KV entry, so its ``k_c`` is shared by all
    ``group`` query heads that read that KV head. Fitting one ``k_c`` per query head would inflate
    the bound by a factor the real design cannot use.

    ``lam`` scales with ``tr(X'X)/p`` so one value works across layers whose ``q`` norms differ by
    orders of magnitude -- the same per-layer-scale trap as ``GATE_SCALE_INIT`` and ``log_gamma``.
    """
    XtX = torch.einsum("gnp,gnq->gpq", X, X)
    Xty = torch.einsum("gnp,gn->gp", X, y)
    p = X.shape[-1]
    scale = torch.diagonal(XtX, dim1=-2, dim2=-1).sum(-1) / p
    eye = torch.eye(p, device=X.device, dtype=X.dtype).expand_as(XtX)
    A = XtX + (lam * scale.clamp(min=1e-12)).view(-1, 1, 1) * eye
    return torch.linalg.solve(A, Xty)


@torch.no_grad()
def run_layer(q, k, v, scores, *, args, scale, group, layer):
    """Every arm, for one layer. Returns a list of per-(chunk_size) result dicts."""
    L, L1 = args.length, args.split
    dev = q.device
    Hq, Hkv, D = q.shape[1], k.shape[1], k.shape[-1]
    kf, vf = k[0].float(), v[0].float()
    key_idx = torch.arange(L, device=dev)

    # ---- eviction over C1, budget-neutral against the CMP slots ------------------------------
    # Sinks are forced and then excluded from the ranking, so the budget means the same thing in
    # every arm. Scores come from the trained router, restricted to C1.
    budget = max(1, int(L1 * args.keep_ratio))
    s_c1 = scores[:, :L1].clone()  # (Hkv, L1)
    s_c1[:, : args.n_sink] = float("inf")  # sinks always retained

    rows = torch.arange(L1, L, args.row_stride, device=dev)
    n_fit = rows.numel() // 2
    fit_rows, val_rows = rows[:n_fit], rows[n_fit:]

    out = []
    for chunk_size in [int(c) for c in args.chunks.split(",")]:
        n_ch = (L1 + chunk_size - 1) // chunk_size
        n_exact = max(args.n_sink, budget - n_ch)  # budget-neutral: CMP slots cost exact slots
        keep_c1 = torch.zeros(Hkv, L1, dtype=torch.bool, device=dev)
        topi = s_c1.topk(min(n_exact, L1), dim=-1).indices
        keep_c1.scatter_(1, topi, True)
        resid_c1 = ~keep_c1  # (Hkv, L1)

        # per-KV-head residual counts per chunk, for the analytic b_c and for empty-chunk masking
        rc = resid_c1.view(Hkv, n_ch, chunk_size).sum(-1)  # (Hkv, n_ch)

        per_head = []
        for hkv in range(Hkv):
            hq0, hq1 = hkv * group, (hkv + 1) * group
            resid = resid_c1[hkv]
            empty_c = rc[hkv] == 0
            # This KV head's own keys/values. GQA: all `group` query heads read these.
            kh, vh = kf[hkv], vf[hkv]

            # ---- targets on fit and val rows, tiled over the query axis ----------------------
            def targets(rr):
                acc = {kk: [] for kk in ("lse_c", "vbar_c", "keeplog", "od", "oS", "rho", "q")}
                for s in range(0, rr.numel(), 512):
                    r = rr[s : s + 512]
                    qt = q[0, hq0:hq1, r].float()  # (group, T, D)
                    lg = torch.einsum("htd,sd->hts", qt, kh) * scale
                    causal = key_idx.view(1, 1, -1) <= r.view(1, -1, 1)
                    # C2 keys are dense (the training-time image of force_local); C1 keys are
                    # retained only if the router kept them.
                    keep = causal & torch.cat(
                        [keep_c1[hkv].view(1, 1, -1).expand(1, r.numel(), L1),
                         torch.ones(1, r.numel(), L - L1, dtype=torch.bool, device=dev)], dim=-1
                    )
                    neg = torch.finfo(torch.float32).min
                    lse_d = torch.logsumexp(lg.masked_fill(~causal, neg), -1)
                    lse_s = torch.logsumexp(lg.masked_fill(~keep, neg), -1)
                    od = torch.einsum(
                        "hts,sd->htd", torch.softmax(lg.masked_fill(~causal, neg), -1), vh
                    )
                    oS = torch.einsum(
                        "hts,sd->htd", torch.softmax(lg.masked_fill(~keep, neg), -1), vh
                    )
                    lc, vc, _ = chunk_targets(lg, vh, resid, chunk_size, L1)
                    acc["lse_c"].append(lc)
                    acc["vbar_c"].append(vc)
                    acc["keeplog"].append(lg.masked_fill(~keep, neg))
                    acc["od"].append(od)
                    acc["oS"].append(oS)
                    acc["rho"].append((1.0 - torch.exp(lse_s - lse_d)).clamp(0, 1))
                    acc["q"].append(qt)
                    del lg, causal, keep
                return {kk: torch.cat(vv, dim=1) for kk, vv in acc.items()}

            tf, tv = targets(fit_rows), targets(val_rows)

            # ---- arm 5: ridge for (k_c, b_c) on log D_c, group-stacked -----------------------
            # finite rows only: a chunk that is empty, or a row whose logits underflow it, carries
            # no equation. Solving through those would fit -inf.
            g, T = tf["lse_c"].shape[0], tf["lse_c"].shape[1]
            Xall = torch.cat([tf["q"], torch.ones(g, T, 1, device=dev)], dim=-1)  # (g, T, D+1)
            X = Xall.reshape(1, g * T, D + 1).expand(n_ch, g * T, D + 1)
            Y = tf["lse_c"].permute(2, 0, 1).reshape(n_ch, g * T)
            fin = torch.isfinite(Y)
            Xm = X * fin.unsqueeze(-1)
            Ym = torch.where(fin, Y, torch.zeros_like(Y))
            w = ridge_fit(Xm.contiguous(), Ym, args.ridge)  # (n_ch, D+1)
            k_ridge, b_ridge = w[:, :D], w[:, D]

            # ---- v_c: mass-weighted mean over FIT rows, query-independent by construction ----
            # v_c = sum_t D_c(t) vbar_c(t) / sum_t D_c(t) = sum_t N_c(t) / sum_t D_c(t), computed
            # with a per-chunk shift so the weights cannot overflow.
            lse_f = tf["lse_c"]  # (g, T, n_ch)
            shift = torch.where(
                torch.isfinite(lse_f), lse_f, torch.full_like(lse_f, -1e30)
            ).amax(dim=(0, 1))
            wt = torch.exp(lse_f - shift.view(1, 1, -1)).nan_to_num(0.0)
            v_agg = torch.einsum("htc,htcd->cd", wt, tf["vbar_c"])
            v_agg = v_agg / wt.sum(dim=(0, 1)).clamp(min=1e-20).unsqueeze(-1)

            # ---- analytic: kbar / vbar over residual tokens, b = log|residual| ---------------
            rmask = resid.float()
            rm = rmask.view(n_ch, chunk_size)
            cnt = rm.sum(-1).clamp(min=1e-20)
            k_an = torch.einsum("cs,csd->cd", rm, kh[:L1].view(n_ch, chunk_size, -1)) / cnt.unsqueeze(-1)
            v_an = torch.einsum("cs,csd->cd", rm, vh[:L1].view(n_ch, chunk_size, -1)) / cnt.unsqueeze(-1)
            b_an = torch.log(cnt)

            # ---- evaluate every arm on the VAL rows -----------------------------------------
            qv = tv["q"]
            live = tv["rho"] > 1e-3

            def relerr(o):
                e = (o - tv["od"]).norm(dim=-1) / tv["od"].norm(dim=-1).clamp(min=1e-9)
                return float(e[live].mean()) if live.any() else float("nan")

            zero_v = torch.zeros(qv.shape[0], qv.shape[1], n_ch, D, device=dev)
            neg_l = torch.full((qv.shape[0], qv.shape[1], n_ch), -1e30, device=dev)
            arms = {}
            arms["none"] = relerr(fuse(tv["keeplog"], vh, neg_l, zero_v, empty_c))
            arms["exact"] = relerr(
                fuse(tv["keeplog"], vh, tv["lse_c"].nan_to_num(-1e30), tv["vbar_c"], empty_c)
            )
            l_an = torch.einsum("htd,cd->htc", qv, k_an) * scale + b_an.view(1, 1, -1)
            arms["analytic"] = relerr(
                fuse(tv["keeplog"], vh, l_an, v_an.view(1, 1, n_ch, D).expand_as(zero_v), empty_c)
            )
            # ridge already absorbs `scale` (it fit q -> log D directly, no scale factor)
            l_rg = torch.einsum("htd,cd->htc", qv, k_ridge) + b_ridge.view(1, 1, -1)
            arms["ridge"] = relerr(
                fuse(tv["keeplog"], vh, l_rg, v_agg.view(1, 1, n_ch, D).expand_as(zero_v), empty_c)
            )
            arms["ridge_vstar"] = relerr(
                fuse(tv["keeplog"], vh, l_rg, tv["vbar_c"], empty_c)
            )

            # ---- attribution arms: which half of (mass, direction) carries the ridge gain? ----
            # This is the crux of the go/no-go. `analytic` and `ridge` differ in BOTH k_c/b_c (mass)
            # and v_c (direction), so their gap alone cannot say which one matters -- and the answer
            # decides whether the next step is trainable. Stage A measured that the MASS learns
            # (d/D_E* 0.000 -> 1.05) while the DIRECTION does not (cos 0.538 -> 0.527 over 60
            # steps), so a gain living in the mass is bankable and one living in the direction is
            # not. Two arms isolate it:
            #
            #  * `an_scalar_b`: analytic k_c and v_c, but b_c gets ONE fitted scalar per chunk --
            #    no q-dependence at all, just a constant offset absorbing the Jensen gap that
            #    `log|resid|` ignores (log sum exp >= log n + q.kbar by convexity, so the analytic
            #    b is biased LOW by the within-chunk logit variance).
            #  * `ridge_van`: ridge k_c/b_c with the analytic v_c, so any difference from `ridge`
            #    is attributable to the direction alone.
            l_an_fit = torch.einsum("htd,cd->htc", tf["q"], k_an) * scale + b_an.view(1, 1, -1)
            okf = torch.isfinite(tf["lse_c"]) & ~empty_c.view(1, 1, -1)
            gap = torch.where(okf, tf["lse_c"] - l_an_fit, torch.zeros_like(l_an_fit))
            delta = gap.sum(dim=(0, 1)) / okf.sum(dim=(0, 1)).clamp(min=1).float()  # (n_ch,)
            l_an_d = l_an + delta.view(1, 1, -1)
            arms["an_scalar_b"] = relerr(
                fuse(tv["keeplog"], vh, l_an_d, v_an.view(1, 1, n_ch, D).expand_as(zero_v), empty_c)
            )
            arms["ridge_van"] = relerr(
                fuse(tv["keeplog"], vh, l_rg, v_an.view(1, 1, n_ch, D).expand_as(zero_v), empty_c)
            )

            # ---- the DEPLOYABILITY check on v_c ---------------------------------------------
            # `v_agg` is mass-weighted over the FIT rows, i.e. over real future queries -- but at
            # deployment the CMP is built before any of them exists. So `ridge` is not a deployable
            # arm; it answers "is the target reachable", not "is it computable in time". This arm
            # replaces the query-derived weights with the ROUTER SCORE, which is available the
            # moment the key arrives: v_c = sum_j softmax(s_j/T) v_j over the chunk's residual set.
            # If it recovers `ridge`, the direction is obtainable query-independently and the design
            # is deployable as stated; if it collapses to `ridge_van`, the gain needs future queries
            # and the arm has the addressability problem again, relocated from the row axis to the
            # construction of v_c.
            s_r = scores[hkv, :L1].clone()
            s_r = s_r.masked_fill(~resid, float("-inf")).view(n_ch, chunk_size)
            for T_ in (1.0,):
                wsc = torch.softmax(s_r / T_, dim=-1).nan_to_num(0.0)
                v_sc = torch.einsum("cs,csd->cd", wsc, vh[:L1].view(n_ch, chunk_size, -1))
                arms["ridge_vscore"] = relerr(
                    fuse(tv["keeplog"], vh, l_rg, v_sc.view(1, 1, n_ch, D).expand_as(zero_v), empty_c)
                )

            # ---- mass and direction, separately (the v_norm lesson) -------------------------
            ok = torch.isfinite(tv["lse_c"]) & ~empty_c.view(1, 1, -1)
            mass_err = float((l_rg - tv["lse_c"])[ok].abs().mean()) if ok.any() else float("nan")
            mass_err_and = float((l_an_d - tv["lse_c"])[ok].abs().mean()) if ok.any() else float("nan")
            mass_err_an = float((l_an - tv["lse_c"])[ok].abs().mean()) if ok.any() else float("nan")
            vv = v_agg.view(1, 1, n_ch, D).expand_as(tv["vbar_c"])
            cosv = torch.nn.functional.cosine_similarity(vv, tv["vbar_c"], dim=-1)
            cos_dir = float(cosv[ok].mean()) if ok.any() else float("nan")
            va = v_an.view(1, 1, n_ch, D).expand_as(tv["vbar_c"])
            cos_an = float(
                torch.nn.functional.cosine_similarity(va, tv["vbar_c"], dim=-1)[ok].mean()
            )
            per_head.append(
                dict(
                    **arms,
                    rho=float(tv["rho"].mean()),
                    mass_err_ridge=mass_err,
                    mass_err_analytic=mass_err_an,
                    mass_err_an_scalar_b=mass_err_and,
                    cos_dir_ridge=cos_dir,
                    cos_dir_analytic=cos_an,
                    v_an_norm=float(v_an.norm(dim=-1).mean()),
                    v_sc_norm=float(v_sc.norm(dim=-1).mean()),
                    cos_sc_agg=float(
                        torch.nn.functional.cosine_similarity(v_sc, v_agg, dim=-1).mean()
                    ),
                    v_agg_norm=float(v_agg.norm(dim=-1).mean()),
                    v_true_norm=float(tv["vbar_c"][ok].norm(dim=-1).mean()) if ok.any() else float("nan"),
                    cos_an_agg=float(
                        torch.nn.functional.cosine_similarity(v_an, v_agg, dim=-1).mean()
                    ),
                    resid_per_chunk=float(rc[hkv].float().mean()),
                    resid_p10=float(rc[hkv].float().quantile(0.1)),
                    resid_p90=float(rc[hkv].float().quantile(0.9)),
                )
            )
            del tf, tv
            torch.cuda.empty_cache()

        agg = {kk: float(sum(h[kk] for h in per_head) / len(per_head)) for kk in per_head[0]}
        agg.update(layer=layer, chunk=chunk_size, n_ch=n_ch, n_exact=n_exact, budget=budget)
        out.append(agg)
        print(
            f"  L{layer:2d} chunk{chunk_size:>4}: none {agg['none']:.4f}  "
            f"analytic {agg['analytic']:.4f}  ridge {agg['ridge']:.4f}  "
            f"(v* {agg['ridge_vstar']:.4f})  exact {agg['exact']:.2e}  "
            f"| rho {agg['rho']:.3f} cos {agg['cos_dir_ridge']:.3f} "
            f"dlogD {agg['mass_err_ridge']:.2f} resid/ch {agg['resid_per_chunk']:.0f}",
            flush=True,
        )
    return out


@torch.no_grad()
def main():
    args = parse_args()
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(dev).eval()
    model.requires_grad_(False)

    rck = torch.load(args.router, map_location="cpu", weights_only=False)
    rsd, rcfg = rck["indexer"], rck.get("config", {})
    scorer, kw = press_kwargs_from_checkpoint(rsd, rcfg)
    assert scorer == "scalar", scorer
    press = GQAIndexerPress(
        compression_ratio=1.0 - args.keep_ratio,
        scorer="scalar",
        gate_scale=any("gate_scale" in k for k in rsd),
        n_sink=args.n_sink,
        n_local=0,
        **kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, rsd)

    tcfg = getattr(model.config, "text_config", model.config)
    n_q, n_kv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // n_q)
    group, scale = n_q // n_kv, head_dim**-0.5
    layers_all = get_language_model(model).layers
    WANT = [int(x) for x in args.layers.split(",")]
    print(f"model: {len(layers_all)} layers H={n_q} Hkv={n_kv} D={head_dim} group={group}")
    print(f"C1 keys [0,{args.split})  C2 rows [{args.split},{args.length})  "
          f"budget {int(args.split*args.keep_ratio)} exact slots over C1")

    # ---- capture q/k/v/hidden for all wanted layers in ONE forward ---------------------------
    # ~100 MB/layer at 8K, so 9 layers fit easily and this replaces 9 forwards with 1.
    grab, hid = {}, {}

    def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
        import torch.nn.functional as F

        idx = int(module.layer_idx)
        if idx in WANT:
            grab[idx] = (q.detach(), k.detach(), v.detach(), scaling)
        g = q.shape[1] // k.shape[1]
        o = F.scaled_dot_product_attention(
            q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
        )
        return o.transpose(1, 2).contiguous(), None

    def pre(module, a, kwargs):
        idx = int(getattr(module, "layer_idx", -1))
        if idx in WANT:
            hs = kwargs.get("hidden_states")
            if hs is None and a:
                hs = a[0]
            hid[idx] = hs.detach()
        return None

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("probe_chunk_cmp", impl)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "probe_chunk_cmp"
    for layer in layers_all:
        layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)

    from kvpress.presses.gqa_indexer.data import (
        LongminoConfig,
        TokenizedConfig,
        build_dataloader,
        build_tokenized_dataloader,
    )

    tokenized = f"{MODELS}/../datasets/longmino_256k_tokenized"
    tokenized = "/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_tokenized"
    if os.path.isdir(tokenized):
        loader = build_tokenized_dataloader(
            TokenizedConfig(root=tokenized, seq_len=args.length, take_from="head"),
            batch_size=1, num_workers=0,
        )
    else:
        loader = build_dataloader(
            LongminoConfig(
                root="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered",
                seq_len=args.length, subsets=("2e16",), take_from="head",
            ),
            tok, batch_size=1, num_workers=0,
        )
    docs = []
    for b in loader:
        docs.append(b["input_ids"][:, : args.length].to(dev))
        if len(docs) >= args.docs:
            break

    results = []
    for di, input_ids in enumerate(docs):
        grab.clear()
        hid.clear()
        model(input_ids=input_ids, use_cache=False)
        print(f"\n=== doc {di} ===", flush=True)
        for idx in WANT:
            q, k, v, scaling = grab[idx]
            sc = press.get_indexer(layers_all[idx].self_attn).score_keys(hid[idx])[0].float()
            rows = run_layer(
                q, k, v, sc, args=args,
                scale=(head_dim**-0.5 if scaling is None else float(scaling)),
                group=group, layer=idx,
            )
            for r in rows:
                r["doc"] = di
            results.extend(rows)
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    summarize(results)


def summarize(results):
    chunks = sorted({r["chunk"] for r in results})
    layers = sorted({r["layer"] for r in results})
    print("\n" + "=" * 100)
    print("HELD-OUT (later C2 rows): relative error of the fused output vs dense, mean over docs")
    for c in chunks:
        print(f"\n--- chunk = {c} ---")
        print(f"{'layer':>5} {'none':>8} {'analytic':>9} {'ratio':>7} {'ridge':>8} {'ratio':>7} "
              f"{'v*':>8} {'exact':>9} {'cos':>6} {'dlogD':>7} {'rho':>6} {'res/ch':>7}")
        for L in layers:
            rr = [r for r in results if r["chunk"] == c and r["layer"] == L]
            if not rr:
                continue
            m = {k: sum(r[k] for r in rr) / len(rr) for k in rr[0] if isinstance(rr[0][k], float)}
            print(
                f"{L:>5} {m['none']:>8.4f} {m['analytic']:>9.4f} "
                f"{m['none']/max(m['analytic'],1e-9):>6.2f}x {m['ridge']:>8.4f} "
                f"{m['none']/max(m['ridge'],1e-9):>6.2f}x {m['ridge_vstar']:>8.4f} "
                f"{m['exact']:>9.2e} {m['cos_dir_ridge']:>6.3f} {m['mass_err_ridge']:>7.2f} "
                f"{m['rho']:>6.3f} {m['resid_per_chunk']:>7.0f}"
            )
    print("\nCONTROLS: `exact` must be ~0 at every chunk size; `analytic` and `ridge` must be ~0 at")
    print("chunk=1 (one residual token per slot). If not, stop -- the plumbing is wrong.")
    print("ratio > 1 means the CMP beats pure eviction at MATCHED total budget.")
    print("`v*` uses the true per-row direction: a COVERAGE bound, never achievable. It is here")
    print("only to split 'direction unreachable' from 'mass unreachable'.")


if __name__ == "__main__":
    main()
