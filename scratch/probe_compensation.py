# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Which compensation route for the evicted branch: linear state, CMP tokens, or a TTT MLP?

All three answer the same question -- given the keys the router threw away, produce something a
future query can read -- so they belong on one ruler. This probe puts them there, plus the two
bounds that decide whether *any* of them can work.

The one bound that dominates everything
---------------------------------------
Every scheme here reads with some function of ``q_t`` (the state, the CMP slots and the TTT weights
are all built before the query arrives, from the document's own KV). So

    **an MLP fitted directly on (q_t, oE*) pairs from this document's own future queries upper
    bounds all of them**

-- it sees the target, which none of them do, and it is nonlinear, which ridge is not. That last
point matters: `memory-direction-not-learnable-from-q` claimed ridge(q) bounded "any design whose
only query-side input is q_t", but ridge is *linear* and a TTT MLP is not, so that claim never
covered this case. ``mlp_q`` closes the hole. If it cannot clear cos 0.8, the family is done
regardless of which parameterization is chosen.

Note what the target actually is::

    oE*(t) = sum_{j in E} softmax(q_t . k_j) v_j

This is an exact, noiseless function of ``q_t``. Nothing is unpredictable in principle -- the
question is only whether it is *compressible* into O(d^2) parameters. Which is why the second bound
is about the shape of the softmax, not about learnability at all.

The second bound: is the payoff recall or averaging?
----------------------------------------------------
``rho`` is heavily skewed (median 0.03-0.17, p90 0.37-0.92), so the compensation is worth having
only on the tail rows. :func:`participation` measures what those rows look like: the participation
ratio ``1 / sum_j p_j^2`` of the softmax over ``E``, bucketed by ``rho``. If the high-``rho`` rows
have effective support of a handful of keys, the payoff is *associative recall* and a 24x-compressed
state is the wrong tool for it (786K numbers of (k,v) association per head against ~32K parameters).
If the support is in the hundreds, the payoff is *averaging* and compression is appropriate.

This is deliberately measured before any arm is fitted, because it bounds the whole family in one
pass and costs nothing.

The arms
--------
========================  ===================================================================
 ``const``                 per-head mean of ``oE*`` over fit rows -- the rank-0 floor
 ``ridge_q``               best LINEAR map ``q -> oE*`` (the old bound, kept for continuity)
 ``mlp_q``                 best nonlinear map ``q -> oE*`` -- **upper bounds every arm below**
 ``lin_state_R``           LESS-style ``(phi(q).H)/(phi(q).z)``, rank R random features
 ``ttt_linear``            least-squares ``W`` fitting ``k_j -> v_j``; read ``W q``
 ``ttt_mlp``               2-layer MLP fitting ``k_j -> v_j`` by SGD; read ``f_W(q)``
 ``cmp_pos_R``             R CMP slots per positional chunk (the arm the last probe killed at R=1)
 ``cmp_kmeans_R``          R CMP slots from k-means over ``(k, v)`` -- ignores position
========================  ===================================================================

``cmp_pos`` vs ``cmp_kmeans`` is the partition test. A positional chunk mixes unrelated ``v_j``, so
its mean is a poor summary however many slots it gets; clustering by content should make each slot's
``v`` coherent. Position is not obviously load-bearing here -- post-RoPE keys do not decohere over a
chunk (measured ``||kbar||/mean||k_j|| = 0.84-0.94``), so dropping the landmark costs less than it
appears to.

Reported the same way for every arm, always separately
------------------------------------------------------
* ``cos(pred, oE*)`` on held-out rows. The threshold is **0.8**, which has now appeared twice with
  unrelated architectures (Stage A: 0.86 wins / 0.53-0.69 loses; chunk-CMP: 0.859 wins / 0.68-0.76
  loses 3-5x), so it is a property of the target rather than of a design.
* fused relative error against ``o_dense``, **given the oracle mass** ``D_E*``. Oracle mass isolates
  direction quality -- and it is the generous choice, since a *correct* mass on a *wrong* direction
  was measured 3-7x worse than plain eviction. If an arm loses here it loses everywhere.

Held out on query rows (fit on the first half of C2, report on the second), because at deployment
every one of these states is built before any future query exists.
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
    )
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--split", type=int, default=6144)
    p.add_argument("--keep-ratio", type=float, default=0.25)
    p.add_argument("--layers", default="0,9,18,27,32,35")
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--row-stride", type=int, default=1)
    p.add_argument("--ranks", default="16,64")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-steps", type=int, default=1500)
    p.add_argument("--state-steps", type=int, default=600)
    p.add_argument("--ttt-steps", type=int, default=1500)
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--out", default="scratch/probe_compensation.json")
    return p.parse_args()


# ==============================================================================================
# bound 1: what does the payoff row look like?
# ==============================================================================================
def participation(p_e, rho):
    """
    Effective support of the softmax over ``E``, bucketed by how much mass eviction removed.

    ``PR = 1 / sum_j p_j^2`` with ``p`` normalized *within* ``E``: the number of evicted keys that
    actually carry the row, independent of ``|E|``. Bucketed by ``rho`` because the compensation is
    only worth having on the high-``rho`` tail -- a family-wide average would be dominated by the
    rows that never needed help.
    """
    p = p_e / p_e.sum(-1, keepdim=True).clamp(min=1e-30)
    pr = 1.0 / p.square().sum(-1).clamp(min=1e-30)
    out = {}
    for lo, hi, name in ((0.0, 0.1, "rho<0.1"), (0.1, 0.3, "0.1-0.3"),
                         (0.3, 0.6, "0.3-0.6"), (0.6, 1.01, "rho>0.6")):
        m = (rho >= lo) & (rho < hi)
        if m.any():
            out[name] = (float(pr[m].median()), float(m.float().mean()))
    out["all"] = (float(pr.median()), 1.0)
    return out


# ==============================================================================================
# the arms
# ==============================================================================================
def ridge_fit(X, Y, lam):
    """Batched ridge ``(G, N, p) x (G, N, m) -> (G, p, m)``, penalty relative to ``tr(X'X)/p``."""
    XtX = torch.einsum("gnp,gnq->gpq", X, X)
    XtY = torch.einsum("gnp,gnm->gpm", X, Y)
    p = X.shape[-1]
    sc = torch.diagonal(XtX, dim1=-2, dim2=-1).sum(-1) / p
    eye = torch.eye(p, device=X.device, dtype=X.dtype).expand_as(XtX)
    return torch.linalg.solve(XtX + (lam * sc.clamp(min=1e-12)).view(-1, 1, 1) * eye, XtY)


def fit_mlp(X, Y, hidden, steps, *, lr=3e-3, seed=0, lam=1e-2, skip=True):
    """
    Per-group 2-layer MLP by Adam, ``(G, N, p) -> (G, N, m)``.

    Grouped as a single ``bmm`` rather than G separate modules: G is 32 query heads and the fits are
    independent, so one batched optimization is both faster and guarantees every head gets the same
    number of steps and the same schedule -- otherwise a head-to-head cos comparison would be
    confounded by optimization effort.
    """
    torch.manual_seed(seed)
    G, N, p = X.shape
    m = Y.shape[-1]
    dev = X.device
    # Standardize the input per group. Without this the arm silently measures the optimizer instead
    # of the hypothesis class: ||q|| reaches ~30 at layer 35, so the preactivations land far out in
    # GELU's saturated tail and mlp_q scored 0.483 -- BELOW ridge_q's 0.940, which is impossible for
    # a strictly larger function class and is the tell that the fit, not the class, was the limit.
    xm = X.mean(1, keepdim=True)
    xs = X.std(1, keepdim=True).clamp(min=1e-6)
    X = (X - xm) / xs
    W1 = (torch.randn(G, p, hidden, device=dev) * p**-0.5).requires_grad_(True)
    b1 = torch.zeros(G, 1, hidden, device=dev, requires_grad=True)
    # Output layer at ZERO and a linear skip seeded with the ridge solution, so the model starts
    # exactly AT the linear optimum and the nonlinearity can only add. Without this the arm is not
    # an upper bound at all: at 65K parameters against a few hundred fit rows the free MLP overfits
    # and lands BELOW ridge on held-out rows, which would understate the very ceiling it exists to
    # establish.
    W2 = torch.zeros(G, hidden, m, device=dev, requires_grad=True)
    b2 = torch.zeros(G, 1, m, device=dev, requires_grad=True)
    ys0 = Y.norm(dim=-1).mean(-1).clamp(min=1e-9).view(G, 1, 1)
    Wsk = (
        ridge_fit(X, Y / ys0, lam) if skip else torch.zeros(G, p, m, device=dev)
    ).clone().requires_grad_(True)
    opt = torch.optim.AdamW([W1, b1, W2, b2, Wsk], lr=lr, weight_decay=lam)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # Normalize the target scale per group so one lr suits every head: ||oE*|| spans two orders of
    # magnitude across depth (|v| 0.32 at layer 0, 30.7 at layer 35).
    ys = Y.norm(dim=-1).mean(-1).clamp(min=1e-9).view(G, 1, 1)
    for _ in range(steps):
        h = torch.nn.functional.gelu(torch.bmm(X, W1) + b1)
        pred = torch.bmm(h, W2) + b2 + torch.bmm(X, Wsk)
        loss = ((pred - Y / ys) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    def apply(Xq):
        with torch.no_grad():
            Xq = (Xq - xm) / xs
            h = torch.nn.functional.gelu(torch.bmm(Xq, W1) + b1)
            return (torch.bmm(h, W2) + b2 + torch.bmm(Xq, Wsk)) * ys

    return apply


def lin_state_read(q_val, k_e, v_e, R, *, seed=0):
    """
    LESS-style linear state: ``n/d = (phi(q).H)/(phi(q).z)``, ``H = sum psi(k) v^T``, ``z = sum psi(k)``.

    ``phi = psi = relu(Wx) + eps`` with a shared random ``W``: non-negativity is what makes ``n/d`` a
    convex combination of value vectors, i.e. a genuine weighted average rather than an arbitrary
    linear map, which is the property the fused softmax relies on. Random features rather than
    trained ones because the point is the *ceiling of the read form* at rank R; a trained phi/psi
    can only interpolate between this and the fitted arms above, both of which are measured.
    """
    torch.manual_seed(seed)
    Hh, Se, D = k_e.shape
    W = torch.randn(D, R, device=k_e.device) * D**-0.5
    psi = torch.nn.functional.relu(torch.einsum("hsd,dr->hsr", k_e, W)) + 1e-3
    Hst = torch.einsum("hsr,hsd->hrd", psi, v_e)
    z = psi.sum(1)  # (H, R)
    phi = torch.nn.functional.relu(torch.einsum("htd,dr->htr", q_val, W)) + 1e-3
    num = torch.einsum("htr,hrd->htd", phi, Hst)
    den = torch.einsum("htr,hr->ht", phi, z).clamp(min=1e-20)
    return num / den.unsqueeze(-1)


def lin_state_learned(q_fit, oE_fit, q_val, k_e, v_e, R, steps, *, lr=3e-3, seed=0):
    """
    The linear state with a **learned** feature map: same normalized read as :func:`lin_state_read`,
    but ``W`` is fitted to the true ``oE*`` on the fit rows instead of being random.

    This is the fair ceiling for two of the three routes at once. A TTT layer whose read is
    normalized IS this arm with a different fitting rule, and the random-feature version is this arm
    at step 0 -- so if the learned one cannot clear the bar, neither can any amount of TTT inner-loop
    tuning. It sees the target and this document's own future queries, so it is an upper bound, not a
    deployable number.
    """
    torch.manual_seed(seed)
    G, Se, D = k_e.shape
    W = (torch.randn(D, R, device=k_e.device) * D**-0.5).requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    ys = oE_fit.norm(dim=-1).mean().clamp(min=1e-9)

    def read(qq):
        psi = torch.nn.functional.relu(torch.einsum("hsd,dr->hsr", k_e, W)) + 1e-3
        Hst = torch.einsum("hsr,hsd->hrd", psi, v_e)
        z = psi.sum(1)
        phi = torch.nn.functional.relu(torch.einsum("htd,dr->htr", qq, W)) + 1e-3
        num = torch.einsum("htr,hrd->htd", phi, Hst)
        den = torch.einsum("htr,hr->ht", phi, z).clamp(min=1e-20)
        return num / den.unsqueeze(-1)

    for _ in range(steps):
        loss = ((read(q_fit) - oE_fit) / ys).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return read(q_val)


def ttt_normalized(q_val, k_e, v_e, R, steps, *, lr=3e-3, seed=0):
    """
    **TTT with the read form the other arms use.** Per-document adaptation, self-supervised.

    The gap this closes: ``lin_state`` (random features, normalized read) is weak, ``lin_learn``
    (features fitted to the true ``oE*``) matches the ceiling, and ``ttt_*`` (per-document fit, but
    an unnormalized ``W q`` read) is catastrophic. Two variables moved at once across those arms, so
    none of them says whether per-document adaptation *by itself* is enough. This arm fixes the read
    to the normalized convex form and keeps the fitting self-supervised, i.e. computable at prefill
    with no future query and no access to the target::

        min_W sum_{j in E} || read_{-j}(k_j) - v_j ||^2

    **Leave-one-out is mandatory, not hygiene.** ``read(k_j)`` with ``j`` still in the state can
    satisfy the objective by routing ``k_j`` to itself, which teaches the features nothing about
    interpolating to an unseen ``q``. Excluding ``j`` is exact and free here -- rank-one removal from
    a sum -- so there is no reason to approximate it.
    """
    torch.manual_seed(seed)
    G, Se, D = k_e.shape
    W = (torch.randn(D, R, device=k_e.device) * D**-0.5).requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    ys = v_e.norm(dim=-1).mean().clamp(min=1e-9)
    for _ in range(steps):
        psi = torch.nn.functional.relu(torch.einsum("hsd,dr->hsr", k_e, W)) + 1e-3
        Hst = torch.einsum("hsr,hsd->hrd", psi, v_e)
        z = psi.sum(1)
        # leave-one-out: drop key j's own rank-one contribution before reading at k_j
        num = torch.einsum("hsr,hrd->hsd", psi, Hst) - psi.square().sum(-1, keepdim=True) * v_e
        den = (torch.einsum("hsr,hr->hs", psi, z) - psi.square().sum(-1)).clamp(min=1e-20)
        loss = ((num / den.unsqueeze(-1) - v_e) / ys).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        psi = torch.nn.functional.relu(torch.einsum("hsd,dr->hsr", k_e, W)) + 1e-3
        Hst = torch.einsum("hsr,hsd->hrd", psi, v_e)
        z = psi.sum(1)
        phi = torch.nn.functional.relu(torch.einsum("htd,dr->htr", q_val, W)) + 1e-3
        return torch.einsum("htr,hrd->htd", phi, Hst) / torch.einsum(
            "htr,hr->ht", phi, z
        ).clamp(min=1e-20).unsqueeze(-1)


def ttt_linear_read(q_val, k_e, v_e, lam):
    """
    TTT at its closed-form fixed point: ``W = argmin sum_j ||W k_j - v_j||^2``, read ``W q``.

    A least-squares associative memory -- what a linear TTT layer converges to given enough inner
    steps, and what DeltaNet approaches. Solved directly so the measurement is not confounded by the
    inner optimizer's schedule.
    """
    Wm = ridge_fit(k_e, v_e, lam)  # (H, D, D)
    return torch.bmm(q_val, Wm)


def cmp_read(q_val, k_slots, v_slots, b_slots, scale):
    """Softmax over R pseudo-KV slots -- the CMP read, in isolation from the retained branch."""
    lg = torch.einsum("htd,hrd->htr", q_val, k_slots) * scale + b_slots.unsqueeze(1)
    return torch.einsum("htr,hrd->htd", torch.softmax(lg, dim=-1), v_slots)


def kmeans(x, R, iters=25, seed=0):
    """Plain Lloyd's, batched over heads. ``(H, S, D) -> (H, R, D)`` centroids + assignment."""
    torch.manual_seed(seed)
    Hh, S, D = x.shape
    idx = torch.stack([torch.randperm(S, device=x.device)[:R] for _ in range(Hh)])
    c = torch.gather(x, 1, idx.unsqueeze(-1).expand(Hh, R, D)).clone()
    a = None
    for _ in range(iters):
        d = torch.cdist(x, c)  # (H, S, R)
        a = d.argmin(-1)
        oh = torch.nn.functional.one_hot(a, R).to(x.dtype)  # (H, S, R)
        cnt = oh.sum(1).clamp(min=1e-9)
        c = torch.einsum("hsr,hsd->hrd", oh, x) / cnt.unsqueeze(-1)
    return c, a


# ==============================================================================================
def run_layer(q, k, v, scores, *, args, scale, group, layer):
    L, L1 = args.length, args.split
    dev = q.device
    Hq, Hkv, D = q.shape[1], k.shape[1], k.shape[-1]
    kf, vf = k[0].float(), v[0].float()
    budget = max(1, int(L1 * args.keep_ratio))
    ranks = [int(r) for r in args.ranks.split(",")]

    s_c1 = scores[:, :L1].clone()
    s_c1[:, : args.n_sink] = float("inf")
    keep_c1 = torch.zeros(Hkv, L1, dtype=torch.bool, device=dev)
    keep_c1.scatter_(1, s_c1.topk(budget, dim=-1).indices, True)

    rows = torch.arange(L1, L, args.row_stride, device=dev)
    nf = rows.numel() // 2
    fit_rows, val_rows = rows[:nf], rows[nf:]

    per_head = []
    part_acc = {}
    for hkv in range(Hkv):
        kh, vh = kf[hkv], vf[hkv]
        ev = ~keep_c1[hkv]  # (L1,) evicted mask over C1
        ei = ev.nonzero(as_tuple=True)[0]
        if ei.numel() < 64:
            continue
        hq0, hq1 = hkv * group, (hkv + 1) * group

        def targets(rr):
            acc = {kk: [] for kk in ("oE", "DE", "od", "oS", "lseS", "rho", "q", "pr")}
            for s in range(0, rr.numel(), 256):
                r = rr[s : s + 256]
                qt = q[0, hq0:hq1, r].float()
                lg = torch.einsum("htd,sd->hts", qt, kh) * scale
                neg = torch.finfo(torch.float32).min
                kmask = torch.cat(
                    [keep_c1[hkv], torch.ones(L - L1, dtype=torch.bool, device=dev)]
                ).view(1, 1, -1)
                causal = torch.arange(L, device=dev).view(1, 1, -1) <= r.view(1, -1, 1)
                lse_d = torch.logsumexp(lg.masked_fill(~causal, neg), -1)
                lse_s = torch.logsumexp(lg.masked_fill(~(causal & kmask), neg), -1)
                acc["od"].append(
                    torch.einsum("hts,sd->htd",
                                 torch.softmax(lg.masked_fill(~causal, neg), -1), vh)
                )
                acc["oS"].append(
                    torch.einsum("hts,sd->htd",
                                 torch.softmax(lg.masked_fill(~(causal & kmask), neg), -1), vh)
                )
                acc["lseS"].append(lse_s)
                acc["rho"].append((1.0 - torch.exp(lse_s - lse_d)).clamp(0, 1))
                # evicted branch, exactly
                lge = lg[:, :, ei]
                pe = torch.softmax(lge, -1)
                acc["oE"].append(torch.einsum("hts,sd->htd", pe, vh[ei]))
                acc["DE"].append(torch.exp(torch.logsumexp(lge, -1)))
                acc["pr"].append(1.0 / pe.square().sum(-1).clamp(min=1e-30))
                acc["q"].append(qt)
                del lg, lge, pe, causal
            return {kk: torch.cat(vv, 1) for kk, vv in acc.items()}

        tf, tv = targets(fit_rows), targets(val_rows)
        # participation, bucketed by rho, from the exact evicted softmax
        prv, rhov = tv["pr"].reshape(-1), tv["rho"].reshape(-1)
        for lo, hi, name in ((0.0, 0.1, "rho<0.1"), (0.1, 0.3, "0.1-0.3"),
                             (0.3, 0.6, "0.3-0.6"), (0.6, 1.01, "rho>0.6")):
            m = (rhov >= lo) & (rhov < hi)
            if m.any():
                part_acc.setdefault(name, []).append((float(prv[m].median()), float(m.float().mean())))
        part_acc.setdefault("all", []).append((float(prv.median()), 1.0))

        k_e, v_e = kh[ei].unsqueeze(0).expand(group, -1, -1), vh[ei].unsqueeze(0).expand(group, -1, -1)
        qv, qf_ = tv["q"], tf["q"]
        preds = {}

        # --- floors / bounds -----------------------------------------------------------------
        preds["const"] = tf["oE"].mean(1, keepdim=True).expand_as(tv["oE"])
        Xf = torch.cat([qf_, torch.ones(group, qf_.shape[1], 1, device=dev)], -1)
        Xv = torch.cat([qv, torch.ones(group, qv.shape[1], 1, device=dev)], -1)
        preds["ridge_q"] = torch.bmm(Xv, ridge_fit(Xf, tf["oE"], args.ridge))
        preds["mlp_q"] = fit_mlp(qf_, tf["oE"], args.mlp_hidden, args.mlp_steps)(qv)

        # --- linear state --------------------------------------------------------------------
        for R in ranks:
            preds[f"lin_state_{R}"] = lin_state_read(qv, k_e, v_e, R)

        # --- TTT ------------------------------------------------------------------------------
        for R in ranks:
            preds[f"lin_learn_{R}"] = lin_state_learned(
                qf_, tf["oE"], qv, k_e, v_e, R, args.state_steps
            )
        for R in ranks:
            preds[f"ttt_norm_{R}"] = ttt_normalized(qv, k_e, v_e, R, args.state_steps)
        preds["ttt_linear"] = ttt_linear_read(qv, k_e, v_e, args.ridge)
        preds["ttt_mlp"] = fit_mlp(k_e, v_e, args.mlp_hidden, args.ttt_steps)(qv)

        # --- CMP: positional chunks vs content clusters ---------------------------------------
        for R in ranks:
            # positional: R slots spread over the chunk grid, i.e. C1 cut into R contiguous spans
            span = (L1 + R - 1) // R
            sl_k, sl_v, sl_b = [], [], []
            for r in range(R):
                m = torch.zeros(L1, dtype=torch.bool, device=dev)
                m[r * span : (r + 1) * span] = True
                m &= ev
                if m.sum() == 0:
                    sl_k.append(torch.zeros(D, device=dev))
                    sl_v.append(torch.zeros(D, device=dev))
                    sl_b.append(torch.tensor(-1e30, device=dev))
                else:
                    sl_k.append(kh[:L1][m].mean(0))
                    sl_v.append(vh[:L1][m].mean(0))
                    sl_b.append(torch.log(m.sum().float()))
            preds[f"cmp_pos_{R}"] = cmp_read(
                qv,
                torch.stack(sl_k).unsqueeze(0).expand(group, -1, -1),
                torch.stack(sl_v).unsqueeze(0).expand(group, -1, -1),
                torch.stack(sl_b).unsqueeze(0).expand(group, -1),
                scale,
            )
            # content clusters: k-means over the evicted keys, ignoring position
            c, a = kmeans(kh[ei].unsqueeze(0), R)
            oh = torch.nn.functional.one_hot(a[0], R).float()
            cnt = oh.sum(0).clamp(min=1e-9)
            vc = torch.einsum("sr,sd->rd", oh, vh[ei]) / cnt.unsqueeze(-1)
            preds[f"cmp_kmeans_{R}"] = cmp_read(
                qv,
                c[0].unsqueeze(0).expand(group, -1, -1),
                vc.unsqueeze(0).expand(group, -1, -1),
                torch.log(cnt).unsqueeze(0).expand(group, -1),
                scale,
            )
            # Temperature on the slot softmax. Layers 27/32 fail for EVERY slot-based read while
            # `const` (which is the T -> inf limit) succeeds there, which is the signature of
            # over-confident routing rather than of bad slots: R centroids carry a much wider logit
            # spread than the hundreds of real keys they replace, so the read collapses onto one
            # centroid where the truth averages over many. T is one scalar per layer and it
            # interpolates the whole way to `const`, so it cannot be worse than the better endpoint.
            for T in (2.0, 4.0, 8.0):
                preds[f"cmp_km{R}_T{T:g}"] = cmp_read(
                    qv,
                    c[0].unsqueeze(0).expand(group, -1, -1),
                    vc.unsqueeze(0).expand(group, -1, -1),
                    torch.log(cnt).unsqueeze(0).expand(group, -1) / T,
                    scale / T,
                )

        # --- score every arm the same way ------------------------------------------------------
        live = tv["rho"] > 1e-3
        row = {}
        rho_hi = tv["rho"] > 0.3

        def fused(oe_dir):
            # oracle mass D_E*, so this isolates DIRECTION quality. Generous by construction: a
            # correct mass on a wrong direction measured 3-7x WORSE than plain eviction.
            DS = torch.exp(tv["lseS"])
            w = tv["DE"] / (DS + tv["DE"]).clamp(min=1e-20)
            return (1 - w).unsqueeze(-1) * tv["oS"] + w.unsqueeze(-1) * oe_dir

        def relerr(o, mask):
            e = (o - tv["od"]).norm(dim=-1) / tv["od"].norm(dim=-1).clamp(min=1e-9)
            return float(e[mask].mean()) if mask.any() else float("nan")

        row["none"] = relerr(tv["oS"], live)
        row["none_hi"] = relerr(tv["oS"], rho_hi)
        row["exact"] = relerr(fused(tv["oE"]), live)
        for name, pr in preds.items():
            row[name] = relerr(fused(pr), live)
            row[name + "_hi"] = relerr(fused(pr), rho_hi)
            row["cos_" + name] = float(
                torch.nn.functional.cosine_similarity(pr, tv["oE"], dim=-1)[live].mean()
            )
        row["rho"] = float(tv["rho"].mean())
        row["frac_hi"] = float(rho_hi.float().mean())
        row["n_evicted"] = float(ei.numel())
        per_head.append(row)
        del tf, tv, preds
        torch.cuda.empty_cache()

    import math as _math

    def _nanmean(key):
        vals = [h[key] for h in per_head if not _math.isnan(h[key])]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    agg = {kk: _nanmean(kk) for kk in per_head[0]}
    agg["layer"] = layer
    agg["participation"] = {
        kk: [sum(a for a, _ in vv) / len(vv), sum(b for _, b in vv) / len(vv)]
        for kk, vv in part_acc.items()
    }
    rmax = ranks[-1]
    print(
        f"  L{layer:2d}: none {agg['none']:.4f} | mlp_q {agg['mlp_q']:.4f} "
        f"(cos {agg['cos_mlp_q']:.3f}) | ttt_lin {agg['ttt_linear']:.4f} "
        f"(cos {agg['cos_ttt_linear']:.3f}) | ttt_mlp cos {agg['cos_ttt_mlp']:.3f} | "
        f"lin{rmax} cos {agg[f'cos_lin_state_{rmax}']:.3f} | "
        f"kmeans{rmax} cos {agg[f'cos_cmp_kmeans_{rmax}']:.3f} "
        f"| pos{rmax} cos {agg[f'cos_cmp_pos_{rmax}']:.3f} | PR {agg['participation']['all'][0]:.0f}",
        flush=True,
    )
    return agg


@torch.no_grad()
def _capture(model, layers_all, WANT, input_ids, grab, hid):
    grab.clear()
    hid.clear()
    model(input_ids=input_ids, use_cache=False)


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
    press = GQAIndexerPress(
        compression_ratio=1.0 - args.keep_ratio, scorer="scalar",
        gate_scale=any("gate_scale" in k for k in rsd), n_sink=args.n_sink, n_local=0, **kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, rsd)

    tcfg = getattr(model.config, "text_config", model.config)
    n_q, n_kv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // n_q)
    group = n_q // n_kv
    layers_all = get_language_model(model).layers
    WANT = [int(x) for x in args.layers.split(",")]

    grab, hid = {}, {}

    def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
        import torch.nn.functional as F

        i = int(module.layer_idx)
        if i in WANT:
            grab[i] = (q.detach(), k.detach(), v.detach(), scaling)
        g = q.shape[1] // k.shape[1]
        o = F.scaled_dot_product_attention(
            q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
        )
        return o.transpose(1, 2).contiguous(), None

    def pre(module, a, kwargs):
        i = int(getattr(module, "layer_idx", -1))
        if i in WANT:
            hs = kwargs.get("hidden_states")
            if hs is None and a:
                hs = a[0]
            hid[i] = hs.detach()
        return None

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("probe_comp", impl)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "probe_comp"
    for layer in layers_all:
        layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)

    from kvpress.presses.gqa_indexer.data import LongminoConfig, build_dataloader

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
    for di, ids in enumerate(docs):
        _capture(model, layers_all, WANT, ids, grab, hid)
        print(f"\n=== doc {di} ===", flush=True)
        for idx in WANT:
            q, k, v, scaling = grab[idx]
            sc = press.get_indexer(layers_all[idx].self_attn).score_keys(hid[idx])[0].float()
            r = run_layer(
                q, k, v, sc, args=args,
                scale=(head_dim**-0.5 if scaling is None else float(scaling)),
                group=group, layer=idx,
            )
            r["doc"] = di
            results.append(r)
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    summarize(results, args)


def summarize(results, args):
    layers = sorted({r["layer"] for r in results})
    ranks = [int(x) for x in args.ranks.split(",")]
    arms = ["const", "ridge_q", "mlp_q", "ttt_linear", "ttt_mlp"]
    arms += [f"ttt_norm_{R}" for R in ranks]
    arms += [f"lin_state_{R}" for R in ranks] + [f"lin_learn_{R}" for R in ranks]
    arms += [f"cmp_pos_{R}" for R in ranks] + [f"cmp_kmeans_{R}" for R in ranks]
    arms += [f"cmp_km{R}_T{T:g}" for R in ranks for T in (2.0, 4.0, 8.0)]

    def M(L):
        rr = [r for r in results if r["layer"] == L]
        return {k: sum(r[k] for r in rr) / len(rr) for k in rr[0] if isinstance(rr[0][k], float)}

    print("\n" + "=" * 108)
    print("BOUND 1 -- participation ratio of the softmax over E (median # of keys carrying the row)")
    print(f"{'layer':>5} " + " ".join(f"{b:>12}" for b in ("rho<0.1", "0.1-0.3", "0.3-0.6", "rho>0.6", "all")))
    for L in layers:
        rr = [r for r in results if r["layer"] == L]
        cells = []
        for b in ("rho<0.1", "0.1-0.3", "0.3-0.6", "rho>0.6", "all"):
            vals = [r["participation"][b][0] for r in rr if b in r["participation"]]
            frac = [r["participation"][b][1] for r in rr if b in r["participation"]]
            cells.append(
                f"{sum(vals)/len(vals):>7.0f}({sum(frac)/len(frac)*100:>3.0f}%)" if vals else f"{'-':>12}"
            )
        print(f"{L:>5} " + " ".join(cells))
    print("  small support on the high-rho rows = associative recall = compression is the wrong tool")

    print("\n" + "=" * 108)
    print("cos(pred, oE*) on held-out rows.  THRESHOLD 0.8 (two prior arms agree on it)")
    print(f"{'layer':>5} " + " ".join(f"{a[:11]:>11}" for a in arms))
    for L in layers:
        m = M(L)
        print(f"{L:>5} " + " ".join(f"{m['cos_'+a]:>11.3f}" for a in arms))

    print("\n" + "=" * 108)
    print("fused rel error GIVEN ORACLE MASS (direction quality only; ratio vs eviction)")
    print(f"{'layer':>5} {'none':>8} " + " ".join(f"{a[:11]:>11}" for a in arms))
    for L in layers:
        m = M(L)
        print(f"{L:>5} {m['none']:>8.4f} " + " ".join(f"{m['none']/max(m[a],1e-9):>10.2f}x" for a in arms))

    print("\n" + "=" * 108)
    print("same, restricted to the rows that matter (rho > 0.3)")
    print(f"{'layer':>5} {'none_hi':>8} {'frac':>6} " + " ".join(f"{a[:11]:>11}" for a in arms))
    for L in layers:
        m = M(L)
        print(
            f"{L:>5} {m['none_hi']:>8.4f} {m['frac_hi']*100:>5.1f}% "
            + " ".join(f"{m['none_hi']/max(m[a+'_hi'],1e-9):>10.2f}x" for a in arms)
        )
    print("\ncontrol: `exact` must be ~0 -- " + ", ".join(f"L{L} {M(L)['exact']:.2e}" for L in layers))
    print("mlp_q sees the TARGET and this document's own future queries: it UPPER BOUNDS every")
    print("other arm. If it cannot clear cos 0.8, no compensation parameterization can.")


if __name__ == "__main__":
    main()
