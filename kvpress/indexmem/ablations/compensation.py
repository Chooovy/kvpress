from __future__ import annotations

import torch


def compare_compensation(queries, keys, values, *, slots=(16, 64), mlp_hidden=256, ttt_steps=1500):
    queries, keys, values = queries.float(), keys.float(), values.float()
    scale = queries.shape[-1] ** -0.5
    exact = torch.softmax(queries @ keys.transpose(-1, -2) * scale, dim=-1) @ values
    outputs = {
        "linear": ttt_linear_read(queries, keys, values, 1e-2),
        "ttt_mlp": fit_mlp(keys, values, mlp_hidden, ttt_steps)(queries),
    }
    for count in slots:
        centers, assignments = kmeans(keys, count)
        membership = torch.nn.functional.one_hot(assignments, count).float()
        populations = membership.sum(1).clamp(min=1e-9)
        centroid_values = torch.einsum("gsr,gsd->grd", membership, values) / populations.unsqueeze(-1)
        outputs[f"centroids_{count}"] = cmp_read(queries, centers, centroid_values, populations.log(), scale)
    return {
        name: {
            "cosine": torch.nn.functional.cosine_similarity(output, exact, dim=-1).mean().item(),
            "relative_l2": ((output - exact).norm(dim=-1) / exact.norm(dim=-1).clamp(min=1e-9)).mean().item(),
        }
        for name, output in outputs.items()
    }


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slots", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--ttt-steps", type=int, default=1500)
    args = parser.parse_args()
    tensors = torch.load(args.tensors, weights_only=True)
    result = compare_compensation(
        tensors["queries"],
        tensors["evicted_keys"],
        tensors["evicted_values"],
        slots=args.slots,
        mlp_hidden=args.mlp_hidden,
        ttt_steps=args.ttt_steps,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def ridge_fit(X, Y, lam):
    XtX = torch.einsum("gnp,gnq->gpq", X, X)
    XtY = torch.einsum("gnp,gnm->gpm", X, Y)
    p = X.shape[-1]
    sc = torch.diagonal(XtX, dim1=-2, dim2=-1).sum(-1) / p
    eye = torch.eye(p, device=X.device, dtype=X.dtype).expand_as(XtX)
    return torch.linalg.solve(XtX + (lam * sc.clamp(min=1e-12)).view(-1, 1, 1) * eye, XtY)


def fit_mlp(X, Y, hidden, steps, *, lr=0.003, seed=0, lam=0.01, skip=True):
    torch.manual_seed(seed)
    (G, N, p) = X.shape
    m = Y.shape[-1]
    dev = X.device
    xm = X.mean(1, keepdim=True)
    xs = X.std(1, keepdim=True).clamp(min=1e-06)
    X = (X - xm) / xs
    W1 = (torch.randn(G, p, hidden, device=dev) * p ** (-0.5)).requires_grad_(True)
    b1 = torch.zeros(G, 1, hidden, device=dev, requires_grad=True)
    W2 = torch.zeros(G, hidden, m, device=dev, requires_grad=True)
    b2 = torch.zeros(G, 1, m, device=dev, requires_grad=True)
    ys0 = Y.norm(dim=-1).mean(-1).clamp(min=1e-09).view(G, 1, 1)
    Wsk = (ridge_fit(X, Y / ys0, lam) if skip else torch.zeros(G, p, m, device=dev)).clone().requires_grad_(True)
    opt = torch.optim.AdamW([W1, b1, W2, b2, Wsk], lr=lr, weight_decay=lam)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ys = Y.norm(dim=-1).mean(-1).clamp(min=1e-09).view(G, 1, 1)
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


def ttt_linear_read(q_val, k_e, v_e, lam):
    Wm = ridge_fit(k_e, v_e, lam)
    return torch.bmm(q_val, Wm)


def cmp_read(q_val, k_slots, v_slots, b_slots, scale):
    lg = torch.einsum("htd,hrd->htr", q_val, k_slots) * scale + b_slots.unsqueeze(1)
    return torch.einsum("htr,hrd->htd", torch.softmax(lg, dim=-1), v_slots)


def kmeans(x, R, iters=25, seed=0):
    torch.manual_seed(seed)
    (Hh, S, D) = x.shape
    idx = torch.stack([torch.randperm(S, device=x.device)[:R] for _ in range(Hh)])
    c = torch.gather(x, 1, idx.unsqueeze(-1).expand(Hh, R, D)).clone()
    a = None
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(-1)
        oh = torch.nn.functional.one_hot(a, R).to(x.dtype)
        cnt = oh.sum(1).clamp(min=1e-09)
        c = torch.einsum("hsr,hsd->hrd", oh, x) / cnt.unsqueeze(-1)
    return (c, a)


if __name__ == "__main__":
    main()
