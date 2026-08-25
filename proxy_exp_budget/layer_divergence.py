# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Localise, per layer, where the real sparse forward pass departs from the dense one.

:mod:`sparse_vs_dense` establishes the fact that needs explaining: on a ``vt`` row at
``topk=2048``, arm C's sparse forward is *near-exact* against dense (KL 6e-4, top-1 agreement
1.00) while arm B's is destroyed (KL 6.31, agreement 0.00) -- and arm A sits between (KL 0.90).
Yet :mod:`attn_recall`, measuring one layer at a time from **dense** hidden states, found their
supports carry nearly the same attention mass (B 0.719 vs A 0.758, C 0.787 on ``vt``). Those two
findings are only compatible if the per-layer measurement is missing a compounding effect: in the
real run, layer ``L``'s hidden states are already the product of sparse attention at layers
``< L``, so a small selection error can feed a larger one.

This script measures that directly. It captures each layer's attention output in a dense pass and
in a real :class:`SparseAttentionContext` pass over the same input, and reports the relative
deviation per layer:

* ``rel_err`` -- ``||sparse - dense|| / ||dense||`` of the attention output, over the answer-region
  query rows. Flat across depth ⇒ each layer is independently a bit lossy. Growing sharply with
  depth ⇒ compounding, and the single-layer recall number was the wrong instrument.
* ``recall_live`` -- attention-mass recall of the support the indexer selects **from the sparse
  run's own hidden states**, i.e. the selection actually used, rather than the counterfactual one
  measured from dense states. If ``recall_live`` collapses where ``attn_recall``'s number did not,
  the drift in ``h`` is what breaks selection, which is a property of the *scorer's sensitivity to
  its input distribution* rather than of its ranking on clean input.

Both are needed: the first says whether the damage accumulates, the second says whether selection
is what accumulates it.
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
EVAL_DIR = REPO_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from kvpress import SparseAttentionContext  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support  # noqa: E402

from dissect_scores import ARMS  # noqa: E402
from sparse_vs_dense import attach_indexer  # noqa: E402


def _register_capture(model, store_h, store_out, store_qk):
    """Hooks capturing each layer's attention input, output and post-RoPE q/k. Returns removers."""
    handles = []

    def pre_hook(module, args, kwargs):
        idx = getattr(module, "layer_idx", None)
        if idx is None:
            return None
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        store_h[int(idx)] = h.detach()
        return None

    def post_hook(module, args, kwargs, output):
        idx = getattr(module, "layer_idx", None)
        if idx is None:
            return None
        out = output[0] if isinstance(output, tuple) else output
        store_out[int(idx)] = out.detach()
        return None

    for layer in model.model.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(
            layer.self_attn.register_forward_hook(post_hook, with_kwargs=True)
        )
    return handles


def capture_dense(model, ids, n_layers):
    """Dense pass: per-layer attention input ``h``, attention output, and post-RoPE q/k."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    h, out, qk = {}, {}, {}
    impl = "_dense_probe"
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capturing(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        qk[int(module.layer_idx)] = (query.detach(), key.detach(), scaling)
        return sdpa(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)

    gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    had, prev = impl in gm, gm.get(impl)
    ALL_ATTENTION_FUNCTIONS.register(impl, capturing)
    handles = _register_capture(model, h, out, qk)
    prev_impl = model.config._attn_implementation
    model.config._attn_implementation = impl
    try:
        with torch.inference_mode():
            model.model(input_ids=ids)
    finally:
        for x in handles:
            x.remove()
        model.config._attn_implementation = prev_impl
        gm[impl] = prev if had else gm.pop(impl, None)
    return h, out, qk


def capture_sparse(model, press, ids, n_layers, **sparse_kwargs):
    """Real ``SparseAttentionContext`` pass: per-layer attention input and output."""
    h, out = {}, {}
    handles = _register_capture(model, h, out, {})
    try:
        with torch.inference_mode():
            with SparseAttentionContext(model, press, **sparse_kwargs):
                model.model(input_ids=ids)
    finally:
        for x in handles:
            x.remove()
    return h, out


def mass_recall(indexer, h, q, k, scaling, q_rows, topk, force_sink, force_local):
    """Attention-mass recall of the support the indexer picks from ``h``, on ``q_rows``.

    ``h`` is cloned out of inference mode first: ``IndexerNorm`` is a custom autograd Function, and
    an inference tensor cannot be saved for backward even when no gradient is wanted.
    """
    with torch.no_grad():
        h = h.clone()
        q_idx = indexer.project_q(h)
        k_idx = indexer.project_k(h)
        support, valid = streaming_topk_support(
            q_idx, k_idx, topk, mask=None, force_sink=force_sink, force_local=force_local
        )
        n_q, k_len, n_kv = q.shape[1], k.shape[2], k.shape[1]
        group = n_q // n_kv
        dev = support.device
        q_rows = q_rows.to(dev)
        key_pos = torch.arange(k_len, device=dev).view(1, -1)
        got = []
        for kv in range(n_kv):
            idx = support[0, :, q_rows, :][kv].clamp_min(0).long()
            ok = valid[0, :, q_rows, :][kv]
            for g in range(group):
                qh = q[0, kv * group + g][q_rows].float()
                logits = (qh @ k[0, kv].float().T) * scaling
                logits = logits.masked_fill(key_pos > q_rows.view(-1, 1), float("-inf"))
                w = torch.softmax(logits, dim=-1)
                got.append((w.gather(1, idx) * ok).sum(-1).mean().item())
    return float(np.mean(got))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--precision", default="tf32")
    ap.add_argument("--task", default="vt")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--n-rows", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/layer_divergence.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    row = df[df["task"] == args.task].iloc[0]
    prompt = row["context"] + "\n" + row["question"] + str(row["answer_prefix"])
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    n = ids.shape[1]
    print(f"task={args.task} tokens={n}", flush=True)

    h_d, out_d, qk = capture_dense(model, ids, n_layers)
    q_rows = torch.linspace(n // 2, n - 1, args.n_rows).long()

    sparse_kwargs = dict(
        topk=args.topk,
        force_sink=args.force_sink,
        force_local=args.force_local,
        block_k=args.block_k,
        precision=args.precision,
    )

    results = {}
    for name, rel in ARMS.items():
        press, cfg_rec = attach_indexer(model, args.ckpt_root + rel)
        h_s, out_s = capture_sparse(model, press, ids, n_layers, **sparse_kwargs)

        per_layer = []
        for li in range(n_layers):
            d = out_d[li][0][q_rows].float()
            s = out_s[li][0][q_rows].float()
            rel_err = float((s - d).norm() / d.norm())
            # Drift of the attention *input* -- how far the sparse run's own hidden states have
            # moved by this depth. This is what the indexer is really scoring at inference.
            hd, hs_ = h_d[li][0][q_rows].float(), h_s[li][0][q_rows].float()
            h_drift = float((hs_ - hd).norm() / hd.norm())
            q, k, scaling = qk[li]
            indexer = press.get_indexer(model.model.layers[li].self_attn)
            per_layer.append(
                {
                    "layer": li,
                    "rel_err": rel_err,
                    "h_drift": h_drift,
                    # Selection from dense h (clean input) vs from the sparse run's own h (live).
                    "recall_dense_h": mass_recall(
                        indexer, h_d[li], q, k, scaling, q_rows, args.topk,
                        args.force_sink, args.force_local,
                    ),
                    "recall_live_h": mass_recall(
                        indexer, h_s[li], q, k, scaling, q_rows, args.topk,
                        args.force_sink, args.force_local,
                    ),
                }
            )
        results[name] = {"arm": name, "ckpt_config": cfg_rec, "per_layer": per_layer}
        print(
            f"arm {name}: final rel_err={per_layer[-1]['rel_err']:.4f} "
            f"recall_live(last)={per_layer[-1]['recall_live_h']:.4f}",
            flush=True,
        )
        del h_s, out_s
        torch.cuda.empty_cache()

    results["_meta"] = {"task": args.task, "n_tokens": n, "topk": args.topk}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(f"\n{'layer':>5} " + " ".join(f"{a}:rel_err {a}:rec_live {a}:h_drift" for a in ARMS))
    for li in range(0, n_layers, 3):
        cells = ""
        for a in ARMS:
            p = results[a]["per_layer"][li]
            cells += f"{p['rel_err']:>11.4f}{p['recall_live_h']:>11.4f}{p['h_drift']:>11.4f}"
        print(f"{li:>5} {cells}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
