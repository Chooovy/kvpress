# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Would letting ``C'`` see its own KV give the router a way to *bypass* the gated keys?

The proposal under test: cross-replay currently masks ``C'`` out entirely (``k_len = |C|``, see
``cross_replay_e2e.md`` §5), so a replay query can only be answered from ``KV(C)``. Admitting
``C'``'s own KV would match KVzip (whose softmax denominator is ``C + C'``, §7.2.1) and would match
inference more closely, where a real question token *does* attend causally to the question tokens
before it.

The risk this script measures, before any code changes. **The gate applies only to ``C``'s keys.**
If a replay query can predict ``C'[j+1]`` from ``C'[0..j]`` instead, it is doing ordinary causal
language modelling on the original text -- exactly what the frozen backbone was pretrained to do, and
it needs no gated key at all. That is a route to a lower loss that routes *around* the router. It
would not look like a bug: the loss curve would improve. §9 catalogues four bugs of precisely that
shape, and §14 just established that this arm's loss already moves opposite to its RULER score (arm B
has the lower replay loss, 1.18 vs 2.70, and the worse task score, 20.43 vs 44.75).

Three geometries, frozen model, **no gate at all** (so this measures the geometry, not a checkpoint):

* ``cross``     -- query ``j`` sees ``C`` only. The current objective's rectangle.
* ``both``      -- query ``j`` sees all of ``C`` plus ``C'[0..j]``. The proposal.
* ``self_only`` -- query ``j`` sees ``C'[0..j]`` only. The bypass route in isolation, i.e. ordinary
  causal LM on the text.

The diagnostic:

    bypass_share = (L_cross - L_both) / (L_cross - L_self_only)

If ``L_both`` collapses onto ``L_self_only`` (share -> 1), the self path dominates the loss and the
gated keys are nearly irrelevant to it -- the gradient reaching ``s_i`` would be scaled down by
whatever attention mass is left on ``C``, which is the failure mode. If ``L_both`` stays near
``L_cross`` (share -> 0), the concern is unfounded and the change is safe on this axis.

Also reported: ``mass_on_C``, the fraction of each replay query's attention that still lands on
``C``'s keys under ``both``. That is the direct multiplier on ``dL/ds``: at 0.1, the router gets a
tenth of the gradient signal it gets today.
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

NEG = -1e30


def build_mask(mode: str, n: int, device, dtype) -> torch.Tensor:
    """Additive ``(1, 1, N, 2N)`` mask over ``[C ; C']`` for one of the three geometries.

    Column block ``[0, N)`` is ``C`` (the read-only prefilled cache); ``[N, 2N)`` is ``C'``'s own
    keys, which only exist because the replay pass is allowed to append them here.
    """
    mask = torch.full((1, 1, n, 2 * n), NEG, device=device, dtype=dtype)
    q = torch.arange(n, device=device).view(n, 1)
    kp = torch.arange(n, device=device).view(1, n)
    if mode in ("cross", "both"):
        # The full rectangle onto C: every replay query sees every C key. NOT causal -- that is the
        # objective (see cross_replay_e2e.md §5); a causal triangle here would be ordinary LM.
        mask[0, 0, :, :n] = 0.0
    if mode in ("both", "self_only"):
        # Causal within C': query j sees C'[0..j], including itself.
        mask[0, 0, :, n:] = torch.where(kp <= q, 0.0, NEG).to(dtype)
    return mask


@torch.inference_mode()
def run(
    model, ids: torch.Tensor, mode: str, n: int, replay_ids: torch.Tensor | None = None
) -> tuple[float, float]:
    """``(loss, mass_on_C)`` for one geometry.

    ``ids`` is ``C`` (prefilled); ``replay_ids`` is ``C'`` (defaults to ``ids``, the reconstruction
    objective). Replay positions are ``N..2N-1`` so relative distances match the trainer's choice
    (§7.3). The loss is next-token cross-entropy on ``C'``, i.e. row ``j`` predicting ``C'[j+1]``.
    """
    from transformers import DynamicCache

    replay = ids if replay_ids is None else replay_ids
    device = model.device
    cache = DynamicCache()
    # Pass 1: dense, ungated prefill of C. Positions 0..N-1.
    model.model(input_ids=ids, past_key_values=cache)

    # Pass 2: the replay. The cache is allowed to grow to 2N here so that C' keys exist to be
    # unmasked; `cross` then masks them out again, which reproduces the read-only layout (verified
    # equal in the docstring of ReadOnlyCache: loss 0.0e+00, gate grad 5.8e-10).
    dtype = next(model.parameters()).dtype
    mask = build_mask(mode, n, device, dtype)
    pos = torch.arange(n, 2 * n, device=device).unsqueeze(0)
    hidden = model.model(
        input_ids=replay, past_key_values=cache, attention_mask=mask, position_ids=pos
    ).last_hidden_state

    # Next-token loss on C', in chunks so the vocab-sized logits never all exist at once.
    total, count = 0.0, 0
    for start in range(0, n - 1, 512):
        stop = min(start + 512, n - 1)
        logits = model.lm_head(hidden[:, start:stop]).float()
        target = replay[:, start + 1 : stop + 1]
        total += torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
        ).item()
        count += target.numel()
    loss = total / count

    # How much attention mass still lands on C under this geometry, at one representative layer.
    # This is the direct multiplier on dL/ds: the gate only touches C's keys.
    layer_idx = model.config.num_hidden_layers // 2
    captured = {}

    def hook(module, args, kwargs):
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        captured["h"] = h
        return None

    handle = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        cache2 = DynamicCache()
        model.model(input_ids=ids, past_key_values=cache2)
        model.model(input_ids=replay, past_key_values=cache2, attention_mask=mask, position_ids=pos)
    finally:
        handle.remove()

    attn = model.model.layers[layer_idx].self_attn
    h = captured["h"]
    q = attn.q_proj(h)
    n_q = model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", model.config.hidden_size // n_q)
    q = q.view(1, n, n_q, hd).transpose(1, 2)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    k = cache2.layers[layer_idx].keys  # (1, Hkv, 2N, D), post-RoPE
    group = n_q // k.shape[1]
    # Rows sampled across the sequence; RoPE is omitted on q here, so this is an approximation of the
    # mass split, reported as an order of magnitude rather than a precise number.
    rows = torch.linspace(n // 4, n - 1, 16).long()
    kk = k.repeat_interleave(group, dim=1).float()
    logits = torch.einsum("bhqd,bhkd->bhqk", q[:, :, rows].float(), kk) * hd**-0.5
    logits = logits + mask[0, 0, rows].view(1, 1, len(rows), 2 * n)
    w = logits.softmax(-1)
    mass = float(w[..., :n].sum(-1).mean())
    return loss, mass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2048, help="|C| in tokens")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--tasks", default="qa_1,vt,niah_single_2")
    ap.add_argument(
        "--cross-doc",
        action="store_true",
        help="replay an unrelated document instead of C. The null control for a copy shortcut. Must "
        "use a DIFFERENT task: RULER rows within one task share the same haystack (the needle "
        "essays), so 'another row of the same task' is a near-duplicate and not a null at all.",
    )
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/bypass_check.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(args.device).eval()

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    task_names = [t.strip() for t in args.tasks.split(",")]
    rows = []
    for i, task in enumerate(task_names):
        r = df[df["task"] == task].iloc[0]
        ids = tok(r["context"], return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] < args.n:
            continue
        ctx = ids[:, : args.n].to(args.device)
        if args.cross_doc:
            # A row from a DIFFERENT task. Within one RULER task every row is built on the same
            # essay haystack, so 'iloc[1] of the same task' shares most of its tokens with C and
            # would let a copy shortcut survive -- which is not a null control.
            donor_task = task_names[(i + 1) % len(task_names)]
            if donor_task == task:
                continue
            donor = df[df["task"] == donor_task].iloc[0]
            other = tok(donor["context"], return_tensors="pt", add_special_tokens=False).input_ids
            if other.shape[1] < args.n:
                continue
            replay = other[:, : args.n].to(args.device)
            # Report the token overlap so the reader can see the control really is one.
            shared = len(set(ctx[0].tolist()) & set(replay[0].tolist()))
            print(
                f"  [control] C={task} C'={donor_task}, distinct-token overlap "
                f"{shared}/{len(set(ctx[0].tolist()))}",
                flush=True,
            )
        else:
            replay = ctx
        rows.append((task, ctx, replay))
    print(
        f"{len(rows)} documents at N={args.n}"
        f"{' (cross-document replay)' if args.cross_doc else ''}",
        flush=True,
    )

    results = []
    for task, ctx, replay in rows:
        entry = {"task": task}
        for mode in ("cross", "both", "self_only"):
            loss, mass = run(model, ctx, mode, args.n, replay_ids=replay)
            entry[mode] = {"loss": loss, "mass_on_C": mass}
            print(f"  {task:<16s} {mode:<10s} loss={loss:.4f} mass_on_C={mass:.4f}", flush=True)
        lc, lb, ls = (entry[m]["loss"] for m in ("cross", "both", "self_only"))
        entry["bypass_share"] = (lc - lb) / (lc - ls) if abs(lc - ls) > 1e-6 else float("nan")
        results.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"n": args.n, "cross_doc": args.cross_doc, "rows": results}, indent=1)
    )

    print(f"\n{'task':<16s}{'L_cross':>10s}{'L_both':>10s}{'L_self':>10s}{'bypass':>9s}{'massC':>9s}")
    print("-" * 64)
    for e in results:
        print(
            f"{e['task']:<16s}{e['cross']['loss']:>10.4f}{e['both']['loss']:>10.4f}"
            f"{e['self_only']['loss']:>10.4f}{e['bypass_share']:>9.3f}"
            f"{e['both']['mass_on_C']:>9.4f}"
        )
    print(
        f"\nmean bypass_share = {np.nanmean([e['bypass_share'] for e in results]):.3f}"
        f"  (1.0 = the self path fully explains the loss drop, 0.0 = it adds nothing)"
    )
    print(f"mean mass_on_C under 'both' = {np.nanmean([e['both']['mass_on_C'] for e in results]):.4f}"
          "  <- the direct multiplier on dL/ds")


if __name__ == "__main__":
    main()
