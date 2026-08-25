# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run the swap oracle on a TRAINED checkpoint, against the real LM loss.

``HANDOFF_exact_k_subset.md`` §7 calls the swap oracle the single most informative diagnostic, and
``exact_k_diagnostics.py`` implements it -- but the unit test measures it at *init* on a synthetic
MSE target. That answers "is the estimator's gradient informative about this loss geometry", not "is
it still informative after training on real text". This script closes that gap.

    python -m scripts.exact_k_swap_oracle --ckpt .../step200.pt --model .../Qwen3-8B

What it does, per layer:

1. one forward + backward through :class:`~kvpress.presses.gqa_indexer.exact_k_trainer.ExactKIndexerTrainer`
   on real tokens, capturing that layer's chunk subset and score gradient;
2. for each sampled boundary pair, re-runs the **whole model** with that layer's subset forced to the
   swapped set, giving a true ``dL = L(S - i + j) - L(S)`` on the actual LM loss;
3. correlates the two.

Why this is expensive and worth it: step 2 is a full forward per pair, so ``--pairs 24`` over 3 layers
is 75 forwards. That is minutes on an 8B model, which is why it is a script rather than a test.

Reading the result
------------------
Report the **rank** statistics. Raw sign accuracy is depressed by a systematic offset between the
selected and unselected gradient populations (measured at init: ``+2.8e-3`` vs ``-9.2e-4``), which
cannot affect which chunk wins a comparison and therefore cannot affect what the router does. See
:mod:`~kvpress.presses.gqa_indexer.exact_k_diagnostics` for the full argument.

A Spearman near 0 on a trained checkpoint would be the interesting negative result: it would say the
estimator stops tracking marginal utility once the router has committed, which no other diagnostic in
this package would reveal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer import (  # noqa: E402
    ExactKIndexerTrainer,
    exact_k_indexer_training_step,
    load_indexer_state_dict,
    swap_oracle_correlation,
)
from kvpress.presses.gqa_indexer.exact_k_attention import exact_k_chunk_attention  # noqa: E402


def real_tokens(tokenizer, seq_len: int, device: str) -> torch.Tensor:
    """
    Real text, not random ids.

    Random ids give a loss near ``log(vocab)`` for *any* method, so every swap looks equally
    irrelevant and the oracle has no signal to recover -- the run would report a meaningless
    correlation over ``dL_true ~ 0``. The repo's own long-form docs are the most convenient real text
    that is guaranteed present on any checkout.
    """
    docs = [
        REPO_ROOT / "kvpress/presses/gqa_indexer/README.md",
        REPO_ROOT / "kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md",
        REPO_ROOT / "kvpress/presses/gqa_indexer/HANDOFF_exact_k_subset.md",
        REPO_ROOT / "README.md",
    ]
    text = "\n\n".join(p.read_text() for p in docs if p.exists())
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :seq_len]
    if ids.shape[1] < seq_len:
        # Tile rather than pad: padding would make most of the sequence maskable and the swaps
        # trivial, which is the same "no signal" failure as random ids.
        reps = -(-seq_len // ids.shape[1])
        ids = ids.repeat(1, reps)[:, :seq_len]
    return ids.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True, help="an exact-K checkpoint (step*.pt or final.pt)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--layers", type=int, default=3,
        help="layers to probe, spread evenly through the stack. Each costs --pairs full forwards.",
    )
    ap.add_argument("--pairs", type=int, default=24, help="boundary pairs per layer")
    ap.add_argument(
        "--truncate", type=int, default=0,
        help="keep only the first N layers, to fit a probe on one GPU. 0 keeps the whole model. "
        "NOTE a truncated model's loss is not the trained model's loss -- use it to check the "
        "machinery, not to draw conclusions.",
    )
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.truncate:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[: args.truncate]))
        model.config.num_hidden_layers = args.truncate
    model = model.to(device).eval()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    print(f"checkpoint step {payload.get('step')}, config: {json.dumps(cfg, default=str)}")

    # Geometry from the CHECKPOINT, not from defaults: chunk_size / query_block are not parameter
    # shapes, so a mismatch here would mis-score silently -- the trap the trainer records them for.
    press = GQAIndexerPress(
        compression_ratio=0.5, gate_scale=True, chunk_size=int(cfg.get("chunk_size", 64))
    )
    press.post_init_from_model(model)
    indexer_sd = payload.get("indexer", payload)
    if args.truncate:
        # A full-model checkpoint carries weights for layers the truncated model does not have, and
        # load_indexer_state_dict is strict about extras (rightly -- silently dropping tensors is how
        # a geometry mismatch hides). Keep only the layers that exist.
        keep = {}
        for name, tensor in indexer_sd.items():
            parts = name.split(".")
            layer = next((int(p) for p, q in zip(parts[1:], parts) if q == "layers"), None)
            if layer is None or layer < args.truncate:
                keep[name] = tensor
        print(f"--truncate {args.truncate}: keeping {len(keep)} of {len(indexer_sd)} indexer tensors")
        indexer_sd = keep
    load_indexer_state_dict(model, indexer_sd, "indexer")

    trainer = ExactKIndexerTrainer(
        press=press,
        chunk_size=int(cfg.get("chunk_size", 64)),
        query_block=int(cfg.get("query_block", 256)),
        topk_chunk=int(cfg.get("topk_chunk", 8)),
        n_candidate=int(cfg.get("n_candidate", 32)),
        explore_frac=0.0,  # a deterministic pool, so the oracle's re-runs see the same candidates
    )
    trainer.freeze_backbone(model)
    ids = real_tokens(tokenizer, args.seq_len, device)
    print(f"probing {args.layers} layer(s) x {args.pairs} pairs at seq_len={ids.shape[1]}")

    n_layers = model.config.num_hidden_layers
    probe_layers = [round(i * (n_layers - 1) / max(args.layers - 1, 1)) for i in range(args.layers)]

    # Capture each layer's (candidates, selected, score_grad) from ONE backward, then replay.
    captured: dict[int, dict] = {}
    original = exact_k_chunk_attention

    import kvpress.presses.gqa_indexer.exact_k_trainer as trainer_mod

    forced: dict[int, torch.Tensor] = {}
    current_layer = {"idx": -1}

    def patched(q, k, v, candidate_scores, candidates, **kw):
        li = current_layer["idx"]
        if li in forced:
            # Force this layer's subset: +1e4 on the target slots plus hard selection reproduces any
            # subset exactly. Determinism is the whole point -- the forward samples, so an unforced
            # re-run would measure sampling noise instead of the swap.
            kw = {**kw, "hard": True}
            candidate_scores = candidate_scores + 1e4 * forced[li]
        out, stats = original(q, k, v, candidate_scores, candidates, **kw)
        # EVERY layer, not just the probed ones: the replay has to hold all 36 subsets fixed, or the
        # unprobed layers re-sample and their variation swamps the swap being measured. That was the
        # second bug here, caught by the forced-vs-unforced baseline check below.
        if li not in captured:
            captured[li] = {"candidates": candidates, "selected": stats["selected"]}
        return out, stats

    # The trainer calls exact_k_chunk_attention by module-global name, so patching the module
    # attribute is what intercepts it; wrapping routed_forward would not see the call.
    trainer_mod.exact_k_chunk_attention = patched
    real_routed = trainer.routed_forward

    def routed(module, *a, **kw):
        current_layer["idx"] = int(module.layer_idx)
        return real_routed(module, *a, **kw)

    trainer.routed_forward = routed

    try:
        # ONE unforced pass gives both the subset and the gradient. Doing it in two passes -- forcing
        # the subset and re-deriving the gradient -- was the first attempt and is WRONG: the +1e4
        # offset that forces a subset saturates the marginals, so |d mu/d score| collapses from
        # 1.5e-1 to 5e-7 and the "gradient" being correlated is numerical noise. Measured. The tell
        # was `bias +0.00e+00` on every layer.
        scores_by_layer: dict[int, torch.Tensor] = {}
        real_chunk_scores = trainer.chunk_scores

        def chunk_scores_capturing(module, hs, kwargs, k_len):
            out = real_chunk_scores(module, hs, kwargs, k_len)
            li = int(module.layer_idx)
            if li in probe_layers:
                out.retain_grad()
                scores_by_layer[li] = out
            return out

        trainer.chunk_scores = chunk_scores_capturing
        loss = exact_k_indexer_training_step(model, trainer, input_ids=ids)
        baseline = float(loss)
        loss.backward()
        trainer.chunk_scores = real_chunk_scores
        print(f"baseline LM loss {baseline:.6f} (unforced)")

        results = {}
        for li in probe_layers:
            info = captured.get(li)
            scores = scores_by_layer.get(li)
            if info is None or scores is None or scores.grad is None:
                print(f"layer {li}: no subset/gradient captured, skipping")
                continue
            cand = info["candidates"]
            score_grad = scores.grad.gather(-1, cand.clamp(min=0))
            if float(score_grad.abs().mean()) == 0.0:
                print(f"layer {li}: score gradient is identically zero, skipping")
                continue

            def loss_fn(mask, _li=li):
                # Force ALL layers to their captured subsets and vary only the probed one, so the
                # replay is deterministic and dL_true isolates this swap. Forcing is fine here --
                # this path needs only the loss, never a gradient (the +1e4 offset saturates the
                # marginals, which is why the gradient is captured from the unforced pass above).
                forced.clear()
                for other, info_other in captured.items():
                    forced[other] = info_other["selected"]
                forced[_li] = mask
                with torch.no_grad():
                    return float(exact_k_indexer_training_step(model, trainer, input_ids=ids))

            # The forced baseline must reproduce the unforced loss, or the forcing mechanism is not
            # replaying the same subset and every dL_true is measured against the wrong reference.
            forced_base = loss_fn(info["selected"])
            # Tolerance, not equality: the forced replay reproduces the same SUBSETS but the candidate
            # pool is rebuilt from detached scores each pass, and bf16 attention is not associative.
            if abs(forced_base - baseline) > 0.05:
                print(
                    f"layer {li}: forcing changed the loss ({baseline:.4f} -> {forced_base:.4f}); "
                    f"the replay does not reproduce the sampled subset, so dL_true would be measured "
                    f"against the wrong baseline. Skipping."
                )
                forced.pop(li, None)
                continue

            res = swap_oracle_correlation(
                loss_fn, info["selected"], score_grad,
                max_pairs=args.pairs, generator=torch.Generator().manual_seed(li),
            )
            print(f"layer {li:2d}: {res}")
            results[li] = dict(res.__dict__)
            forced.clear()
    finally:
        trainer_mod.exact_k_chunk_attention = original
        trainer.routed_forward = real_routed

    if results:
        spearmans = [r["spearman"] for r in results.values() if r["spearman"] == r["spearman"]]
        centered = [
            r["centered_sign_accuracy"] for r in results.values()
            if r["centered_sign_accuracy"] == r["centered_sign_accuracy"]
        ]
        print("\n=== summary over probed layers ===")
        if spearmans:
            print(f"mean Spearman              {sum(spearmans) / len(spearmans):+.3f}  (chance 0)")
        if centered:
            print(f"mean centered sign accuracy {sum(centered) / len(centered):.3f}  (chance 0.5)")
        print("Read the rank statistics; raw sign accuracy is offset -- see exact_k_diagnostics.")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"ckpt": args.ckpt, "step": payload.get("step"), "layers": results}, indent=2)
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
