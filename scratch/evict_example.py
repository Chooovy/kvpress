"""Minimal driver for the hard-eviction decode path, at real geometry.

    python scratch/evict_example.py --ckpt .../final.pt --table .../head_budget_2048_fl512.pt

Prefills each context under the mask path, compresses it, then decodes the whole batch out of the
paged pool. Prints the cache size actually held and the decode throughput.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import GQAIndexerPress, load_indexer_state_dict
from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext
from kvpress.presses.gqa_indexer.train import press_kwargs_from_checkpoint

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--table", default="", help="head_budget table; omit for a uniform --topk")
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--force-sink", type=int, default=4)
    p.add_argument("--force-local", type=int, default=128)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:  # older transformers spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    indexer_sd = ckpt.get("indexer", ckpt)
    # Rebuild the geometry the checkpoint was trained at, exactly as evaluate_sparse.py does:
    # pos_slope is not a parameter, so a wrong value mis-scores while every weight loads cleanly.
    scorer, scorer_kwargs = press_kwargs_from_checkpoint(indexer_sd, ckpt.get("config") or {})
    has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)
    press = GQAIndexerPress(
        compression_ratio=0.0, gate_scale=has_gate, scorer=scorer, **scorer_kwargs
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, indexer_sd, "indexer")
    print(f"loaded scorer={scorer} gate_scale={has_gate}")

    cfg = model.config
    n_layers, n_kv = cfg.num_hidden_layers, cfg.num_key_value_heads
    if args.table:
        payload = torch.load(args.table, map_location="cpu", weights_only=False)
        budgets = torch.as_tensor(payload["table"], dtype=torch.int64)
        fitted = payload.get("topk")
        if fitted is not None and int(fitted) != args.topk:
            raise SystemExit(
                f"table was fitted at topk={fitted} but --topk={args.topk}; its rows sum to "
                "that budget, so mixing them changes the total cache."
            )
    else:
        budgets = torch.full((n_layers, n_kv), args.topk, dtype=torch.int64)

    torch.manual_seed(0)
    vocab = int(cfg.vocab_size)
    contexts = [torch.randint(0, vocab, (1, args.ctx), device=device) for _ in range(args.batch)]
    questions = [torch.randint(0, vocab, (1, 28), device=device) for _ in range(args.batch)]

    sparse_kwargs = dict(
        topk=args.topk, force_sink=args.force_sink, force_local=args.force_local
    )
    with EvictInferenceContext(
        model, press, budgets=budgets, n_sink=args.force_sink,
        n_local=args.force_local, batch_size=args.batch,
    ) as ec:
        t0 = time.time()
        for seq, ctx in enumerate(contexts):
            ec.prefill_and_commit(ctx, seq, sparse_kwargs)
        torch.cuda.synchronize()
        t_prefill = time.time() - t0
        print(f"pool: {ec.pool.summary()}")
        dense = n_layers * args.batch * n_kv * args.ctx * 2 * cfg.head_dim * 2
        print(f"dense would be {dense / 2 ** 30:.2f} GiB "
              f"-> {dense / ec.pool.memory_bytes():.2f}x smaller")
        print(f"prefill+commit: {t_prefill:.1f} s for {args.batch} x {args.ctx} tokens")

        t0 = time.time()
        answers = ec.generate(questions, max_new_tokens=args.new_tokens, eos_token_ids=[-1])
        torch.cuda.synchronize()
        dt = time.time() - t0
        steps = max(len(a) for a in answers)
        print(f"decode: {dt * 1000 / steps:.2f} ms/step for batch {args.batch} "
              f"({dt * 1000 / steps / args.batch:.2f} ms/token/seq), {steps} steps")
        print("peak GiB:", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
