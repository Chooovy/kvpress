#!/usr/bin/env python
"""
Sanity-check KVzipPress.query_pool before spending an eval on it.

Three things, in the order that would catch a mistake earliest:

1. ``query_pool=1`` reproduces the unmodified press *bit for bit* -- the pooling branch must be
   genuinely inert at its default, or the A/B is not single-variable.
2. ``query_pool=8`` changes the score but keeps it sane (finite, non-degenerate, still ranking).
3. The pooled pass really is ~8x fewer query rows, i.e. the cost claim holds.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kvpress import KVzipPress  # noqa: E402

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"


def run(press, context, question):
    torch.manual_seed(0)
    pipe = run.pipe
    scores = {}
    orig = KVzipPress.compress_post

    def spy(self, model):
        scores["val"] = self.score_val.detach().float().cpu().clone()
        return orig(self, model)

    KVzipPress.compress_post = spy
    try:
        t0 = time.time()
        out = pipe(context, question=question, press=press, max_new_tokens=16)
        dt = time.time() - t0
    finally:
        KVzipPress.compress_post = orig
    return scores["val"], out["answer"], dt


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    run.pipe = pipeline(
        "kv-press-text-generation",
        model=MODEL,
        model_kwargs={"torch_dtype": "auto", "attn_implementation": "sdpa"},
        device="cuda:0",
    )

    # ~3k tokens so the chunking path (chunk_size=2048) is actually exercised.
    context = (
        "The Apollo program was a series of missions run by NASA. " * 180
        + "The secret access code is 74915. "
        + "Lunar samples were returned by six separate crewed landings. " * 180
    )
    n_ctx = len(tok.encode(context))
    question = "\n\nWhat is the secret access code?"
    print(f"context = {n_ctx} tokens\n")

    s_ref, a_ref, t_ref = run(KVzipPress(compression_ratio=0.75), context, question)
    s_one, a_one, _ = run(KVzipPress(compression_ratio=0.75, query_pool=1), context, question)
    s_p8, a_p8, t_p8 = run(KVzipPress(compression_ratio=0.75, query_pool=8), context, question)

    print("1) query_pool=1 vs unmodified press")
    same = torch.equal(s_ref, s_one)
    print(f"   score_val identical: {same}   maxdiff={(s_ref - s_one).abs().max():.3e}")
    print(f"   answers match: {a_ref.strip() == a_one.strip()}")
    if not same:
        raise SystemExit("FAIL: query_pool=1 is not inert -- the A/B would not be single-variable")

    print("\n2) query_pool=8 changes the score, sanely")
    print(f"   shape {tuple(s_p8.shape)}  finite={bool(torch.isfinite(s_p8).all())}")
    print(f"   maxdiff vs reference: {(s_ref - s_p8).abs().max():.4f}")
    print(f"   ref  range [{s_ref.min():.4f}, {s_ref.max():.4f}]  std {s_ref.std():.4f}")
    print(f"   p8   range [{s_p8.min():.4f}, {s_p8.max():.4f}]  std {s_p8.std():.4f}")

    # Top-k agreement at the eval's own budget, pooled over layers and heads.
    k = max(1, int(0.25 * s_ref.shape[-1]))
    ia = s_ref.topk(k, dim=-1).indices
    hits = torch.zeros_like(s_ref, dtype=torch.bool)
    hits.scatter_(-1, s_p8.topk(k, dim=-1).indices, True)
    print(f"   top-25% overlap with reference: {hits.gather(-1, ia).float().mean():.4f}")

    print("\n3) cost")
    print(f"   reference {t_ref:.1f}s   pool8 {t_p8:.1f}s   speedup {t_ref / t_p8:.2f}x")

    print("\n4) answers")
    for name, a in (("kvzip", a_ref), ("pool8", a_p8)):
        print(f"   {name:>6}: {a.strip()[:70]!r}")


if __name__ == "__main__":
    main()
