#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Probes A/B/C for the *mean-pooled KVzip* idea: is KVzip's second-pass query set compressible?

The idea under test. KVzip scores key ``j`` by replaying the context against its own cache and
taking, per KV head, ``max`` over every (repeat query, GQA group) of the softmax weight landing on
``j``. That ``max`` over ``|C|`` real queries is what costs 2-3x prefill. The proposal is to average
the second segment's embeddings in blocks of ``P`` and replay only ``|C|/P`` pooled tokens, so the
overhead falls to ``~(2/P)x``.

Why it might not work, and what each probe measures
---------------------------------------------------
``<mean_i(q_i), k_j> = mean_i(<q_i, k_j>)``, so pooling silently replaces KVzip's ``max`` over
queries with an ``average`` over them. KVzip's semantics are "keep ``j`` if *any* reconstruction
query needs it"; pooling makes it "keep ``j`` if the *average* query needs it". A key that only one
query in its block depends on -- a needle -- has its signal diluted ~P-fold, while a key that every
query mildly likes is unaffected. The ``max`` *between* blocks survives, so ``P`` is exactly the
knob trading "how many distinct information needs can be expressed" against cost.

* **Probe A** (decisive): top-k overlap at keep 25% and Spearman of each variant's per-head score
  against the true ``P=1`` score, per layer per head. This is the whole question.
* **Probe B** (explains A): within-block pairwise cosine and effective rank of the real queries,
  plus the *argmax-winner concentration* -- how many distinct queries actually win the ``max`` for
  90% of keys. If the repeat queries inside a block are near-collinear, pooling is near-lossless and
  the winner count bounds how small ``P`` can be pushed. The prior for optimism is that "Repeat the
  previous context exactly" makes every repeat token ask nearly the same thing.
* **Probe C** (control): keep ``|C|/P`` *real* tokens as queries instead of averaging, at the same
  count, same positions and same cost. Mean-pool must beat this or pooling itself contributes
  nothing and the cheaper, simpler subsample is the method. This is the control the prefix-indexer
  arm lacked.

Two variants of C are reported. ``subsample`` re-runs the model on the kept tokens only, so it is
genuinely cost-matched to mean-pool. ``sub_oracle`` reads the same query rows out of the *full*
second pass, so its queries saw the whole repeat segment -- not cheap, but it separates "which
queries you keep" from "how the queries are computed" for free.

Implementation notes that matter for correctness
------------------------------------------------
*Explicit ``position_ids``.* Pooling shortens the second segment, so the default positions would
place the pooled queries ``P`` times closer to the context and compress every RoPE relative
distance -- recency decay would silently vanish and the resulting bad score would look like
"pooling does not work". Each pooled block is given the position of the block's **centre**, which
preserves its geometry against the context keys. ``subsample`` keeps its tokens' true positions,
which are those same centres, so C is single-variable against mean-pool: identical count, identical
positions, only "average of the block" vs "one real token from it" differs.

*The softmax denominator.* ``score_kvzip`` normalizes over ``[sink, chunk, repeat-self]`` and only
then slices out the chunk columns, so each row's denominator includes the repeat segment attending
to itself. Pooling changes how many such columns there are, and the ``max`` is taken *across* rows,
so this shifts the ranking for reasons unrelated to the idea. Both are reported: ``self`` matches
KVzip exactly, ``noself`` drops those columns from the denominator (the form a real implementation
should prefer).

*Only ``a_ids`` is pooled.* The instruction tokens ("Repeat the previous context exactly", plus the
chat suffix) carry the reconstruction semantics and are kept verbatim, matching
``KVzipPress.prepare``'s ``[q_ids, suffix_ids, a_ids]`` layout including its ``prev_postfix_size``
carry-over between chunks.

*fp32 throughout.* KVzip itself scores in the model dtype; here both the reference and the variants
use fp32 so bf16 rounding does not enter the overlap numbers being compared.

Usage
-----
    python scratch/probe_meanpool_kvzip.py --model /path/Qwen3-8B --length 8192 --n_docs 2 \\
        --pool 4 16 64 256 --out scratch/probe_meanpool_8k.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_registry import DATASET_REGISTRY  # noqa: E402
from evaluate_sparse import load_cached_dataset  # noqa: E402

from kvpress.utils import get_prerope_query_states  # noqa: E402


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Ordinal ranks along the last axis (ties broken by index -- scores here are continuous)."""
    return x.argsort(dim=-1).argsort(dim=-1).to(torch.float64)


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rank correlation along the last axis."""
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    num = (ra * rb).sum(-1)
    den = ra.norm(dim=-1) * rb.norm(dim=-1)
    return num / den.clamp_min(1e-12)


def topk_overlap(a: torch.Tensor, b: torch.Tensor, keep: float) -> torch.Tensor:
    """Fraction of the true top-k that the variant's top-k also selects, along the last axis."""
    k = max(1, int(round(keep * a.shape[-1])))
    ia = a.topk(k, dim=-1).indices
    ib = b.topk(k, dim=-1).indices
    hits = torch.zeros_like(a, dtype=torch.bool)
    hits.scatter_(-1, ib, True)
    return hits.gather(-1, ia).sum(-1).to(torch.float64) / k


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalized singular-value spectrum) -- a soft count of directions."""
    if x.shape[0] < 2:
        return float(x.shape[0])
    s = torch.linalg.svdvals(x.to(torch.float32))
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


# ----------------------------------------------------------------------------------------------
# the scoring pass
# ----------------------------------------------------------------------------------------------
class KVzipScorer:
    """
    Replays one chunk and computes KVzip's per-head key scores, plus the probe's extra readouts.

    Mirrors :meth:`kvpress.presses.kvzip_press.KVzipPress.score_kvzip` -- same key subsampling,
    same causal mask over the repeat block, same ``amax`` over (GQA group, query) -- so the ``P=1``
    numbers are KVzip's own and every variant is compared against them on identical footing.
    """

    def __init__(self, n_sink: int, query_layers: set[int]):
        self.n_sink = n_sink
        self.query_layers = query_layers
        # Set per chunk / per variant before each forward.
        self.start_idx = 0
        self.end_idx = 0
        self.sub_query_index: dict[int, torch.Tensor] = {}  # P -> query rows, for sub_oracle
        self.collect_extras = False
        # Filled by the hook.
        self.out: dict[str, dict[int, torch.Tensor]] = {}
        self.queries: dict[int, torch.Tensor] = {}
        self.argmax_winner: dict[int, torch.Tensor] = {}

    def reset(self):
        self.out = {}
        self.queries = {}
        self.argmax_winner = {}

    def hook(self, module: nn.Module, args, kwargs, output):
        layer_idx = int(module.layer_idx)
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values") or kwargs.get("past_key_value")
        keys = cache.layers[layer_idx].keys

        bsz, q_len, _ = hidden_states.shape
        n_heads = module.config.num_attention_heads
        n_kv = module.config.num_key_value_heads
        head_dim = module.head_dim
        n_groups = n_heads // n_kv

        # Pre-RoPE queries (this also applies Qwen3's q_norm), then RoPE with whatever
        # position_ids this forward was given -- which is how the pooled variants get the block
        # centres rather than a compressed 0..|C|/P ramp.
        queries = get_prerope_query_states(module, hidden_states)
        cos, sin = kwargs["position_embeddings"]
        if cos.dim() == 4:
            cos, sin = cos.squeeze(1), sin.squeeze(1)
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, n_kv, n_groups, q_len, head_dim).float()

        sink = min(self.n_sink, self.start_idx)
        ctx_len = self.end_idx - self.start_idx
        keys_sub = torch.cat(
            [
                keys[:, :, :sink],  # attention sinks (typically the system prompt)
                keys[:, :, self.start_idx : self.end_idx],  # the chunk being scored
                keys[:, :, -q_len:],  # the replayed chunk itself
            ],
            dim=2,
        ).float()
        keys_sub = keys_sub.unsqueeze(2).transpose(-2, -1)

        logits = torch.matmul(queries, keys_sub) / math.sqrt(head_dim)
        # Causal mask over the repeat block, as in KVzipPress._mask_causal.
        neg = torch.finfo(logits.dtype).min
        causal = torch.full((q_len, q_len), neg, device=logits.device, dtype=logits.dtype).triu(1)
        logits[..., -q_len:, -q_len:] += causal.view(1, 1, 1, q_len, q_len)

        # "self": KVzip's own normalization, which includes the repeat-self columns.
        attn = logits.softmax(-1)[..., sink : sink + ctx_len]
        # "noself": drop those columns from the denominator. Pooling changes how many there are,
        # and the amax runs across rows, so this isolates the idea from a bookkeeping artifact.
        attn_ns = logits[..., : sink + ctx_len].softmax(-1)[..., sink:]

        self.out.setdefault("self", {})[layer_idx] = attn.amax(dim=(2, 3))[0].cpu()
        self.out.setdefault("noself", {})[layer_idx] = attn_ns.amax(dim=(2, 3))[0].cpu()

        if self.collect_extras:
            # Which query row wins the max for each key (after collapsing the GQA group) --
            # the "how many affine pieces does the max actually use" measurement.
            per_query = attn.amax(dim=2)  # (B, Hkv, Sq, ctx)
            self.argmax_winner[layer_idx] = per_query.argmax(dim=2)[0].cpu()
            for P, idx in self.sub_query_index.items():
                # Naming convention shared with the mean-pool/subsample variants: the plain key is
                # KVzip's own normalization and "_noself" is the variant that drops the repeat-self
                # columns from the denominator. main() strips that suffix to pair them up.
                self.out.setdefault(f"sub_oracle_{P}", {})[layer_idx] = (
                    attn[:, :, :, idx, :].amax(dim=(2, 3))[0].cpu()
                )
                self.out.setdefault(f"sub_oracle_{P}_noself", {})[layer_idx] = (
                    attn_ns[:, :, :, idx, :].amax(dim=(2, 3))[0].cpu()
                )
            if layer_idx in self.query_layers:
                # (Hkv, G, Sq, D) -> keep for probe B
                self.queries[layer_idx] = queries[0].transpose(1, 2).cpu()
        return output


def truncate_cache(cache, length: int):
    """Drop everything past ``length`` so the next variant replays against the same context."""
    for layer in cache.layers:
        layer.keys = layer.keys[:, :, :length].contiguous()
        layer.values = layer.values[:, :, :length].contiguous()


def chat_affixes(tokenizer):
    """Reproduce KVzipPress.__call__'s prefix/suffix extraction from the chat template."""
    if tokenizer.chat_template is None:
        return 0, tokenizer.encode("\n", return_tensors="pt", add_special_tokens=False)
    dummy = "dummy context"
    sep = "\n" + "#" * len(dummy)
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": dummy + sep}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    context, suffix_text = templated.split(sep)
    prefix_text = context.split(dummy)[0]
    prefix_len = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
    suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)
    return prefix_len, suffix_ids


def block_bounds(n: int, P: int) -> list[tuple[int, int]]:
    """Blocks of ``P`` over ``n`` positions, last one ragged."""
    return [(s, min(s + P, n)) for s in range(0, n, P)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--dataset", default="ruler")
    ap.add_argument("--length", default="8192")
    ap.add_argument("--n_docs", type=int, default=2)
    ap.add_argument(
        "--task",
        default=None,
        help="RULER task to draw documents from, e.g. niah_single_1. Pooling replaces KVzip's max "
        "over queries with an average, which dilutes a key only one query depends on -- so the "
        "damage is predicted to be task-dependent and needle tasks are where to look for it.",
    )
    ap.add_argument("--pool", type=int, nargs="+", default=[4, 16, 64, 256])
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--n_sink", type=int, default=4)
    ap.add_argument("--prev_postfix_size", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--query_layers", type=int, nargs="+", default=[0, 9, 18, 27, 35])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO_ROOT / "scratch" / "probe_meanpool_kvzip.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", attn_implementation="sdpa"
    ).to(args.device)
    model.eval()
    embed = model.get_input_embeddings()

    prefix_len, suffix_ids = chat_affixes(tokenizer)
    print(f"chat prefix_len={prefix_len} suffix_len={suffix_ids.shape[-1]}", flush=True)

    df = load_cached_dataset(DATASET_REGISTRY[args.dataset], str(args.length)).to_pandas()
    if args.task is not None:
        if "task" not in df.columns:
            raise SystemExit(f"{args.dataset} has no 'task' column; drop --task")
        avail = sorted(df["task"].unique())
        if args.task not in avail:
            raise SystemExit(f"unknown task {args.task!r}; available: {avail}")
        df = df[df["task"] == args.task]
    contexts = df["context"].drop_duplicates().tolist()[: args.n_docs]
    print(
        f"{len(contexts)} document(s) from {args.dataset} @ {args.length}"
        f"{'' if args.task is None else f' task={args.task}'}",
        flush=True,
    )

    scorer = KVzipScorer(args.n_sink, set(args.query_layers))
    handles = [
        layer.self_attn.register_forward_hook(scorer.hook, with_kwargs=True)
        for layer in model.model.layers
    ]

    n_layers = model.config.num_hidden_layers
    variant_names: list[str] = []
    # accumulators: variant -> list over docs of (n_layers, Hkv) tensors
    per_doc: list[dict[str, dict[str, torch.Tensor]]] = []
    probe_b: list[dict] = []

    try:
        for doc_i, context in enumerate(contexts):
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                add_generation_prompt=False,
                tokenize=False,
                enable_thinking=False,
            )
            ctx_ids = tokenizer(templated, return_tensors="pt", add_special_tokens=False).input_ids
            ctx_len = ctx_ids.shape[1]
            print(f"\n=== doc {doc_i}: context {ctx_len} tokens", flush=True)

            cache = DynamicCache()
            with torch.no_grad():
                # Prefill without the hook's scoring work: start_idx == end_idx makes ctx_len 0.
                scorer.start_idx = scorer.end_idx = 0
                scorer.collect_extras = False
                model(input_ids=ctx_ids.to(args.device), past_key_values=cache)

            body = ctx_ids[:, prefix_len:]
            chunks = [
                body[:, s : s + args.chunk_size] for s in range(0, body.shape[1], args.chunk_size)
            ]
            chunks = [c for c in chunks if c.shape[1] > 0]

            # scores[variant][layer] -> (Hkv, ctx_len), assembled chunk by chunk
            scores: dict[str, torch.Tensor] = {}
            winners: dict[int, list] = {}
            doc_probe_b: dict[int, dict] = {}

            start_idx = prefix_len
            for ci, a_ids in enumerate(chunks):
                end_idx = start_idx + a_ids.shape[1]
                scorer.start_idx, scorer.end_idx = start_idx, end_idx

                if ci == 0:
                    prompt = "\n\nRepeat the previous context exactly."
                    q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                else:
                    prompt = "\n\nRepeat the part of the previous context exactly, starting with"
                    q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                    q_ids = torch.cat(
                        [q_ids, chunks[ci - 1][:, -args.prev_postfix_size :]], dim=1
                    )
                head_ids = torch.cat([q_ids, suffix_ids], dim=1)
                n_prefix = head_ids.shape[1]
                len_a = a_ids.shape[1]

                # Block centres, shared by mean-pool (as pooled positions) and by subsample (as the
                # kept token) so the two differ only in averaging vs picking.
                centres = {
                    P: [(lo + hi - 1) // 2 for lo, hi in block_bounds(len_a, P)] for P in args.pool
                }

                # ---- reference pass: the true P=1 score, plus the oracle-subsample readouts
                scorer.reset()
                scorer.collect_extras = True
                scorer.sub_query_index = {
                    P: torch.tensor([n_prefix + c for c in cs], device=args.device)
                    for P, cs in centres.items()
                }
                full_ids = torch.cat([head_ids, a_ids], dim=1).to(args.device)
                pos = torch.arange(ctx_len, ctx_len + full_ids.shape[1], device=args.device)
                with torch.no_grad():
                    model(
                        input_ids=full_ids,
                        past_key_values=cache,
                        position_ids=pos.unsqueeze(0),
                        num_logits_to_keep=1,
                    )
                truncate_cache(cache, ctx_len)

                for base, layers in scorer.out.items():
                    # "self"/"noself" are the reference score; the sub_oracle_* keys already
                    # carry their own normalization suffix.
                    key = f"true_{base}" if base in ("self", "noself") else base
                    tgt = scores.setdefault(
                        key, torch.zeros(n_layers, layers[0].shape[0], ctx_len, dtype=torch.float64)
                    )
                    for li, v in layers.items():
                        tgt[li, :, start_idx:end_idx] = v.to(torch.float64)
                for li, w in scorer.argmax_winner.items():
                    winners.setdefault(li, []).append(w)
                # ---- probe B, on this chunk's real queries
                for li, q in scorer.queries.items():
                    # q: (Hkv, Sq, G, D) after the transpose in the hook -> take the a_ids rows
                    qa = q[:, n_prefix:, :, :]
                    entry = doc_probe_b.setdefault(li, {"cos": {}, "erank": {}})
                    for P in args.pool:
                        cs, es = [], []
                        for lo, hi in block_bounds(qa.shape[1], P)[:64]:  # cap the work
                            blk = qa[:, lo:hi, :, :]
                            if blk.shape[1] < 2:
                                continue
                            # Flatten the GQA group into the vector axis: the max runs over it too.
                            v = blk.reshape(blk.shape[0], blk.shape[1], -1).to(torch.float32)
                            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                            g = v @ v.transpose(1, 2)
                            m = ~torch.eye(g.shape[-1], dtype=torch.bool)
                            cs.append(g[:, m].mean().item())
                            es.append(
                                sum(effective_rank(v[h]) for h in range(v.shape[0])) / v.shape[0]
                            )
                        if cs:
                            entry["cos"].setdefault(P, []).extend(cs)
                            entry["erank"].setdefault(P, []).extend(es)

                # ---- the variants
                scorer.collect_extras = False
                scorer.sub_query_index = {}
                head_emb = embed(head_ids.to(args.device))
                a_emb = embed(a_ids.to(args.device))
                pos_head = torch.arange(ctx_len, ctx_len + n_prefix, device=args.device)

                for P in args.pool:
                    bounds = block_bounds(len_a, P)
                    pooled = torch.stack([a_emb[0, lo:hi].mean(0) for lo, hi in bounds]).unsqueeze(0)
                    pos_blocks = torch.tensor(
                        [ctx_len + n_prefix + c for c in centres[P]], device=args.device
                    )
                    variants = {
                        f"meanpool_{P}": (
                            torch.cat([head_emb, pooled], dim=1),
                            torch.cat([pos_head, pos_blocks]),
                        ),
                        f"subsample_{P}": (
                            torch.cat([head_emb, a_emb[:, centres[P], :]], dim=1),
                            torch.cat([pos_head, pos_blocks]),
                        ),
                    }
                    for vname, (emb_in, pos_in) in variants.items():
                        scorer.reset()
                        with torch.no_grad():
                            model(
                                inputs_embeds=emb_in,
                                past_key_values=cache,
                                position_ids=pos_in.unsqueeze(0),
                                num_logits_to_keep=1,
                            )
                        truncate_cache(cache, ctx_len)
                        for base, layers in scorer.out.items():
                            key = f"{vname}" if base == "self" else f"{vname}_noself"
                            tgt = scores.setdefault(
                                key,
                                torch.zeros(
                                    n_layers, layers[0].shape[0], ctx_len, dtype=torch.float64
                                ),
                            )
                            for li, v in layers.items():
                                tgt[li, :, start_idx:end_idx] = v.to(torch.float64)
                print(f"  chunk {ci}: [{start_idx}, {end_idx}) done", flush=True)
                start_idx = end_idx

            scored_end = start_idx
            # Compare only the positions that were actually scored, and drop the sinks: KVzip
            # overwrites those with 1.0, so including them would inflate every overlap.
            lo, hi = args.n_sink, scored_end
            doc_metrics: dict[str, dict[str, list]] = {}
            for norm in ("self", "noself"):
                ref = scores[f"true_{norm}"][:, :, lo:hi]
                for key, val in scores.items():
                    if key.startswith("true_"):
                        continue
                    is_ns = key.endswith("_noself")
                    if (norm == "noself") != is_ns:
                        continue
                    name = key[: -len("_noself")] if is_ns else key
                    v = val[:, :, lo:hi]
                    doc_metrics.setdefault(norm, {})[name] = {
                        "spearman": spearman(v, ref).tolist(),
                        "overlap": topk_overlap(ref, v, args.keep).tolist(),
                    }
            per_doc.append(doc_metrics)
            variant_names = sorted(doc_metrics["self"].keys())

            # probe B: winner concentration, from the reference pass's argmax
            conc = {}
            for li, ws in winners.items():
                n_win, n_90 = [], []
                for w in ws:  # one per chunk; w: (Hkv, ctx_len_of_chunk)
                    for h in range(w.shape[0]):
                        counts = torch.bincount(w[h].reshape(-1))
                        counts = counts[counts > 0].sort(descending=True).values
                        n_win.append(int(counts.numel()))
                        csum = counts.cumsum(0).to(torch.float64) / counts.sum()
                        n_90.append(int((csum < 0.9).sum().item()) + 1)
                conc[li] = {
                    "n_distinct_winners": sum(n_win) / len(n_win),
                    "n_winners_for_90pct": sum(n_90) / len(n_90),
                }
            probe_b.append(
                {
                    "winner_concentration": conc,
                    "block_stats": {
                        li: {
                            "cos": {P: sum(v) / len(v) for P, v in e["cos"].items()},
                            "erank": {P: sum(v) / len(v) for P, v in e["erank"].items()},
                        }
                        for li, e in doc_probe_b.items()
                    },
                }
            )
            del scores
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    # ------------------------------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------------------------------
    def pooled(norm: str, name: str, metric: str):
        vals = []
        for d in per_doc:
            vals.append(torch.tensor(d[norm][name][metric], dtype=torch.float64))
        return torch.stack(vals)  # (docs, layers, heads)

    summary: dict = {"config": vars(args), "variants": {}}
    for norm in ("self", "noself"):
        print(f"\n================ probe A/C, normalization = {norm} "
              f"(keep {args.keep:.0%}, vs true P=1) ================")
        print(f"{'variant':>22}  {'overlap':>16}  {'spearman':>16}")
        for name in variant_names:
            ov = pooled(norm, name, "overlap")
            sp = pooled(norm, name, "spearman")
            summary["variants"].setdefault(norm, {})[name] = {
                "overlap_mean": ov.mean().item(),
                "overlap_std": ov.std().item(),
                "spearman_mean": sp.mean().item(),
                "spearman_std": sp.std().item(),
                "overlap_per_layer": ov.mean(dim=(0, 2)).tolist(),
                "spearman_per_layer": sp.mean(dim=(0, 2)).tolist(),
            }
            print(
                f"{name:>22}  {ov.mean():.4f} +- {ov.std():.4f}  "
                f"{sp.mean():.4f} +- {sp.std():.4f}"
            )

    summary["probe_b"] = probe_b
    print("\n================ probe B: winner concentration (reference pass) ================")
    print(f"{'layer':>6}  {'distinct winners':>18}  {'winners for 90% of keys':>24}")
    for li in sorted(probe_b[0]["winner_concentration"], key=int):
        d = [p["winner_concentration"][li] for p in probe_b]
        print(
            f"{li:>6}  {sum(x['n_distinct_winners'] for x in d)/len(d):>18.1f}  "
            f"{sum(x['n_winners_for_90pct'] for x in d)/len(d):>24.1f}"
        )

    print("\n================ probe B: within-block query redundancy ================")
    print(f"{'layer':>6}  " + "  ".join(f"P={P}: cos/erank" for P in args.pool))
    for li in sorted(probe_b[0]["block_stats"], key=int):
        cells = []
        for P in args.pool:
            c = [p["block_stats"][li]["cos"].get(P) for p in probe_b]
            e = [p["block_stats"][li]["erank"].get(P) for p in probe_b]
            c = [x for x in c if x is not None]
            e = [x for x in e if x is not None]
            cells.append(f"{sum(c)/len(c):.3f}/{sum(e)/len(e):.1f}" if c else "-")
        print(f"{li:>6}  " + "  ".join(f"{x:>14}" for x in cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
