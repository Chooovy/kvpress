# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Where does each router actually look? Distillation vs end-to-end, on a RULER needle.

Motivation. On `niah_multikey_3` the e2e checkpoint scores 4.35 at 16K against distillation's
95.65, and the failure mode in `predictions.csv` is not a miss but a **partial read**: the first
~17 of 36 answer characters are right and the tail degenerates into `9999`/`0000`, with the model
sometimes annotating its own output as fabricated. That is the signature of a support set that
contains *part* of the needle. This script tests that directly instead of inferring it.

What it measures, per layer and per KV head, for the query rows that generate the answer:

* **rank** of each needle token under the router's score -- where it sits in the full ordering.
* **coverage**: how many of the needle's tokens the actual `streaming_topk_support` selection
  keeps. This is the quantity the failure mode points at, and it is computed with the *same*
  function evaluation uses, not a reimplementation.
* **locality**: the mean normalized distance `(q_pos - k_pos) / q_pos` of the selected keys, which
  is what "the router learned to prefer local context" would mean concretely.

Why coverage and rank are both needed: a needle whose tokens all rank inside topk is retrievable;
one whose tokens rank far outside is invisible; one that is *split* -- head tokens in, tail tokens
out -- reproduces exactly the observed truncation. Only the third is consistent with a right
prefix and a garbage suffix.

Usage::

    python -m evaluation.probe_router_selection \\
        --model /path/Qwen3-8B \\
        --distill-ckpt /path/Qwen-3-8B-gqa_indexer/stage1/step600.pt \\
        --e2e-ckpt /path/Qwen-3-8B-gqa_indexer_e2e/stage1/final.pt \\
        --data-dir 16384 --topk 2048 --n-samples 8

Needs a GPU: the backbone has to run to produce the hidden states the indexer scores from. Nothing
here trains, so it is all under `no_grad`, and the scoring is streamed by
`streaming_topk_support` -- the `(Sq, Sk)` score matrix is never materialized.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer import load_indexer_state_dict  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support  # noqa: E402
from kvpress.presses.gqa_indexer.train import press_kwargs_from_checkpoint  # noqa: E402

logger = logging.getLogger("probe_router_selection")


@dataclass
class NeedleSpan:
    """Where the gold answer sits in the tokenized prompt, and what the query rows are."""

    start: int
    stop: int  # exclusive
    n_tokens: int
    prompt_len: int


def find_needle_span(input_ids: torch.Tensor, tokenizer, answer: str) -> NeedleSpan | None:
    """
    Locate the gold answer's token span in the prompt.

    Searched on the **decoded** text and mapped back through `offset_mapping` rather than by
    matching token ids: the answer appears mid-sentence in the haystack, and a UUID tokenizes
    differently there than it does in isolation (leading-space and digit-grouping differences), so
    an id-level search silently finds nothing. Returns ``None`` when the answer does not occur --
    RULER's distractor items are meant to contain near-misses, so a caller should skip rather than
    treat that as an error.
    """
    ids = input_ids.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    # `find`, not `rfind`: the gold uuid occurs exactly once in the RULER context (verified on the
    # 16K niah_multikey_3 split), and the trailing question/answer_prefix quote the *query* uuid,
    # which is a different string. rfind would be equivalent here but would silently prefer a
    # tail occurrence if the prompt format ever changed to echo the answer.
    position = text.find(answer)
    if position < 0:
        return None

    # Re-encode with offsets to map the character span onto tokens. Encoding the decoded text can
    # differ from `ids` by a token or two at the edges; that is fine here because the span is used
    # to index keys, and a one-token slack at the boundary does not change the conclusion.
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    span = [
        index
        for index, (begin, end) in enumerate(encoded["offset_mapping"])
        if begin < position + len(answer) and end > position
    ]
    if not span:
        return None
    return NeedleSpan(
        start=min(span), stop=max(span) + 1, n_tokens=len(span), prompt_len=len(ids)
    )


def build_model(model_path: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval(), tokenizer


def attach_indexer(model, ckpt_path: str):
    """
    Attach and load one indexer checkpoint, mirroring evaluate_sparse.py exactly.

    The scorer and its geometry come from :func:`press_kwargs_from_checkpoint`, the same helper
    ``evaluate_sparse.py`` uses -- so a probe cannot score with a different router than the
    evaluation does. Without that, a scalar or prefix checkpoint would be loaded into a freshly
    built *pairwise* indexer, which fails on every key; and for a scorer that shares parameter
    names it would be worse, loading a subset and probing a half-initialized router.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("indexer", checkpoint)
    config = checkpoint.get("config") or {}
    has_gate = any(str(key).endswith("gate_scale") for key in state_dict)
    scorer, scorer_kwargs = press_kwargs_from_checkpoint(state_dict, config)
    if scorer in ("scalar", "prefix") and "scalar_pos_slope" not in scorer_kwargs:
        logger.warning(
            "%s records no scalar_pos_slope; using the module default. pos_slope is not a "
            "parameter, so a mismatch against training cannot be caught by weight loading.",
            Path(ckpt_path).name,
        )

    press = GQAIndexerPress(
        compression_ratio=0.0,
        gate_scale=has_gate,
        scorer_attr="indexer",
        scorer=scorer,
        **scorer_kwargs,
    )
    press.post_init_from_model(model, force_reinit=True)
    load_indexer_state_dict(model, state_dict, "indexer")
    logger.info("loaded %s (scorer=%s, %s)", Path(ckpt_path).name, scorer, scorer_kwargs or "{}")
    return press, config


@torch.no_grad()
def capture_hidden_states(model, input_ids: torch.Tensor) -> tuple[list, tuple]:
    """
    Every layer's input hidden states, plus the RoPE tables, in one forward pass.

    Captured with pre-hooks on the attention modules -- the same place
    ``E2EIndexerTrainer._capture_hook`` reads them, so the indexer scores from exactly the tensor
    it would score from at inference. Returned on CPU: 36 layers x (1, 16384, 4096) in bf16 is
    4.8 GiB, which is worth moving off the device before the second checkpoint runs.
    """
    captured: dict[int, torch.Tensor] = {}
    rope: dict[str, tuple] = {}
    handles = []

    def make_hook(index):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            captured[index] = hidden.detach().to("cpu")
            if "position_embeddings" in kwargs and "tables" not in rope:
                cos, sin = kwargs["position_embeddings"]
                rope["tables"] = (cos.detach(), sin.detach())
            return None

        return hook

    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for index, layer in enumerate(language_model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(index), with_kwargs=True))
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[i] for i in sorted(captured)], rope["tables"]


@torch.no_grad()
def probe_one_checkpoint(
    model, press, hidden_states, rope_tables, span: NeedleSpan, *, topk, force_sink,
    force_local, query_rows, device,
) -> dict:
    """
    Rank / coverage / locality of the needle, per layer, for this checkpoint.

    ``query_rows`` are the absolute positions whose support is inspected -- the last few prompt
    positions, which are the rows that actually generate the answer. Scoring every row would be
    ``Sq`` times the work for no extra signal, since the answer is produced from the tail.
    """
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cos, sin = rope_tables
    per_layer = []

    for index, layer in enumerate(language_model.layers):
        indexer = press.get_indexer(layer.self_attn)
        hidden = hidden_states[index].to(device)
        q_cos, q_sin = press.get_rope_tables(indexer, {"position_embeddings": (cos, sin)})

        q_idx = indexer.project_q(hidden, q_cos, q_sin)      # (1, h, S, D)
        k_idx = indexer.project_k(hidden, q_cos, q_sin)      # (1, S, D)

        rows = torch.tensor(query_rows, device=device)
        q_rows = q_idx[:, :, rows, :]

        # Full ordering: needed for rank, and affordable because only a few rows are scored.
        scores = torch.einsum("bhqd,bkd->bhqk", q_rows.float(), k_idx.float())
        causal = torch.arange(k_idx.shape[1], device=device)[None, None, None, :] > rows[None, None, :, None]
        scores = scores.masked_fill(causal, float("-inf"))
        order = scores.argsort(dim=-1, descending=True)
        rank_of = torch.empty_like(order)
        rank_of.scatter_(-1, order, torch.arange(order.shape[-1], device=device)
                         .expand_as(order))
        needle_ranks = rank_of[..., span.start : span.stop].float()   # (1, h, rows, n_needle)

        # The real selection, via the same function evaluation calls. Run on the FULL query axis
        # and then sliced: streaming_topk_support derives causality from query_offset, so feeding
        # it only the trailing rows would place them at positions 0..n and truncate their history.
        support_full, _ = streaming_topk_support(
            q_idx, k_idx, topk, force_sink=force_sink, force_local=force_local
        )
        support = support_full[:, :, rows, :]

        selected = torch.zeros(
            (support.shape[0], support.shape[1], support.shape[2], k_idx.shape[1] + 1),
            dtype=torch.bool, device=device,
        )
        selected.scatter_(-1, support.clamp(min=0) + (support < 0).long() * k_idx.shape[1], True)
        selected = selected[..., : k_idx.shape[1]]
        needle_selected = selected[..., span.start : span.stop]
        coverage = needle_selected.float().mean(-1)                   # fraction of needle kept

        # Locality: mean normalized distance of selected keys from the query.
        valid = support >= 0
        distance = (rows[None, :, None] - support.clamp(min=0)).float()
        normalized = (distance / rows[None, :, None].clamp(min=1).float()).clamp(min=0)
        locality = (normalized * valid).sum(-1) / valid.sum(-1).clamp(min=1)

        per_layer.append({
            "layer": index,
            "needle_rank_mean": float(needle_ranks.mean()),
            "needle_rank_median": float(needle_ranks.median()),
            "needle_rank_max": float(needle_ranks.max()),
            "coverage_mean": float(coverage.mean()),
            "coverage_min": float(coverage.min()),
            "fully_covered_frac": float((coverage >= 1.0).float().mean()),
            "locality_mean": float(locality.mean()),
        })
        del hidden, q_idx, k_idx, scores, order, rank_of, support_full, selected
        torch.cuda.empty_cache()

    return {"per_layer": per_layer}


def summarize(name: str, result: dict, topk: int) -> dict:
    layers = result["per_layer"]
    mean = lambda key: sum(row[key] for row in layers) / len(layers)  # noqa: E731
    return {
        "checkpoint": name,
        "coverage_mean": mean("coverage_mean"),
        "fully_covered_frac": mean("fully_covered_frac"),
        "needle_rank_median": mean("needle_rank_median"),
        "needle_rank_max": mean("needle_rank_max"),
        "locality_mean": mean("locality_mean"),
        "layers_with_partial_coverage": sum(
            1 for row in layers if 0.0 < row["coverage_mean"] < 1.0
        ),
        "layers_with_zero_coverage": sum(1 for row in layers if row["coverage_mean"] == 0.0),
        "topk": topk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--distill-ckpt", required=True)
    parser.add_argument("--e2e-ckpt", required=True)
    parser.add_argument("--data-dir", default="16384", help="RULER length subset")
    parser.add_argument("--task", default="niah_multikey_3", help="the task that regressed most")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--force-local", type=int, default=64)
    parser.add_argument("--query-rows", type=int, default=4, help="trailing rows to inspect")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default="evaluation/router_selection_probe.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit(
            "this probe needs a GPU: the backbone must run to produce the hidden states the "
            "indexer scores from"
        )

    from datasets import load_dataset

    dataset = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    dataset = dataset[dataset["task"] == args.task]
    logger.info("%s: %d rows at %s", args.task, len(dataset), args.data_dir)

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.device)

    records = []
    used = 0
    for _, row in dataset.iterrows():
        if used >= args.n_samples:
            break
        # `to_pandas()` hands back a numpy array here, not a list -- an isinstance check against
        # (list, tuple) misses it and str() then yields "['3c7ed208-...']", brackets and quotes
        # included, which never occurs in the prompt. That silently skipped all 500 rows.
        answers = row["answer"]
        answer = str(answers[0]) if hasattr(answers, "__len__") and not isinstance(answers, str) \
            else str(answers)
        # The prompt the evaluation actually builds: context + question + answer_prefix. Dropping
        # answer_prefix would shift every position by its token count, so the needle span and the
        # query rows would both be measured against the wrong offsets.
        prompt = row["context"] + row["question"] + row["answer_prefix"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)

        span = find_needle_span(input_ids[0], tokenizer, answer)
        if span is None:
            logger.info("skipping a row: gold answer %r not found in the prompt", answer[:24])
            continue
        if span.stop > input_ids.shape[1]:
            continue

        query_rows = list(range(input_ids.shape[1] - args.query_rows, input_ids.shape[1]))
        # Distance matters for reading the result: a needle 4k tokens back and one 20k back are
        # different tests of the same router, and the regression grew with context length.
        distance = input_ids.shape[1] - span.stop
        logger.info(
            "sample %d: prompt %d tok, needle at [%d,%d) = %d tokens, %d back, answer %r",
            used, input_ids.shape[1], span.start, span.stop, span.n_tokens, distance, answer[:24],
        )

        hidden_states, rope_tables = capture_hidden_states(model, input_ids)

        sample = {
            "answer": answer, "span": span.__dict__, "prompt_len": input_ids.shape[1],
            "needle_distance": distance,
        }
        for name, ckpt in (("distill", args.distill_ckpt), ("e2e", args.e2e_ckpt)):
            press, config = attach_indexer(model, ckpt)
            result = probe_one_checkpoint(
                model, press, hidden_states, rope_tables, span,
                topk=args.topk, force_sink=args.force_sink, force_local=args.force_local,
                query_rows=query_rows, device=args.device,
            )
            sample[name] = result
            sample[f"{name}_summary"] = summarize(name, result, args.topk)
            logger.info("  %-8s %s", name, json.dumps(sample[f"{name}_summary"]))
        records.append(sample)
        used += 1
        del hidden_states
        torch.cuda.empty_cache()

    if not records:
        raise SystemExit("no usable samples: the gold answer was never located in a prompt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    logger.info("wrote %s (%d samples)", out, len(records))

    print(f"\n{'':<10} {'coverage':>10} {'full-cov':>10} {'rank med':>10} {'rank max':>10} {'locality':>10}")
    for name in ("distill", "e2e"):
        keys = [row[f"{name}_summary"] for row in records]
        avg = lambda k: sum(row[k] for row in keys) / len(keys)  # noqa: E731
        print(f"{name:<10} {avg('coverage_mean'):10.4f} {avg('fully_covered_frac'):10.4f} "
              f"{avg('needle_rank_median'):10.1f} {avg('needle_rank_max'):10.1f} "
              f"{avg('locality_mean'):10.4f}")
    print(f"\ntopk={args.topk}: a needle token ranked above {args.topk} is selectable; below is not.")

    # Per sample, ordered by how far back the needle sits: if the router is biased towards local
    # context then coverage should fall as the distance grows, and that is the shape to look for
    # rather than the average alone.
    print(f"\n{'dist back':>10} {'d-cov':>8} {'e-cov':>8} {'d-rank':>9} {'e-rank':>9}")
    for row in sorted(records, key=lambda r: r["needle_distance"]):
        print(f"{row['needle_distance']:10d} "
              f"{row['distill_summary']['coverage_mean']:8.3f} "
              f"{row['e2e_summary']['coverage_mean']:8.3f} "
              f"{row['distill_summary']['needle_rank_median']:9.0f} "
              f"{row['e2e_summary']['needle_rank_median']:9.0f}")


if __name__ == "__main__":
    main()
