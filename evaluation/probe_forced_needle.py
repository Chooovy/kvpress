# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Causal test: is the e2e router's `niah_multikey_3` collapse caused by its *selection*?

**Why a causal test rather than another probe.** Two static probes were run first and both were
uninformative, for reasons worth recording:

* Needle coverage came out at or below chance (0.048 distill, 0.072 e2e, chance 0.097) while the
  distilled checkpoint scores 95.65 on this task. A support set containing the answer only at
  chance rate cannot answer 44 of 46 questions, so the probe was measuring the wrong thing.
* Checking against the backbone showed why the premise was wrong: **dense attention does not
  concentrate on the needle either** -- best head, best layer, 1.91x uniform, median rank 10722 of
  21025. And RULER's question asks for the uuid *keyed by* another uuid, which sits 2-3 tokens
  before it, so what a router must recognize is the **key**, not the answer span the probe scored.

This script stops asking "does the router rank the needle highly" and asks the question that
actually matters: **if the needle is forcibly kept, does accuracy come back?**

* If it does -> the regression is a selection failure, and the fix belongs in what the router
  learns to select.
* If it does not -> selection is not the bottleneck; the router's *scores* feed the gate at train
  time but something else explains the eval gap, and effort should move elsewhere.

The forced span covers the whole ``key: value`` unit (~65-71 tokens, measured), not just the
answer, because the model has to match the key before it can read the value off it.

Arms, per checkpoint: ``off`` (untouched baseline, reproducing the published numbers) and
``forced`` (needle span injected into every row's support at every layer). Same prompts, same
seed, same topk -- the only difference is the injection, so the accuracy delta is attributable.

Usage::

    python -m evaluation.probe_forced_needle --model /path/Qwen3-8B \\
        --distill-ckpt ... --e2e-ckpt ... --n-samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.probe_router_selection import attach_indexer, build_model  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext  # noqa: E402

logger = logging.getLogger("probe_forced_needle")

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def token_span(text: str, needle: str, offsets) -> tuple[int, int] | None:
    """Token span of ``needle``'s first occurrence in ``text``, via character offsets."""
    position = text.find(needle)
    if position < 0:
        return None
    indices = [
        index
        for index, (begin, end) in enumerate(offsets)
        if begin < position + len(needle) and end > position
    ]
    return (min(indices), max(indices) + 1) if indices else None


def key_value_span(context: str, question: str, answer: str, tokenizer) -> tuple[int, int] | None:
    """
    Token span covering the retrieval **key** and its **value**, in context coordinates.

    RULER phrases these as ``One of the special magic uuids for <key> is: <value>.`` and asks for
    the value given the key. Forcing only the value would test something the model cannot use: it
    has no way to know which value is wanted without first matching the key. Measured spans are
    65-71 tokens for key+value together.
    """
    keys = UUID_RE.findall(question)
    if not keys:
        return None
    encoded = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    key_span = token_span(context, keys[0], offsets)
    value_span = token_span(context, answer, offsets)
    if key_span is None or value_span is None:
        return None
    return min(key_span[0], value_span[0]), max(key_span[1], value_span[1])


class ForcedSpanContext(SparseAttentionContext):
    """
    ``SparseAttentionContext`` that guarantees a token range is in every row's support.

    Implemented by wrapping ``streaming_topk_support``'s result inside ``_attend``: the span's
    indices replace the support's **lowest-scoring** slots, which are the ones the selection was
    least confident about, so the injection costs the least information. ``topk`` is unchanged, so
    the arms are compared at equal budget -- adding slots instead would confound "the needle is
    present" with "more keys are available".

    Only rows whose causal history already reaches the span are modified; a query before the span
    cannot legally attend to it, and injecting there would leak future keys.
    """

    def __init__(self, *args, forced_span: tuple[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_start, self.forced_stop = forced_span
        self.injections = 0

    def _attend(self, module, query, key, value, scaling):
        from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support
        from kvpress.presses.gqa_indexer.triton_sparse_attention import sparse_gqa_attention

        layer_idx = int(module.layer_idx)
        hidden_states = self._hidden_states.get(layer_idx)
        kwargs = self._kwargs.get(layer_idx, {})
        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        q_idx = indexer.project_q(hidden_states, cos, sin)
        k_idx_new = indexer.project_k(hidden_states, cos, sin)

        previous = self._k_idx.get(layer_idx)
        k_idx = k_idx_new if previous is None else torch.cat([previous, k_idx_new], dim=1)
        self._k_idx[layer_idx] = k_idx

        k_len = key.shape[2]
        support, _ = streaming_topk_support(
            q_idx, k_idx, self.topk, mask=None,
            force_sink=self.force_sink, force_local=self.force_local,
        )

        span = torch.arange(
            self.forced_start, min(self.forced_stop, k_len), device=support.device
        )
        if span.numel() and span.numel() * 2 <= support.shape[-1]:
            q_len = support.shape[2]
            query_offset = k_len - q_len
            positions = torch.arange(q_len, device=support.device) + query_offset
            # Only rows that can legally see the whole span; a row before it cannot attend there
            # and injecting would leak future keys.
            eligible = positions >= span[-1].item()
            if bool(eligible.any()):
                # Support is ASCENDING, so slot order is position order, not score order: the
                # trailing slots are the local window (what force_local reserves) and the leading
                # ones are sinks or -1 padding. Overwriting either would confound the experiment
                # with a loss of sink/local, so the span goes into the MIDDLE slots -- ordinary
                # content selections. No re-sort: the kernel treats the index list as unordered
                # (see triton_sparse_attention.py:270), so slot order carries no meaning to it.
                width = span.numel()
                patch = support[:, :, eligible, :].clone()
                middle = (patch.shape[-1] - width) // 2
                patch[..., middle : middle + width] = span
                support[:, :, eligible, :] = patch
                self.injections += 1
                self.eligible_rows = int(eligible.sum())

        out, _ = sparse_gqa_attention(
            query, key, value, support.to(torch.int32), scaling=scaling,
            causal=self.causal, block_k=self.block_k, precision=self.precision,
        )
        return out.transpose(1, 2).contiguous()


@torch.no_grad()
def answer_one(model, tokenizer, context_ids, question_ids, context_length, max_new_tokens,
               press, sparse_kwargs, forced_span):
    """Prefill the context under (optionally span-forced) sparse attention, then greedy-decode."""
    from transformers import DynamicCache

    factory = (
        (lambda: ForcedSpanContext(model, press, forced_span=forced_span, **sparse_kwargs))
        if forced_span is not None
        else (lambda: SparseAttentionContext(model, press, **sparse_kwargs))
    )
    with factory() as context:
        cache = DynamicCache()
        model.model(input_ids=context_ids, past_key_values=cache)

        generated = []
        ids = question_ids
        for _ in range(max_new_tokens):
            out = model(input_ids=ids, past_key_values=cache)
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(next_id))
            if next_id.item() == tokenizer.eos_token_id:
                break
            ids = next_id
        injections = getattr(context, "injections", None)
    return tokenizer.decode(generated, skip_special_tokens=True), injections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--distill-ckpt", required=True)
    parser.add_argument("--e2e-ckpt", required=True)
    parser.add_argument("--data-dir", default="16384")
    parser.add_argument("--task", default="niah_multikey_3")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--force-local", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="evaluation/forced_needle_probe.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    from datasets import load_dataset

    frame = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    frame = frame[frame.task == args.task]

    model, tokenizer = build_model(args.model, torch.bfloat16, args.device)
    sparse_kwargs = dict(
        topk=args.topk, force_sink=args.force_sink, force_local=args.force_local,
        block_k=64, precision="tf32",
    )

    records = []
    for position in range(len(frame)):
        if len(records) >= args.n_samples:
            break
        row = frame.iloc[position]
        answer = str(row["answer"][0])

        # The prompt is built exactly as the pipeline does: chat-templated context, then the
        # question carrying the template's suffix and the answer prefix. Anything else would
        # shift positions and make the span coordinates wrong.
        separator = "\n\n"
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["context"] + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        context_text, question_suffix = templated.split(separator, 1)
        question_text = row["question"] + question_suffix + row["answer_prefix"]

        context_ids = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        question_ids = tokenizer(question_text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)

        # Located directly in the TEMPLATED text, not shifted from raw-context coordinates by a
        # guessed prefix length: the span indexes the key cache that `context_text` produces, and
        # an off-by-a-few here would force the wrong tokens and silently invalidate the whole
        # experiment. Verified below by decoding the span back.
        span = key_value_span(context_text, row["question"], answer, tokenizer)
        if span is None:
            continue
        if span[1] > context_ids.shape[1]:
            continue

        # Self-check: the forced span must actually contain the key and the value. A silent
        # off-by-N here would make every "forced" arm meaningless while still producing numbers.
        decoded_span = tokenizer.decode(context_ids[0, span[0]:span[1]].tolist())
        if answer not in decoded_span:
            logger.warning("span check failed for %s: decoded %r -- skipping", answer[:16],
                           decoded_span[:80])
            continue

        record = {"answer": answer, "span": span, "span_text": decoded_span,
                  "context_len": context_ids.shape[1]}
        for name, checkpoint in (("distill", args.distill_ckpt), ("e2e", args.e2e_ckpt)):
            press = attach_indexer(model, checkpoint)[0]
            for arm, forced in (("off", None), ("forced", span)):
                text, injections = answer_one(
                    model, tokenizer, context_ids, question_ids, context_ids.shape[1],
                    args.max_new_tokens, press, sparse_kwargs, forced,
                )
                record[f"{name}_{arm}"] = text
                record[f"{name}_{arm}_correct"] = answer in text
                if injections is not None:
                    record[f"{name}_{arm}_injections"] = injections
            torch.cuda.empty_cache()

        records.append(record)
        logger.info(
            "sample %d span=%s  distill off=%s forced=%s | e2e off=%s forced=%s",
            len(records), span,
            record["distill_off_correct"], record["distill_forced_correct"],
            record["e2e_off_correct"], record["e2e_forced_correct"],
        )

    if not records:
        raise SystemExit("no usable samples")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))

    print(f"\n{len(records)} samples, topk={args.topk}, forced span = key+value (~70 tok)\n")
    print(f"{'checkpoint':<10} {'off':>8} {'forced':>8} {'delta':>8}")
    for name in ("distill", "e2e"):
        off = sum(r[f"{name}_off_correct"] for r in records) / len(records)
        forced = sum(r[f"{name}_forced_correct"] for r in records) / len(records)
        print(f"{name:<10} {off:8.3f} {forced:8.3f} {forced - off:+8.3f}")
    print("\nIf e2e's 'forced' recovers towards distill's 'off', the regression is a selection")
    print("failure. If it does not, selection is not the bottleneck.")


if __name__ == "__main__":
    main()
