# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.indexmem.training.data import document_loader, next_batch, parse_schedule
from kvpress.indexmem.training.losses import token_cross_entropy
from kvpress.indexmem.training.train import TrainingConfig


def tokenize_shard(task):
    source, output, model, sequence_length, minimum_tokens = task
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens, doc_ids = [], []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            document = json.loads(line)
            text = document["text"]
            estimate = document.get("metadata", {}).get("len_cl100k_base")
            if estimate is not None and estimate < minimum_tokens:
                continue
            if estimate is None and len(text) < minimum_tokens * 6.5:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= sequence_length:
                tokens.append(ids[:sequence_length])
                doc_ids.append(str(document["id"]))
    array = np.asarray(tokens, dtype=np.uint32).reshape(-1, sequence_length)
    target = Path(output) / source.parent.name / (source.name.split(".")[0] + ".npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "doc_ids": doc_ids,
                "num_docs": len(doc_ids),
                "seq_len": sequence_length,
            }
        )
    )
    return len(doc_ids)


def tokenize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--minimum-tokens", type=int, default=65536)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = [path for subset in args.subsets for path in sorted((Path(args.data_root) / subset).glob("*.json*.gz"))]
    tasks = [(path, args.out, args.model, args.sequence_length, args.minimum_tokens) for path in paths]
    with ProcessPoolExecutor(args.workers) as pool:
        counts = list(pool.map(tokenize_shard, tasks))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(
        json.dumps(
            {
                "seq_len": args.sequence_length,
                "subsets": args.subsets,
                "model": args.model,
                "total_docs": sum(counts),
                "complete": True,
            },
            indent=2,
        )
    )


def cache_document_ids(config, data_world_size):
    selected = set()
    for sequence_length, steps in parse_schedule(config.schedule):
        for rank in range(data_world_size):
            loader = document_loader(
                config.tokenized,
                sequence_length,
                config.subsets,
                batch_size=config.batch_size,
                workers=config.workers,
                seed=config.seed,
                shuffle_buffer=config.shuffle_buffer,
                rank=rank,
                world_size=data_world_size,
                take_from="head",
            )
            iterator = iter(loader)
            batches = steps * config.global_batch_size // (data_world_size * config.batch_size)
            for _ in range(batches):
                batch, iterator = next_batch(loader, iterator)
                selected.update(batch["doc_ids"])
    return selected


@torch.no_grad()
def longce_for_document(model, input_ids, *, truncation=1024, window=1024, gamma=5.0, chunk_size=2048):
    from kvpress.indexmem.ablations.objectives import long_context_weights

    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    long_loss = token_cross_entropy(hidden, input_ids, model.lm_head, chunk_size=chunk_size)
    short_loss = torch.zeros_like(long_loss)
    scored = torch.zeros_like(long_loss, dtype=torch.bool)
    length = input_ids.shape[1]
    for start in range(0, length - truncation, window):
        span = min(window, length - start - truncation)
        ids = input_ids[:, start : start + truncation + span]
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        losses = token_cross_entropy(hidden, ids, model.lm_head, chunk_size=chunk_size)
        position = start + truncation - 1
        short_loss[position : position + span] = losses[truncation - 1 : truncation - 1 + span]
        scored[position : position + span] = True
    return long_context_weights(long_loss, short_loss, scored, gamma)


def token_digest(tokens):
    tokens = np.asarray(tokens, dtype=np.int64)
    return int((tokens * np.arange(1, tokens.size + 1, dtype=np.int64)).sum() % (2**61 - 1))


def token_checksum(tokens):
    return hashlib.blake2b(np.ascontiguousarray(tokens, dtype="<u4").tobytes(), digest_size=8).hexdigest()


def prepare_cache_main(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    if kind == "longce":
        parser.add_argument("--truncation", type=int, default=1024)
        parser.add_argument("--window", type=int, default=1024)
        parser.add_argument("--gamma", type=float, default=5.0)
    args = parser.parse_args()
    config = TrainingConfig(**json.loads(Path(args.config).read_text()))
    if config.take_from != "head":
        parser.error("Cached teacher states and LongCE weights require take_from=head.")
    selected = cache_document_ids(config, args.data_world_size)
    widths = sorted({length for length, _ in parse_schedule(config.schedule)})
    sequence_length = max(widths)
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    paths = [path for subset in config.subsets for path in sorted((Path(config.tokenized) / subset).glob("*.npy"))]
    for source in paths[args.shard_index :: args.shard_count]:
        doc_ids = json.loads(source.with_suffix(".json").read_text())["doc_ids"]
        rows = [row for row, doc_id in enumerate(doc_ids) if doc_id in selected]
        if not rows:
            continue
        array = np.load(source, mmap_mode="r")
        values, ids, digests, checksums = [], [], [], []
        for row in rows:
            tokens = np.array(array[row, :sequence_length], dtype=np.int64)
            input_ids = torch.from_numpy(tokens).unsqueeze(0).to(args.device)
            with torch.no_grad():
                if kind == "teacher":
                    value = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
                else:
                    value = longce_for_document(
                        model,
                        input_ids,
                        truncation=args.truncation,
                        window=args.window,
                        gamma=args.gamma,
                        chunk_size=config.chunk_size,
                    )
            values.append(value.float().cpu().numpy().astype(np.float16))
            ids.append(doc_ids[row])
            digests.append(token_digest(tokens))
            checksums.append([token_checksum(tokens[:width]) for width in widths])
        target = Path(args.out) / source.parent.name / source.stem
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "teacher":
            np.save(target.with_suffix(".npy"), np.stack(values))
            target.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "meta": {
                            "version": 1,
                            "seq_len": sequence_length,
                            "hidden_size": model.config.hidden_size,
                            "model": config.model,
                        },
                        "doc_ids": ids,
                        "digest": digests,
                    }
                )
            )
        else:
            metadata = {
                "version": 3,
                "seq_len": sequence_length,
                "trunc_len": args.truncation,
                "window": args.window,
                "gamma": args.gamma,
                "model": config.model,
                "scored_from": args.truncation - 1,
                "checksum_widths": widths,
            }
            np.savez(
                target.with_suffix(".npz"),
                doc_ids=np.asarray(ids, dtype="U64"),
                weights=np.stack(values),
                checksums=np.asarray(checksums, dtype="U16"),
                meta=np.asarray(json.dumps(metadata)),
            )
        print(f"{target}: {len(ids)} documents", flush=True)
