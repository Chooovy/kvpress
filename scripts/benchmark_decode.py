# SPDX-FileCopyrightText: 2026 Xintong Yang
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scorer-checkpoint", type=Path)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--cache-budget", type=int, default=2048)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--method", choices=["full", "indexmem"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--documents", type=int, default=3)
    args = p.parse_args()
    if min(args.length, args.batch, args.steps, args.repeats, args.documents) < 1 or args.warmups < 0:
        p.error("length, batch, steps, repeats and documents must be positive; warmups must be nonnegative")
    if args.length < 2:
        p.error("length must include a prefill token and the first decode input")
    return args


def main():
    a = arguments()
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache

    from kvpress.indexmem import IndexMemConfig, IndexMemInferenceContext
    from kvpress.indexmem.checkpoint import load_scorer_checkpoint
    from kvpress.indexmem.inference.generation import sample_token
    from kvpress.indexmem.inference.paged_cache import PAGE_BLOCK

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.device_count() == 1, "Bind one GPU UUID per process"
    base = {
        "model": a.model,
        "length": a.length,
        "batch": a.batch,
        "method": a.method,
        "steps": a.steps,
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "pid": os.getpid(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError("Use a new output file for retries: " + str(a.output))

    def emit(event, **kw):
        row = {**base, "event": event, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw}
        with a.output.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    phase = "load"
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        assert model.config._attn_implementation == "flash_attention_2"
        if a.method == "indexmem":
            checkpoint = load_scorer_checkpoint(model, a.scorer_checkpoint)
            emit("scorer_loaded", scorer_config=checkpoint["scorer_config"], tensor_count=len(checkpoint["scorers"]))
            del checkpoint
        finite_flags = []

        def forward(token, cache, pos, ec):
            out = model(
                input_ids=token.reshape(a.batch, 1),
                past_key_values=cache,
                position_ids=pos.view(1, 1).expand(a.batch, 1),
                cache_position=pos,
                use_cache=True,
                logits_to_keep=1,
            )
            if ec is not None:
                ec.finish_step()
            finite_flags.append(torch.isfinite(out.logits[:, -1]).all())
            return sample_token(out.logits[:, -1], None)

        docs = json.loads(a.prompts.read_text())[: a.documents]
        if len(docs) != a.documents or any((len(doc["token_ids"]) < a.length for doc in docs)):
            raise ValueError("prompts.json does not contain the requested documents/length")
        for doc_idx, doc in enumerate(docs):
            phase = "prefill"
            ids = torch.tensor(doc["token_ids"][: a.length], dtype=torch.long, device="cuda").unsqueeze(0)
            assert ids.shape[1] == a.length
            cm = (
                IndexMemInferenceContext(model, IndexMemConfig(cache_budget=a.cache_budget), batch_size=a.batch)
                if a.method == "indexmem"
                else contextlib.nullcontext(None)
            )
            with torch.inference_mode(), cm as ec:
                if ec is not None:
                    ec.prefill_and_commit(ids[:, :-1], 0)
                    phase = "replicate"
                    for seq in range(1, a.batch):
                        ec.replicate(0, seq)
                    ec.activate()
                    cache = ec.new_cache()
                    token = forward(ids[:, -1].repeat(a.batch), cache, torch.tensor([a.length - 1], device="cuda"), ec)
                    assert torch.all(
                        ec.pool.budgets.sum(-1).cpu() + model.config.num_key_value_heads * 64
                        == model.config.num_key_value_heads * a.cache_budget
                    )
                    assert int(ec.pool.budgets.min()) + 64 >= 512
                    assert torch.equal(ec.pool.filled.to(torch.int64), ec.pool.row_budget)
                    valid = (
                        torch.arange(ec.pool.block_table.shape[1], device="cuda")[None, :]
                        < ((ec.pool.row_budget + PAGE_BLOCK - 1) // PAGE_BLOCK)[:, None]
                    )
                    blocks = ec.pool.block_table[valid]
                    assert torch.unique(blocks).numel() == blocks.numel(), "Sequences must own disjoint physical pages"
                    assert torch.isfinite(ec.cmp.centroid_keys).all() and torch.isfinite(ec.cmp.centroid_values).all()
                    phase = "snapshot"
                    snapshots = [
                        {k: v.detach().cpu().clone() for (k, v) in vars(obj).items() if isinstance(v, torch.Tensor)}
                        for obj in (ec.pool, ec.cmp)
                    ]
                    expected_pop = float(ec.cmp.cluster_counts.sum())
                    emit(
                        "state_validated",
                        document=doc_idx,
                        id=doc.get("id", doc_idx),
                        budgets=ec.pool.budgets.cpu().tolist(),
                        allocated_page_bytes=ec.pool.memory_bytes(),
                        cmp_population=expected_pop,
                        snapshot_bytes=sum((v.numel() * v.element_size() for s in snapshots for v in s.values())),
                        disjoint_pages=True,
                    )
                    ec._hidden.clear()
                    ec._pending.clear()
                else:
                    cache = DynamicCache()
                    model.model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True)
                    out = model(
                        input_ids=ids[:, -1:],
                        past_key_values=cache,
                        position_ids=torch.tensor([[a.length - 1]], device="cuda"),
                        cache_position=torch.tensor([a.length - 1], device="cuda"),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    finite_flags.append(torch.isfinite(out.logits[:, -1]).all())
                    token = sample_token(out.logits[:, -1], None).repeat(a.batch)
                    del out
                    phase = "snapshot"
                    snapshots = [(layer.keys.cpu(), layer.values.cpu()) for layer in cache.layers]
                    emit(
                        "state_validated",
                        document=doc_idx,
                        id=doc.get("id", doc_idx),
                        retained_tokens=snapshots[0][0].shape[2],
                        snapshot_bytes=sum((x.numel() * x.element_size() for pair in snapshots for x in pair)),
                    )
                assert torch.stack(finite_flags).all()
                seed = token.detach().cpu()
                del cache, token, ids
                gc.collect()
                torch.cuda.empty_cache()
                positions = torch.arange(a.length, a.length + a.steps, device="cuda")
                reference_trace = None
                for rep in range(-a.warmups, a.repeats):
                    phase = "restore"
                    if ec is not None:
                        ec._hidden.clear()
                        ec._pending.clear()
                        for obj, snap in zip((ec.pool, ec.cmp), snapshots):
                            for key, value in snap.items():
                                getattr(obj, key).copy_(value)
                        cache = ec.new_cache()
                        assert torch.all(ec.pool.seen == a.length)
                        assert float(ec.cmp.cluster_counts.sum()) == expected_pop
                    else:
                        cache = DynamicCache()
                        for layer_idx, (k, v) in enumerate(snapshots):
                            (keys, values) = (
                                k.to("cuda").repeat(a.batch, 1, 1, 1),
                                v.to("cuda").repeat(a.batch, 1, 1, 1),
                            )
                            assert keys.is_contiguous() and keys.stride(0) > 0
                            cache.update(keys, values, layer_idx)
                        del keys, values
                    token = seed.to("cuda")
                    trace = []
                    finite_flags.clear()
                    torch.cuda.synchronize()
                    initial_allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    phase = "warmup" if rep < 0 else "decode"
                    start = time.perf_counter()
                    for step in range(a.steps):
                        token = forward(token, cache, positions[step : step + 1], ec)
                        trace.append(token)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    assert torch.stack(finite_flags).all(), "Non-finite logits invalidate the measurement"
                    output = torch.stack(trace).cpu().numpy().tobytes()
                    trace_sha = hashlib.sha256(output).hexdigest()
                    if reference_trace is None:
                        reference_trace = trace_sha
                    assert trace_sha == reference_trace, "Reset must reproduce the same token trace"
                    if ec is not None:
                        assert torch.all(ec.pool.seen == a.length + a.steps)
                        assert torch.equal(ec.pool.filled.to(torch.int64), ec.pool.row_budget)
                        pop_delta = float(ec.cmp.cluster_counts.sum()) - expected_pop
                        assert pop_delta == a.steps * ec.pool.rows, (pop_delta, a.steps * ec.pool.rows)
                    else:
                        retained = a.length + a.steps
                        assert all((layer.keys.shape[2] == retained for layer in cache.layers))
                        pop_delta = None
                    emit(
                        "warmup" if rep < 0 else "measurement",
                        document=doc_idx,
                        id=doc.get("id", doc_idx),
                        repeat=rep,
                        seconds=elapsed,
                        batch_tokens_s=a.batch * a.steps / elapsed,
                        sequence_tokens_s=a.steps / elapsed,
                        ms_step=elapsed * 1000 / a.steps,
                        initial_allocated_bytes=initial_allocated,
                        peak_allocated_bytes=peak,
                        peak_reserved_bytes=reserved,
                        output_sha256=trace_sha,
                        nonfinite_rows=0,
                        cmp_population_delta=pop_delta,
                    )
                    del cache, token, trace
                    if ec is not None:
                        ec._hidden.clear()
                        ec._pending.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                del snapshots, seed, positions
            del cm, ec
            gc.collect()
            torch.cuda.empty_cache()
        emit("complete", measured_trials=a.documents * a.repeats)
        return 0
    except Exception as exc:
        oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        emit(
            "failure",
            status="oom" if oom else "error",
            phase=phase,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 20 if oom else 1


if __name__ == "__main__":
    raise SystemExit(main())
