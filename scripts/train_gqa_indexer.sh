#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer distillation on the longmino corpus.
#
#   scripts/train_gqa_indexer.sh tokenize   # once: pre-tokenize to 64K (serves both stages)
#   scripts/train_gqa_indexer.sh smoke      # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer.sh check      # can the teacher lse come from flash-attn?
#   scripts/train_gqa_indexer.sh profile    # once per machine: measure batch/tile per length
#   scripts/train_gqa_indexer.sh stage1     # dense, 8K -> 16K -> 32K curriculum
#   scripts/train_gqa_indexer.sh stage2     # sparse @ 64K, from stage 1's checkpoint
#
# Run tokenize first, then stage1, then stage2 (stage 2 loads stage 1's final.pt). Override
# anything with env vars, e.g.  STEPS=200 scripts/train_gqa_indexer.sh stage1
#
# Runs on NGPU GPUs of one node via torchrun. Per-GPU batch size is 1, so the effective batch is
# NGPU sequences. Batch 1 is always correct -- the loss is a plain row mean, so batch changes the
# effective batch and the device occupancy, never the objective. It does leave the 8K stage using
# a quarter of the memory 32K does, i.e. partly idle; AUTOTUNE=1 recovers that by sizing the
# batch per length, at the cost of a profiling sweep up front.
#
# LR is WSD throughout: warmup 10% to PEAK_LR, hold 60%, decay linearly to FINAL_LR on the
# last step. The LR is NOT scaled by NGPU -- the gradient is averaged across ranks, not summed,
# so its magnitude is unchanged and the peak that works on one GPU works on eight.
#
# STEPS counts OPTIMIZER steps, not samples, so on 8 GPUs the same count sees 8x the data. The
# script logs the actual token count for the schedule it is about to run; read that rather than
# deriving it from STEPS, since the stages have different seq_len.
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen-3-8B-gqa_indexer}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"

PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

# Autotune batch size and tile shape per curriculum length. OFF by default: a full sweep is
# ~180 real forward+backward steps, and at 32K that is most of an hour before the first
# optimizer step -- too much to pay up front for a throughput gain that batch=1 does not need
# in order to be correct.
#
# AUTOTUNE=1 turns it back on, and the result is cached per (GPU, model, stage, backend), so
# the cost is paid once per machine rather than once per run. `$0 profile` does exactly that
# without training, and bounds itself with AUTOTUNE_TIME_BUDGET.
AUTOTUNE="${AUTOTUNE:-0}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
AUTOTUNE_TIME_BUDGET="${AUTOTUNE_TIME_BUDGET:-900}"

# Reuse the logsumexp flash-attention already computed, instead of recomputing it with
# teacher_lse_from_qk on every layer of every step (which runs on BOTH backends -- the Triton
# kernel takes lse as an input, so it does not avoid the recompute).
#
# Default "never" until it has been exercised on real hardware: it changes what the teacher is
# normalized against, so it is opt-in rather than silently on. `$0 check` reports whether this
# box can use it; "auto" falls back per layer when the mask is not purely causal.
CAPTURE_LSE="${CAPTURE_LSE:-never}"

cd "$(dirname "$0")/.."

# The tokenizer's thread pool is forked by the dataloader workers; it warns per worker and
# buys nothing, since each worker handles one document at a time.
export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator badly. expandable_segments lets it grow a
# segment instead of failing on a large contiguous request, which is exactly the shape of
# request the tiled loss makes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Surface the rank that actually failed instead of a generic "one worker died".
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# torchrun only when there is more than one GPU: a single-process run skips the process group
# entirely, which keeps the smoke test debuggable (real traceback, no rank multiplexing).
if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=(python)
fi

case "$MODE" in
  tokenize)
    # Store 64K tokens per document: stage 2 needs all of it, and stage 1 takes a 32K prefix.
    # Storing 32K instead would make stage 2 impossible, since the tokens simply would not
    # exist -- so the width is set by the LONGEST stage, not the first one.
    #
    # Only 2e16 and 2e17 have documents that reach 64K (measured 100% of each, versus 0% for
    # 2e15/synth_*, whose medians are ~43K). Including the others would spend hours reading
    # documents that are then rejected. Add them back if you lower --seq-len.
    exec python -m scripts.pretokenize_longmino \
      --data-root "$DATA_ROOT" --out "$TOKENIZED" --model "$MODEL" \
      --subsets ${TOK_SUBSETS:-2e16 2e17} \
      --seq-len "${TOK_SEQ_LEN:-65536}" \
      --workers "${TOK_WORKERS:-16}"
    ;;

  smoke)
    # Deliberately single-process: a real traceback beats eight interleaved ones when the
    # thing being verified is whether the path works at all.
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SCHEDULE:-8192:2}" \
      --stage dense --backend torch \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1)
    # Dense distillation on a length curriculum: 8K -> 16K -> 32K, in ONE lr schedule.
    #
    # One schedule, not three. Measured on this loss: |grad| grows only 1.018x over an 8x
    # length change, so the optimal lr does not move with seq_len and there is nothing for a
    # per-stage schedule to adapt to. Three schedules would instead anneal to FINAL_LR and
    # reheat to PEAK_LR twice over -- paying for the decay three times and keeping the benefit
    # once, since reheating discards most of what annealing consolidates.
    #
    # The step split puts both boundaries inside the stable phase (at 10%/60% they fall at
    # steps 300 and 600 of 1500, where lr is flat at PEAK_LR), so a length change is never
    # confounded with an lr change. The decay then happens entirely at 32K, so the model
    # anneals at the length it will be evaluated at.
    #
    # Cheap, too: the stage-1 loss is O(L^2), so 300 steps at 8K cost about 19 steps at 32K.
    #
    # EXPECTED: the reported loss JUMPS by ~0.69 = log 2 at each boundary. That is arithmetic,
    # not a regression -- loss(L) ~ log(L) + const, measured +0.7076/+0.7057/+0.6990 per
    # doubling. Nothing needs to react to it.
    #
    # Batch size is a flat BATCH_SIZE (default 1) unless AUTOTUNE=1. Batch 1 at every length is
    # always correct -- the loss is a plain mean over rows, so batch only changes the effective
    # batch and the device occupancy, never the objective. What it costs is throughput at the
    # short end: 8K at batch 1 uses a quarter of the memory 32K does, so the card is partly
    # idle there. AUTOTUNE=1 (or `$0 profile` once per machine) recovers that by holding
    # batch x seq_len roughly constant, giving about 4 / 2 / 1 across 8K / 16K / 32K.
    #
    # Leaving it off keeps tokens/step proportional to seq_len instead of constant. That is fine
    # for the single lr schedule -- the boundaries are still pure length changes, since batch is
    # not changing either -- it just means the short stages see fewer tokens per step, which is
    # also why they are cheap.
    #
    # Note --key-tile/--query-tile are NOT passed here. They do not reach the Triton kernels at
    # all (those take block_m/block_n, default 64), and backend=auto selects Triton for every
    # mask stage 1 builds -- so the only thing they affected was teacher_lse_from_qk. The
    # profiler sweeps both and records which backend actually ran.
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      # No retokenization needed for the curriculum: every stage length is <= the stored
      # width, and a shorter stage slices a window out of it.
      EXTRA+=(--tokenized "$TOKENIZED" --subsets 2e16 2e17)
      echo "using pre-tokenized corpus at $TOKENIZED"
    else
      EXTRA+=(--subsets 2e15 2e16 synth_cwe synth_rex)
      echo "no $TOKENIZED/index.json; tokenizing on the fly (run '$0 tokenize' to avoid this)"
    fi
    if [[ "$AUTOTUNE" != "0" ]]; then
      # The cache is keyed on GPU, model geometry, stage and backend, so it is safe to share
      # and safe to keep: a different machine misses rather than silently reusing.
      EXTRA+=(--autotune --token-budget "$TOKEN_BUDGET"
              --autotune-time-budget "$AUTOTUNE_TIME_BUDGET"
              --autotune-cache "$OUT/autotune.json")
    fi
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      --stage dense \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --capture-lse "$CAPTURE_LSE" \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  check)
    # Report whether the teacher logsumexp can come from flash-attention rather than being
    # recomputed by teacher_lse_from_qk on every layer of every step. Exits non-zero when it
    # cannot, so it can gate a launch.
    exec python -m scripts.check_flash_lse --model "$MODEL" --dtype "${DTYPE:-bfloat16}"
    ;;

  profile)
    # Measure batch/tile per curriculum length and write the cache, without training.
    #
    # Single-process on purpose: only rank 0 measures during training anyway (the ranks must
    # agree on batch size, since a differing batch would desynchronize the gradient allreduce),
    # so eight processes here would be eight times the work for one answer. --dry-run stops
    # after 2 steps; the profiling itself happens before the loop, so the cache is complete.
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --tokenized "$TOKENIZED" --subsets 2e16 2e17 \
      --schedule "${SCHEDULE:-8192:1,16384:1,32768:1}" \
      --stage dense \
      --autotune --autotune-force --token-budget "$TOKEN_BUDGET" \
      --autotune-time-budget "$AUTOTUNE_TIME_BUDGET" \
      --autotune-cache "$OUT/autotune.json" \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/profile" --dry-run
    ;;

  stage2)
    # Sparse refinement at 64K. --topk is fixed, not a ratio: a ratio makes the retained
    # support O(L^2). topk-tile 128 rather than 512 because the per-tile gather is
    # O(query_tile * topk_tile * head_dim) and is stage 2's dominant transient.
    INIT="${INIT:-$OUT/stage1/final.pt}"
    if [[ ! -f "$INIT" ]]; then
      echo "stage 2 needs stage 1's checkpoint at $INIT (run stage1 first, or set INIT=)" >&2
      exit 1
    fi
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      EXTRA+=(--tokenized "$TOKENIZED")
    fi
    # At 64K only these two subsets have qualifying documents at all.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" \
      --subsets 2e16 2e17 \
      --schedule "65536:${STEPS:-600}" \
      --stage sparse --topk "${TOPK:-512}" --teacher-mode global \
      --force-local "${FORCE_LOCAL:-64}" --force-sink "${FORCE_SINK:-4}" \
      --peak-lr "${PEAK_LR}" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 32 \
      --num-workers "${WORKERS:-2}" --topk-tile 128 --query-tile 512 \
      --init-from "$INIT" \
      --out "$OUT/stage2" --metrics-file "$OUT/stage2/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {tokenize|smoke|check|profile|stage1|stage2}" >&2
    exit 1
    ;;
esac
