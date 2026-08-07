#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer distillation on the longmino corpus.
#
#   scripts/train_gqa_indexer.sh smoke     # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer.sh stage1    # dense warmup, 8K -> 32K
#   scripts/train_gqa_indexer.sh stage2    # sparse refinement from stage 1's checkpoint
#
# Stage 2 loads stage 1's final checkpoint, so run them in order. Override anything with
# env vars, e.g.  SEQ_SCHEDULE=8192:50 scripts/train_gqa_indexer.sh stage1
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_256k_filtered}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT="${OUT:-checkpoints/gqa_indexer}"
LR="${LR:-1e-4}"

cd "$(dirname "$0")/.."

# The tokenizer's thread pool is forked by the dataloader workers; it warns per worker and
# buys nothing, since each worker tokenizes one document at a time.
export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator badly. expandable_segments lets it grow a
# segment instead of failing on a large contiguous request, which is exactly the shape of
# request the tiled loss makes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$MODE" in
  smoke)
    # 2 steps at a short length, torch backend: shortest path that still exercises the
    # loader, the hooks, the loss, backward, and checkpointing.
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SEQ_SCHEDULE:-8192:2}" \
      --stage dense --backend torch \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1)
    # Dense warmup. 8K first because stage 1 is O(L^2) -- 200 steps at 8K cost about what
    # 13 steps at 32K do -- then 32K for the bulk. 2e17 is left out: its median is 168K
    # tokens, so at these lengths most of it would be read and thrown away.
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k 2e15 synth_cwe synth_rex 2e16 \
      --schedule "${SEQ_SCHEDULE:-8192:300,32768:1200}" \
      --stage dense --lr "$LR" \
      --take-from random --shuffle-buffer 64 --num-workers 2 \
      --key-tile 512 --query-tile 512 \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every 200 --log-every 10
    ;;

  stage2)
    # Sparse refinement at the eviction budget used at eval. --topk is fixed, not a ratio:
    # a ratio makes the retained support O(L^2). topk-tile 128 rather than 512 because the
    # per-tile gather is O(query_tile * topk_tile * head_dim) and dominates stage-2 memory.
    INIT="${INIT:-$OUT/stage1/final.pt}"
    if [[ ! -f "$INIT" ]]; then
      echo "stage 2 needs stage 1's checkpoint at $INIT (run stage1 first, or set INIT=)" >&2
      exit 1
    fi
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 2e15 synth_cwe synth_rex 2e16 \
      --schedule "${SEQ_SCHEDULE:-32768:600}" \
      --stage sparse --topk "${TOPK:-512}" --teacher-mode global \
      --force-local "${FORCE_LOCAL:-64}" --force-sink "${FORCE_SINK:-4}" \
      --topk-tile 128 --query-tile 512 \
      --lr "${LR:-5e-5}" --init-from "$INIT" \
      --take-from random --shuffle-buffer 64 --num-workers 2 \
      --out "$OUT/stage2" --metrics-file "$OUT/stage2/metrics.jsonl" \
      --save-every 200 --log-every 10
    ;;

  *)
    echo "usage: $0 {smoke|stage1|stage2}" >&2
    exit 1
    ;;
esac
