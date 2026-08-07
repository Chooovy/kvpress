#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer distillation on the longmino corpus.
#
#   scripts/train_gqa_indexer.sh tokenize   # once: pre-tokenize to 64K (serves both stages)
#   scripts/train_gqa_indexer.sh smoke      # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer.sh stage1     # dense  @ 32K
#   scripts/train_gqa_indexer.sh stage2     # sparse @ 64K, from stage 1's checkpoint
#
# Run tokenize first, then stage1, then stage2 (stage 2 loads stage 1's final.pt). Override
# anything with env vars, e.g.  STEPS=200 scripts/train_gqa_indexer.sh stage1
#
# LR is WSD throughout: warmup 10% to PEAK_LR, hold 60%, decay linearly to FINAL_LR on the
# last step. Batch size is 1 -- at these lengths the sequence axis already saturates the GPU,
# so a larger batch would just halve the usable length.
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen-3-8B-gqa_indexer}"

PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

cd "$(dirname "$0")/.."

# The tokenizer's thread pool is forked by the dataloader workers; it warns per worker and
# buys nothing, since each worker handles one document at a time.
export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator badly. expandable_segments lets it grow a
# segment instead of failing on a large contiguous request, which is exactly the shape of
# request the tiled loss makes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    # Shortest path that still exercises the loader, hooks, loss, backward and checkpointing.
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SCHEDULE:-8192:2}" \
      --stage dense --backend torch \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1)
    # Dense distillation at 32K. Uses the pre-tokenized corpus if it exists, since tokenizing
    # a 32K sample costs ~0.2 s of serial work on the data path.
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      EXTRA+=(--tokenized "$TOKENIZED" --subsets 2e16 2e17)
      echo "using pre-tokenized corpus at $TOKENIZED"
    else
      EXTRA+=(--subsets 2e15 2e16 synth_cwe synth_rex)
      echo "no $TOKENIZED/index.json; tokenizing on the fly (run '$0 tokenize' to avoid this)"
    fi
    exec python -m scripts.train_gqa_indexer \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" \
      --schedule "32768:${STEPS:-1500}" \
      --stage dense \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" --key-tile 512 --query-tile 512 \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
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
    exec python -m scripts.train_gqa_indexer \
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
    echo "usage: $0 {tokenize|smoke|stage1|stage2}" >&2
    exit 1
    ;;
esac
