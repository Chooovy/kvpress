#!/usr/bin/env bash
# LongBench-v2 baseline: base Llama 3.3 70B Instruct, no KV press (no_press).
# Only rows with domain == LONG_BENCH_V2_DOMAIN, then fraction=0.2 within remaining domains
# (here a single domain → 20% of that domain's rows).
#
# Requires evaluate_bak.py with --longbenchv2_domain (filters before stratified fraction).
#
# Run: cd KVCache/kvpress/evaluation && bash rebuttal_llama70B_baseline_longbench.sh
# Logs: REBUTTAL_LOG_DIR (default /aifs4su/guhao/logs)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${REBUTTAL_LOG_DIR:-/aifs4su/guhao/logs}"
mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_ROOT}/rebuttal-llama70b-longbenchv2-baseline-$(date +%Y%m%d-%H%M%S)-$(hostname -s).log"

dataset="longbench-v2"
# Base model only (no indexer checkpoint)
model_path="/aifs4su/guhao/Models/Llama-3.3-70B-Instruct"
press_name="no_press"
fraction="0.2"

# Must match HF `domain` string exactly (see predictions.csv / dataset card if unsure)
LONG_BENCH_V2_DOMAIN="Long Structured Data Understanding"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 USE_TORCH=1
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# Short tag for result folder (no spaces)
domain_tag="long_structured_data_understanding"
out_tag="longbenchv2_baseline_${domain_tag}_cloze_fwd_65536_frac${fraction}_8gpu"
output_dir="./results_${out_tag}"

{
  echo "=== $(date -Is) rebuttal_llama70B_baseline_longbench.sh ==="
  echo "host=$(hostname) pwd=$PWD"
  echo "log=$LOG_FILE"
  echo "python=$(command -v python)"
  echo "----------------------------------------"
  echo "[RUN] baseline model=$model_path press=$press_name"
  echo "[RUN] domain filter=$LONG_BENCH_V2_DOMAIN fraction=$fraction (then per-domain sample)"
  echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES device=auto out=$output_dir"

  python evaluate_bak.py \
    --dataset "$dataset" \
    --model "$model_path" \
    --press_name "$press_name" \
    --fraction "$fraction" \
    --seed 42 \
    --device auto \
    --longbenchv2_eval cloze \
    --longbenchv2_cloze_impl forward \
    --longbenchv2_domain "$LONG_BENCH_V2_DOMAIN" \
    --max_input_tokens 65536 \
    --output_dir "$output_dir"

  echo "[DONE] exit=$?"
} 2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
