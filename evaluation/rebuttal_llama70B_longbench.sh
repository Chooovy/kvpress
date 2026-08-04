#!/usr/bin/env bash
# LongBench-v2 (cloze + forward), Llama 3.3 70B, single process + 8 GPUs (device auto).
#
# Intentional comparison (different checkpoints — not a same-weight press ablation):
#   A) Fine-tuned QueryIndexer checkpoint + query_indexer_score press
#   B) Llama-3.3-70B-Instruct + TOVA only
#
# QueryIndexerScorePress scores via an einsum over seq×seq (~O(L²) VRAM). On ~80GB GPUs,
# --max_input_tokens 65536 can ask for 100GB+ in one allocation and OOM; TOVA does not use
# that path, so indexer and TOVA use different caps below (tune max_input_tokens_indexer up
# only if you add chunked scoring in the press or have more memory).
#
# Stratified sampling: `fraction` of each LongBench-v2 domain (see `fraction=` below).
# Run: cd KVCache/kvpress/evaluation && bash rebuttal_llama70B_longbench.sh
# Logs: REBUTTAL_LOG_DIR (default /aifs4su/guhao/logs)
#   - Per-phase: ...-query_indexer-<RUN_ID>.log and ...-tova-<RUN_ID>.log (same RUN_ID)
#   - Combined: ...-BOTH-<RUN_ID>.log

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${REBUTTAL_LOG_DIR:-/aifs4su/guhao/logs}"
mkdir -p "$LOG_ROOT"

RUN_ID="$(date +%Y%m%d-%H%M%S)-$(hostname -s)"
MASTER_LOG="${LOG_ROOT}/rebuttal-llama70b-longbenchv2-BOTH-${RUN_ID}.log"
echo "[INFO] Combined log (phase A + B): $MASTER_LOG" >&2

dataset="longbench-v2"
fraction="0.1" # 10% per domain (both runs A and B use this)

# Long-context cap per phase (see header: indexer needs a much lower L than TOVA).
max_input_tokens_indexer="8192"
max_input_tokens_tova="65536"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 USE_TORCH=1
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# A) Trained indexer model + query_indexer_score
# -----------------------------------------------------------------------------
LOG_FILE="${LOG_ROOT}/rebuttal-llama70b-longbenchv2-query_indexer-${RUN_ID}.log"
model_path="/aifs4su/guhao/KVCache/checkpoints/llama-3.3-70B-query-indexer-math-longalpaca"
compression_ratio="0.5"
press_name="query_indexer_score"
out_tag="longbenchv2_cloze_forward_${max_input_tokens_indexer}_cr${compression_ratio}_frac${fraction}_8gpu_query_indexer"
output_dir="./results_${out_tag}"

{
  echo "=== $(date -Is) rebuttal_llama70B_longbench.sh PHASE A (query_indexer) ==="
  echo "host=$(hostname) pwd=$PWD"
  echo "phase_log=$LOG_FILE"
  echo "combined_log=$MASTER_LOG"
  echo "python=$(command -v python)"
  echo "----------------------------------------"
  echo "[RUN] dataset=$dataset model=$model_path press=$press_name cr=$compression_ratio fraction=$fraction"
  echo "[RUN] max_input_tokens=$max_input_tokens_indexer (indexer press; lower than TOVA — see script header)"
  echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES device=auto out=$output_dir"
  if [[ ! -d "$model_path" ]]; then
    echo "[FATAL] indexer checkpoint directory missing: $model_path" >&2
    exit 2
  fi

  python evaluate_bak.py \
    --dataset "$dataset" \
    --model "$model_path" \
    --press_name "$press_name" \
    --compression_ratio "$compression_ratio" \
    --fraction "$fraction" \
    --seed 42 \
    --device auto \
    --longbenchv2_eval cloze \
    --longbenchv2_cloze_impl forward \
    --max_input_tokens "$max_input_tokens_indexer" \
    --output_dir "$output_dir"

  py_exit=$?
  echo "[DONE] python exit=$py_exit"
  exit "$py_exit"
} 2>&1 | tee "$LOG_FILE" | tee -a "$MASTER_LOG"

ec1="${PIPESTATUS[0]}"

if [[ "$ec1" -ne 0 ]]; then
  echo "[FAIL] query_indexer phase exit=$ec1 (see phase log + $MASTER_LOG)" >&2
  exit "$ec1"
fi
echo "[OK] query_indexer phase done. metrics under $output_dir"
exit 0

# -----------------------------------------------------------------------------
# Optional Phase B (TOVA): duplicate the Phase A block, swap model_path / press /
# max_input_tokens to max_input_tokens_tova, then replace the exit block above with:
#   ec2="${PIPESTATUS[0]}"; [[ "$ec1" -ne 0 || "$ec2" -ne 0 ]] && exit 1; exit 0
# Or keep a second script / sbatch job for TOVA only (you already have a finished run).
# -----------------------------------------------------------------------------
