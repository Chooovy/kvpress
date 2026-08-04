#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
log_file="logs/rebuttal_run_indexmem_$(date +%Y%m%d_%H%M%S).log"
exec >"$log_file" 2>&1

echo "========== Script started at $(date) =========="
echo "Log file: $log_file"

dataset="local_ruler"
model="/aifs4su/guhao/KVCache/checkpoints/llama-3-8b-1m-query-indexer"
press_names=("query_indexer_score_block")
# compression_ratios=(0 0.1 0.3 0.5 0.7)
compression_ratios=(0.8)

tasks=(
  "niah_single_1"
  "niah_single_2"
  "niah_single_3"
  "niah_multikey_1"
  "niah_multikey_2"
  "niah_multivalue"
  "niah_multiquery"
  "niah_multiturn_1"
  "niah_multiturn_2"
  "vt"
  "fwe"
  "qa_1"
  "qa_2"
)

base_data_dir="/aifs4su/guhao/KVCache/xKV/evaluate/data/ruler/data/llama-3/32768"
output_dir="./results_others_dataset"

gpu_ids=(0 1 2 3 4 5 6 7)
sleep_seconds=20

declare -A GPU_PID_MAP


cleanup_finished_jobs() {
  for gpu in "${!GPU_PID_MAP[@]}"; do
    pid="${GPU_PID_MAP[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      unset GPU_PID_MAP["$gpu"]
    fi
  done
}


wait_for_available_gpu() {
  while true; do
    cleanup_finished_jobs

    for gpu in "${gpu_ids[@]}"; do
      if [[ -z "${GPU_PID_MAP[$gpu]+x}" ]]; then
        echo "$gpu"
        return 0
      fi
    done

    sleep "${sleep_seconds}"
  done
}


launch_one_job() {
  local task="$1"
  local press_name="$2"
  local compression_ratio="$3"
  local gpu_id="$4"

  CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate_bak.py \
    --dataset "$dataset" \
    --model "$model" \
    --data_dir "${base_data_dir}/${task}" \
    --press_name "$press_name" \
    --compression_ratio "$compression_ratio" \
    --device "cuda:0" \
    --fraction 1 \
    --use_chunk_prefill True \
    --chunk_prefill_size 512 \
    --output_dir "$output_dir" &

  local pid=$!
  GPU_PID_MAP["$gpu_id"]="$pid"
}


for task in "${tasks[@]}"; do
  for press_name in "${press_names[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
      gpu_id=$(wait_for_available_gpu)
      launch_one_job "$task" "$press_name" "$compression_ratio" "$gpu_id"
      sleep 3
    done
  done
done

wait
echo "========== All evaluation tasks completed at $(date) =========="