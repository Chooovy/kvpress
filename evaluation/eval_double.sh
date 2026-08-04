set -euo pipefail

dataset="ruler"
data_dir="4096"
compression_ratios=(0.25 0.5 0.75 0.9)

device="cuda:0"

model_1="/aifs4su/guhao/checkpoints/llama3-8b-query_indexer-max"
press_name_1="query_indexer_max_mode_128"
gpu_ids_1=(0 1 2 3)
output_dir_1="./results_debug"


model_2="/aifs4su/guhao/checkpoints/llama3.1-8b-instruct-max-mode-memory"
press_name_2="query_indexer_max_mode_128"
gpu_ids_2=(4 5 6 7)
output_dir_2="./results_debug"

if [ "${#compression_ratios[@]}" -ne "${#gpu_ids_1[@]}" ] || [ "${#compression_ratios[@]}" -ne "${#gpu_ids_2[@]}" ]; then
  echo "ERROR: compression_ratios(${#compression_ratios[@]}) 必须等于每组 gpu_ids 的长度：gpu_ids_1(${#gpu_ids_1[@]}), gpu_ids_2(${#gpu_ids_2[@]})" >&2
  exit 1
fi

for i in "${!compression_ratios[@]}"; do
  compression_ratio="${compression_ratios[$i]}"
  gpu_id="${gpu_ids_1[$i]}"
  echo "Running model_1: $model_1 | press_name: $press_name_1 | compression_ratio: $compression_ratio | GPU $gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate_bak.py \
    --dataset "$dataset" \
    --data_dir "$data_dir" \
    --model "$model_1" \
    --press_name "$press_name_1" \
    --compression_ratio "$compression_ratio" \
    --key_channel_compression_ratio "$compression_ratio" \
    --compress_questions False \
    --device "$device" \
    --fraction 0.1 \
    --output_dir "$output_dir_1" &
done

for i in "${!compression_ratios[@]}"; do
  compression_ratio="${compression_ratios[$i]}"
  gpu_id="${gpu_ids_2[$i]}"
  echo "Running model_2: $model_2 | press_name: $press_name_2 | compression_ratio: $compression_ratio | GPU $gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" python evaluate_bak.py \
    --dataset "$dataset" \
    --data_dir "$data_dir" \
    --model "$model_2" \
    --press_name "$press_name_2" \
    --compression_ratio "$compression_ratio" \
    --key_channel_compression_ratio "$compression_ratio" \
    --compress_questions False \
    --device "$device" \
    --fraction 0.1 \
    --output_dir "$output_dir_2" &
done

wait
echo "All evaluation tasks completed."

