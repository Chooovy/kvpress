dataset="ruler"
data_dir="4096"
model="/aifs4su/guhao/checkpoints/llama3-8b-query_indexer-max"
compression_ratios=(0.25 0.5 0.75 0.9)
press_names=(
    "query_indexer_max_mode_layer_mean"
)
device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0

for press_name in "${press_names[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
        # Wait for a free GPU slot if all are busy
        while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
            wait -n
        done

        device_id=${gpu_ids[$next_gpu_idx]}
        next_gpu_idx=$(( (next_gpu_idx + 1) % max_jobs ))

        echo "Running press_name: $press_name with compression_ratio: $compression_ratio on GPU $device_id"
        CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
            --dataset $dataset \
            --model $model \
            --data_dir "$data_dir" \
            --press_name $press_name \
            --compression_ratio $compression_ratio \
            --device ${device}:0 \
            --compress_questions False \
            --fraction 0.1 \
            --output_dir "./results_debug" &
    done
done


wait
echo "All evaluation tasks completed."