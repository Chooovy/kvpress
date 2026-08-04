dataset="ruler"
data_dir=("4096" "16384")
model=(
    "/aifs4su/guhao/checkpoints/qwen3-8b-indexer-max"
    )
compression_ratios=(0.10 0.25 0.5 0.75 0.9)
press_names=(
    "query_indexer_ea_mode"
)
device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0

for press_name in "${press_names[@]}"; do
    for model in "${model[@]}"; do
        for data_dir in "${data_dir[@]}"; do
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
            --model "$model" \
            --data_dir "$data_dir" \
            --press_name $press_name \
            --compression_ratio $compression_ratio \
            --device ${device}:0 \
            --fraction 0.1 \
            --compress_questions False \
            --output_dir "./results_finals" &
            done
        done
    done
done


wait
echo "All evaluation tasks completed."