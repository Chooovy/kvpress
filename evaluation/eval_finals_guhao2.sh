dataset="ruler"
# format: "model_path|press_name|data_dir1,data_dir2,..."
configs=(
    # "/aifs4su/guhao/checkpoints/mistral-7b-indexer-max|query_indexer_max_mode|16384"
    # "/aifs4su/guhao/checkpoints/qwen3-8b-indexer-max|query_indexer_max_mode|4096,16384"
    "/aifs4su/guhao/checkpoints/mistral-7b-memory-max-mode|memory_query_indexer_max|16384"
    # "/aifs4su/guhao/checkpoints/qwen3-8b-memory-max-mode|memory_query_indexer_max|4096,16384"
)
compression_ratios=(0.10 0.25 0.5 0.75 0.9)
device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0

for config in "${configs[@]}"; do
    IFS='|' read -r model press_name data_dirs_csv <<< "$config"
    IFS=',' read -ra data_dirs <<< "$data_dirs_csv"
    for data_dir in "${data_dirs[@]}"; do
        for compression_ratio in "${compression_ratios[@]}"; do
            # Wait for a free GPU slot if all are busy
            while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
                wait -n
            done

            device_id=${gpu_ids[$next_gpu_idx]}
            next_gpu_idx=$(( (next_gpu_idx + 1) % max_jobs ))

            echo "Running model: $model press_name: $press_name data_dir: $data_dir compression_ratio: $compression_ratio on GPU $device_id"
            CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
                --dataset $dataset \
                --model "$model" \
                --data_dir "$data_dir" \
                --press_name "$press_name" \
                --compression_ratio $compression_ratio \
                --device ${device}:0 \
                --compress_questions False \
                --output_dir "./results_finals" &
        done
    done
done


wait
echo "All evaluation tasks completed."