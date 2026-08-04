dataset="aime25"
model="/aifs4su/guhao/checkpoints/qwen3-8b-indexer-max"
compression_interval=128
target_sizes=(2048 1536 1024 512)
hidden_states_buffer_size=128
output_dir="./results_aime25"
# press_names=("decoding_tova" "decoding_qfilter" "decoding_adakv_snapkv" "decoding_keydiff" "decoding_query_indexer")
press_names=("decoding_query_indexer" "decoding_tova")

device_id=0

echo "Running decode evaluation for $dataset with model $model"

for press_name in "${press_names[@]}"; do
    for target_size in "${target_sizes[@]}"; do
        echo "Running press_name: $press_name with target_size: $target_size on GPU $device_id"
        CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
            --dataset $dataset \
            --model $model \
            --press_name $press_name \
            --compression_interval $compression_interval \
            --target_size $target_size \
            --hidden_states_buffer_size $hidden_states_buffer_size \
            --output_dir $output_dir \
            --device "cuda:0" \
            --samples 20 &
        device_id=$((device_id + 1))
    done
done

wait
echo "All evaluations completed."
