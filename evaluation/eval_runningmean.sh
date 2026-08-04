dataset="ruler"
model="/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
compression_ratios=(0.25 0.5 0.75 0.9)
press_names=(
    "expected_attention"
    "expected_attention_layer_mean"
    "expected_attention_layer_mean_ent_skip_high_negonly"
    "expected_attention_layer_mean_ent_skip_high_softmax"
    "expected_attention_layer_mean_ent_skip_low_negonly"
    "expected_attention_layer_mean_ent_skip_low_softmax"
    "expected_attention_layer_mean_ent_skip_low"
    "expected_attention_layer_mean_ent_skip_high"
)
device="cuda:0"
device_id=0

for press_name in "${press_names[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
        echo "Running press_name: $press_name with compression_ratio: $compression_ratio on GPU $device_id"
        CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
            --dataset $dataset \
            --model $model \
            --data_dir "" \
            --press_name $press_name \
            --compression_ratio $compression_ratio \
            --device $device \
            --compress_questions False \
            --fraction 0.1 \
            --output_dir "./results_others_dataset" &
        device_id=$((device_id + 1))
    done
done


wait
echo "All evaluation tasks completed."