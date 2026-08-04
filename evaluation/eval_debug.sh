dataset="ruler"
data_dir="4096"
model="/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
compression_ratios=(0.25 0.5 0.75 0.9)
press_names=("snapkv_128")
device="cuda:0"
device_id=0

for press_name in "${press_names[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
        echo "Running press_name: $press_name with compression_ratio: $compression_ratio on GPU $device_id"
        CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
            --dataset $dataset \
            --data_dir $data_dir \
            --model $model \
            --press_name $press_name \
            --compression_ratio $compression_ratio \
            --device $device \
            --compress_questions False \
            --output_dir "./results_debug" \
            --model_kwargs attn_implementation=eager \
            --fraction 0.1 &
        device_id=$((device_id + 1))
    done
done


wait
echo "All evaluation tasks completed."