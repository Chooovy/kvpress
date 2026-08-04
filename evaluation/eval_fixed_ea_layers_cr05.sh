dataset="ruler"
data_dir="4096"
model="/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
compression_ratio=0.5
device="cuda:0"

# Uses 8 GPUs in parallel, 4 waves to cover layer_idx 0..31
# Each run uses a different press_name: fixed_EA_layer{idx}
output_dir="./results_fixed_ea_layers"
fraction=0.1

for start in 0 8 16 24; do
  echo "=== Wave starting at layer_idx=${start} (GPUs 0-7) ==="
  device_id=0
  for layer_idx in $(seq $start $((start + 7))); do
    press_name="fixed_EA_layer${layer_idx}"
    echo "Running ${press_name} cr=${compression_ratio} on GPU ${device_id}"
    CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
      --dataset $dataset \
      --data_dir $data_dir \
      --model $model \
      --press_name $press_name \
      --compression_ratio $compression_ratio \
      --device $device \
      --compress_questions False \
      --output_dir "$output_dir" \
      --fraction $fraction &
    device_id=$((device_id + 1))
  done
  wait
done

echo "All fixed-layer EA evaluation tasks completed."

