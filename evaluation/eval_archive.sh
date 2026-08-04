# #!/bin/bash

# dataset="ruler"
# data_dir="4096"
# model="/aifs4su/guhao/checkpoints/llama3-1b-instruct-indexer_score"
# compression_ratios=(0.1 0.25 0.5 0.7)
# press_name="learned_score"
# device="cuda:0"

# # Run evaluation for each compression ratio
# for compression_ratio in "${compression_ratios[@]}"; do
#     echo "Running press_name: $press_name with compression_ratio: $compression_ratio on GPU $device"
#     python evaluate.py \
#         --dataset $dataset \
#         --data_dir $data_dir \
#         --model $model \
#         --press_name $press_name \
#         --compression_ratio $compression_ratio \
#         --device $device \
#         --output_dir "./results_comparison"
# done

# echo "All evaluations completed."



# #!/bin/bash

# dataset="ruler"
# data_dir="4096"
# model="/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
# compression_ratios=(0.1 0.25 0.5)
# press_names=("learned_score" "knorm" "snapkv" "expected_attention")

# # Check if the number of press names is less than or equal to the number of available GPUs
# num_gpus=$(nvidia-smi --list-gpus | wc -l)
# if [ ${#press_names[@]} -gt $num_gpus ]; then
#   echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
#   exit 1
# fi

# # Iterate over press names and compression ratios
# for i in "${!press_names[@]}"; do
#   press="${press_names[$i]}"
  
#   # Run each press_name on a different GPU in the background
#   (
#     for compression_ratio in "${compression_ratios[@]}"; do
#       echo "Running press_name: $press with compression_ratio: $compression_ratio on GPU cuda:$i"
#       python evaluate.py \
#           --dataset $dataset \
#           --data_dir $data_dir \
#           --model $model \
#           --press_name $press \
#           --compression_ratio $compression_ratio \
#           --device "cuda:$i" \
#           --output_dir "./results_comparison"
#     done
#   ) &
# done

# # Wait for all background jobs to finish
# wait
# echo "All evaluations completed."


# export CUDA_VISIBLE_DEVICES=1,2,3
# dataset="ruler"
# data_dir="4096"
# model="/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
# compression_ratios=(0.5  0.7  0.9)
# press_names=("knorm" "snapkv" "expected_attention")

# # Check if the number of press names is less than or equal to the number of available GPUs
# num_gpus=$(nvidia-smi --list-gpus | wc -l)
# if [ ${#press_names[@]} -gt $num_gpus ]; then
#   echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
#   exit 1
# fi

# # Iterate over press names and compression ratios
# for i in "${!press_names[@]}"; do
#   press="${press_names[$i]}"
  
#   # Run each press_name on a different GPU in the background
#   (
#     for compression_ratio in "${compression_ratios[@]}"; do
#       echo "Running press_name: $press with compression_ratio: $compression_ratio on GPU cuda:$i"
#       python evaluate.py \
#           --dataset $dataset \
#           --data_dir $data_dir \
#           --model $model \
#           --press_name $press \
#           --compression_ratio $compression_ratio \
#           --device "cuda:$i" \
#           --output_dir "./results_comparison"
#     done
#   ) &
# done

# # Wait for all background jobs to finish
# wait
# echo "All evaluations completed."



dataset="ruler"
data_dir="4096"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
compression_intervals=(64 128 256)
target_sizes=(512 1024 2048)
hidden_states_buffer_size=128
press_names=("decoding_knorm" "decoding_streaming_llm" "decoding_tova" "decoding_qfilter" "decoding_adakv_expected_attention_e2" "decoding_adakv_snapkv" "decoding_keydiff" "decoding_indexer_score")

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  
  # Run each press_name on a different GPU in the background
  (
    for compression_interval in "${compression_intervals[@]}"; do
      for target_size in "${target_sizes[@]}"; do
        echo "Running press_name: $press with compression_interval: $compression_interval, target_size: $target_size on GPU cuda:$i"
        python evaluate.py \
          --dataset $dataset \
          --data_dir $data_dir \
          --model $model \
          --press_name $press \
          --compression_interval $compression_interval \
          --target_size $target_size \
          --hidden_states_buffer_size $hidden_states_buffer_size \
          --device "cuda:$i"
      done
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."




# export CUDA_VISIBLE_DEVICES=1

# dataset="math500"
# model="/aifs4su/guhao/Models/DeepSeek-R1-Distill-Qwen-1.5B"
# press_name="no_press"
# output_dir="./results_math"

# python evaluate.py \
#     --dataset $dataset \
#     --model $model \
#     --press_name $press_name \
#     --device "cuda:0" \
#     --output_dir $output_dir \
#     --samples 20