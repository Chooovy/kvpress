dataset="ruler"
data_dir="4096"
model="/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
compression_ratios=(0.0)
press_names=("snapkv" "kvzip" "query_indexer_kvzip_max")
device="cuda:0"
device_id=0

score_dump_root="/aifs4su/guhao/KVCache/kvpress/evaluation/results_dump"

for press_name in "${press_names[@]}"; do
  for compression_ratio in "${compression_ratios[@]}"; do
    dump_dir="${score_dump_root}/${dataset}__${data_dir}__$(basename "$model")__${press_name}__cr${compression_ratio}"
    echo "Dumping scores to: $dump_dir"

    KVPRESS_SCORE_DUMP_PATH="$dump_dir" \
    CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak.py \
      --dataset $dataset \
      --data_dir $data_dir \
      --model $model \
      --press_name $press_name \
      --compression_ratio $compression_ratio \
      --device $device \
      --compress_questions False \
      --output_dir "./results_dump" \
      --fraction 0.01
  done
done