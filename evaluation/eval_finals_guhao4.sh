# dataset="ruler"
# model="/aifs4su/guhao/checkpoints/mistral-7b-indexer-max"
# data_dir="16384"
# press_name="query_indexer_kvzip_max"
# compression_ratio=0.10
# device="cuda"


# CUDA_VISIBLE_DEVICES=0 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &


# compression_ratio=0.25
# CUDA_VISIBLE_DEVICES=1 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &


# compression_ratio=0.50
# CUDA_VISIBLE_DEVICES=2 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &


# compression_ratio=0.75
# CUDA_VISIBLE_DEVICES=3 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &


# compression_ratio=0.90
# CUDA_VISIBLE_DEVICES=4 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &





dataset="ruler"
model="/aifs4su/guhao/checkpoints/qwen3-8b-indexer-max"
data_dir="16384"
press_name="query_indexer_kvzip_max"
compression_ratio=0.25
device="cuda"


CUDA_VISIBLE_DEVICES=5 python evaluate_bak.py \
    --dataset $dataset \
    --model "$model" \
    --data_dir "$data_dir" \
    --press_name "$press_name" \
    --compression_ratio $compression_ratio \
    --device ${device}:0 \
    --compress_questions False \
    --output_dir "./results_finals" &



# compression_ratio=0.75
# CUDA_VISIBLE_DEVICES=6 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &





# compression_ratio=0.50
# CUDA_VISIBLE_DEVICES=7 python evaluate_bak.py \
#     --dataset $dataset \
#     --model "$model" \
#     --data_dir "$data_dir" \
#     --press_name "$press_name" \
#     --compression_ratio $compression_ratio \
#     --device ${device}:0 \
#     --compress_questions False \
#     --output_dir "./results_finals" &

wait