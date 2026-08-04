#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kvpress_env
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="ruler"
DATA_DIR="4096"
MODEL="/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"

FRACTION="0.1"
MAX_CTX="4096"
MAX_NEW="32"
CR="0.50"

OUT_ROOT="./pilot_ruler100_llama31_8b_instruct_h800"
CR_TAG="${CR/./p}"
GT_OUT="${OUT_ROOT}/gt_tokens_cr${CR_TAG}"

rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

python evaluate.py \
  --device 0 \
  --dataset "${DATASET}" --data_dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --press_name no_press \
  --fraction "${FRACTION}" \
  --max_context_length "${MAX_CTX}" \
  --max_new_tokens "${MAX_NEW}" \
  --gt_mode True \
  --gt_out_dir "${GT_OUT}" \
  --output_dir "${OUT_ROOT}/gt_run_cr${CR_TAG}"

echo "GT saved to: ${GT_OUT}"
