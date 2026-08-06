#!/bin/bash

set -euo pipefail


export HF_TOKEN="${HF_TOKEN:-}"

DATASET_ID="${DATASET_ID:-sagels/longmino_256k_filtered}"
LOCAL_DIR="${LOCAL_DIR:-/apdcephfs_zw31/share_303843174/user/marcushaogu/datasets/longmino_256k_filtered}"

hf download \
  "$DATASET_ID" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR" \
  --token "$HF_TOKEN" \
