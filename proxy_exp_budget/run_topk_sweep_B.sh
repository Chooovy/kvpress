#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Arm B (cross-replay, mid_dim=256, budget=2048) topk sweep at RULER 8K -- diagnosis only, no
# training. B's true gate participation is 0.062, i.e. ~508 keys at N=8192, against the topk=2048
# the 20.43 was measured at. If the collapse is a budget/concentration misalignment, the score
# should recover near topk ~512-1024; if no topk recovers it, the score itself is broken and
# retraining at B=1 would be pointless.
#
# Everything except --topk matches the run that produced 20.43 (FRACTION=0.1, SEED=42,
# force_local=64, force_sink=4, block_k=64, tf32), so the rows scored are identical.
set -euo pipefail

cd "$(dirname "$0")/../evaluation"

export CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_cross_replay/stage1_256/final.pt}"
export LENGTHS="${LENGTHS:-8192}"
export TOPKS="${TOPKS:-512 1024 256 4096}"
export NGPU="${NGPU:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-./results_sparse_scalar_topksweep}"
export PYTHON="${PYTHON:-/opt/conda/envs/torch-base/bin/python}"

exec bash evaluate_sparse_scalar_shard.sh
