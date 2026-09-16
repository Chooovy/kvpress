#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch multi-node JOINT training from ONE node, over taiji's management SSH port.
#
#   HOST_NUM=4 bash launch_joint_dist.sh smoke      # verify first, 4 nodes, 2 steps
#   HOST_NUM=4 bash launch_joint_dist.sh joint_8k   # the real run
#   HOST_NUM=1 bash launch_joint_dist.sh smoke      # single node, no SSH at all
#
# Adapted from launch_dist.sh. Same mechanism: taiji containers cannot SSH each other on 22, but
# port 36000 accepts password login, so expect drives ssh, starts ranks 1..N-1 with nohup, and
# rank 0 runs in the foreground as the rendezvous master.
#
# WHY THIS IS NOT launch_dist.sh VERBATIM
# ---------------------------------------
# Four things differ, and each of them fails silently if left alone:
#
# 1. **PYBIN, not PATH.** launch_dist.sh forwards rank-0's $PATH so workers find conda. That is
#    fragile here because torchrun must come from the SAME env as torch -- a PATH that resolves
#    `torchrun` from one env and `torch` from another rendezvouses and then dies on import. This
#    passes an absolute interpreter instead, which cannot be shadowed.
# 2. **pkill patterns.** Its cleanup kills `tasks/pretrain.py`, `CPT_dist.sh` and so on -- names
#    from another project that would never match, leaving stale ranks holding GPU memory. The
#    patterns here name THIS run's processes.
# 3. **The GPUs must be free.** Joint training at 8K wants most of the card. These nodes were seen
#    with ~51 GiB/GPU already in use, so this script CHECKS and refuses rather than OOMing 20
#    minutes in.
# 4. **HOST_NUM=1 short-circuits.** No SSH, no expect, no cleanup trap -- just run. Useful because
#    the smoke test should not need working inter-node SSH to tell you the model loads.
#
# Logs: rank 0 to the terminal and log_joint_rank0.txt; workers to log_joint_rank<i>.txt.
#
# Run under tmux so an IDE disconnect does not kill the job:
#   tmux new -s joint
#   HOST_NUM=4 bash launch_joint_dist.sh joint_8k
#   # Ctrl+B D to detach, `tmux attach -t joint` to return
set -euo pipefail

MODE="${1:-smoke}"
WORK_DIR="${WORK_DIR:-$(pwd)}"
NNODES="${HOST_NUM:-1}"
MASTER_PORT="${MASTER_PORT:-29514}"
SSH_PORT="${SSH_PORT:-36000}"
SSH_USER="${SSH_USER:-root}"
SSH_PASS="${SSH_PASS:-epUsleVZYDPXI6b,}"

# Absolute interpreter, forwarded to every rank. See note 1 above.
PYBIN="${PYBIN:-/opt/conda/envs/torch-base/bin/python}"

# Sequences per optimizer step across all replicas. Held CONSTANT as HOST_NUM grows so a 4-node
# run is comparable to a 1-node run at the same step number; accumulation absorbs the difference.
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"

TRAIN_SH="scripts/train_gqa_indexer_joint_gy.sh"

if [[ ! -x "$PYBIN" ]]; then
    echo "[launch] ERROR: PYBIN not executable: $PYBIN" >&2
    exit 1
fi
if [[ ! -f "$WORK_DIR/$TRAIN_SH" ]]; then
    echo "[launch] ERROR: $WORK_DIR/$TRAIN_SH not found" >&2
    exit 1
fi

# ── Free-memory guard ────────────────────────────────────────────────────────
# Joint training needs most of the card. A run started next to a 51 GiB resident job will get
# through model load and then OOM inside the first backward, which is a slow and confusing way to
# discover the node was busy. MIN_FREE_GIB=0 skips the check.
MIN_FREE_GIB="${MIN_FREE_GIB:-70}"
if [[ "$MIN_FREE_GIB" != "0" ]]; then
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk -v lim=$(( (97871 - MIN_FREE_GIB * 1024) )) '$1 > lim {n++} END {print n+0}')
    if [[ "$busy" -gt 0 ]]; then
        echo "[launch] ERROR: ${busy} GPU(s) on THIS node have less than ${MIN_FREE_GIB} GiB free." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
        echo "[launch] Free them, or pass MIN_FREE_GIB=0 to override (expect OOM)." >&2
        exit 1
    fi
fi

echo "[launch] MODE=${MODE}  NNODES=${NNODES}  GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "[launch] PYBIN=${PYBIN}"
echo "[launch] WORK_DIR=${WORK_DIR}"

# ── Single node: no SSH needed ───────────────────────────────────────────────
if [[ "$NNODES" -le 1 ]]; then
    echo "[launch] single node -- running directly, no SSH"
    cd "$WORK_DIR"
    NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT} \
    NGPU="${NGPU:-8}" GLOBAL_BATCH="${GLOBAL_BATCH}" PYBIN="${PYBIN}" \
        bash "$TRAIN_SH" "$MODE" 2>&1 | tee "${WORK_DIR}/log_joint_rank0.txt"
    exit ${PIPESTATUS[0]}
fi

MASTER_ADDR="${NODE_IP_0:?multi-node needs NODE_IP_0 (source taiji/setup_env.sh first)}"
echo "[launch] MASTER=${MASTER_ADDR}:${MASTER_PORT}"

ssh_exec() {
    local ip="$1" cmd="$2"
    expect -c "
        set timeout 60
        spawn ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no ${SSH_USER}@${ip} \"${cmd}\"
        expect {
            \"assword:\" { send \"${SSH_PASS}\r\"; exp_continue }
            eof
        }
        catch wait result
        exit [lindex \$result 3]
    "
}

# Every NODE_IP_* forwarded, so a worker's own setup_env-derived vars are not needed.
NODE_IP_EXPORTS=""
for i in $(seq 0 $((NNODES - 1))); do
    var="NODE_IP_${i}"
    NODE_IP_EXPORTS+="export ${var}=${!var}; "
done

# Patterns matching THIS run only -- see note 2. Kept in one place so cleanup and pre-launch
# teardown cannot drift apart.
KILL_PATTERNS=(
    'train_gqa_indexer_joint_gy.sh'
    'scripts.train_gqa_indexer_joint'
    'torch.distributed.run'
    'torchrun'
)
kill_cmd() {
    local out=""
    for pat in "${KILL_PATTERNS[@]}"; do
        out+="pkill -f '${pat}' 2>/dev/null; "
    done
    echo "${out}true"
}

echo "[launch] clearing stale ranks on workers..."
for i in $(seq 1 $((NNODES - 1))); do
    var="NODE_IP_${i}"; ip="${!var}"
    ssh_exec "$ip" "$(kill_cmd)" &
done
wait
sleep 3

cleanup_workers() {
    echo ""
    echo "[launch] stopping workers..."
    for i in $(seq 1 $((NNODES - 1))); do
        var="NODE_IP_${i}"; ip="${!var}"
        echo "[launch]   rank-${i} on ${ip}"
        ssh_exec "$ip" "$(kill_cmd)" &
    done
    wait
    echo "[launch] done."
}
trap cleanup_workers EXIT

for i in $(seq 1 $((NNODES - 1))); do
    var="NODE_IP_${i}"; ip="${!var}"
    echo "[launch] starting rank-${i} on ${ip}:${SSH_PORT}"
    # NODE_RANK is passed EXPLICITLY rather than derived on the worker: the training script
    # defaults it to 0, so a worker that failed to compute it would silently become a second
    # rank 0 and torchrun would hang at rendezvous with nothing in the log.
    ssh_exec "$ip" \
        "${NODE_IP_EXPORTS} \
         export NNODES=${NNODES}; \
         export NODE_RANK=${i}; \
         export MASTER_ADDR=${MASTER_ADDR}; \
         export MASTER_PORT=${MASTER_PORT}; \
         export NGPU=${NGPU:-8}; \
         export GLOBAL_BATCH=${GLOBAL_BATCH}; \
         export PYBIN=${PYBIN}; \
         ${EXTRA_ENV:-} \
         cd ${WORK_DIR} && \
         nohup bash ${TRAIN_SH} ${MODE} > ${WORK_DIR}/log_joint_rank${i}.txt 2>&1 & \
         echo rank-${i} PID=\$!" &
done

echo "[launch] waiting 5s for workers..."
sleep 5

echo "[launch] starting rank-0 locally"
cd "$WORK_DIR"
NNODES=${NNODES} NODE_RANK=0 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
NGPU="${NGPU:-8}" GLOBAL_BATCH="${GLOBAL_BATCH}" PYBIN="${PYBIN}" \
    bash "$TRAIN_SH" "$MODE" 2>&1 | tee "${WORK_DIR}/log_joint_rank0.txt"
