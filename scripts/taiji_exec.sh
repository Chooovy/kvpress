#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command inside a running Taiji training pod (or open a shell there).
#
#   scripts/taiji_exec.sh                          # interactive shell
#   scripts/taiji_exec.sh nvidia-smi               # one command
#   scripts/taiji_exec.sh -- bash -lc 'cd /x && y'  # anything after -- goes through verbatim
#
# TASK/INSTANCE default to the H20 pod; override via env:
#   TASK=... INSTANCE=1 scripts/taiji_exec.sh ...
#
# `taiji_client exec` defaults to --tty=true --stdin=true, which needs a terminal. With no
# command we keep those (interactive shell); with a command we turn both off and turn --stderr
# on, because the default drops stderr and a failing command would otherwise look silent.
set -uo pipefail

TASK="${TASK:-basic_train_marcushaogu_20260814142753_87094d49}"
INSTANCE="${INSTANCE:-0}"
CONTAINER="${CONTAINER:-taiji-mpiv2}"

[[ "${1:-}" == "--" ]] && shift

if [[ $# -eq 0 ]]; then
  exec taiji_client exec -c "$CONTAINER" "$TASK" "$INSTANCE" bash
fi

# Not exec'd: we want the pipeline's exit status to survive, and taiji_client returns 0 even
# when the remote command fails, so the marker below is the only reliable signal.
out=$(taiji_client exec --tty=false --stdin=false --stderr -c "$CONTAINER" \
        "$TASK" "$INSTANCE" bash -lc "$* ; echo \"__RC=\$?\"" 2>&1)
printf '%s\n' "$out" | grep -v '^__RC='
rc=$(printf '%s\n' "$out" | sed -n 's/^__RC=\([0-9]*\)$/\1/p' | tail -1)
exit "${rc:-1}"
