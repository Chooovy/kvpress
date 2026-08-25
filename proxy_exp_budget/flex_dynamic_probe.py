# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Which of the two compiled callables owns the flex path's ``illegal memory access``?

The crash (`qi_flex_attention.py`, all 4 shards of a RULER 8K run, and again at ``fraction=0.02``)
is reported at a `torch.tensor` construction under `CUDA_LAUNCH_BLOCKING=0` but resolves, with
blocking on, to an **inductor-compiled Triton kernel** launched inside a compiled region -- i.e. it
is a compilation/guard problem, not an out-of-bounds index in ``deadlines`` (which passes standalone
at every RULER length, including non-multiples of the 128 block size, plus 60 sequential decode
steps).

``_flex()`` and ``_block_mask()`` *both* compile with ``dynamic=None`` (torch's automatic mode:
specialise on the first shape, re-specialise to dynamic when a second arrives). Every RULER context
has a different length, so a sweep drives that transition constantly. Forcing **both** to
``dynamic=True`` makes the crashing run complete -- but the module's perf note measures
``dynamic=None`` at 45 ms steady state against ``dynamic=True``'s 86 ms, so flipping both is a ~2x
tax.

The crashing kernel is named ``triton_per_fused__to_copy_slice_sum_transpose_6`` -- a
copy/slice/sum/transpose reduction, which is the shape of ``create_block_mask``'s block-wise
reduction rather than the attention kernel. So the useful question is not "does ``dynamic=True`` fix
it" (answered: yes) but **which callable needs it**, since fixing only ``_block_mask`` would keep
``flex_attention`` itself at the fast setting.

Run as::

    python flex_dynamic_probe.py none        # reproduce: both dynamic=None -> crashes
    python flex_dynamic_probe.py bm_true     # block_mask=True, flex=None   <- the hypothesis
    python flex_dynamic_probe.py flex_true   # flex=True, block_mask=None
    python flex_dynamic_probe.py true        # both True -> known to survive

Whichever minimal setting survives is the fix; the surviving combination also has to be checked for
*selection equality* against the gather path, since a compile mode must not change which keys are
chosen (`tests/presses/test_gqa_indexer_qi_flex.py` covers that at fixed shapes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/apdcephfs_tj5/share_300719894/user/guhao/kvpress")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import kvpress.presses.gqa_indexer.qi_flex_attention as qf  # noqa: E402

#: (flex dynamic, block_mask dynamic) per mode name.
MODES = {
    "none": (None, None),
    "bm_true": (None, True),
    "flex_true": (True, None),
    "true": (True, True),
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "none"
FLEX_DYN, BM_DYN = MODES[MODE]

from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: E402

# Pre-populate the module's memoised compiles so its own lazy _flex()/_block_mask() never build the
# default ones. Same objects, only `dynamic` differs.
qf._flex_compiled = torch.compile(flex_attention, dynamic=FLEX_DYN)
qf._block_mask_compiled = torch.compile(create_block_mask, dynamic=BM_DYN)
print(f"[probe] mode={MODE}: flex dynamic={FLEX_DYN!r}, block_mask dynamic={BM_DYN!r}", flush=True)

from evaluate_sparse import SparseEvaluationConfig, SparseEvaluationRunner  # noqa: E402

config = SparseEvaluationConfig(
    dataset="ruler",
    data_dir="8192",
    model="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B",
    indexer_ckpt=(
        "/apdcephfs_gy8/share_303843174/guhao/models/"
        "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256/final.pt"
    ),
    topk=2048,
    force_local=64,
    force_sink=4,
    block_k=64,
    precision="tf32",
    # Small on purpose: the crash reproduced at 0.02, so this is the cheapest signal.
    fraction=0.02,
    seed=42,
    device="cuda:0",
    output_dir=f"/tmp/flexbug_{MODE}",
)
SparseEvaluationRunner(config).run()
print(
    f"[probe] mode={MODE} (flex={FLEX_DYN!r}, block_mask={BM_DYN!r}) "
    "COMPLETED WITHOUT CRASHING",
    flush=True,
)
