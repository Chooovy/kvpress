# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from fire import Fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvpress.indexmem.evaluation import main

if __name__ == "__main__":
    Fire(main)
