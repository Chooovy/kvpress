# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class DecodePress(BasePress):
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        raise NotImplementedError("compress method must be implemented in subclass")
