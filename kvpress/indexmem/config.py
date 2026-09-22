from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexMemConfig:
    cache_budget: int = 2048
    sink_size: int = 4
    window_size: int = 128
    cmp_slots: int = 64
    head_budget: str = "mass"
    head_budget_table: str = ""
    min_head_budget: int = 512
    cache_budget_ratio: float | None = None
    decode_batch: int = 1
    inference_mode: str = "evict"
