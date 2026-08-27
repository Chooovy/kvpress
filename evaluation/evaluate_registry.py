# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

from kvpress import (
    AdaKVPress,
    BlockPress,
    CAMPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVComposePress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LUKVPress,
    MergingPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)


def _lazy_scorer(module: str, function: str = "calculate_metrics"):
    """Load a benchmark's optional metric dependencies only when it is scored."""

    def score(*args, **kwargs):
        return getattr(import_module(module), function)(*args, **kwargs)

    return score


# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

SCORER_REGISTRY = {
    "loogle": _lazy_scorer("benchmarks.loogle.calculate_metrics"),
    "ruler": _lazy_scorer("benchmarks.ruler.calculate_metrics"),
    "zero_scrolls": _lazy_scorer("benchmarks.zero_scrolls.calculate_metrics"),
    "infinitebench": _lazy_scorer("benchmarks.infinite_bench.calculate_metrics"),
    "longbench": _lazy_scorer("benchmarks.longbench.calculate_metrics"),
    "longbench-e": _lazy_scorer("benchmarks.longbench.calculate_metrics", "calculate_metrics_e"),
    "longbench-v2": _lazy_scorer("benchmarks.longbenchv2.calculate_metrics"),
    "needle_in_haystack": _lazy_scorer("benchmarks.needle_in_haystack.calculate_metrics"),
    "aime25": _lazy_scorer("benchmarks.aime25.calculate_metrics"),
    "math500": _lazy_scorer("benchmarks.math500.calculate_metrics"),
}


PRESS_REGISTRY = {
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "cur": CURPress(),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "fastkvzip": FastKVzipPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvcompose": KVComposePress(),
    "kvcompose_unstructured": KVComposePress(structured=False),
    "kvzip": KVzipPress(),
    "kvzip_plus": KVzipPress(kvzip_plus_normalization=True),
    "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
    "kvzap_mlp_head": KVzapPress(model_type="mlp"),
    "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
    "lagkv": LagKVPress(),
    "lukv": LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "no_press": None,
    "cam_streaming_llm": CAMPress(base_press=StreamingLLMPress()),
    "cam_knorm": CAMPress(base_press=KnormPress()),
    "cam_adakv_snapkv": CAMPress(base_press=AdaKVPress(SnapKVPress())),
    "cam_tova": CAMPress(base_press=TOVAPress()),
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
    # MergingPress: merge-on-evict during prefill (values-only merge preserves RoPE keys)
    "merging_knorm": MergingPress(KnormPress()),
    "merging_snapkv": MergingPress(SnapKVPress()),
    "merging_expected_attention": MergingPress(ExpectedAttentionPress(epsilon=1e-2)),
    "merging_kvzap_mlp": MergingPress(KVzapPress(model_type="mlp")),
}
