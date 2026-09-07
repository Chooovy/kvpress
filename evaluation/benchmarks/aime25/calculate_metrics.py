# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from benchmarks.boxed_answer import extract_boxed, is_answered, score_boxed


def score_aime(pred_answer, true_answer) -> bool:
    """Whether the prediction's final \\boxed{} matches the reference."""
    return score_boxed(pred_answer, true_answer)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Accuracy over \\boxed{} answers, plus how many rows produced a box at all.

    ``answered`` matters more here than anywhere else: AIME traces are long (the dataset ships
    ``max_new_tokens=32000``), so a truncated trace is a routine outcome rather than an anomaly, and
    it scores identically to a confident wrong answer unless the two are reported separately.

    Extraction is shared with math500 (``benchmarks.boxed_answer``): brace-matched, last box,
    formatting-normalized. AIME answers are integers 0-999, so the brace fix does not change this
    benchmark's numbers -- it is here so the two math scorers cannot drift apart, which they already
    had (this one read the last box, math500 the first).
    """
    correct = sum(bool(score_aime(row["predicted_answer"], row["answer"])) for _, row in df.iterrows())
    answered = sum(is_answered(row["predicted_answer"]) for _, row in df.iterrows())
    return {
        "correct": int(correct),
        "answered": int(answered),
        "accuracy": correct / len(df) if len(df) else 0.0,
        "total": len(df),
    }


__all__ = ["calculate_metrics", "score_aime", "extract_boxed"]
