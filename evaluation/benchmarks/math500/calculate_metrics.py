# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from benchmarks.boxed_answer import extract_boxed, is_answered, score_boxed


def score_math500(pred_answer, true_answer) -> bool:
    """Whether the prediction's final \\boxed{} matches the reference."""
    return score_boxed(pred_answer, true_answer)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Accuracy over \\boxed{} answers, plus how many rows produced a box at all.

    ``answered`` is reported because a missing box and a wrong number fail for different reasons: a
    truncated or looping trace against a completed-but-wrong one. A compression arm can cause the
    first directly, and accuracy alone cannot tell them apart.

    Extraction is brace-matched and formatting-normalized -- see ``benchmarks.boxed_answer``. The
    scorer this replaced split on the first ``}``, which truncated the 21% of math500 references that
    contain a brace (``\\boxed{\\frac{1}{4}}`` -> ``\\frac{1``) and under-reported every arm.
    """
    correct = sum(bool(score_math500(row["predicted_answer"], row["answer"])) for _, row in df.iterrows())
    answered = sum(is_answered(row["predicted_answer"]) for _, row in df.iterrows())
    return {
        "correct": int(correct),
        "answered": int(answered),
        "accuracy": correct / len(df) if len(df) else 0.0,
        "total": len(df),
    }


__all__ = ["calculate_metrics", "score_math500", "extract_boxed"]
