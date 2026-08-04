# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd


def extract_boxed(pred_answer):
    if not isinstance(pred_answer, str):
        return None
    try:
        # Check if boxed is present
        if "boxed{" in pred_answer:
            return str(pred_answer.split("boxed{")[1].split("}")[0])
        return None
    except IndexError:
        return None


def score_aime(pred_answer, true_answer):
    extracted = extract_boxed(pred_answer)
    if extracted is None:
        return False
    # AIME answers are typically integers, strip whitespace for comparison
    return extracted.strip() == str(true_answer).strip()


def calculate_metrics(df: pd.DataFrame) -> dict:
    correct = 0
    answered = 0
    for index, row in df.iterrows():
        correct += score_aime(row["predicted_answer"], row["answer"])
        if row["predicted_answer"] and isinstance(row["predicted_answer"], str) and "boxed{" in row["predicted_answer"]:
            answered += 1
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df) if len(df) > 0 else 0, "total": len(df)}
