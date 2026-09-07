# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the shared ``\\boxed{}`` scorer behind math500 and aime25.

The scorer these replaced extracted with ``split("boxed{")[i].split("}")[0]``, which stops at the
first closing brace instead of the matching one. 103 of math500's 500 reference answers (21%) contain
a brace, so a perfect model was scored wrong on every one of them -- measured, a 10-row dense
Qwen3-8B sample reported 0.30 where 0.80 was right. The bug under-reports every arm, so it also
shrinks the gap between arms, which is what these evals exist to measure.

Half of these tests are therefore about *not* being too lenient either: a normalizer that is loose
enough to call 1/2 and 1/3 equal would hide real regressions just as effectively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

EVALUATION = Path(__file__).resolve().parents[2] / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from benchmarks.boxed_answer import (  # noqa: E402
    extract_boxed,
    is_answered,
    normalize,
    score_boxed,
)


class TestExtraction:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (r"so \boxed{6}", "6"),
            # The regression: nested braces must not truncate at the first "}".
            (r"answer: \boxed{\frac{1}{4}}", r"\frac{1}{4}"),
            (r"\boxed{\frac{3\sqrt{3}}{4}}", r"\frac{3\sqrt{3}}{4}"),
            (r"\boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)"),
            (r"\boxed{\text{Evelyn}}", r"\text{Evelyn}"),
        ],
    )
    def test_brace_matching(self, text, expected):
        assert extract_boxed(text) == expected

    def test_takes_the_last_box_not_the_first(self):
        """An earlier box is a step in the trace; the final one is the answer.

        The two scorers used to disagree about this -- aime25 read [-1] and math500 [1] -- so the
        same trace could score differently on the two benchmarks for no stated reason.
        """
        trace = r"first \boxed{3}, on reflection \boxed{7}"
        assert extract_boxed(trace) == "7"

    def test_no_box_is_none(self):
        assert extract_boxed("I could not solve it") is None
        assert extract_boxed(None) is None
        assert extract_boxed(float("nan")) is None

    def test_truncated_box_returns_the_partial_answer(self):
        """A trace cut off mid-answer yields what there is, not None.

        None is reserved for "never attempted a box" so that `answered` stays a meaningful signal;
        a truncated box did attempt one.
        """
        assert extract_boxed(r"\boxed{123") == "123"


class TestNormalization:
    @pytest.mark.parametrize(
        "a, b",
        [
            (r"16 \sqrt{3}", r"16\sqrt{3}"),  # interior whitespace is not semantic
            (r"\left( 3, 4 \right)", r"(3, 4)"),  # size hints render identically
            ("$42$", "42"),
            ("  7  ", "7"),
            (r"\frac{1}{2}", r"\frac{1}{2} "),
            (r"2\!x", "2x"),
            # Display-size fraction variants: same number, different rendered size. Qwen3 in
            # thinking mode prefers \dfrac while the references use \frac, and treating them as
            # different scored 9 of 50 math500 rows wrong -- reporting thinking mode as 0.70 against
            # non-thinking's 0.80, an apparent regression that was purely this gap.
            (r"\dfrac{1}{4}", r"\frac{1}{4}"),
            (r"\tfrac{3}{8}", r"\frac{3}{8}"),
            (r"-\dfrac{3}{8}", r"-\frac{3}{8}"),
            (r"\dfrac{3\sqrt{3}}{4}", r"\frac{3\sqrt{3}}{4}"),
            (r"\dbinom{5}{2}", r"\binom{5}{2}"),
        ],
    )
    def test_equivalent_formatting_compares_equal(self, a, b):
        assert normalize(a) == normalize(b)

    @pytest.mark.parametrize(
        "a, b",
        [
            (r"\frac{1}{2}", r"\frac{1}{3}"),  # different values
            ("42", "43"),
            ("-5", "5"),
            (r"\frac{1}{2}", "2"),
            ("90", "90^\\circ"),  # a degree marker changes the meaning
            ("50", "50\\%"),
            # The \dfrac collapse must not extend to different values or dropped subscripts.
            (r"\dfrac{1}{4}", r"\frac{1}{5}"),
            (r"\dfrac{721}{9}", r"\frac{3}{56}"),
            ("52", "52_8"),  # a base subscript is part of the answer
        ],
    )
    def test_different_values_stay_different(self, a, b):
        """The normalizer must not be lenient enough to hide a real regression."""
        assert normalize(a) != normalize(b)


class TestScoring:
    def test_nested_brace_answer_scores_correct(self):
        assert score_boxed(r"... so \boxed{\frac{1}{4}}", r"\frac{1}{4}")

    def test_wrong_answer_scores_wrong(self):
        assert not score_boxed(r"\boxed{\frac{1}{5}}", r"\frac{1}{4}")

    def test_missing_box_scores_wrong(self):
        assert not score_boxed("no idea", "4")

    def test_is_answered_separates_truncation_from_error(self):
        assert is_answered(r"\boxed{4}")
        assert not is_answered("a long trace that never concluded")


class TestMetrics:
    def test_math500_reports_accuracy_and_answered(self):
        from benchmarks.math500.calculate_metrics import calculate_metrics

        df = pd.DataFrame(
            {
                # right, right-with-braces, wrong, never answered
                "predicted_answer": [
                    r"\boxed{4}",
                    r"\boxed{\frac{1}{4}}",
                    r"\boxed{9}",
                    "ran out of tokens",
                ],
                "answer": ["4", r"\frac{1}{4}", "8", "5"],
            }
        )
        metrics = calculate_metrics(df)
        assert metrics == {"correct": 2, "answered": 3, "accuracy": 0.5, "total": 4}

    def test_aime25_matches_math500_on_the_same_frame(self):
        """The two must not drift: they were reading different boxes."""
        from benchmarks.aime25.calculate_metrics import calculate_metrics as aime
        from benchmarks.math500.calculate_metrics import calculate_metrics as math500

        df = pd.DataFrame(
            {"predicted_answer": [r"step \boxed{1} then \boxed{70}"], "answer": ["70"]}
        )
        assert aime(df) == math500(df)

    def test_empty_frame_does_not_divide_by_zero(self):
        from benchmarks.math500.calculate_metrics import calculate_metrics

        metrics = calculate_metrics(pd.DataFrame({"predicted_answer": [], "answer": []}))
        assert metrics["total"] == 0 and metrics["accuracy"] == 0.0
