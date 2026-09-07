# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract and compare ``\\boxed{...}`` answers, shared by the math500 and aime25 scorers.

Both benchmarks ask the model to "put your final answer within \\boxed{}" and then compare that
against a reference string. Both used to do it with

    pred_answer.split("boxed{")[i].split("}")[0]

which is wrong for any answer whose own LaTeX contains a brace, because ``split("}")[0]`` stops at
the *first* closing brace rather than the one that matches the opening one. ``\\boxed{\\frac{1}{4}}``
extracts as ``\\frac{1`` and compares unequal to the reference ``\\frac{1}{4}``.

This is not a rare shape: **103 of math500's 500 reference answers (21%) contain a brace**, so a
model that answers all of them perfectly is scored wrong on every one. Measured on a 10-row dense
Qwen3-8B sample, the naive extractor reported accuracy 0.30 where 0.80 was correct -- and it
under-reports every arm, so it also compresses the gap between them, which is the quantity these
evals exist to measure.

The two scorers also disagreed on *which* box to read: aime25 took ``[-1]`` (the last) and math500
``[1]`` (the first). For a reasoning trace the last one is the answer -- an earlier box is a step --
so both use the last here.

Comparison is exact after normalizing LaTeX that carries no mathematical content: surrounding
whitespace and ``$``, the ``\\left``/``\\right`` size hints, ``\\!``-style spacing macros, and
whitespace *inside* the expression (``16 \\sqrt{3}`` and ``16\\sqrt{3}`` are the same number, and
which one a model emits is a formatting coin flip). Nothing that could change a value is touched: no
fraction rewriting, no numeric parsing, no sympy round-trip. So this stays a *string* metric, just
one that is not defeated by a space.
"""

from __future__ import annotations

import re
from typing import Optional

#: Spacing macros that render as nothing and appear only for typesetting.
_SPACING_MACROS = ("\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad")


def extract_boxed(pred_answer: Optional[str]) -> Optional[str]:
    """
    Return the contents of the LAST ``\\boxed{...}``, brace-matched, or None if there is none.

    Brace-matched rather than split on ``}``: the answer's own LaTeX routinely contains braces, and
    stopping at the first one truncates 21% of math500's answers (see the module docstring).
    """
    if not isinstance(pred_answer, str):
        return None
    start = pred_answer.rfind("boxed{")
    if start == -1:
        return None
    i = start + len("boxed{")
    depth = 1
    out = []
    while i < len(pred_answer):
        char = pred_answer[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(char)
        i += 1
    # Unbalanced: the trace was truncated mid-answer (a real outcome when max_new_tokens bites).
    # Return what there is so it can still match a short reference, rather than None -- None would
    # be indistinguishable from "never attempted an answer", which `answered` reports separately.
    return "".join(out)


def normalize(answer: Optional[str]) -> Optional[str]:
    """Strip LaTeX that carries no mathematical content, so formatting cannot fail a right answer."""
    if answer is None:
        return None
    text = str(answer).strip()
    text = text.strip("$").strip()
    # Size hints: \left( ... \right) renders identically to ( ... ).
    text = text.replace("\\left", "").replace("\\right", "")
    # Display-size fraction/binomial variants. \dfrac and \tfrac differ from \frac ONLY in rendered
    # size -- \dfrac{1}{4} and \frac{1}{4} are the same number. This is not a nicety: Qwen3 in
    # thinking mode prefers \dfrac while the references use \frac, so without this 9 of 50 math500
    # rows were scored wrong for a typographic choice, and thinking mode measured 0.70 against
    # non-thinking's 0.80 -- an apparent regression that was entirely this normalization gap.
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\dbinom", "\\binom").replace("\\tbinom", "\\binom")
    for macro in _SPACING_MACROS:
        text = text.replace(macro, " ")
    # Trailing "\\text{ units}" and a trailing period are presentation, not value.
    text = text.rstrip(".").strip()
    # Whitespace inside LaTeX is not semantic: "16 \sqrt{3}" == "16\sqrt{3}".
    text = re.sub(r"\s+", "", text)
    # A trailing degree/percent marker is kept -- those DO change the value's meaning.
    return text


def is_answered(pred_answer: Optional[str]) -> bool:
    """Whether the model produced a box at all.

    Reported alongside accuracy because a missing box and a wrong number are different failures:
    the first means the trace was truncated or looping (which a top-k budget can cause directly),
    the second means it reasoned and erred. Scoring only accuracy conflates them.
    """
    return isinstance(pred_answer, str) and "boxed{" in pred_answer


def score_boxed(pred_answer: Optional[str], true_answer) -> bool:
    """True when the prediction's last box matches the reference after normalization."""
    predicted = normalize(extract_boxed(pred_answer))
    if predicted is None:
        return False
    return predicted == normalize(true_answer)
