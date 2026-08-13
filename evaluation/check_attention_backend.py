# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Is the backbone producing sane logits at all, and does the attention kernel change the answer?

Run this before trusting any evaluation number. It generates a few tokens from a prompt whose
answer is stated in the prompt itself, once per attention implementation, and reports whether the
model reproduces it.

    python check_attention_backend.py --model /path/Qwen3-8B

Why it exists: ``evaluate.py`` selects ``flash_attention_2`` whenever ``import flash_attn``
succeeds, without checking that the build matches the installed torch. A mismatched build does not
raise -- it silently returns wrong attention output, and the only visible symptom is that every
task scores 0.0 with the first generated token already garbage ("matplotlib matplotlib ..."), which
reads like a broken model or a broken metric rather than a broken kernel. Two backends disagreeing
here localizes that in about a minute instead of a full evaluation run.

Exits non-zero if any available backend fails the check, so it can gate a launch.
"""

from __future__ import annotations

import sys

import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer

#: Stated in the prompt, so any model that attends correctly repeats it. A number the tokenizer
#: splits into several pieces, so copying it requires the attention to be right for several steps.
MAGIC = "6188935"
PROMPT = (
    f"The special magic number for questionable-tangerine is {MAGIC}. "
    "What is the special magic number for questionable-tangerine?"
)


def available_backends() -> list[str]:
    """The implementations worth testing on this box, cheapest signal first."""
    backends = ["sdpa", "eager"]
    try:
        import flash_attn  # noqa: F401

        backends.insert(0, "flash_attention_2")
    except ImportError:
        print("flash_attn not installed -- flash_attention_2 will not be tested (or used).")
    return backends


def main(
    model: str = "Qwen/Qwen3-8B",
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    max_new_tokens: int = 24,
    backends: str = "",
) -> int:
    """
    Generate the same prompt under each attention backend and report which reproduce the answer.

    Parameters
    ----------
    backends : str
        Comma-separated subset to test; defaults to every one available here.
    """
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    try:
        import flash_attn

        print(f"flash_attn {flash_attn.__version__}")
    except ImportError:
        pass
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; flash_attention_2 cannot be tested here.", file=sys.stderr)

    to_test = [b.strip() for b in backends.split(",") if b.strip()] or available_backends()
    tokenizer = AutoTokenizer.from_pretrained(model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    results: dict[str, bool] = {}
    for backend in to_test:
        print(f"\n=== {backend} ===", flush=True)
        try:
            loaded = AutoModelForCausalLM.from_pretrained(
                model, dtype=getattr(torch, dtype), attn_implementation=backend
            )
        except TypeError:  # older transformers used torch_dtype
            loaded = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=getattr(torch, dtype), attn_implementation=backend
            )
        except (ValueError, ImportError) as exc:
            print(f"  unavailable: {exc}")
            continue
        loaded = loaded.to(device).eval()

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            # Greedy: sampling would make a wrong answer look like bad luck.
            out = loaded.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        ok = MAGIC in generated
        results[backend] = ok
        print(f"  generated: {generated!r}")
        print(f"  contains {MAGIC}: {'YES' if ok else 'NO  <-- this backend is broken'}")

        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    for backend, ok in results.items():
        print(f"  {backend:20s} {'ok' if ok else 'BROKEN'}")
    if not results:
        print("  no backend could be tested")
        return 1
    broken = [b for b, ok in results.items() if not ok]
    if broken:
        print(
            f"\n{', '.join(broken)} produced the wrong answer to a question the prompt answers.\n"
            "Do not evaluate with it: attention is returning wrong values, and every task will\n"
            "score ~0 in a way that looks like a model or metric problem. Pass a working backend\n"
            "explicitly, e.g. --attn_implementation sdpa (evaluate.py / evaluate_sparse.py) or\n"
            "ATTN=sdpa (evaluate_dense_baseline.sh)."
        )
        return 1
    print("\nAll tested backends reproduce the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(Fire(main))
