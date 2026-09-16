# Is a `--evict` score comparable to a masking-path score?

Measured 2026-09-14 because the math tables use `--evict` (5.7x faster) while every RULER and
LongBench number in the paper came from the masking path. If the two select different keys, the
math table cannot sit beside the others without a caveat.

Scripts: `/tmp/parity_evict_vs_mask.py` (generated tokens), `/tmp/parity_logits.py` (first-step
logits), `/tmp/parity_sets2.py` (retained key sets, the decisive one).
Setup: Qwen3-8B, rvkl_16k ckpt, 4096-token random context, topk=1024, static table, sink4/local128.

## Answer: the same selection rule, two reference rows, and evict retains a superset.

### Retained key sets, per (layer, head)

Ranking the mask at the MIDDLE row (its documented default under decay):

| | evict | mask | evict-only | mask-only | jaccard |
|---|---|---|---|---|---|
| L0h0 | 2070 | 1995 | 79 | 4 | 0.960 |
| L17h3 | 403 | 367 | 52 | 16 | 0.838 |
| L35h0 | 584 | 507 | 107 | 30 | 0.777 |

Ranking the mask at the LAST row (what `prefill_and_commit` uses):

| | evict-only | mask-only | jaccard |
|---|---|---|---|
| L0h0 | 79 | **4** | 0.960 |
| L17h3 | 29 | **4** | 0.919 |
| L35h0 | 82 | **4** | 0.854 |

`mask-only` collapses to a constant **4** for every head — and 4 is `force_sink`, which the probe
adds to the mask side by hand while the pool stores sinks separately. So it is a probe artifact:
**at a common reference row the masking path selects a SUBSET of what the pool holds.**

### The two real differences

1. **Reference row.** The masking path ranks at the middle prefill row (documented: 1.64% deadline
   error vs 4.52% at row 0), the evict commit at the last row. That choice accounts for most of
   the set difference — `mask-only` falls from 4-36 to a constant 4 when matched.
2. **The pool is FULL at its budget; the deadline mask is not.** `evict` holds 2070 of a budgeted
   2074 at L0h0, while the mask keeps 1995. This is one-directional: evict retains a superset, so
   it is a slightly *weaker* compression at nominally the same budget.

### Consequences for the numbers

* First-step logits: `max |dlogit| = 0.281`, `mean 0.054`, **argmax agrees**, top-50 overlap 49/50.
  Note 0.281 is ~40x bf16 output rounding (7.5e-3), so this is NOT mere reduction-order noise --
  it is the superset above. My first script printed an "arithmetic only" interpretation; that was
  wrong and is retracted here.
* Greedy generation: **36 of 40 tokens identical**, first divergence at step 36, and the texts are
  paraphrases of each other ("straightforward. Wait," vs "straightforward, but I need").

### What to say in the paper

The math/AIME numbers come from the eviction path and the RULER/LongBench numbers from the masking
path. They are the same selection rule at the same budget, differing in the reference row and in
that eviction keeps its pool full. **Do not present them as bit-identical measurements.** The
honest framing is that eviction is the deployable arm (it physically frees memory) and masking is
the quality-measurement arm, and at matched budget eviction retains marginally MORE, so a math
number is not flattered by compression it did not pay for.

Cheap follow-up if a reviewer presses: run one math500 config on the masking path (~5.7x slower,
~13 GPU-h at K=1024) and report the delta.
