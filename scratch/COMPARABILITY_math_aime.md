# Are our math500/AIME25 numbers comparable to the baseline table in `ICLR-27/results/math_aime_current.md`?

Checked 2026-09-14 because that table (TrimKV-Math, RKV, Tri-Attn on Qwen3-8B) is what IndexMem++
will sit beside, and it was produced by a different harness on a different cluster
(`/aifs4su/...`, allocation 386431, dgx-097) with its own pinned contract. Verified empirically,
not by reading the protocol prose.

## Verdict: comparable, with ONE bounded penalty against us. Do not silently "fix" it.

### 1. The problem sets are IDENTICAL — verified by fetching their pinned files

The baseline contract pins `sha256` over VaSE jsonl files. I fetched both at their pinned commit
(`terarachang/VaSE@bfa2692`) and the hashes reproduce exactly:

    math    446563 B  sha256 a239272930e5202b...  == pinned   True
    aime25   15812 B  sha256 12467eb9bb742396...  == pinned   True

Against our `alessiodevoto/{math500,aime25}` HF copies:

| | exact overlap | after stripping our `\boxed{}` instruction suffix |
|---|---|---|
| math500 | 0/500 | **500/500** |
| aime25 | 0/30 | **30/30** |

So the underlying problems and answers are the same set; the only textual difference is that our
dataset appends `"\nRemember to put your final answer within \boxed{}."` to each prompt. That is a
prompt difference, not a data difference — worth one sentence in the paper, since it plausibly
*helps* answer extraction and therefore helps us slightly.

### 2. Sampling matches exactly

`do_sample=True`, `temperature 0.6`, `top_p 0.95`, `top_k 20`, `enable_thinking=True`, and the
same pass@1 definition (`sum(correct) / (problems x rollouts)`, never best-of-R or majority vote).
Rollouts match too: math 2, aime25 8.

### 3. The one real discrepancy: our math500 cap is 16384, theirs is 32768

Their contract is `max_length 32768` for **prompt + output**; our math500 runs used
`max_new_tokens 16384` (our aime25 runs used 32768 and match).

**Measured cost on our own K=1024 predictions (n=1000):**

    median 3096 tok   p90 9112   p99 16196   max 17906
    traces >= 97% of cap:  11 (1.10%)
    unanswered:            58 (5.8%)
      of which cap-pressed: 10
      of which short (genuine non-termination / no box): 48

So the cap can account for **at most 1.0 of the 100 pass@1 points**, and it is one-directional: it
can only *understate* IndexMem++. The other 48 unanswered traces are short, i.e. the model stopped
without boxing — a real failure, not a truncation artifact.

**Recommendation: report as-is and state the 16384-vs-32768 difference plus this 1.0-point bound.**
Re-running math500 at 32768 to close a <=1 point gap that runs against us costs ~13 GPU-h per arm
and would delay the table; and a number that moves by <=1 point does not change any claim. What
would be wrong is to quietly present it as though the caps matched.

### 4. Differences that do NOT affect comparability

* **Seed scheme.** Ours is `seed + 1000*r` (per rollout); theirs is
  `42 + 1e6*[aime] + 100*problem_index` with both rollouts sharing one batch RNG stream. Different
  streams, but pass@1 over 1000 (or 240) sampled traces does not depend on the stream, only on the
  distribution. It does mean our dense-vs-sparse pairing is internal to our own arms and cannot be
  extended to theirs — a paired test against a baseline row is not available.
* **Harness.** They use the Random-Attention native loop at commit `64db9688`; we use kvpress.
  Same sampling parameters and same grader semantics (last `\boxed{}`, brace-matched), so the
  measured quantity is the same.
* **`answered` denominator.** We report `correct/total`, matching their `pass@1` definition. Their
  table reports `Answers` as a completeness check (`1000/1000`) rather than a box-rate, so their
  "Answers" column and our `answered` column are NOT the same quantity — do not put them in one
  column. Ours is "produced a `\boxed{}`"; theirs is "the expected rollout identities all exist".

## Numbers as they stand

| arm | pass@1 | answered (box rate) |
|---|---|---|
| IndexMem++ math500 K=1024 | **83.20%** | 942/1000 |
| IndexMem++ aime25 K=2048 | 55.42% | 181/240 |
| IndexMem++ aime25 K=1024 | 47.92% | 169/240 |
| TrimKV-Math math500 K=1024 (baseline) | 94.20% | 1000/1000 (completeness) |
| TrimKV-Math aime25 K=2048 (baseline) | 62.50% | 240/240 (completeness) |

**Read this honestly: TrimKV-Math is ahead of IndexMem++ on both math benchmarks at matched
budget.** TrimKV-Math uses task-specific fine-tuned weights (`ngocbh/TrimKV-Qwen3-8B-Math`) while
IndexMem++ runs the stock `Qwen3-8B` with a trained router only, so the comparison is not
like-for-like on training budget — but that is an explanation, not a win, and the table should not
be arranged to obscure it. RKV and Tri-Attn rows are still `in_progress`/`pending` in that
snapshot, so the full baseline picture is not yet known.
