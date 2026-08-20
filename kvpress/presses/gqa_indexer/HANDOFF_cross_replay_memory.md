# RESOLVED: cross-replay 8K 73 GiB peak — it was the SDPA backend

**Closed.** 8K peak **73.2 → 27.8 GiB** (16K: 33.5 GiB), 1.8× faster, losses and training signal
unchanged. Full accounting in `cross_replay_e2e.md` **§6.3** (cause) and **§6.4** (measured table,
which replaces §6.1's estimate). The original handoff brief is preserved below the line.

## Verdict on each hypothesis

**H1 — SDPA falls back to MATH, materializing the attention matrix.**
**CONFIRMED — this was it.** 1288 MiB/layer × 36 = **46.7 GiB**, the whole gap. The stated mechanism
was incomplete, though; see the next section.

**H2 — `max_memory_allocated` never reset, so the peak includes model loading.**
**REFUTED.** The reset was already in `smoke_cross_replay.py`, the weights report separately at
15.3 GiB, and the per-step peak was still 73.2 GiB. The constancy across steps was real, not an
artifact: the step genuinely cost that.

**H3 — `h_C` is retained twice.**
**CONFIRMED**, at exactly 2 × 2.25 GiB — but it is 8% of the peak, not the bug. The second copy is
`in_norm(h)`, saved by autograd as `w_out`'s input, so `hidden.clear()` cannot free it. §6.1(b)'s
two-stage backward would remove both.

**H4 — something in pass 1 retains a graph.**
**REFUTED** on the real model: pass 1 accounts for exactly `KV(C)` + `h_C` = 3.38 GiB, matching the
no-graph prediction.

## The one correction to H1 worth carrying forward

H1 named **one** condition (flash rejects any `attn_mask`). There are **three**, and all must hold:

1. non-`None` `attn_mask` → excludes **flash**
2. GQA head mismatch with a dense mask present → excludes **mem-efficient**
3. mask **requires grad** → excludes **cuDNN**

Drop any one and a fused kernel survives. Condition 3 is the trap: with a **detached** gate, cuDNN is
eligible and the same call retains **8.1 MiB** — so a backend probe written the obvious way reports
"fused, no problem". But `dL/ds` arriving through the mask *is* the objective, so production always
requires grad. Had I probed with a detached tensor I would have concluded H1 was refuted too.

## The fix, and the three ways it can bite back

`flex_attention` + `score_mod` (as instructed — no Triton kernel). Verified against the SDPA mask
route: forward 5.6e-16 fp64, `dL/dq` 1.1e-15, `dL/dgate` 4.8e-07 (the fp32 floor the gate path already
sits at). Each hazard below is measured and guarded, and two of them are **worse than the original
bug** — an optimisation with a catastrophic degradation path needs tests on that path:

* **The `torch.compile` *is* the fix.** Eager `flex_attention` materializes: **18730 MiB** for one
  layer vs 40 MiB compiled — 14× worse than the MATH bug. Guarded by `_note_flex_shape`, which warns
  at dynamo's 8-shape `recompile_limit` (the 9th shape silently reverts to eager).
* **Inductor has no valid Triton config for `64 <= Sq < 128`** at `Sk=8192, D=128` (shared-memory
  limit) and *raises* rather than degrading. Any ragged final chunk in that band hits it — e.g.
  `--context-len 8292 --query-chunk 1024` → 100. Fixed by padding queries to 128 and slicing back
  (exact); mutation-tested by setting `_FLEX_Q_ALIGN = 1`, which reproduces the crash.
* **`donated_buffer` must be off.** It asserts no compiled backward runs with `retain_graph=True`,
  which is exactly what `logit_chunk` does — so every `--logit-chunk` run raised. Found only by
  sweeping both chunk knobs together; the repro command in the brief below does not pass
  `--logit-chunk`, which is why it was invisible.

## Guards and tests

The three silent-failure guards are untouched (`pin_mode="self"` rejection, explicit rectangle mask,
`k_len == |C|` check). Tests: **35 passed + 12 skipped on CPU** (was 33), **47 passed on GPU**. The two
unrelated pre-existing failures elsewhere in `tests/presses/` are unchanged.

New tests, all aimed at *what the optimisation optimises* rather than only at exactness — MATH computes
the right answer, so every existing exactness test passed straight through this bug:

* `test_a_masked_gqa_sdpa_call_has_no_fused_backend` — no fused backend exists for this call, and
  each of the three conditions above is load-bearing (verified by dropping them one at a time).
* `test_the_cuda_path_never_calls_sdpa_with_a_gate_mask` — the gated replay never reaches SDPA on CUDA.
* `test_flex_and_sdpa_paths_agree_on_forward_and_gate_gradient` — the two routes are the same
  function, to the fp32 floor.
* `test_flex_runs_for_every_ragged_chunk_length` — `Sq` ∈ {1, 17, 64, 100, 127, 128, 129} all run.
* `test_query_padding_does_not_change_the_result` — padding does not perturb `dL/ds`.
* `test_flex_fallback_is_diagnosed_rather_than_silent` — every fallback reason is reported
  (a pure function, so it runs on CPU too).
* `test_recompile_pressure_is_warned_before_dynamo_gives_up` — the eager-revert warning fires before
  the cliff, not after.

Also recovered `test_the_rectangle_is_not_the_causal_fast_path`, which had lost its `def` line and was
dead code appended to the end of `test_a_real_mask_is_still_honoured` — its assertions had not been
running.

## Next lever

Not memory-critical any more, but the profile has inverted: the *fixed* cost now grows with `N`
(`KV(C)` + two copies of `h_C` = 5.6 GiB at 8K, 11.2 at 16K) while the chunk cost does not. So
§6.1(b)'s `h_C`-free two-stage backward — verified bit-exact, previously deferred as "only pays off
above 16K" — is now the largest remaining term rather than a nicety.

---

# Original handoff brief (preserved)

You have a GPU. I do not (CPU-only box), which is why this is being handed over: the remaining
question is a **CUDA SDPA backend-selection question** that cannot be answered without a device.

## Repo / files

`/apdcephfs_tj5/share_300719894/user/guhao/kvpress`, branch `feat/gqa-indexer`.

| file | role |
|---|---|
| `kvpress/presses/gqa_indexer/cross_replay.py` | the objective (`CrossReplayTrainer`, `cross_replay_training_step`) |
| `kvpress/presses/gqa_indexer/cross_replay_e2e.md` | design notes + every number verified so far. **Read §6.1 and §6.2.** |
| `tests/presses/test_gqa_indexer_cross_replay.py` | 33 tests, all passing on CPU |
| `scripts/smoke_cross_replay.py` | the script that produced the numbers below |

## What the objective does (one paragraph)

Train a query-independent per-key indexer (`s_i = f(h_i)`, `ScalarIndexer`) from a cross-replay LM
loss. Pass 1: dense ungated prefill of context `C` under `no_grad`, keeping `h_C` and `KV(C)`.
Pass 2: replay the same tokens as `C'` against `KV(C)` **only** (`ReadOnlyCache`, so `C'` never
enters the cache), every replay query seeing every `C` key (full rectangle, explicit all-zero 4D
mask), with the per-key score added to the attention logits as a `(B, H, 1, N)` additive
`attn_mask`. LM loss on `C'`. Replay queries are chunked (`query_chunk`) and each chunk is
backwarded immediately.

## The symptom

`--context-len 8192 --steps 5 --query-chunk 1024` on Qwen3-8B, bf16:

```
step 0 | loss 15.3042 | grad_norm 3.474 | participation 0.8827 | gate_scale 1.00000 | peak 73.1 GiB
step 1 | loss 14.7545 | grad_norm 2.462 | participation 0.8395 | gate_scale 0.99927 | peak 73.2 GiB
step 4 | loss 13.6490 | grad_norm 2.777 | participation 0.4226 | gate_scale 0.99680 | peak 73.2 GiB
shuffle control | learned 13.4137 | shuffled 15.3645 | delta +1.9508
```

**Estimated peak was 23.9 GiB** (16.0 weights + 2.25 `h_C` + 1.12 `KV(C)` + 3.66 one chunk's
activations + 0.87 one chunk's logits). Gap ≈ **49 GiB**.

## What I already tried, and why it was wrong

I hypothesised that `mask = bias + attention_mask` was broadcasting `(B,H,1,N) + (1,1,Sq,N)` into a
materialized `(B,H,Sq,N)` tensor (0.50 GiB/layer bf16 × 36 = 18 GiB). I fixed it (the rectangle mask
is all zeros, so it is now dropped rather than added, recognized by a `_kvpress_all_zero` tag).

**The fix changed nothing: peak went 73.1 → 73.1 GiB, and the losses are byte-identical.** So either
that term was never the cost, or something else dominates and hides it. My diagnosis was wrong.
The fix is correct on its own terms (verified: SDPA now receives `(1,4,1,16)`, query axis intact) and
is worth keeping, but it is **not** the answer.

## Verified facts (do not re-litigate these)

Verified on CPU with tiny models unless noted:

- The `_kvpress_all_zero` tag **does** survive to `_attention` in both fp32 and bf16, and SDPA
  receives a mask of shape `(1, 4, 1, 16)` — query axis is **not** materialized.
- `query_chunk` is exact: loss bit-identical across chunk sizes, gradients agree to fp32 epsilon.
- `query_chunk` *does* reduce retained memory on CPU: peak 2461 → 1428 → 910 → 651 KiB for
  `None/32/16/8` on a 4-layer fp64 model (3.8×).
- `logit_chunk` is exact and also reduces retention.
- Pass 1 is ungated and builds no graph (`h.requires_grad == False`).
- 33/33 tests pass on CPU. Two unrelated pre-existing failures elsewhere in `tests/presses/`
  (`test_per_tile_upcast_is_bit_identical[4-9]`, `test_capacity_model_prices_the_stage2_tile_gather`)
  — confirmed pre-existing, not caused by this work.

## Ranked hypotheses for the 49 GiB

**H1 — SDPA falls back to the MATH backend, materializing the attention matrix. (most likely)**
The gate *is* a non-None `attn_mask`, and PyTorch's flash backend does not accept `attn_mask` at all
(only `is_causal`). If the mem-efficient backend also rejects the stride-0 broadcast
`(B, H, 1, N)`, SDPA lands on math, which retains the full `(B, H, Sq, Sk)` score matrix **and** its
softmax output. At `H=32, Sq=1024, Sk=8192`: 0.50 GiB/layer bf16 (1.00 fp32) → **18 GiB (or 36 GiB
if upcast) over 36 layers**, twice over if both the pre- and post-softmax tensors are kept.

This is the risk `cross_replay_e2e.md` §6.1 explicitly flagged as unverified. It is the first thing
to measure.

**H2 — `torch.cuda.max_memory_allocated()` is never reset, so the reported peak includes model
loading.** The number is suspiciously *constant* (73.1, 73.2, 73.2, 73.2, 73.2) and identical across
two different code versions. A genuine 18 GiB reduction should have moved it. Add
`torch.cuda.reset_peak_memory_stats()` after the model is on device and before the training loop, and
report per-step peaks. **Check this before anything else — it is cheap and would explain the
insensitivity to my fix.**

**H3 — `h_C` is retained twice.** `score_context` builds a graph from `h`, so `h` stays alive through
the score graph even after `hidden.clear()`. Expected (2.25 GiB), but worth confirming it is not
2.25 × 2 or held in fp32.

**H4 — something in pass 1 retains a graph after all.** `prefill` is decorated `@torch.no_grad()`,
and CPU tests confirm `h.requires_grad == False`, so this should be excluded — but confirm on the
real model, since a 36-layer 8K graph would be ~29 GiB, which is the right order of magnitude.

## What to do

1. **H2 first.** Insert `torch.cuda.reset_peak_memory_stats()` before the loop in
   `scripts/smoke_cross_replay.py`, rerun, and report whether the per-step peak drops.
2. **Then H1.** Instrument `CrossReplayTrainer._attention` (`cross_replay.py`, around line 435) to
   report which backend SDPA chooses, e.g. with
   `torch.backends.cuda.can_use_efficient_attention(...)` / `can_use_flash_attention(...)` on the
   actual tensors, or by running one step under
   `with torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):` and seeing whether it
   raises. Report the verdict and the reason string.
3. **If H1 confirmed**, the intended fix is **`torch.nn.attention.flex_attention`**, not a
   hand-written Triton kernel: a `score_mod` returning `score + gate[h, kv_idx]` is exactly this
   operation and compiles to a fused kernel with no materialized mask. `flex_attention` is available
   (torch 2.13). Note `cross_replay_e2e.md` §1.1 explains why the existing
   `triton_gated_attention.py` is *not* needed here — please do not reintroduce it without reading
   that section.
   - A cheaper stopgap worth measuring first: make the gate a **contiguous** `(B, H, Sq, N)` expand
     only if that flips SDPA to mem-efficient; if not, skip it.
4. Report the corrected memory accounting so `cross_replay_e2e.md` §6.1's table can be fixed. That
   table is currently an **estimate**, and the estimate is evidently wrong by 3×.

## Constraints and conventions

- **Do not weaken the three silent-failure guards** (`cross_replay_e2e.md` §0): `pin_mode="self"`
  rejection, the explicit rectangle mask, and the `k_len == |C|` check. Each corresponds to a
  failure that produces a clean loss curve and an untrained router. Mutation-tested.
- **Any optimisation needs a test for the thing it optimises**, not only for the invariant it must
  not break. Three bugs in this work were "correct but achieved nothing" (chunking saved no memory;
  `skip_logits` was silently a no-op; the mask broadcast) and the exactness tests passed through all
  three. See `cross_replay_e2e.md` §9.
- Keep corrections visible in the `.md` rather than editing them away — including reverting my
  wrong §6.2 diagnosis once you know the real cause. That file's value is the record of which
  conclusions survived.
- Line length 120. Run `python -m pytest tests/presses/test_gqa_indexer_cross_replay.py -q` (CPU,
  ~7 s) before and after.

## Reproduce

```bash
cd /apdcephfs_tj5/share_300719894/user/guhao/kvpress
python -m scripts.smoke_cross_replay \
  --model /apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B \
  --context-len 8192 --steps 5 --query-chunk 1024
```

`--logit-chunk N` bounds the `lm_head` logits separately; `--no-shuffle-control` skips two extra
replay passes.

## Not part of this task

The training signal itself looks healthy and is **not** what needs fixing: participation falls
0.883 → 0.423 (the gate concentrating, which is what eviction needs) and the shuffle control is
+1.95 nats/token. Caveat for whoever reads those numbers later: the smoke script feeds **random
token ids**, so they establish the mechanism, not that anything useful was learned. Do not tune the
objective; just fix the memory.
