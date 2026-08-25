# Cross-replay LM loss for a query-independent indexer: design notes

Working notes for the `[C ; C']` cross-replay objective sketched in
`query_independent_indexer_cross_replay.md`, written while auditing whether it can be built on the
existing e2e path (`e2e_trainer.py`, `gated_attention.py`, `gate_pin.py`).

**Provenance of the numbers.** Every figure below was produced on a **CPU box with no GPU**, on a
randomly-initialised 2–3 layer Qwen3 (`hidden_size=64`, `head_dim=16`) unless stated otherwise.
They are therefore **mechanism and invariant checks only** — "does this tensor flow", "are these two
layouts equal", "does this constant cancel". They are **not** performance evidence and do not
predict anything about Qwen3-8B. Claims that were not verified are labelled as such.

Following `ROUTER_LEARNABILITY.md`, corrections are kept visible rather than edited away. Four
conclusions reached in this line of work were **wrong and are retracted below** (§2, §7.1, §7.2,
§6.2); one of them (§2) was itself a retraction and had to be **re-retracted** in §2.5 after real
measurements contradicted it, and two had already been used to make a recommendation or a code change.

---

## 0. TL;DR

The design that survived the audit:

```
pass 1: dense prefill, no_grad, NO gate.  Keep h_C (values only) and KV(C).
pass 2: read-only cache (k_len = N)
        + EXPLICIT all-zero 4D mask          (the full rectangle; see §5)
        + position_ids = N .. 2N-1
        + gate as a flex_attention score_mod         (no Triton, no history_lse; see §1, §6.3)
        + pin_mode="sink", n_sink > 0        (mandatory; see §3)
        + log_budget = log(topk)             (sets the CONCENTRATION; see §2.5)
        + dense scope
        + chunk the replay queries           (O(chunk) memory; exact in math, see §6.1 and §11.4)
loss:   LM loss on C' only
```

Memory is **not** dominated by `h_C` (~8%, measured) but by pass 2's own graph, and — until §6.3 — by
46.7 GiB of attention scores that SDPA retained because a differentiable mask under GQA leaves it no
fused backend. With `flex_attention`'s `score_mod` in place of the mask, 8K peaks at **27.8 GiB** and
16K at **33.5 GiB** (measured, `query_chunk=1024`). Still no hand-written kernel (§1.1).

Three things fail **silently** if got wrong, and are the right targets for assertions:

1. `pin_mode="self"` pins **zero** keys in this geometry → the gate reverts to a reachable no-op (§3).
2. `mask=None` with `q_len == k_len` takes SDPA's `is_causal` fast path → you train on a causal
   triangle while believing it is a rectangle (§5).
3. `d_max = 2N-1 > max_position_embeddings` → replay queries sit at untrained positions (§7.3).
   Not an issue at N ≤ 16384 on Qwen3-8B; only bites at 32K.

A fourth is silent in the loss but visible in a diagnostic: **the gate's converged concentration is
set by `log_budget`, not learned**. `B=1` (the default before the term existed) collapsed
participation to 0.0070 against an eval budget of 0.125 and cost ~21 RULER points; `B=n_gated` is the
flat-gate no-op. Watch `participation` against `topk/N` (§2.5).

---

## 1. The gate does not, and need not, enter the KV cache

The premise "prefill a KV cache that has been adjusted by the log gate" is a category error worth
spelling out, because it changes what has to be implemented.

`g_i` is an additive term on the **logits**, acting on `(query, key)` pairs:

```
A_ji = softmax_i( q'_j · k_i / sqrt(d) + g_i )
```

An additive term inside `exp` cannot be folded into `k_i` itself. **`K_C` and `V_C` are bit-identical
across the two passes** (verified). The gate changes how pass 2's queries *weight* those keys, not
what they are. This is the desired behaviour, not a limitation: if the gate did alter `k_i`, pass 1's
`h` and its cache would no longer be mutually consistent.

### 1.1 Query-independence removes the two heaviest pieces of the pairwise e2e path

Because `s_i = f(h_i)` does not depend on the query, the gate is **one per-key vector shared by every
replay query** (verified: the pairwise score matrix from `ScalarIndexer` is identical across all query
rows). So `gate` is `(B, H_kv, N)` and can ride as a `(B, H, 1, N)` additive `attn_mask`, broadcast
over queries.

Verified against an explicit materialised implementation:

| route | forward | gradient w.r.t. gate |
|---|---|---|
| `attn_mask` of shape `(B, H, 1, N)` | maxdiff **0.00e+00** | maxdiff **0.00e+00** |
| concat one extra head dim | maxdiff 3.6e-07 | maxdiff 9.5e-07 |

Consequences, and they are the real engineering dividend of this route:

* **`triton_gated_attention` is not needed.** The kernel exists because a pairwise gate needs
  `Dqk != Dv` (flash-ineligible) or a 256-wide padded head (past what flash supports in backward) —
  both OOM'd at 8K. A per-key gate needs neither.
  *(Corrected: this bullet originally read "Plain SDPA suffices … a per-key gate is just a mask."
  The first half is **false** and cost 46.7 GiB — passing the gate to SDPA as a mask leaves it no
  fused backend and lands it on MATH; see §6.3. The conclusion survives, but via `flex_attention`'s
  `score_mod`, not plain SDPA. "It's just a mask" was true as algebra and wrong as engineering.)*
* **`gate_pin.history_lse` / `_HistoryLSE` are not needed.** That streaming logsumexp exists to avoid
  `O(Sq*Sk)` retention for a pairwise score. A per-key score's logsumexp is a sum over `N` numbers:
  `O(N)`.
* Memory for the gate path drops from `O(N^2)` to `O(N)`.

SDPA propagates gradient into a float `attn_mask` (verified), which is what makes this legal.

---

## 2. RETRACTED, then RE-RETRACTED: the `log B` budget constant

**Read §2.5 first.** §2–§2.4 record a retraction that was itself wrong. The budget term is real, it
is now implemented (`CrossReplayTrainer.log_budget`, `--budget`), and it should be set to the
inference top-k. §2–§2.4 are kept because the *reasoning error* is the reusable part, and because
§2.1's identity and §2.3's evidence are still valid.

**This recommendation was made during the session and is wrong.** Recorded because the reasoning
error is a reusable one.

### 2.1 What was claimed, and the one true step in it

With `pin_mode="sink"`, the normalised gate satisfies an identity (verified):

```
sum_{i in gated} exp(g_i) = K        (independent of N)
```

i.e. the whole gated history carries the weight of `K` sink tokens. A first pass at this claimed
`+log K` "cancels in the softmax". That sub-claim is **also wrong**, and the distinction is
instructive — whether it cancels depends on *whom it is added to* (verified, N=8192):

| where the constant is added | gated attention mass, K = 1 / 64 / 2048 |
|---|---|
| to **all** keys, including sinks | 0.465 / 0.465 / 0.465 — **cancels** |
| to **gated** keys only (sinks pinned at 0) | 0.221 / 0.956 / 0.998 — **does not cancel** |

Pinning breaks the per-row-constant symmetry, so with a pin the constant *is* a live knob on the
gated-vs-sink mass split. That much is real, and it means "no constant" is arithmetically the same as
**K = 1**, not "no budget".

### 2.2 Why the recommendation was still wrong

The inference from "K=1 suppresses gated mass to 0.22" to "the router cannot learn" was never
checked. It is false. `log K` is added to **every** gated key, so it cancels exactly inside the gated
softmax — and inference only ever reads `TopK(s)`, i.e. the **ranking**. Verified with `s` and `qk`
held fixed across K:

| K | gated total mass | gated **internal** conditional distribution vs K=1 |
|---|---|---|
| 1 | 0.272 | 0 (definition) |
| 16 | 0.857 | **7.4e-09** |
| 256 | 0.990 | **7.4e-09** |
| 2048 | 0.999 | **1.9e-08** |

The ranking is invariant. On gradients (same `s`, same downstream cotangent):

| K | cos(dL/ds, K=1) | Spearman of induced ranking |
|---|---|---|
| 4 | +0.996 | +0.990 |
| 64 | +0.983 | +0.951 |
| 2048 | +0.981 | +0.944 |

Direction is near-identical; the difference is **magnitude** (`|g|` grows ~3.5× from K=1 to K=2048).
**`log K` is effectively a gradient-scale / learning-rate knob, not a semantic one.**

### 2.3 The decisive evidence was already in the repo

`results_sparse_e2e` was trained at exactly K=1 (confirmed: `gate_from_score` is `score - lse` with
no constant anywhere in the package; `COMPRESSION_RATIO` feeds the press for eval, not the gate),
`pin_mode=sink`. Computed from its `metrics.json`:

* **@8K: 87.06**, **@16K: 79.04** (13 RULER tasks)
* and not uniformly degraded: `niah_single_1/2/3` and `vt` are **100.0**, `niah_multiquery`
  99.56/100, `cwe` 95.12 @8K.

If K=1 suppressed the router, `niah_single` could not be saturated. The weak cells are
`niah_multikey_3` (41.3 @8K → 4.35 @16K) and `qa_2` (54.55 → 47.73) — multi-target retrieval
collapsing with length, with no evident link to the gated/sink mass split.

**Decision: keep dense scope + sink pin, no budget constant.** ← **superseded, see §2.5**

### 2.4 The reusable lesson

Total gated mass was treated as a proxy for learnability. It is not: **the only deliverable is a
ranking**, and this constant is invisible to `TopK`. For any scalar introduced on the gate, ask first
whether it survives into the final `TopK`.

*(A first version of the K-invariance probe re-drew `randn` on every call and so compared different
samples — it printed "diff ~1e-9" while actually outputting 4.6e-02, a self-contradiction that went
unnoticed for one turn. This is the paired-vs-unpaired error `proxy_exp/HANDOFF.md` §9.4 warns
about, reproduced inside this session.)*

### 2.5 RE-RETRACTED: the budget does not change the ranking, but it does change the *concentration*

§2.2's measurements were right; the conclusion drawn from them was not. The invalid step:

> `log B` does not change the gated ranking at fixed parameters **⟹** it is only a gradient-scale
> knob.

The premise holds the parameters **fixed**. Concentration is not a property of a fixed score — it is
where **training converges**. Nothing in §2.2 tested that, and §2.4's "ask whether it survives into
`TopK`" is the wrong question: the constant does not survive into `TopK`, yet it decides *which*
score training arrives at.

**What the term actually is.** From §2.1's identity, `sum_{j gated} exp(g_j) = B`. Each pinned sink
sits at multiplier 1, so **`B` is the number of sink-equivalents the whole gated history is worth**.
With a flat score every gated key gets `B / n_gated`. Two consequences:

* `B = n_gated` is the **flat-gate no-op** (`g_j = 0` everywhere) — the hole pinning exists to close.
* `B = 1` (what omitting the term gives) makes the entire history worth **one** sink. At
  `n_gated = 16384` a flat gate is then `6.1e-05` per key, so the router can only be heard by piling
  mass onto a handful of keys. The suppression gets **worse** as the context grows, since it scales
  as `B / n_gated`.

**Measured, on the real model** (`proxy_exp_budget/{0.6B,8B}.jpg`; Qwen3-8B 4K/100 steps and
Qwen3-0.6B 4K/300 steps, identical seed/init/data order, only `B` varying). Final participation ratio,
higher = flatter:

| mode | 0.6B | 8B |
|---|---|---|
| raw (no normalizer) | 0.951 | 0.710 |
| ratio = 0.5 | 0.915 | 0.726 |
| ratio = 0.05 | 0.746 | 0.598 |
| fixed B = 4 | 0.358 | 0.391 |
| **fixed B = 1** | **0.236** | **0.254** |

Small `B` ⟹ concentrated; raw or large ratio ⟹ flat. Reproduced from the gate arithmetic alone under
training on a toy model (8 seeds, `n_H=512`, no transformer): mean participation
`raw 0.412 > B=n_H 0.405 > ratio0.5 0.397 > ratio0.05 0.333 > B=4 0.192 > B=1 0.066`, the same
ordering, stable across three regression targets. The mechanism there: `std(s)` is essentially
constant across modes (1.274–1.331), so the *learned score scale* is not what differs — what differs
is that at small `B` the history group cannot carry enough mass to fit the target at all (loss stalls
at 1.65e-1 for `B=1` against ~1e-27 for `raw`), and gradient descent responds by concentrating.

**What it cost us.** The first real cross-replay run trained at `B = 1`, because the term did not
exist. Its participation collapsed to **0.0070** at 16K while eval retains `topk/N = 0.125` — an 18×
mismatch — and the RULER failure pattern is exactly what that predicts: worse on every task needing
many keys retained, better on the one needing few.

| task | LM-loss arm | cross-replay (B=1) |
|---|---|---|
| `niah_single_3` | 9.52 | **95.24** ← needs *few* keys |
| `cwe` | 85.12 | 17.67 |
| `vt` | 99.51 | 49.76 |
| `niah_multiquery` | 94.30 | 35.09 |
| `niah_multivalue` | 90.79 | 36.40 |
| **mean** | **66.24** | **44.75** |

⚠️ **That comparison is confounded anyway** and cannot be read as loss-vs-loss: the LM-loss arm ran
`scalar_mid_dim=256` (38.14M indexer params) and cross-replay ran `scalar_mid_dim=0` (1.48M, a plain
linear score) — 26× the capacity. Only the checkpoints' own `config` revealed it. Both the budget and
the capacity are fixed in `scripts/train_gqa_indexer_cross_replay_gy.sh` for the rerun.

**Setting.** `B = topk`. That is the only value at which what inference does is exactly representable
during training: a hard top-k gate holds `topk` keys at multiplier 1 and drops the rest, whose
`sum exp(g)` is exactly `topk`. Implemented as `CrossReplayTrainer.log_budget` / `--budget`; unset
resolves to the no-op point and warns, `B <= 1` warns.

**Caveats.** The tables are 100/300 steps and our run is 600 — B=1 gave 0.254 there against 0.0070
here, so training longer concentrates further, and the tables predict the *direction* only, not that
`B=2048` will land participation at 0.125. And `B = topk` is derived from the `sum exp(g) = B`
identity, not taken from the source (`proxy_exp_budget/ChatGPT - 分析Budget选择.pdf`, which analyses
`B` versus the dense no-op condition and never mentions top-k; it also states — correctly — that `B`
leaves the participation of `softmax(s)` untouched *at fixed parameters*, which is precisely the
statement that does not settle the converged value).

**The reusable lesson, corrected.** §2.4 asked the wrong question. The right one: *does this term
change the loss landscape the router descends?* A term can be invisible in the forward ranking and
still decide the answer, because what it changes is the gradient's preferred direction — here via the
one thing pinning deliberately breaks, the symmetry between gated keys and sinks. "Invariant at fixed
parameters" and "invariant at convergence" are different claims, and only the second one matters for
a training knob.

---

## 3. `pin_mode="self"` silently pins nothing in this geometry

`e2e_trainer.py` defaults the dense stage to `pin_mode="self"` (`:233`). Under `[C ; C']` with `C'`
masked out, query `j`'s diagonal key lies in the **`C'` block**, which is exactly the region the
objective removes. Verified (N=16, n_sink=2):

```
pin_mode='self':  pins inside C block  = 0      <- none at all
pin_mode='self':  pins inside C' block = 16     <- all in the masked-out region
pin_mode='sink':  pins inside C block  = 32     <- effective
```

With zero pins the normaliser degenerates to a plain log-softmax, and by `gate_pin.py:38-41` that is
*exactly* interchangeable with raw `s` in forward and `d/ds`. The flat-gate no-op becomes reachable
again: **training runs cleanly, the loss looks fine, and the router learns no ranking.**

`pin_mode="sink"` with `n_sink > 0` is therefore mandatory here, and the pin count is worth
asserting rather than trusting.

---

## 4. Why the LM loss can train `s_i` at all, with the gate outside the cache

The mechanism, since it is the question the design turns on. The library is fixed (`KV(C)`); pass 1
only shelves the books. The indexer assigns each book a brightness `s_i`. Pass 2's readers can see
brighter books more easily. When they answer badly, the gradient asks: which books misled me, and
which needed books were too dim? The shelf never moved — the lighting did.

`s_i` sits on the forward path, so the chain rule gives:

```
dL/ds_i = sum_j A_ji <dL/do_j , v_i - o_j>
          ^ summed over ALL replay queries j
```

Three properties follow:

1. **`v_i - o_j`**, not attention mass. A token whose value matches the current output contributes
   zero gradient even at high attention — it carries no information. This is the "value geometry"
   the top-level README refers to.
2. **`sum_j`** aggregates every replay query's demand onto the *same* `s_i`. This is
   triangle→rectangle: `s_i` is forced to compromise across many unknown queries, which is the
   eviction semantics wanted.
3. **Implicit competition.** Verified: `dL/ds_i = u_i - p_i * sum(u)` (maxdiff 1.5e-7) and the
   gradient **sums to zero** over gated keys (-2.4e-07). So it learns a *ranking*, not absolute
   values — raising one token requires lowering another, which is the right behaviour under a fixed
   budget.

Then `dL/dw_out = sum_i (dL/ds_i) * h_i`, which is why `h_C` must be kept (§6).

**Directional check (toy model, mechanism only — not performance evidence).** Comparing
`-dL/ds_i` at a flat gate against leave-one-out true damage (hard-mask token `i`, measure the rise
in replay loss), N=24:

| signal | Spearman vs true damage |
|---|---|
| cross-replay gradient | **+0.587** |
| ordinary causal LM gradient | +0.312 |
| the two against each other | +0.625 |

Read only as: the signal exists and points the right way, and it is not the same quantity the causal
LM loss provides. The magnitudes mean nothing on a randomly-initialised model.

---

## 5. Layout: read-only cache, and the mask must be explicit

Two layouts are numerically identical (verified — loss `diff 0.00e+00`, gate-gradient maxdiff
**5.8e-10**, so training is equivalent too):

| layout | k_len | note |
|---|---|---|
| append `C'`, then mask it out | 2N | `query_offset = N`; causal happens to admit all of `C` |
| **read-only cache, `C'` never appended** | **N** | half the attention work; `C'` K/V never computed |

Read-only is preferred. Note `use_cache=False` does **not** prevent the append (verified: cache still
grows N → 2N) — a Cache subclass whose `update()` returns the existing entries is needed.

**The trap.** Under read-only, `q_len == k_len == N`, and `gated_attention_full` takes SDPA's
`is_causal=True` fast path when `mask is None`, which **ignores `query_offset`** (verified: differs
from the explicit all-visible mask by maxdiff **1.44**, and from the reference at `query_offset=0` by
1.44 — i.e. it silently gives a top-left causal triangle). Separately, `gated_attention._visible()`
**unconditionally** intersects with a bottom-right causal mask, so a rectangle intent does not
survive it.

Hence: **pass an explicit all-zero 4D mask**, and prefer the plain `(B, H, 1, N)` additive-mask route
of §1.1 over `gated_attention_full` for this objective.

Two facts that make the whole scheme work, both verified against transformers 4.57.3:

* A **4D** `attention_mask` is returned as-is by `masking_utils._preprocess_mask_arguments`
  ("If the mask is already 4D, simply return as-is"), so a custom rectangle passes straight through.
* With `C'` fully masked from itself, **replay queries are independent**: running a subset of replay
  positions gives hidden states bit-identical to the corresponding rows of the full pass (maxdiff
  8.3e-07). So replay queries may be subsampled without bias, and the pass is trivially
  sequence-parallel.

---

## 6. Pass 1 must be dense, ungated, and can be `no_grad`

**Ungated.** The inference path is dense prefill → evict → decode, so `KV(C)` must be the dense
values. If pass 1 were also gated, the objective would train streaming prefill, not eviction — and
worse, it closes a loop. Verified: perturbing layer 0's attention output by 5% changes downstream
hidden states (`layer1` 2.38, `layer2` 2.93, `layer0` exactly 0.0), so `h` depends on earlier
attention. Since `s_i = f(h_i)`, gating pass 1 makes `gate → h → s → gate` circular.

**`no_grad` is fine.** `dL/dw = sum_i (dL/ds_i) * h_i` needs the *values* of `h_i`, not their graph.
Verified: with `h` captured under `no_grad` (`h.requires_grad=False`), re-feeding it to the indexer
still yields `s.requires_grad=True` and a non-zero `w_out.grad`, because the weights are the leaves.
So pass 1 builds no autograd graph at all; only `h_C` is retained.

Cost on Qwen3-8B (36 layers, `hidden_size=4096`, 8 KV heads, `head_dim=128`):

| N | `h_C` (bf16) | `KV(C)` (bf16) | scores (fp32) |
|---|---|---|---|
| 8K | 2.25 GiB | 1.12 GiB | 9 MiB |
| 16K | 4.50 GiB | 2.25 GiB | 18 MiB |
| 32K | 9.00 GiB | 4.50 GiB | 36 MiB |

Comfortable at 8K/16K — **but `h_C` is not the dominant term, and §6.1 removes it entirely.**

### 6.1 Memory: `h_C` is ~10% of the problem; two exact reductions

Sizing pass 2's own retained activations (Qwen3-8B, bf16, all 36 layers, ~104 KiB/token/layer for
qkv + attn out + the three FFN intermediates — an estimate, not a measurement):

| N | pass 2 activations | `h_C` | `KV(C)` | `h_C`+`KV` share |
|---|---|---|---|---|
| 8K | ~29 GiB | 2.25 | 1.12 | 10% |
| 16K | ~59 GiB | 4.50 | 2.25 | 10% |

**The dominant term is pass 2's own graph**, which must be built for the gradient to reach the gate.
So the question "will `h_C` fit" is the wrong worry. Two reductions apply, and **both are exact — no
approximation, no kernel**:

**(a) Chunk the replay queries.** §5 established that replay queries are independent once `C'` is
masked from itself. So `C'` can be split into micro-chunks, each attending to **all** of `KV(C)`,
with gradients accumulated. Verified — loss and gate-gradient against the unchunked run:

| chunk | loss diff | gate-grad maxdiff | relative |
|---|---|---|---|
| 4 | 1.9e-06 | 1.1e-08 | 1.1e-07 |
| 8 | 1.1e-05 | 7.5e-09 | 9.2e-08 |
| 16 | 0.0e+00 | 7.5e-09 | 5.0e-08 |

i.e. fp accumulation noise. **This is unlike KVzip's chunking**: KVzip also restricts the *keys* to
the chunk (hence block-diagonal, §7.2), whereas here every chunk sees the whole key axis, so the
rectangle is preserved exactly. Peak activation drops from `O(N)` to `O(chunk)`.

> ⚠️ **The word "exact" in this subsection is measured on fp64 + SDPA and does NOT transfer to the
> production bf16 + `flex_attention` path.** There the loss is *not* bit-identical (1e-3 at
> `query_chunk=512`) and gradients deviate ~9e-03 relative. The decomposition is still mathematically
> exact — the error converges to 7e-07 at fp64 — but bf16 reassociation is visible. **See §11.4**,
> which also establishes that this path has a ~4e-03 gradient nondeterminism floor of its own.

> ⚠️ **The saving requires backwarding each chunk immediately.** The first implementation summed the
> chunks' losses and backwarded once at the end. That is numerically identical but saves **nothing** —
> every chunk's graph stays alive until the final `backward()`. Measured retention actually *grew*
> (1164 KiB unchunked → 1195 KiB at 8 chunks) from per-chunk overhead, and an 8K smoke run OOM'd.
> After moving `backward()` inside the loop, on a 4-layer fp64 model at N=64: peak **2461 → 1428 →
> 910 → 651 KiB** for chunk `None/32/16/8` (**3.8×** at chunk 8), with the loss **bit-identical**
> (`5.549922466278` at every chunk size) and gradients agreeing to fp32 epsilon.
>
> This forces a second change: the per-key scores are shared by all chunks, so a per-chunk backward
> would free their graph on the first chunk. `s` is therefore detached into a **leaf** for the
> replay, each chunk accumulates `dL/ds` into it, and one final `torch.autograd.backward` pushes the
> accumulated cotangent through `s = f(h)`. That is exactly the two-stage structure of (b) below —
> so (b) is now partly implemented, out of necessity rather than for the memory.
>
> Consequence for the API: `cross_replay_training_step` performs the backward itself and returns a
> **detached** loss. `backward=False` is available but rejected when chunking would split the
> sequence (unless grad is off).

**Gradient tolerance is fp32, not fp64.** The gate path is deliberately fp32 in two places —
`ScalarIndexer.score_keys` returns `.float()`, and `gate_scale` is upcast to fp32 — so `dL/ds`
accumulates in fp32 and the chunked/unchunked difference floors at fp32 epsilon (1.2e-07). Measured
relative differences are 6e-08 to 1.6e-07, i.e. exactly that floor. Note `in_norm.bias` reports
relative differences up to 9.0 purely because its gradient magnitude is ~1e-11 (numerically zero for
that input) — a case where only an absolute tolerance is meaningful.

**Do not forget the `lm_head` logits.** At `vocab=151936` they are a first-order term the §6.1 table
omits: per chunk, bf16 logits plus the fp32 copy the cross-entropy needs cost 0.87 GiB at
`chunk=1024` and **6.96 GiB at chunk=8192**. Held across all chunks (the original bug), an 8K run
retains ~7.5 GiB in logits alone on top of ~29 GiB of activations, ~16 GiB of weights and the AdamW
state — which is the OOM. Bounded independently by `logit_chunk`, which splits `lm_head` +
cross-entropy over row blocks (verified exact, and verified to lower peak retention).

Liger's `skip_logits` is **not** usable here, and was threaded through for one revision before that
was noticed: it fuses `lm_head` into the loss inside `*ForCausalLM.forward` and needs `labels` passed
to it, whereas this objective calls the **base** model (it must, to control the cache and the mask)
and computes the loss itself. The flag was silently doing nothing.

Post-fix budget at 8K, `query_chunk=1024` (estimate): 16.0 weights + 2.25 `h_C` + 1.12 `KV(C)` +
3.66 one chunk's activations + 0.87 one chunk's logits ≈ **23.9 GiB**, against ~55.6 GiB before.

> ⚠️ **That 23.9 GiB estimate was wrong by 3×, and the measured accounting is in §6.4.** It predicted
> the *right terms* but omitted the dominant one, which was not in this file's model at all: the SDPA
> backend. Superseded — read §6.4 for the real numbers.

### 6.2 RETRACTED: "the mask must not be *added* to the gate" — it was not the cause

The first GPU run reported **73.1 GiB** peak at 8K, 3× the 23.9 GiB estimate above. I hypothesised
that the attention override's

```python
mask = bias if attention_mask is None else bias + attention_mask   # (B,H,1,N) + (1,1,Sq,N)
```

was broadcasting into a materialized `(B, H, Sq, N)` tensor — 0.50 GiB per layer bf16, **18 GiB** over
36 layers — and fixed it by dropping the all-zero rectangle instead of adding it.

**The fix changed nothing: peak went 73.1 → 73.1 GiB, losses byte-identical.** So that term was
either never the cost or is hidden behind something larger. **The diagnosis was wrong** and the
49 GiB is still unexplained.

The change is kept, because avoiding a materialized `(B, H, Sq, N)` mask is correct on its own terms
(verified: SDPA receives `(1, 4, 1, 16)`, query axis intact; the `_kvpress_all_zero` tag survives in
both fp32 and bf16), and it is regression-tested by
`test_gate_reaches_sdpa_with_the_query_axis_unmaterialized`, with `test_a_real_mask_is_still_honoured`
guarding the obvious wrong fix. But it is **not** the answer to the memory question.

**Why it was insensitive, now that the answer is known (§6.3):** the shape of the mask never mattered
because *any* non-`None` mask sends the call to the MATH backend, whose cost is set by `(Sq, Sk)` —
the score matrix — not by the mask. Shrinking the mask from `(B,H,Sq,N)` to `(B,H,1,N)` saved 0.50
GiB/layer of mask and left 1.29 GiB/layer of retained scores untouched, and the allocator's peak sat
above both. The reasoning error was treating "the biggest tensor I can see in my own code" as the
cost, when the cost was a tensor allocated inside a kernel I had not checked the choice of.

### 6.3 The real cause: a differentiable mask under GQA leaves SDPA no fused backend

**Measured on an H20, `B=1, H=32, H_kv=8, Sq=1024, Sk=8192, D=128`, bf16, retained after forward:**

| SDPA call | flash | mem_eff | cudnn | runs | retained |
|---|---|---|---|---|---|
| `(B,H,1,N)` mask, `enable_gqa`, **requires_grad** | ✗ | ✗ | ✗ | **MATH** | **1288 MiB** |
| `(B,H,1,N)` mask, `enable_gqa`, detached | ✗ | ✗ | ✓ | cudnn | 8.1 MiB |
| no mask, `enable_gqa` | ✓ | ✗ | ✓ | flash | 16 MiB |
| `(B,H,1,N)` mask, K/V replicated (no gqa) | ✗ | ✓ | ✓ | mem_eff | 177 MiB |
| **`flex_attention` + `score_mod`** | — | — | — | fused | **48 MiB** |

1288 MiB × 36 layers = **46.7 GiB**. That is the 49 GiB gap, to within the noise of the other terms.

H1 was right in outcome and incomplete in mechanism. It named one condition (flash rejects any
`attn_mask`); there are **three**, and all three must hold, which is why this was not obvious:

1. a non-`None` `attn_mask` — excludes **flash** ("Flash Attention does not support non-null
   attn_mask");
2. a GQA head mismatch with a dense mask present — excludes **mem-efficient** ("both fused kernels
   require query, key and value to have the same num_heads");
3. a mask that **requires grad** — excludes **cuDNN**.

Remove any one and a fused kernel survives (rows 2–4). Condition 3 is the one worth recording,
because it is a trap for the next person to measure this: with a **detached** gate cuDNN is eligible
and the same call retains 8.1 MiB — *no bug at all*. A backend probe written the obvious way therefore
reports "fused, fine". But `dL/ds` arriving through the mask **is** the objective, so the production
gate always requires grad and row 1 is always the row that runs. Nothing in the loss distinguishes
them. `test_a_masked_gqa_sdpa_call_has_no_fused_backend` builds the mask with `requires_grad=True` for
exactly this reason, and asserts each of the three conditions is load-bearing by dropping it.

**H2 was refuted, and cheaply:** `reset_peak_memory_stats()` was already in the smoke script (added by
the handoff author), the weights report separately at 15.3 GiB, and the per-step peak was still
73.2 GiB. The constancy across steps was real, not an artifact — the step genuinely costs that.

**H3 was confirmed as a side finding**, at exactly the size predicted: `score_context` adds **+2.25
GiB**, a second copy of `h_C`. Not a duplicate tensor but `x = in_norm(h)`, saved by autograd as
`w_out`'s input for the backward. `hidden.clear()` cannot help — the graph holds it. This is real but
it is 8% of the peak, not the bug; §6.1(b)'s two-stage backward would remove both copies at once.

**The fix.** `flex_attention` with `score_mod = score + gate[h, kv_idx]`, per §1.1's instruction not
to write a kernel. Verified against the SDPA mask route: forward maxdiff **5.6e-16** at fp64,
`dL/dq` **1.1e-15**, `dL/dgate` **4.8e-07** — the fp32 floor this gate path already sits at, since
`score_keys` returns `.float()`. Three sharp edges, each measured and each guarded in code:

* **The `torch.compile` is not optional, it is the entire fix.** Eager `flex_attention` falls back to
  a materializing reference implementation: **18730 MiB** for one layer at the shape above, against 40
  MiB compiled. That is 14× *worse* than the MATH backend it replaces. Anything that bypasses the
  compiled path silently inverts the optimization, so `_note_flex_shape` warns when dynamo's
  8-shape `recompile_limit` is approached (the 9th distinct shape reverts to eager).
* **Inductor has no valid Triton config for `64 <= Sq < 128`** at `Sk=8192, D=128` — every candidate
  exceeds the H20's 232448-byte shared-memory limit, and the call *raises* `No valid triton configs`
  rather than falling back. Measured: `Sq` = 63, 64, 65, 96, 100, 127 all raise; 17 and 128 are fine.
  A ragged final chunk lands there for any `|C| % query_chunk` in that band (e.g. `--context-len 8292
  --query-chunk 1024` → 100). Fixed by padding queries to a multiple of 128 and slicing the output
  back; exact, since the padded rows' cotangent is zero (verified: `dL/ds` unchanged).
* **`donated_buffer` must be disabled.** Inductor's donated-buffer optimization asserts that no
  compiled backward is called with `retain_graph=True`, and `logit_chunk` backwards each row block
  with exactly that. So every `--logit-chunk` run raised `RuntimeError: This backward function was
  compiled with non-empty donated buffers`. Found only by sweeping both chunk knobs together —
  `query_chunk` alone never trips it, and the original smoke command does not pass `--logit-chunk`.

flex is also **1.8× faster**: 9.4 → 5.1 s/step at 8K.

### 6.4 Measured memory accounting (replaces §6.1's estimate)

Qwen3-8B bf16, `n_sink=4`, `scalar_mid_dim=0`, AdamW on the indexers only, H20 96 GiB,
`torch.cuda.max_memory_allocated()` reset per step, random token ids. **Measurement, not arithmetic.**

`--context-len 8192 --query-chunk 1024`, before and after:

| term | §6.1 estimate | measured, before | measured, after |
|---|---|---|---|
| weights + indexers | 16.0 | 15.26 | 15.26 |
| AdamW state (indexers only) | — | 0.08 | 0.08 |
| `KV(C)` | 1.12 | **1.12** | **1.12** |
| `h_C` | 2.25 | **2.25** | **2.25** |
| `in_norm(h_C)` saved by the score graph | *omitted* | **2.25** | **2.25** |
| per-key scores, fp32 | 0.01 | 0.01 | 0.01 |
| one chunk's activations + logits | 4.53 | ~6.9 | ~6.9 |
| **retained attention scores (MATH backend)** | **omitted** | **≈45.4** | **0** |
| **total peak** | **23.9** | **73.2** | **27.8** |

The estimate's error was entirely one omitted row. Every term it *did* list was right to within a few
hundred MiB — `KV(C)` and `h_C` are exact — which is why the arithmetic looked trustworthy.

**Peak is now dominated by what chunking controls,** which was not true before: at 73.2 GiB the MATH
scores swamped the chunk, so `query_chunk` moved the peak barely at all. Now it scales as designed
(8K, `logit_chunk` off unless stated):

| `query_chunk` | 128 | 256 | 512 | 1024 | 2048 | 8192 (unchunked) |
|---|---|---|---|---|---|---|
| peak GiB | 21.9* | 22.7* | 24.4 | **27.8** | 34.6 | 61.3* |

*with `logit_chunk` set (128/256/1024 respectively), which is independent — at `query_chunk=1024`:
`logit_chunk` `None`/512/256/128 → 27.8 / 26.8 / 26.3 / 26.0 GiB.

Longer context, `query_chunk=1024`: **16K = 33.5 GiB** (versus the 65.2 GiB §6.1 projected for naive
16K, and it would have OOM'd at 96 GiB under MATH). The fixed cost grows with `N` (`KV(C)` + two
copies of `h_C` = 5.6 GiB at 8K, 11.2 at 16K) while the chunk cost does not, so §6.1(b)'s
`h_C`-free two-stage backward is now the *next* lever rather than a deferred nicety — it would remove
4.5 GiB at 8K and 9.0 at 16K, both copies at once.

Losses across all of the above agree to 3 decimal places (14.763–14.769 at step 1, the variation being
the fp32 accumulation order the §6.1 chunking analysis already characterises), and the training signal
is unchanged: participation 0.883 → 0.426, shuffle control **+1.95** nats/token.

**(b) Drop `h_C` via a two-stage backward.** `dL/dw = sum_i (dL/ds_i) h_i` needs `h_i`'s values, but
not simultaneously. Stage A: run pass 1 layer by layer, compute `s_l` immediately, keep only `s_l`
(fp32, 256x smaller than `h_l`) and discard `h_l`; run pass 2 treating `s` as a **leaf** and collect
`dL/ds`. Stage B: recompute pass 1 layer by layer and inject `dL/ds_l` into each layer's indexer
backward. Verified **bit-exact** against keeping `h_C` throughout:

```
loss diff 0.00e+00;  every indexer parameter's grad maxdiff 0.00e+00
(in_norm, w_in, mid_norm, w_out, all 3 layers)
```

Cost: one extra pass-1 forward — ordinary gradient checkpointing, applied across the two passes.

Resulting peaks (excluding ~16 GiB of bf16 weights):

| N | naive | (a) q-chunk 1024 | (a)+(b) |
|---|---|---|---|
| 8K | 32.6 G | 7.0 G | **4.8 G** |
| 16K | 65.2 G | 10.4 G | **5.9 G** |
| 32K | 130.5 G | 17.2 G | **8.2 G** |

**Recommendation: implement (a) first** — it is a loop, gets 16K to ~10 GiB, and needs no
restructuring. Add (b) only if 32K is wanted. **Write no kernel**: the `(B, H, 1, N)` additive-mask
route of §1.1 already avoids the `O(N^2)` materialisation that `triton_gated_attention` was written
to avoid, and query-chunking handles the rest.

*(Unverified, and it decides whether a kernel is eventually needed: whether CUDA SDPA keeps a
memory-efficient backend when given a broadcast `(B, H, 1, N)` float mask, or falls back to the math
backend and materialises `(Sq, Sk)`. A CPU backend probe is not representative and was discarded.
Flash-attention on CUDA generally does not accept arbitrary additive masks. If it does fall back,
the fix is `torch.nn.attention.flex_attention` — a `score_mod` adding `gate[h, kv_idx]` is exactly
this operation and compiles to a fused kernel — not a hand-written Triton kernel.)*

---

## 7. Positions, and what KVzip's chunking actually does

### 7.1 RETRACTED: "KVzip chunks to bound the replay distance"

**Wrong.** Reproducing `kvzip_press.py`'s index bookkeeping exactly (CTX=32768, `chunk_size=2048`,
16 chunks):

| chunk | scored keys | replay q positions | **relative distance** | scoring matrix |
|---|---|---|---|---|
| 0 | [20, 2068) | [32768, 34828) | **30701 .. 34815** | 2060 x 4112 |
| 1 | [2068, 4116) | [32768, 34836) | 28653 .. 32767 | 2068 x 4120 |
| 15 | [30740, 32788) | [32768, 34836) | -19 .. 4095 | 2068 x 4120 |

The cache is truncated back to `context_length` at the end of every chunk (`kvzip_press.py:355`), so
replay queries always restart at CTX. **The maximum distance is ~CTX regardless of `chunk_size`.**
Chunking bounds two different things:

* **the materialised scoring matrix** — 32780 x 65552 ≈ 2.1G elements/layer/head unchunked vs
  2060 x 4112 ≈ 8.5M chunked, **~496x smaller**. `score_kvzip` really does materialise it (it needs
  `amax` over queries), so this is a memory optimisation — the opposite of what was claimed.
* **which queries score which keys** — `score_val[..., start:end]` (`:352`) is written only by the
  current chunk's queries.

### 7.2 Consequence: KVzip's supervision is block-diagonal, not a rectangle

This is the most useful finding of the session and it repositions the objective. Chunk `c`'s keys are
scored **only** by chunk `c`'s replay queries. KVzip never realises the "every key scored by all `N`
queries" rectangle that `query_independent_indexer_cross_replay.md` argues for. Each key is judged by
~`chunk_size` queries, and different chunks sit in **different positional regimes** (chunk 0 at
distance ~32K, chunk 15 at ~0–4K) — an inconsistency in KVzip that is not discussed in the paper, and
a plausible contributor to score instability at long context.

So the novelty of cross-replay versus KVzip is **not only** "keep the LM loss instead of distilling a
per-key label" — it is **also** "actually do the full rectangle". *(Unverified: whether the rectangle
is empirically better than block-diagonal. This note only establishes that KVzip is block-diagonal.)*

### 7.2.1 The paper read at last: `C'` **does** attend to `C'`, and that is not a problem

Read directly from the paper (`KVZip.pdf` §3.2, Algorithm 1, Figure 5, §C.2) rather than from the
implementation, because a question came up that the code alone cannot answer: *should `C'` be allowed
to see its own KV, given the paper reportedly says `C'`→`C'` attention is low?*

**The premise is the wrong way round.** KVzip's scoring forward is a *single ordinary causal pass*
over `[repeat_prompt ; C]` with `KV(C)` already in cache. The paper's own shapes say so:

* keys are `K_{l,h} ∈ R^{(n_c + n_in) × d}` — the cached `C` **plus** the new input's own keys;
* the attention matrix is `A_{l,h} = Softmax(Q_{l,h} K_{l,h}^T) ∈ R^{G × n_in × (n_c + n_in)}`, i.e.
  the softmax denominator spans **both** blocks;
* only *afterwards* is it sliced — "Extracting entries corresponding to keys in `KV_c` gives
  `Ā_{l,h} ∈ R^{G × n_in × n_c}`" — and the max is taken over that slice.

Algorithm 1 confirms it (`K ← Keys in the l-th attention layer # shape: H×(n_c+n_in)×d`, and
`Ā ← Subsample keys in KV_c` only for the *scoring* slice), and the FLOP count is quoted as
`O(n_c m + m²/2)` **causal**-attention per chunk — the `m²/2` term *is* `C'` attending to `C'`.

So `C'`→`C'` attention is present in the forward and merely **excluded from the score**. It is not
suppressed, and the paper never claims it is small. What the paper does say about small scores is a
different statement, and it is worth not conflating the two:

* **Figure 5 / §3.3:** *cross*-attention (`C'`→`C`) during reconstruction is **sparser** than the
  *self*-attention seen during the initial prefill of `C`. That is the argument for why
  reconstruction scores compress better than H2O's prefill scores — it is about `C'`→`C`, not
  `C'`→`C'`.
* **§C.2:** for a 2K NIAH context, **98.1%** of KV pairs take their max attention from the *repeated
  context* rather than from the repeat prompt (99.4% among those surviving 30% compression). This is
  an argument that the 7-token prompt is negligible — again nothing about `C'`→`C'`.

**Consequence for our layout.** §5's read-only cache masks `C'` out entirely (`k_len = N`), so our
softmax denominator is `C`-only where KVzip's is `C + C'`. That is a **real difference from KVzip**,
and §5 justified it on grounds that remain valid (replay queries become independent — verified
bit-identical under subsampling, maxdiff 8.3e-07 — which is what makes subsampling unbiased and the
pass sequence-parallel). But it should be recorded as *a deliberate divergence*, not as a
reproduction of KVzip: excluding `C'` from the denominator makes every retained probability larger
than KVzip's by the factor `1/(1 - mass on C')`, which is a per-row rescaling and therefore does not
change a per-row ranking — but our gate's normaliser (§2.1's `Σexp(g) = B`) is *not* per-row in the
same way, so the two are not automatically equivalent under training.

**"Do plain replay instead of cross-replay" does not follow from the paper.** KVzip *is* the
cross-replay geometry: queries from a repeated copy of the context, scoring the *prefilled* keys. The
only thing plain replay would change is whether `C'`'s own keys join the denominator. If that is
worth testing, the honest experiment is a flag on the mask (denominator `C` vs `C + C'`) with
everything else fixed — not a change of objective. ⚠️ Not tested; and note it interacts with the
three §0 guards, since admitting `C'` reintroduces exactly the region `pin_mode="self"` pins into
(§3).

### 7.2.2 MEASURED: admitting `C'` destroys the objective (do not do it)

§7.2.1 left "let `C'` see `C'`" as an untested flag. It is now tested, and the answer is a hard no —
the flag was not built, because a 20-minute measurement showed it removes the supervision entirely.
`proxy_exp_budget/bypass_check.py`, frozen Qwen3-8B, **no gate at all** (so this is a property of the
*geometry*, not of any checkpoint), `N=2048`, next-token loss on `C'`:

| geometry | what query `j` sees | `qa_1` | `vt` | `niah_single_2` |
|---|---|---|---|---|
| `cross` (today) | all of `C` | 6.431 | 2.736 | 6.778 |
| `both` (the proposal) | all of `C` **+** `C'[0..j]` | **0.0098** | **0.0132** | **0.0041** |
| `self_only` | `C'[0..j]` only | 2.457 | 0.277 | 2.652 |

**`both` reaches ~0.01 nats — two to three orders of magnitude below `cross`, and far below
`self_only` too.** A loss of 0.01 nats/token is not language modelling; it is a *copy*. The mechanism
is induction: `C'` is a verbatim repeat of `C`, so once query `j` can see `C'[0..j]`, the pair
(`C` in cache, `C'` prefix) lets an induction head match the current prefix against its earlier
occurrence and read off the next token. That is far *easier* than either reconstructing from `KV(C)`
alone or modelling the text from its own prefix — which is why `both` beats `self_only` rather than
interpolating between the two.

**The null control confirms the mechanism.** Replay an *unrelated* document (`C'` from a different
RULER task, so no copy is available):

| geometry | `qa_1` | `vt` | `cwe` |
|---|---|---|---|
| `cross` | 7.970 | 7.511 | 9.881 |
| `both` | 0.290 | 1.412 | 2.411 |
| `self_only` | 0.277 | 1.430 | 2.457 |

`both` now equals `self_only` to within noise — **bypass share 1.00** (0.998 / 1.003 / 1.006). With
no copy to exploit, `C` contributes *nothing at all* once `C'` is visible: the model simply does
ordinary causal LM on `C'` and ignores the entire cache. Either way the gated keys stop mattering.

⚠️ **A methodological note on that control.** The first attempt drew `C'` from another *row of the
same task* and reported bypass ≈ 1.4, which looked like evidence but was not: within one RULER task
every row is built on the same essay haystack, so "another row" is a near-duplicate of `C` and the
copy shortcut survives. Only a *different task* is a null. The corrected control is the table above.

**Why this kills the proposal, in terms of §4's gradient.** `dL/ds_i = Σ_j A_ji ⟨dL/do_j, v_i − o_j⟩`.
The gate applies to `C`'s keys only, so `dL/ds` is scaled by the attention mass still landing on `C`.
Measured under `both`: **0.41–0.64** of the mass leaves `C` for `C'`, so the router's gradient is
roughly halved even before the loss collapse. And with the loss at 0.01 nats there is almost no
`dL/do_j` left to distribute — the objective is satisfied without the router doing anything. This is
the reachable-no-op family the whole pin mechanism exists to close (§0, §3), arrived at from a new
direction: the pin closes the *flat-gate* no-op, but nothing closes a *bypass-the-gated-keys* no-op.

**And it would be silent.** The replay loss would drop dramatically and every dashboard would look
better. That is the exact failure shape §9 catalogues four times, and §14 has just shown this arm's
loss already anti-correlates with its task score (arm B: replay loss 1.18 vs A's 2.70, RULER 20.43 vs
44.75). A change that improves the loss by 100× while removing the supervision would be
indistinguishable from success on any metric now logged, except `gate_participation` and the shuffle
control.

**What the concern behind the proposal points at, and the version that survives.** The intuition —
"inference lets a query attend to itself, training does not" — is real, but the mismatch is not that
`C'` cannot see `C'`; it is that at inference the query is **short and not a copy of the context**,
while `C'` is `|C|` tokens and *is* a copy. So the defensible variant is not "admit `C'`" but
**"replay a short, non-duplicate query"** — which is what §7.3's train/inference gap note is really
about, and what KVzip's own repeat prompt approximates. If self-visibility is wanted, it must be
confined to a query span that cannot copy: e.g. real questions over `C` with the loss on the answer
tokens only, letting the answer see the question. That admits self-attention without admitting
induction, and it is a different (larger) change than a mask flag.

**Note KVzip is not a counterexample.** Its scoring forward does include `C'`→`C'` (§7.2.1), and it is
fine there because KVzip takes `max` attention onto `KV(C)` as a *label* — it never differentiates a
loss. There is no gradient to starve, so a shortcut in the forward costs it nothing. Cross-replay
differentiates, so the same geometry is load-bearing in a way it is not for KVzip. **This is the
cleanest example so far of why KVzip's design choices cannot be copied across to this objective one at
a time.**



### 7.3 What position does and does not affect, with the full rectangle

Decision taken: **full rectangle, no chunking, `C'` at `[N, 2N)`.** So query `j` reads key `i` at
relative distance `d = N + j - i`, ranging over `1 .. 2N-1`.

One path is ruled out. **`s_i` carries no positional information at all** (verified): `score_keys`
takes only `h_i` and `key_offset`, and `key_offset=0` vs `N` differs by exactly
`pos_slope * N = 3.2e-05` — the deliberate recency tilt, nothing else. RoPE acts on `q·k`, never on
`s`. So positions cannot contaminate the score directly.

The live path is the **gradient**: `A_ji` contains `q'_j · k_i` with RoPE(`N+j-i`), and a small
`A_ji` means weak supervision for key `i`. But Qwen3-8B's RoPE barely decays (verified, 4000 random
q/k pairs, real config `rope_theta=1e6`, `max_position_embeddings=40960`):

| d | 1 | 512 | 4096 | 16384 | 32768 | 65535 |
|---|---|---|---|---|---|---|
| mean &#124;q·k&#124; relative to d=1 | 1.000 | 1.014 | 1.013 | 1.015 | 1.015 | 1.018 |

So "RoPE decay starves early keys" is **not** the mechanism. Two real effects remain:

**(a) `N = 32768` leaves the trained position range.** `d_max = 2N - 1`:

| N | d_max | vs 40960 |
|---|---|---|
| 4096 | 8191 | ok |
| 8192 | 16383 | ok |
| 16384 | 32767 | ok |
| **32768** | **65535** | **1.6x over** |

A hard edge, not a gradient. N ≤ 16384 is safe; **not a concern for current runs**, which stay at
≤16K. Worth an assertion so it surfaces if the curriculum is later extended (the existing e2e
schedule `8192:300,16384:300,32768:900` would hit it in its third leg).

**(b) A key's positional regime depends on its index in `C`** — intrinsic to the rectangle, not
removable (N=16384):

| key `i` | pass 1 distance (dense prefill) | pass 2 distance (replay) |
|---|---|---|
| 0 | 1 .. 16383 | **16384 .. 32767** |
| 8192 | 1 .. 8191 | 8192 .. 24575 |
| 16383 | — | 1 .. 16384 |

Early keys can only be read from far away in pass 2; late keys span the whole range. Combined with
the model's own recency preference, **tokens near the start of `C` receive systematically weaker
supervision**. Partly this is faithful to deployment (real queries do follow the context), but it
means `s_i` absorbs a "distance from the end of `C`" component that does **not** hold at inference,
where queries sit immediately after `C`. An inherent train/inference gap of the
rectangle + shifted-position choice; recorded, not fixed. *(Unverified: its magnitude.)*

The alternative (`C'` reusing `0..N-1`) trades this for negative relative distances — distance 0 when
query `j` reads key `j`, which is maximally in-distribution, but negative when reading later keys,
which pretraining never saw. An earlier claim that this is simply "worse" was asserted without
evidence and is withdrawn; the toy-model losses across four position schemes (5.516–5.524) are pure
noise. Treat it as a sweepable flag, not a settled default.

---

## 8. Relation to SAS, and why it needs no budget constant either

SAS's gate is the same object as ours (their Eq. 6):

```
log g_m = s_m - LSE(s),  B_m in H        # historical blocks
log g_0 = 0                             # always-retained current block, unnormalised
```

and the paper gives our §2.1 argument verbatim: the shared `-LSE(s)` "does not cancel in the
subsequent attention softmax **because it is not applied to the current block**". Their current block
is our sink pin.

The difference is their design choice #4, **sparse scope** (Algorithm 1): `g = softmax(s)` is
normalised over **all** `N` historical blocks, but only `Top-K` enter the forward. So the gated mass
actually present in the forward is "the mass captured by the top-K", which is `< 1` and **set by the
selector itself** (verified, N=1024):

| score spread | top-8 captures | top-64 | top-512 |
|---|---|---|---|
| 0.0 (flat) | 0.008 | 0.063 | 0.500 |
| 1.0 | 0.066 | 0.285 | 0.836 |
| 4.0 | 0.940 | 0.995 | 1.000 |

A flat selector's chosen blocks capture almost nothing, so historical context goes mute and the LM
loss punishes it: **an emergent budget**. Under dense/full scope the same formula gives forward mass
**identically 1.0000** at every spread (verified) — flatness costs nothing, which is precisely why
`pin_mode` exists in this package.

But per §2, this does **not** mean SAS "avoids" a constant we need: the constant is invisible to the
ranking on both sides. What sparse scope actually buys is **soft/hard consistency** — the forward
attends to exactly the set inference will select. That is a real difference, and it is what
`e2e_trainer.py`'s stage-2 (`scope="sparse"`, `keep_ratio=0.25`) already implements.

**Note for a future sparse-scope stage.** Under query-independence, `TopK(s)` is taken **once over
the key axis** and the resulting subset `S` is shared by every replay query — literally eviction,
not per-query sparse attention. No kernel needed: gather `K_S, V_S` and call SDPA. The cost is
heavier than SAS's, though: SAS gives a key a gradient if *any* query selects it, whereas a shared
`S` leaves every key outside `S` with zero gradient for *all* queries, permanently. This is SAS
Figure 5's concern in a stronger form and is **unexplored**. Hence: stage 1 dense scope + sink pin
first (every key gets its own content-dependent gradient), sparse scope only after a ranking exists.

---

## 9. Implementation

Shipped as `cross_replay.py` (`CrossReplayTrainer`, `cross_replay_training_step`, `ReadOnlyCache`,
`rectangle_mask`), tested in `tests/presses/test_gqa_indexer_cross_replay.py` (23 tests), with
`scripts/smoke_cross_replay.py` as a real-model smoke check that reports gate participation,
`gate_scale`, and a shuffle control.

The three silent failure modes of §0 are assertions, not comments: `pin_mode="self"` and `n_sink=0`
are rejected at construction, the rectangle mask is always built explicitly, and the attention
override raises if the key axis ever differs from `|C|`. Verified by mutation — reverting each of
four properties (drop the rectangle mask, let the cache grow, gate pass 1, remove the sink pin)
fails the corresponding test.

**Two bugs the tests caught, both of which would have been silent:**

1. **Pass 1 ran gated.** `prefill` is called from inside `hooks()`, where the attention
   implementation is already swapped, so pass 1 used the gated path — exactly the feedback loop §6
   forbids, and it cannot work anyway since the scores do not exist yet. Fixed by
   `CrossReplayTrainer.ungated`, which restores the model's own attention for pass 1. Caught by
   `test_step_uses_a_full_rectangle_not_a_triangle`.
2. **`.float()` on the logits downcast fp64.** Intended to keep the summed cross-entropy precise in
   bf16, it silently *reduced* precision on the fp64 test model, making the chunked/unchunked
   gradient comparison fail at ~1e-7. Fixed with `torch.promote_types(logits.dtype, torch.float32)`.
   Caught by `test_query_chunking_reproduces_the_unchunked_gradient`.

**A third bug the tests did NOT catch, found by an 8K OOM:** chunking saved no memory at all,
because the chunks' losses were summed and backwarded once at the end. See the warning box in
§6.1(a). The tests asserted only that chunking was *exact*, which it was — nothing asserted that it
was *cheaper*, so the property the feature exists for went unverified. Now covered by
`test_chunking_releases_each_chunk_graph`, which compares peak retained bytes.

The lesson generalises: an optimisation needs a test for the thing it optimises, not only for the
invariant it must not break.

**A fourth, found on GPU (§6.3): the gate reached SDPA as a mask and SDPA silently chose MATH,
retaining 46.7 GiB of attention scores.** Same shape as the other three — every exactness test passed,
because MATH computes the *right answer*, just expensively. What no test asserted was **which kernel
ran**. Now covered by `test_a_masked_gqa_sdpa_call_has_no_fused_backend` (no fused backend is available
for this call, so the fallback is genuinely catastrophic) and
`test_the_cuda_path_never_calls_sdpa_with_a_gate_mask` (the gated replay must not reach SDPA at all on
CUDA). The refinement over the earlier three: the untested property was not in *this* code, it was a
dispatch decision made inside a library call. "Which kernel did I get" belongs in the same category as
"was the graph freed" and "did the flag do anything".

Two of the new guards exist because the *fix* has failure modes that are worse than the bug, which is
its own lesson — an optimisation that degrades catastrophically needs a test on the degradation path,
not just on the happy path:

* eager `flex_attention` retains **18730 MiB** where compiled retains 40, so losing the compile is 14×
  worse than the MATH bug. `test_recompile_pressure_is_warned_before_dynamo_gives_up` covers the
  realistic way that happens (dynamo's 8-shape recompile limit).
* `flex_attention` *raises* rather than degrades for `64 <= Sq < 128`.
  `test_flex_runs_for_every_ragged_chunk_length` parametrises the whole band, and the padding that
  fixes it was mutation-tested (`_FLEX_Q_ALIGN = 1` reproduces the crash at `N=8292`).

**Deferred, deliberately:** the `h_C`-free two-stage backward of §6.1(b). It is verified bit-exact
but only pays off above 16K, and the query-chunking of §6.1(a) — which is implemented — already
brings 16K to ~10 GiB.

**Not yet done:** a training driver (the equivalent of `scripts/train_gqa_indexer_e2e.py`) and its
launcher. `cross_replay_training_step` is the whole objective, so wiring it into the existing loop is
mostly data plumbing.

---

## 10. Open items

* **Not verified on a real model.** Everything here is CPU / toy-scale mechanism checking. The
  §4 directional check in particular needs re-running on Qwen3-8B from a training checkpoint before
  any of its magnitudes are quoted.
* A diagnostic comparing the learned `s_j` against accumulated attention mass `sum_i A_ij`
  (H2O-style) would test whether the rectangle buys anything beyond replay-distribution H2O. Per
  `proxy_exp/HANDOFF.md` §9.4 it must report **per-(layer, doc) geometric ratios, never an aggregate
  over layers** (that trap has fired at least three times in this project — and once more inside this
  session, see §2.4), be read at a **real training checkpoint** rather than at flat init, and be
  judged by a **sign test against a shuffle control** rather than by a correlation magnitude.
  Deferred by decision.
* Whether the rectangle beats KVzip's block-diagonal supervision (§7.2) is untested.
* Whether `[N, 2N)` or aligned `[0, N)` positions train better (§7.3) is untested.
**CLOSED (§6.3, §6.4): whether CUDA SDPA stays memory-efficient under a broadcast `(B, H, 1, N)` float
mask.** It does **not** — it has no fused backend at all for this call and runs MATH, retaining
46.7 GiB over 36 layers, which was the entire 49 GiB gap. Fixed with `flex_attention`'s `score_mod`
(§1.1's instruction, and still no hand-written kernel): **73.2 → 27.8 GiB at 8K**, 33.5 GiB at 16K,
1.8× faster, losses and training signal unchanged. §6.4 has the measured accounting that replaces
§6.1's estimate; `HANDOFF_cross_replay_memory.md` is resolved.

Two open items *created* by the fix, both bounded and guarded rather than solved:

* **`_FLEX_Q_ALIGN` is calibrated on an H20.** The `64 <= Sq < 128` autotuner gap is a shared-memory
  limit (232448 bytes), so a different GPU may have a different band. The failure is loud (an
  exception, not silent slowness) and `test_flex_runs_for_every_ragged_chunk_length` covers the band,
  but the constant is not proven portable.
* **`torch._functorch.config.donated_buffer = False` is set process-wide** by
  `compiled_flex_attention()`, because `logit_chunk`'s `retain_graph=True` is incompatible with it. It
  is a mild pessimization for any *other* compiled code in the same process. Narrowing it to this
  call would be better if a scoped API appears.

**Answered, then superseded (§6.2):** a broadcast `(B, H, 1, N)` mask does reach SDPA un-materialized,
but that was never what made the first GPU run peak at 73.1 GiB, and fixing it left the peak
unchanged. The mask's *shape* was irrelevant; its mere *existence* (plus GQA, plus `requires_grad`)
was the problem. Kept as a record of a wrong diagnosis whose reasoning error — auditing only the
tensors visible in one's own code — is the reusable part.

**First GPU signal, 5 steps at 8K on Qwen3-8B (random tokens, so mechanism only):** loss 15.30 →
13.66, participation **0.883 → 0.425** (the gate concentrating, which is what eviction needs), and a
shuffle control of **+1.93 nats/token** — destroying the ranking hurts a lot, so the scores carry
real ordering information rather than a flat gate. Encouraging, but on random tokens it cannot
distinguish "learned useful ranking" from "learned that some positions matter"; rerun on real text
before quoting.

---

## 11. First real-text run (§10's first open item, partly closed)

Everything above was random token ids or toy models. This is the first run on the longmino corpus, via
`scripts/train_gqa_indexer_cross_replay.py` — the driver §9 listed as "not yet done" — launched as
`scripts/train_gqa_indexer_cross_replay_gy.sh probe`: 100 steps at 8K, 4 ranks of H20, scalar indexer
at `mid_dim=0` (SparseK's linear score), `pin_mode=sink`, `n_sink=4`, `query_chunk=1024`. 8.9 minutes.

| step | loss | participation | gate_scale |
|---|---|---|---|
| 0 | 6.71 | 0.823 | 0.9999 |
| 20 | 6.17 | 0.380 | 0.9908 |
| 50 | 6.02 | 0.077 | 0.9635 |
| 99 | 5.51 | **0.036** | 0.9329 |

**The shuffle control, which is the readout that matters:**

| | delta (nats/token) |
|---|---|
| step 0, before anything is learned | **-0.0008** |
| the step-100 checkpoint, three fresh documents | **+1.21 / +1.57 / +1.78** |

That before/after is the cleanest evidence this objective has produced, and it is stronger than the
+1.93 on random ids precisely because of the paired zero: at init, permuting the scores along the key
axis costs **nothing** (-0.0008, zero to fp noise) — correct, since a flat gate has no ordering to
destroy — and after 100 steps the same permutation costs **1.2-1.8 nats/token**. A real per-key ranking
exists where none did. Peak was **27.8 GiB at 8K and 33.5 at 16K, matching §6.4 exactly** on a real
corpus under a real DDP launch, which also confirms the §6.3 fix outside the smoke harness.

Two caveats, since this section will get quoted:

* **100 steps at 8K is a probe, not a training run.** The WSD schedule was compressed into those 100
  steps, so part of the participation collapse to 0.036 is a fast LR decay rather than a converged
  router. Whether 0.036 is *too* concentrated for eviction at ratio 0.5 is unknown and is a real
  question, not a rhetorical one.
* **No downstream number yet.** RULER eviction scores from `evaluation/evaluate_indexer_press.py` are
  what decide whether the rectangle beats distillation (`results_sparse_e2e`: 87.06 @8K, 79.04 @16K).
  §10's open items about that are untouched.

### 11.1 A launcher bug worth recording, being the same shape as all the others

The first version of the launcher set `SHUFFLE_EVERY="${SHUFFLE_EVERY:-100}"` near the top and then
wrote `--shuffle-control-every "${SHUFFLE_EVERY:-20}"` inside the 100-step `probe` mode. The inner
default is **dead** — the variable is already set — so the probe ran at interval 100 and produced
exactly **one** control, at step 0, before anything could be learned. It printed `-0.0008` and nothing
further: the run's single most important readout silently absent, on the one run whose entire purpose
was to produce it. The loss curve and participation looked great throughout.

Same shape as the four bugs in §9 and the wrong diagnosis in §6.2 — a knob that looks set and is not,
with output that looks like output. The +1.21 above had to be recovered afterwards by re-running the
control against the saved checkpoint. Fixed by leaving `SHUFFLE_EVERY` empty at the top so each mode
supplies its own default, with a comment recording why. The general lesson §9 states about
optimisations applies verbatim to diagnostics: **a diagnostic needs a check that it actually ran.**

### 11.2 Why cross-replay is cheaper and faster than e2e, measured

Both arms train the *same* scalar indexer on the *same* corpus, so the difference is purely how the
router's gradient is obtained. Measured head to head on **one** H20 at N=8192, same scorer
(`mid_dim=0`), e2e run without `--liger`/`--ffn-sp-size` so the objectives are compared and not their
launch tricks:

| | peak | s/seq |
|---|---|---|
| e2e (gated causal LM) | **74.2 GiB** | 6.96 |
| cross-replay (`query_chunk=1024`) | **27.8 GiB** | 5.19 |

**Memory: 2.7x, and the cause is chunkability, not the attention kernel.** Split of the e2e step
shows the forward alone retains **49.7 GiB** — the router's gradient travels from `lm_head` back
through all 36 layers, so the *entire sequence's* backbone activations must stay alive until the
backward reaches layer 0. Nothing about that is wasteful; it is what an end-to-end causal LM loss
costs. Arithmetic at 104 KiB/token/layer:

* e2e: `36 x 8192 x 104 KiB` = **29.2 GiB** of activations + **7.0 GiB** of `(L, vocab)` logits, all
  live at once. `--liger` exists to remove the second term and FFN-SP to shard the first, which is why
  the e2e launcher needs both to reach 16K (89.1 GiB even *with* them, from its metrics).
* cross-replay: replay queries are **independent** once `C'` is masked from itself (§5), so only one
  chunk's graph is ever alive: `36 x 1024 x 104 KiB` = **3.66 GiB**, 8x less, plus 0.87 GiB of logits.
  The price is a fixed term e2e does not pay — `KV(C)` 1.12 + two copies of `h_C` 4.50 = 5.62 GiB —
  which *grows with N* while the chunk term does not.

So the saving is structural: **the rectangle geometry makes the loss decomposable over query blocks,
and the causal LM loss does not.** It is not that cross-replay found a cheaper kernel; both use one
fused attention call per layer. This also means the two arms respond differently to length — see the
fixed term above, and §6.4's note that §6.1(b) is now the next lever.

**Speed: only 1.34x, and the gap is far smaller than the memory gap.** Worth stating plainly because
it is tempting to read 2.7x memory as 2.7x speed. Cross-replay does *strictly more* attention work
(a full N x N rectangle against a causal triangle's N²/2) and an extra ungated prefill pass; it wins
on wall-clock anyway because chunking keeps everything in a fused kernel with far better allocator
behaviour, and because pass 1 is `no_grad`. Do not expect the ratio to hold at other lengths — the
rectangle's relative disadvantage grows with N.

**Both arms see identical tokens/step**, which is the point of matching everything that is not the
objective, though they reach it differently:

| | geometry | seqs/step | tokens/step @8K / @16K |
|---|---|---|---|
| scalar/e2e `stage1_16k` | NGPU 8, FFN_SP 8 -> **1** replica x accum **8** | 8 | 65536 / 131072 |
| cross-replay `stage1_16k` | NGPU 8, FFN_SP 1 -> **8** replicas x accum **1** | 8 | 65536 / 131072 |

Confirmed from both runs' `metrics.jsonl` rather than derived. `--global-batch-size` is what holds the
invariant: it divides by the replica count, so tokens/step does not move with `NGPU` or `FFN_SP`.

### 11.3 The 8-GPU 16K run (supersedes §11's probe)

`stage1_16k` on 8 GPUs, 8192:300 then 16384:300, `MAX_STEPS=600`, logged to
`Qwen-3-8B-gqa_indexer_cross_replay/stage1/metrics.jsonl`. Reached step 400 before being stopped:

| step | L | loss | participation | shuffle delta |
|---|---|---|---|---|
| 0 | 8K | 7.02 | 0.823 | **-0.014** |
| 100 | 8K | 5.22 | 0.054 | **+2.72** |
| 200 | 8K | 4.28 | 0.023 | **+2.97** |
| 300 | 16K | 5.39 | 0.014 | **+3.03** |
| 400 | 16K | 3.83 | 0.009 | **+3.41** |

The shuffle control rises **monotonically from ~0 to +3.41 nats/token** while participation falls to
0.009. Peak was 27.8 GiB at 8K and 33.5 at 16K — §6.4's numbers exactly, now under an 8-rank DDP
launch on real text. This is the strongest evidence the objective has produced, and it supersedes
§11's 100-step probe.

Two things still to watch rather than celebrate:

* **participation 0.009 is very concentrated** — the effective support is ~0.9% of the gated keys,
  well below the 50% the press keeps at `compression_ratio=0.5`. Whether that helps eviction (a sharp
  ranking) or hurts it (a ranking that only trusts a handful of keys, so the other ~49% are ordered
  by noise) is **not answerable from these numbers**, and it is the first thing a RULER eval should
  be read against.
* **The 16K loss jumps at the curriculum boundary** (4.20 at step 250 -> 5.39 at 300). Expected here
  and *not* the arithmetic artifact distillation has: an LM loss carries no `log(L)` term, so per §11
  this is a real change in difficulty (a longer rectangle is a harder reconstruction), not a scale
  shift. Worth confirming it decays back, which by step 400 (3.83) it has.

### 11.4 CORRECTION: chunking is exact in *math*, not in bf16 arithmetic

§6.1(a) says the query chunking is "exact", with "loss bit-identical" and "gradients agreeing to fp32
epsilon". Those measurements are real but were taken on **fp64 + SDPA on CPU**. Production is
**bf16 + flex_attention on CUDA**, a path that did not exist when that claim was written, and the
claim does **not** transfer. Measured on Qwen3-8B, N=2048, bf16, flex:

| | dloss vs unchunked | worst grad rel. dev. | cosine |
|---|---|---|---|
| `query_chunk=1024` | +9.5e-07 | 8.8e-02 | 0.99998 |
| `query_chunk=512` | **-1.0e-03** | 9.4e-01 | 0.99969 |
| `query_chunk=256` | -1.0e-03 | 9.2e-01 | 0.99969 |
| `logit_chunk` 512 / 128 at qc=1024 | +9.5e-07 | — | 0.99998 |

Relative deviations are taken only over parameters with `|g|max > 1e-4`, because `in_norm.bias`
carries a gradient of ~4.6e-06 where a relative measure is meaningless — the same near-zero artifact
§6.1 already flags (it reports 2.72 while differing by ~1e-5).

**Two things had to be separated to interpret this, and both were measured:**

1. **A nondeterminism floor exists.** Running the *identical* configuration twice gives
   **bit-identical loss but non-identical gradients** — relative deviation **~4e-03**, cosine
   1.0000000. `flex_attention`'s backward accumulates with atomics, so the gradient is not
   reproducible run to run even with no chunking involved. Any chunking comparison has to be read
   above this floor, and the qc=512 effect (9.4e-01) is clearly above it while qc=1024 (8.8e-02) is
   only ~20x above it.
2. **The chunking logic is correct; bf16 is the coarse part.** Same code, same chunk sizes, on a
   small real-architecture Qwen3 at three precisions — the error converges monotonically with
   precision, which it could not do if the decomposition were wrong:

   | dtype | worst grad rel. dev. across qc = 256/128/64 |
   |---|---|
   | bf16 | ~9.1e-03 |
   | fp32 | ~1.4e-05 |
   | fp64 | **~7.2e-07** |

**So: exact as mathematics, approximate as arithmetic.** The decomposition itself introduces no
approximation — replay queries genuinely are independent (§5), every chunk sees the whole key axis,
and at fp64 the residual is 7e-07, i.e. fp64 epsilon. What bf16 adds is reassociation error: chunking
regroups the summation over the ~N replay queries, floating-point addition is not associative, and
bf16 has ~3 decimal digits, so regrouping is visible at 1e-3 on the loss.

**Does it matter for training?** On the evidence here, no, but the honest statement is narrower than
"exact":

* The **cosine similarity stays >= 0.9997** at every chunk size, and §4 established that this
  objective only ever delivers a **ranking** to `TopK(s)`. A gradient 0.03% off in direction does not
  reorder keys. (This argument was *also* used in §2 to dismiss `log B`, where it was invalid — see
  §2.5. It is sound here because the question really is about the ranking at fixed parameters; it was
  unsound there because the question was about where training converges.)
* The deviation is **the same order as the nondeterminism the path already has** (4e-03) and well
  below normal SGD gradient noise across micro-batches.
* `logit_chunk` is much cleaner than `query_chunk` (9.5e-07): it splits the cross-entropy row-wise
  and does not regroup the attention sum at all.

What would be wrong is to keep calling it "bit-identical" on the production path, or to use it as
evidence that two runs at different `query_chunk` are comparable *checkpoint for checkpoint*. They are
comparable in distribution, not step for step. **`query_chunk` should therefore be treated as a fixed
part of a run's configuration**, not a free knob to retune mid-run — which is also why the driver
records it in every checkpoint's `config`.

*(Unverified, and the one thing that would sharpen this: whether accumulating `dL/ds` in fp32 —
which it already is, since `score_keys` returns `.float()` — could be extended to the attention
backward, or whether the bf16 reassociation is entirely inside `flex_attention`'s kernel and so not
reachable from this side. The measurements above do not distinguish those.)*

---

## 13. First RULER numbers, and the confound that makes them unreadable

`evaluation/results_sparse_scalar/` now holds both arms at `topk=2048, fraction=0.100`, step 600,
identical eval config apart from the checkpoint. Per-task, computed from the `metrics.json` files:

| task | 8K arm 1 (LM loss) | 8K arm 2 (cross-replay) | 16K arm 1 |
|---|---|---|---|
| `cwe` | 85.12 | 17.67 | 34.19 |
| `fwe` | 84.67 | 66.67 | 77.33 |
| `niah_multikey_1` | 96.30 | 53.70 | 59.26 |
| `niah_multikey_2` | 2.70 | 2.70 | 0.00 |
| `niah_multikey_3` | 0.00 | 0.00 | 0.00 |
| `niah_multiquery` | 94.30 | 35.09 | 50.44 |
| `niah_multivalue` | 90.79 | 36.40 | 57.02 |
| `niah_single_1` | 100.00 | 100.00 | 100.00 |
| `niah_single_2` | 100.00 | 59.09 | 92.42 |
| `niah_single_3` | 9.52 | **95.24** | 0.00 |
| `qa_1` | 59.57 | 40.43 | 48.94 |
| `qa_2` | 38.64 | 25.00 | 29.55 |
| `vt` | 99.51 | 49.76 | 95.12 |
| **mean** | **66.24** | **44.75** | **49.56** |

(`results_sparse_scalar/ruler__16384__*/2` exists but is empty — the 16K cross-replay eval did not
produce metrics.)

**This is not a loss-vs-loss comparison, and must not be quoted as one.** Two variables moved:

1. **Capacity, 26×.** From the checkpoints' own `config`: arm 1 ran `scalar_mid_dim=256` → **38.14 M**
   indexer parameters (`w_in (256,4096)`, `w_out (8,256)`, plus norms); arm 2 ran
   `scalar_mid_dim=0` → **1.48 M** (`w_out (8,4096)` only). `scalar_indexer.py` documents `mid_dim`
   as this arm's main capacity knob, with a nonlinear readout worth +0.12/+0.09 held-out Spearman
   over a linear one in the probe study. Fixed: the launcher now defaults `MID_DIM=256`, matching
   `train_gqa_indexer_scalar_gy.sh`.
2. **Budget, hence concentration.** Arm 2 trained at `B=1` (§2.5) and converged to participation
   **0.0070** while eval keeps `topk/N = 0.125`. Fixed: `BUDGET=2048`.

**The failure pattern is nonetheless informative, and it is the concentration signature rather than
noise.** `niah_single_3` is the *only* task that improves, and it improves enormously (9.52 → 95.24);
everything that needs many keys retained simultaneously falls hard (`cwe` counts frequent words, `vt`
tracks variable chains, `multiquery`/`multivalue` have several targets). An over-concentrated score is
exactly what would do that. It is consistent with the 18× mismatch, but with two variables moving it
is consistent, not established.

**A zero-cost check available before any retraining:** re-run the eval on the *existing* arm-2
checkpoint at `topk ≈ 128–256` (0.0070 × 16384 ≈ 115 keys, i.e. the budget it actually trained for).
If the score recovers sharply, the concentration-mismatch reading is confirmed without waiting for a
run. If it does not, the reading is wrong and the capacity confound is carrying more of the 21 points
than assumed.

**Everything that is not the objective must move together.** Both fixes are in
`scripts/train_gqa_indexer_cross_replay_gy.sh` with the reasoning inline, so the rerun is A/B against
`train_gqa_indexer_scalar_gy.sh stage1_16k` on every axis except the loss. The general failure here is
the same one §9 records for optimisations: **an A/B needs a check that only one thing differs, and the
checkpoint config is the only place that check can be made** — which is why `budget` is now recorded
there alongside `query_chunk` and `scalar_mid_dim`.

⚠️ §13's "zero-cost check" was run and it **falsified the reading above**. See §14: lowering `topk`
made arm 2 *worse*, not better, and the whole `topk`-vs-quality curve turned out to be the thing that
was never measured. Read §14 before quoting anything in §13 or in §2.5's "What it cost us" paragraph.

---

## 14. The arm-2 rerun collapse: it was the eval budget, not the score

Arm B — the rerun that fixed both of §13's confounds (`scalar_mid_dim=256`, `budget=2048`) — scored
**20.43** at `topk=2048`, against 44.75 for arm A and 66.24 for arm C. Every prior said it should
beat A: 26× the capacity, `B` raised from 1 to 2048, participation up 9× (0.0070 → 0.062), replay
loss 1.18 vs 2.70, shuffle control +5.46 vs +3.41.

| arm | objective | `mid_dim` | `budget` | RULER 8K @ `topk=2048` |
|---|---|---|---|---|
| A | cross-replay | 0 (linear, 1.47M) | — (⇒ `B=1`) | 44.75 |
| B | cross-replay | 256 (MLP, 38.14M) | 2048 | **20.43** |
| C | e2e LM loss | 256 (MLP, 38.14M) | — | 66.24 |

### 14.1 The `topk` sweep, which inverts §13's prediction

§13 predicted that if the concentration-mismatch reading were right, *lowering* `topk` toward the
budget B actually trained for would recover the score. Run on B's existing checkpoint, nothing
retrained, `fraction=0.100, seed=42`, everything but `topk` identical:

| `topk` | RULER 8K mean | `vt` | `niah_single_1` | `niah_single_3` | `cwe` |
|---|---|---|---|---|---|
| 1024 | **8.32** | 0.00 | 3.03 | 0.00 | 8.37 |
| 2048 | 20.43 | 0.00 | 62.12 | 50.00 | 19.77 |
| 4096 | **64.13** | 44.39 | 92.42 | 100.00 | 66.51 |

**Monotonically increasing, and steeply.** `topk=1024` — the direction §13 proposed — is *worse*
(8.32). `topk=4096` recovers B to 64.13, i.e. **+43.7 points from a budget change alone, with the
same weights**. So:

* **The score is not broken.** A broken score cannot be rescued to 64.13 by doubling the budget.
* **§13's proposed check had the sign backwards.** It reasoned from "participation 0.0070 ⇒ the score
  wants a *small* budget", but participation is a property of the *gate's* soft mass distribution
  during training, not of how many keys a *hard* top-k needs in order to contain the answer. The two
  are not the same quantity, and §13 treated them as one.

#### 14.1.1 A confound in this very table, found and cleared

`sparse_inference.py` gained the query-independent `flex_attention` selection path
(`qi_sparse_attention`, `_use_qi`) **while this sweep was running**. Timeline, from file mtimes and
the shard logs' own timestamps:

| event | time |
|---|---|
| `topk=2048` run (the 20.43) | 19:35 – 19:45 |
| `sparse_inference.py` modified — flex path added | **20:45** |
| `qi_flex_attention.py` modified | **21:06** |
| `topk=1024` run | 20:35 – 20:58 |
| `topk=4096` run | 20:59 – 21:17 |

`_use_qi` is chosen automatically from `ScalarIndexer.is_query_independent`, so the `topk=2048` row
was produced on the **gather** path and the other two rows (at least partly) on the **flex** path.
The three points were therefore not guaranteed to be one experiment, and the +43.7 recovery could
have been a kernel artefact rather than a budget effect.

**Checked, and the paths agree** (`proxy_exp_budget/path_agreement.py`, same model/checkpoint/input,
`query_independent=` forced either way, last 256 logit rows of a `niah_single_1` context):

| `topk` | mean abs logit diff | top-1 agreement | KL(gather‖flex) |
|---|---|---|---|
| 1024 | 0.380 | 0.9961 | 2.4e-04 |
| 2048 | 0.335 | 1.0000 | 2.6e-04 |
| 4096 | 0.231 | 1.0000 | 1.4e-04 |

A KL of ~2e-04 and top-1 agreement of 1.000 is bf16-rounding equivalence, not a behavioural
difference, so the sweep reads as one experiment and the table above stands. (`max abs logit diff`
is ~14, but on unnormalised logits at a handful of positions — the KL and the argmax agreement are
the meaningful columns.) An end-to-end re-run of `topk=2048` on the current code is the direct
confirmation; see §14.7.

⚠️ **The methodological point is worth more than the result.** The sweep was launched, the working
tree changed underneath it, and nothing in the output would have said so — the results directory
records `topk`, `fraction`, `seed` and the checkpoint, but **not the code version that produced
them**. This is the same class of failure as §13's capacity confound (which only the checkpoint's own
`config` revealed) and §9's optimisation confounds. `config.yaml` should record the repo's `git rev-parse HEAD`
plus a dirty flag; without it, "identical eval config" is a claim about the flags, not about the run.


### 14.2 The main hypothesis was wrong, and the negative results are worth recording

The hypothesis under test was that B's score is effective on the replay objective but degenerates
under a hard top-k — content-independent selection, or domination by one component. **Measured on
real RULER text (3 documents from distinct tasks, 36 layers, `topk=2048`, `N=8192`), it is not.**
Scored via `ScalarIndexer.score_keys` from hidden states captured on the real model
(`proxy_exp_budget/dissect_scores.py`):

| measurement | A | B | C | discriminates B? |
|---|---|---|---|---|
| **top-k Jaccard, 2 documents** | 0.1522 | **0.1490** | 0.1531 | **no** — all at the 0.1429 chance floor |
| top-k Jaccard, 2 heads | 0.1826 | 0.1977 | 0.2279 | no |
| per-head Spearman | 0.0758 | 0.0856 | 0.1700 | no |
| score mean / std | -1.73 / 3.84 | -1.69 / 2.52 | -3.27 / 2.31 | no |
| std / abs(mean) | 3.22 | 8.94 | 1.36 | no (B is the *least* bias-dominated) |
| corr(score, position) | 0.0049 | **-0.0284** | 0.0917 | no — B is the least positional |
| `pos_slope` tilt / std | 2.3e-03 | 3.4e-03 | 3.9e-03 | no — negligible in all three |
| `w_out` top-1 SV energy | 0.2105 | 0.2429 | 0.3266 | no |
| `w_out` effective rank | 7.14 | 6.52 | 5.56 | no (of 8; C is the *most* collapsed) |
| `w_in` effective rank | — | 28.85 | 35.21 | no (of 256) |

**Every axis is negative.** In particular the load-bearing one: B's cross-document top-k Jaccard is
**0.1490 against a chance floor of `topk/(2N-topk)` = 0.1429**. B's selection is content-driven, not
a fixed set of positions. No per-head collapse (effective rank 6.52 of 8 — and C, which works best,
is the most collapsed at 5.56). No position tilt: `pos_slope`'s total contribution is 3.4e-03 of a
standard deviation, and B's score-position correlation is the *smallest* of the three. No
constant-bias domination.

**So if B had been judged on the score-anatomy axes alone, it would have passed.** The defect is
quantitative and lives in *how many* keys it takes to contain the answer — which none of the above
measures.

### 14.3 What does discriminate B: needle coverage, and its depth profile

The measurement that separates the arms asks directly whether the answer's own key positions are
inside the support (`proxy_exp_budget/needle_coverage.py`, 16 rows with a locatable needle over 4
tasks, support built through the real `project_q`/`project_k`/`streaming_topk_support` path with the
eval's `force_sink=4, force_local=64`):

| `topk` | chance (`topk/N`) | A | B | C | B − A |
|---|---|---|---|---|---|
| 1024 | 0.129 | 0.329 | **0.240** | 0.348 | −0.089 |
| 2048 | 0.258 | 0.480 | **0.364** | 0.446 | −0.115 |
| 4096 | 0.515 | 0.689 | **0.562** | 0.601 | −0.128 |

B is the worst at every budget, and the deficit is *depth-dependent* — at `topk=2048`, by layer third:

| layers | A | B | C |
|---|---|---|---|
| 0–11 (early) | 0.4575 | 0.4020 | 0.4335 |
| 12–23 (mid) | 0.4550 | 0.3796 | 0.4747 |
| 24–35 (late) | **0.5261** | **0.3107** | 0.4296 |

A's coverage *rises* into the late layers (0.458 → 0.526) while B's *falls* (0.402 → 0.311, with 5 of
36 layers below chance against 1 for A and C). The late layers are where retrieval is consumed, so a
uniform ~0.1 deficit concentrated at depth is exactly the shape that costs answers.

This is a **quantitative** deficit, not a degeneracy: B needs a larger budget than A to contain the
same needle. Combined with §14.1, that is the whole story — B's score ranks the right keys *lower*
than A's does, so at `topk=2048` the needle falls outside the support and at `topk=4096` it comes
back inside.

### 14.4 Three measurements that looked decisive and were not

Recorded because each one is a trap this investigation walked into, and the shape of the mistake
generalises.

1. **Averaged attention-mass recall says all three arms are equal.**
   `proxy_exp_budget/attn_recall.py`, mass of the true attention distribution landing inside the
   support, 36 layers, real q/k post-RoPE:

   | document | A | B | C | oracle |
   |---|---|---|---|---|
   | `niah_multikey_2` | 0.7443 | 0.7170 | 0.7400 | 0.9843 |
   | `vt` (B scores **0.00**, A 49.76) | 0.7581 | **0.7185** | 0.7868 | 0.9792 |

   A 4-point recall gap on the very task where B scores 0.00 and A scores 49.76. **Why it fails:**
   most of a long context's attention mass sits on the sink and the local window, which
   `force_sink`/`force_local` pin unconditionally. The needle is a few tokens carrying ~1% of the
   mass. Averaged mass recall is therefore mostly measuring the *forced* slots — it cannot see the
   thing that decides the answer. Per-layer inspection does not rescue it: no arm has a collapsed
   layer, and B tracks A and C at all 36.

2. **Mid-context logit divergence says all three arms are equal.**
   `proxy_exp_budget/logit_divergence.py`, 2048 positions sampled from the second half of 8 prompts,
   real `SparseAttentionContext` vs dense: KL(dense‖sparse) A 0.051 / B 0.064 / C 0.035, top-1
   agreement 0.928 / 0.926 / 0.951. On `vt`, B's top-1 agreement is **1.000**. **Why it fails:** an
   ordinary mid-context token is predicted from its local window, which is pinned. That measures
   language modelling under sparsity, not retrieval. A retrieval failure is invisible at every
   position except the one that needs the distant key.

3. **A 6-position version of the same measurement said the opposite, and was noise.** Scoring only
   the 3 gold tokens of 2 `vt` rows gave B KL 6.31 / top-1 agreement 0.00 against C's KL 0.0006 /
   1.00 — which looked like the decisive discriminator and was quoted as such for one iteration. Six
   positions from two rows is not a measurement. The 2048-position version (2 above) shows the
   arms nearly equal. **Lesson:** a difference of three orders of magnitude is not self-validating;
   it is exactly when the sample size deserves the most scrutiny.

   (A related invalid attempt is worth naming: scoring the answer position from a raw
   `context + question + answer_prefix` string, without the chat template `KVPressTextGenerationPipeline`
   applies, put *dense* gold NLL at 16.6 nats with the gold token top-1 in **0 of 42** rows. When the
   dense reference is that bad, no sparse-vs-dense delta computed against it means anything.)

### 14.5 What this revises

* **§13's zero-cost check: retracted.** Re-running at `topk ≈ 128–256` would have shown ~0, and §13
  would have read that as "the concentration reading is wrong and the capacity confound carries the
  21 points". Both halves of that inference would have been wrong. The correct sweep direction was
  *up*.
* **§2.5's "What it cost us": the RULER-points attribution is unsafe.** It attributes arm A's 21-point
  deficit to `B=1`'s participation collapse via an "18× mismatch" against `topk/N`. The mechanism
  (small `B` ⇒ concentrated gate) is still measured and still real; what does not follow is reading
  `topk/N` as the budget the *hard* selector needs. Arm A, the `B=1` run, has the **best** needle
  coverage of the three at every `topk`. Participation and needle coverage are different quantities.
* **§0's fourth silent failure** should be read with that caveat: watching `participation` against
  `topk/N` is a reasonable diagnostic for the gate, but it does not predict the `topk` a hard top-k
  eval needs, and this section is the counterexample.
* **Unchanged:** everything about the budget term's effect on converged concentration (§2.5's tables),
  the three silent-failure guards (§0, §3, §5), and the memory accounting (§6.4). None of this
  section touches them.

### 14.6 The open question this leaves

Why does B's score rank needles lower than A's, given more capacity and a better-conditioned budget?
Not answered here. What is now established is the shape of the answer: it is a *ranking-margin*
question at fixed budget, not a degeneracy — so the diagnostic to build is needle coverage as a
function of `topk` per layer (§14.3's instrument), and the arms must be compared at **matched needle
coverage** rather than matched `topk`. Comparing two scorers at one arbitrary `topk` conflates "ranks
the needle higher" with "needs a smaller budget", which is what produced the 20.43-vs-44.75 mystery
in the first place.

**The reusable lesson.** Three separate averaged metrics (mass recall, per-layer recall, mid-context
KL) all said "these arms are equivalent" while the task metric said 20.43 vs 66.24. Every one of them
averaged over positions the pinned slots already handle. **When a proxy metric disagrees with the task
metric by 40 points, the proxy is measuring the pinned part of the problem** — find the measurement
that isolates the ~1% of mass the budget is actually deciding, which here meant locating the answer's
own tokens and asking whether they are in the set.

### 14.7 Status of the confirmations, and the flex-path bug that blocked one (now fixed)

**The blocker, and its diagnosis.** The end-to-end re-run of B at `topk=2048` on the current code —
which would turn §14.1.1's logit-level path agreement into a task-level one — crashed on all four
shards:

```
qi_flex_attention.py:331  in qi_sparse_attention
    dl = deadlines(scores[0].float(), topk, force_sink=..., force_local=...)
qi_flex_attention.py:174  in deadlines
    neg = torch.tensor(-float("inf"), device=device, dtype=scores.dtype)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**That traceback is misleading.** With `CUDA_LAUNCH_BLOCKING=1` the fault resolves to an
**inductor-compiled Triton reduction** (`triton_per_fused__to_copy_slice_sum_transpose_6`) launched
inside a compiled region — the `torch.tensor` line is merely where the asynchronous error surfaced.
So it is a **`torch.compile` specialisation bug, not an out-of-bounds index in `deadlines`**:

* `deadlines` and `qi_block_mask` called directly at `k_len` ∈ {2048, 2112, 4096, 8030, 9326, 9605,
  10931, 11080, 12288, 16384} — including non-multiples of the 128 block size — all pass.
* A prefill plus 60 sequential decode steps, each a new `k_len` — passes.
* The four shards crashed at *different* contexts (45/163, 121/163, 117/162, 109/162), i.e.
  shape-sequence dependent rather than systematic.

**Localised to one of the two compiled callables.** `_flex()` and `_block_mask()` both used
`dynamic=None` (automatic mode: specialise on the first shape, re-specialise when a second arrives).
Every RULER context has a different length, so a sweep drives that transition constantly. Bisecting
the two (`proxy_exp_budget/flex_dynamic_probe.py`, modes `none` / `bm_true` / `flex_true` / `true`):

| mode | `flex_attention` | `create_block_mask` | RULER 8K `fraction=0.02` |
|---|---|---|---|
| `none` | `None` | `None` | **crash** (illegal memory access) |
| `bm_true` | `None` | `True` | 20.69, completed |
| `true` | `True` | `True` | 20.69, completed |

`bm_true` and `true` agree on **all 13 tasks task-for-task**, so the compile mode does not change
which keys are selected. **Fix applied: only `create_block_mask` is pinned to `dynamic=True`**
(`_BLOCK_MASK_DYNAMIC`), which leaves `flex_attention` — where the 45 ms steady state lives — at the
fast setting. The kernel name (`_to_copy_slice_sum_transpose`) is consistent with the mask build's
block-wise reduction rather than the attention kernel, so the localisation and the symptom agree.
Suspected upstream bug in torch 2.10.0-rc6; worth rechecking on a release build.

**Regression tests added** to `tests/presses/test_gqa_indexer_qi_flex.py` (now 16 passing):

* `test_block_mask_is_compiled_dynamic` — asserts the constant. **This is the test that actually
  guards the revert:** mutation-tested by setting it back to `None`, and it fails.
* `test_many_lengths_through_one_compiled_callable` — several key lengths (non-monotonic,
  non-128-multiples) through *one* process and *one* compiled callable, checking support exactness
  against `streaming_topk_support` at each. ⚠️ **Honest limitation, measured:** this test does *not*
  fail when the fix is reverted — at sub-1K lengths the faulty kernel is never generated, and
  reproducing the crash needed the real model at 8-11K keys over 36 layers. Its value is pinning
  selection exactness across a recompile sequence, not catching this specific crash.

**Why the suite missed it originally:** every other test parametrises the length, so pytest gives
each one a fresh compile and the re-specialisation path is never reached. Same shape of gap as §5's
`is_causal` trap — the fast path was verified against the reference *at one shape*, and the bug lived
in the transition between shapes.

**The confirmation the fix unblocked.** B at `topk=2048`, full `fraction=0.100` run on the fixed flex
path, against the original gather-path number the whole investigation started from:

| | gather (original) | flex (fixed) |
|---|---|---|
| `cwe` | 19.77 | 23.02 |
| `niah_single_1` | 62.12 | 59.09 |
| `niah_single_3` | 50.00 | 54.76 |
| `vt` | 0.00 | 0.00 |
| `niah_single_2` | 7.58 | 7.58 |
| **mean** | **20.43** | **21.16** |

A 0.73-point difference with per-task moves in both directions, which is exactly the module's own
"comparable in aggregate but not row-identical" characterisation (the two kernels accumulate the same
softmax in a different order; 5.0e-3 apart, both ~2.05e-3 from an fp32 reference). **So §14.1.1's
kernel confound is now closed at the task level, not just the logit level**, and §14's conclusions
stand: 20.43/21.16 at `topk=2048` against 64.13 at `topk=4096` is a budget effect, ~60× larger than
the kernel difference.

**Still not run:** arms A and C at `topk=4096`. Without them the §14.1 sweep shows that *B* recovers
to 64.13, but not whether B ever *catches* A and C — they may recover too. Since §14.3 has B's needle
coverage below both at every budget, the likeliest outcome is that all three rise and B stays last,
which would mean B is simply a worse scorer at matched budget and the `topk=2048` gap overstated it.
Until measured, **the honest claim is "B's collapse is budget-sensitive", not "B is fixed by
`topk=4096`"** — 64.13 against C's 66.24 at a *different* budget is not a comparison.

*(Now run — see §15. The predicted outcome held: all three rise, B stays last.)*


⚠️ One residual caveat for §14.1's table: the `topk=1024` and `topk=4096` rows ran on the flex path
with this bug latent in the process. They did not crash, and the `topk=2048` row has now been
reproduced on both paths to within 0.73, so there is no evidence they are wrong — but they have not
themselves been re-taken on the fixed code.






---

## 15. Why cross-replay loses to the e2e LM loss: it is the objective, not the budget

§14 closed the "why did B collapse" question (eval budget). This section answers the larger one:
**cross-replay is genuinely worse than the plain e2e LM loss, at matched budget, and the budget term
`B` is not what separates them.** Two measurements settle it.

### 15.1 At matched `topk=4096` the ordering is unchanged

§14.1 measured only arm B across `topk`. Doing all three at `topk=4096`, `fraction=0.100, seed=42`,
step 600, everything but the checkpoint identical:

| task | A cross-replay (mid 0, B=1) | B cross-replay (mid 256, B=2048) | C **e2e LM loss** (mid 256) |
|---|---|---|---|
| `cwe` | 53.26 | 66.51 | **96.98** |
| `fwe` | 82.00 | 79.33 | 94.00 |
| `niah_multikey_1` | 96.30 | 87.04 | 100.00 |
| `niah_multikey_2` | 8.11 | 27.03 | 100.00 |
| `niah_multikey_3` | 0.00 | 0.00 | 0.00 |
| `niah_multiquery` | 95.18 | 87.28 | 100.00 |
| `niah_multivalue` | 92.54 | 75.88 | 98.68 |
| `niah_single_1` | 100.00 | 92.42 | 100.00 |
| `niah_single_2` | 95.45 | 71.21 | 100.00 |
| `niah_single_3` | 100.00 | 100.00 | 85.71 |
| `qa_1` | 65.96 | 61.70 | 74.47 |
| `qa_2` | 40.91 | 40.91 | 59.09 |
| `vt` | 91.71 | 44.39 | **100.00** |
| **mean** | **70.88** | **64.13** | **85.30** |

Compare with `topk=2048` (44.75 / 20.43 / 66.24). **Every arm rises, the ordering is identical, and
the C−A gap barely moves (21.5 → 14.4).** So:

* **§14's budget story is confirmed and bounded.** Raising `topk` is worth +26 to +44 points, far
  more than anything else measured here — but it does not reorder the arms. B was both
  budget-starved *and* the worst scorer; those were two separate facts.
* **Cross-replay's deficit against e2e is real, not a budget artefact.** It survives a 2× budget
  change intact.
* `niah_single_3` is the one task where cross-replay wins (100.00 vs 85.71 at 4096), consistent
  across every budget. It is the task needing the *fewest* keys — the same signature §2.5 noted.
* `niah_multikey_3` is 0.00 for all three at both budgets: a floor, not a discriminator. Worth
  excluding from means when comparing, since it only dilutes.

### 15.2 RETRACTED (§2.5): `B=1` is not what causes low participation — the objective is

§2.5 attributed arm A's deficit to training at `B=1`, via the participation collapse to 0.0070
against an eval `topk/N` of 0.125. **The decisive counterexample was in the repo the whole time:
arm C, the best arm, also trains at `B=1`.** `E2EIndexerTrainer.gate_budget` defaults to `1.0`, and
arm C's launcher never overrode it. Final-step training metrics:

| arm | objective | `mid_dim` | `B` | final loss | participation | `gate_scale` | RULER @2048 | @4096 |
|---|---|---|---|---|---|---|---|---|
| A | cross-replay | 0 | **1** | 2.695 | 0.0070 | 0.568 | 44.75 | 70.88 |
| B | cross-replay | 256 | 2048 | **1.182** | 0.0622 | 1.119 | 20.43 | 64.13 |
| C | **e2e LM loss** | 256 | **1** | 1.740 | 0.1652 | 1.131 | **66.24** | **85.30** |

Read the A/C pair: **same `B=1`, same schedule/LR/data/seed, and participation differs by 24×**
(0.0070 vs 0.1652). A budget constant cannot explain a 24× difference it is held fixed across. And
B — the arm that *fixed* `B` to 2048 — has **lower** participation than C at `B=1` and the worst RULER
of the three. `B` is therefore not the lever §2.5 claimed; what §2.5 measured (small `B` ⇒ more
concentrated, at fixed objective and 100–300 steps) may still hold, but it does not survive as an
explanation of the A-vs-C gap, and the "~21 RULER points" attribution is withdrawn.

⚠️ **A confound in that participation column, which I nearly reported without noticing.** The two
numbers are computed against **different denominators** and are not directly comparable as printed:

* cross-replay (`gate_participation`): one rectangle row, denominator `N − n_sink`.
* e2e (`E2EIndexerTrainer._gate_participation`): per **causal** row `t`, denominator `t`, averaged
  over rows — so early rows, which can only see a few keys, are scored against a small history.

If the underlying gate were *identical* and peaked on `k` keys, the two metrics would still differ:

| peaked on `k` keys | cross-replay reports | e2e reports | inflation |
|---|---|---|---|
| 128 | 0.0156 | 0.0805 | 5.2× |
| 512 | 0.0625 | 0.2357 | 3.8× |
| 2048 | 0.2500 | 0.5965 | 2.4× |

So C's 0.1652 is inflated by roughly **2.4–5.2×** relative to A's 0.0070 and B's 0.0622. **The
conclusion survives the correction:** deflating C by even 5.2× leaves ~0.032, still 4.5× above A's
0.0070 at the same `B=1`, and the A-vs-C comparison is what the retraction rests on. But the raw
columns must not be quoted side by side, and a cross-objective participation comparison needs one
metric with one denominator — currently it does not exist. That is a diagnostic to build before any
further participation-based reasoning.

### 15.3 RESULT: the 2×2 is complete, and the budget "fix" is the harmful variable

A and C differ in **two** variables (objective *and* `mid_dim`), which is §13's confound again in a new
place, so the missing cell was trained: `stage1_256_B1`, 600 steps, `MID_DIM=256 BUDGET=1`, same
`8192:300,16384:300` schedule and seed 0 as all three existing runs (checkpoint config confirms
`budget: 1, scalar_mid_dim: 256`, 38.14 M params). Call it **arm D**. The grid at `topk=2048`:

| | `mid_dim=0` | `mid_dim=256` |
|---|---|---|
| cross-replay, `B=1` | **A** 44.75 | **D** 48.20 |
| cross-replay, `B=2048` | — | **B** 20.43 |
| e2e LM loss, `B=1` | — | **C** 66.24 |

Three predictions were recorded before the run: near C (⇒ capacity), near A (⇒ objective), near B
(⇒ `mid_dim` harmful). **D landed at 48.20 — near A, so prediction 2 held.** But the useful reading is
the set of contrasts the grid now makes single-variable for the first time, and both budgets were
measured so the effects can be checked for budget-dependence:

| contrast | change | @`topk=2048` | @`topk=4096` |
|---|---|---|---|
| capacity | A → D (`mid_dim` 0 → 256, `B=1` both) | **+3.45** | **+1.98** |
| **objective** | D → C (cross-replay → e2e, all else matched) | **+18.04** | **+12.44** |
| **budget** | D → B (`B` 1 → 2048, `mid_dim=256` both) | **−27.77** | **−8.73** |

Full arm means: A 44.75 / 70.88, D 48.20 / 72.86, B 20.43 / 64.13, C 66.24 / 85.30. **The ordering
C > D > A > B is identical at both budgets**, so none of the three conclusions is a budget artefact.

**The budget term is the largest single effect and its sign is negative.** `B=2048` — the value §2.5
derived from the `Σexp(g)=B` identity, prescribed as the fix, and §13 listed as one of two confounds to
eliminate — costs **27.8 points** at `topk=2048` and **8.7** at 4096. Even the smaller figure exceeds
what 26× the scorer capacity buys. Note the damage *shrinks* as the eval budget grows, which is
consistent with §14: over-concentration hurts most when the budget is tight.

**So §2.5's prescription `B = topk` is retracted, not just its RULER attribution.** §15.2 already
withdrew the claim that `B=1` caused A's deficit (C runs `B=1` and is the best arm); this goes further:
at this scale, setting `B` to the inference top-k is *actively harmful*. The identity is still true and
§2.5's short-run measurements (small `B` ⇒ more concentrated at fixed objective) may still hold, but
"exactly representable during training" is not the property that matters, and the argument from
representability was wrong. **`BUDGET=1` in `train_gqa_indexer_cross_replay_gy.sh` is now the
measured-best setting rather than the legacy one** — the comment block there has been corrected, since
it argued the opposite at length.

**Capacity is nearly inert under this objective, which is a finding in its own right.** `mid_dim`
0 → 256 is 26× the parameters for **+3.45 / +1.98**. The same knob is worth far more under the e2e loss
(it is why §13's confound mattered). A router that cannot use 26× capacity is not capacity-limited, so
"cross-replay just needs a bigger scorer" is ruled out.

**The objective gap is the irreducible one: +18.0 / +12.4 for e2e over cross-replay** at matched
capacity, matched budget, matched schedule/LR/seed/data. That is the answer to "why is cross-replay
worse than the e2e LM loss" — it is the objective itself, not any of the three knobs that were
suspected.

**This time the training-side diagnostics predicted the outcome, which §14 showed they usually do
not.** D tracked A on both: `gate_scale` 1.00 → **0.573** (A 0.568; B and C both *rose* to ~1.12) and
participation → **0.0136** (A 0.0070; B 0.0622, C 0.1652). Both said "A-like run" from ~step 220, well
before the eval. Note the implication: at fixed objective, `mid_dim` did **not** move converged
concentration, while changing the objective did — concentration under cross-replay is set by the loss
geometry, not by scorer capacity and not by `B`. `shuffle_delta` reached **+6.00**, the largest of any
arm, so the score carries a genuine ranking; it is simply the wrong one for a hard top-k.

**Per-task, D is a better A, not a smaller C.** It improves on A almost everywhere
(`niah_multiquery` 35.09 → 47.37, `niah_single_2` 59.09 → 78.79, `cwe` 17.67 → 28.84) and keeps A's
signature win on `niah_single_3` (95.24 → 97.62, against C's 9.52); the one clear regression is `vt`
(49.76 → 32.20). A and D therefore share a *shape* C does not: cross-replay arms win the few-keys task
and lose the many-keys ones (`cwe`, `vt`, `multiquery`, `multivalue`). That is the concentration
signature — and D has it at `B=1`, which is §15.2's point again from the other direction.

At `topk=4096` D's `vt` recovers to 96.10 (A 91.71, C 100.00) and the whole grid compresses, but the
one task cross-replay wins outright is still `niah_single_3` (100.00 for A/B/D against C's 85.71) —
the fewest-keys task, at every budget and every capacity. That is now a four-arm, two-budget
regularity rather than an observation about one run: **cross-replay systematically over-concentrates
relative to the e2e loss, and it is the objective that does it.**

(`niah_multikey_3` is 0.00 for all four arms at both budgets, so it only dilutes the mean.
`niah_multikey_2` separates C sharply from every cross-replay arm — 100.00 against 8–27 at `topk=4096`
— which is worth a look on its own, since it is the largest single-task gap in the grid.)

### 15.4 What NextMem does differently, and what transfers

`NextMem/models/NextMemQwen.py` is a repeat-reconstruction architecture, so it is worth asking what
it does that this objective does not. Read from the code and README (the repo ships inference only;
training is not included, so the reading is structural rather than from a loss function).

**What it is.** An autoregressive **autoencoder**: `batch_encode` appends a `<|start_of_document|>`
token, then autoregressively emits `latent_length` hidden vectors, each fed back as the next input
embedding (`NextMemQwen.py:164-179`). `batch_decode` prepends those latents to a suffix and decodes.
Memory is a handful of **latent vectors**, not a KV cache — so it is a different compression target
(no per-key selection at all) and most of it does not apply.

Three things do:

1. **A dedicated `<|start_of_document|>` / "repeat" token, with its own trained embedding table.**
   `special_input_embeddings` is a separate `nn.Embedding` added to the ordinary token embedding
   (`:123-134`), initialised from the embedding of the word "repeat" (`:27-30, :109-114`). The repeat
   instruction is a *learned* signal, not a fixed prompt. **Cross-replay has no such marker**: `C'`
   is the raw context tokens at shifted positions, with nothing telling the model "this is a
   reconstruction, not a continuation". KVzip does use a text repeat prompt ("Repeat the previous
   context:") and §C.2 measures it as negligible *for KVzip* — but KVzip does not differentiate a
   loss, so "negligible for scoring" does not imply "negligible for a gradient". Adding a learned
   marker token is cheap (one embedding row) and is the most directly transferable idea here.
2. **A two-stage curriculum: "autoregressive reconstruction alignment" then "progressive latent
   substitution"** (README). The second stage swaps ground truth for the model's own latents
   *gradually*. The `wo_PS` ablation exists as a separate model class, i.e. they measured its removal.
   Cross-replay has no curriculum on the replay side — the gate is at full strength from step 0. If
   the difficulty of the replay task is what limits it, a ramp is the analogous fix.
3. **Their reconstruction is deliberately made hard.** The latent bottleneck is a few vectors, so
   copying is impossible and reconstruction must actually compress. That is the mirror image of
   §7.2.2's finding: when `C'` could see itself, the task became a copy (loss 0.01 nats) and the
   supervision vanished. **Both point the same way — a reconstruction objective only teaches
   something if reconstruction is hard.** Cross-replay currently sits at loss 1.18–2.70 nats, which
   is not trivially easy; but this is the axis to think along, and it is an argument *against*
   loosening the mask and *for* asking whether the current task is too easy in some other way.

Not transferable: the latent bottleneck itself, the PEFT/LoRA two-adapter structure (`:118-122`, and
note `decode` runs under `disable_adapter()` — the encoder is tuned, the decoder is the frozen base
model), and NF4 quantisation of the latents (`NextMemQwenSparse.py`).

---

## 16. Why cross-replay loses: the rectangle is the wrong supervision shape

§15.3 localised the deficit to the objective (+18.0 / +12.4 for e2e at matched capacity, budget,
schedule, seed and data). This section asks *why*, tests the obvious mechanism, and states what to
change. The framing that makes it tractable:

* **e2e LM loss** = loss on `C`, where query `t` sees `C[0..t]` — a **causal triangle**.
* **cross-replay** = loss on `C'`, where query `j` sees all of `C` — a **full rectangle**.

Both are next-token losses on the same text with the same frozen backbone. The only difference is the
*shape of the key set each query chooses from*.

### 16.1 REFUTED: it is not a label leak

The first hypothesis, and it would have explained everything: under the rectangle, query `j` predicts
`C'[j+1]`, and with `C' == C` the key `C[j+1]` **is the target token** and is visible. The e2e loss
cannot do this (`C[t+1]` is strictly outside `C[0..t]`). A gate could then serve the loss by pointing
at one key, which is exactly the over-concentration §15.3 measured.

Measured on the frozen model with **no gate** (`proxy_exp_budget/label_leak.py`, `N=2048`, four masks
over `C`, replay positions `N..2N-1` as the trainer uses):

| task | `rect` | `rect_noself` (target key removed) | `rect_causal` | `causal` | leak share | argmax on target |
|---|---|---|---|---|---|---|
| `qa_1` | 6.4308 | 6.5963 | 7.0437 | 7.0437 | 0.025 | 0.000 |
| `vt` | 2.7358 | 2.7985 | 2.8141 | 2.8141 | 0.022 | 0.000 |
| `niah_single_2` | 6.7782 | 6.9331 | 7.1301 | 7.1301 | 0.022 | 0.000 |

**Removing the target key changes the loss by 2.3%, and no attention head's argmax lands on the target
position (0.000 of 64 sampled rows × 32 heads at layer 18).** The leak exists arithmetically and is
negligible in practice. Hypothesis dead — and worth recording, because it is the kind of explanation
that is very hard to unbelieve once stated.

*(Contrast §7.2.2, where admitting `C'`'s own keys collapsed the loss to 0.01 nats — a 99.8% effect.
That leak was catastrophic; this one is 2.3%. The difference is that `C'[0..j]` gives an induction head
a *prefix to match*, while a lone target key gives it nothing to match against.)*

### 16.2 What the same table does show: the rectangle is an easier task

`rect` 6.43 / 2.74 / 6.78 against `causal` 7.04 / 2.81 / 7.13. The rectangle is **easier by
0.07–0.61 nats** on the frozen model, and `rect_noself` sits in between, so the gain is not the leak —
it is simply that seeing the whole context beats seeing a prefix.

**This invalidates every loss-vs-loss comparison made in this document.** Arm B's 1.18 against arm C's
1.74 are not two scores on one task; cross-replay's loss is measured on a *strictly easier* problem.
§14 noted the loss anti-correlates with RULER and treated that as a curiosity; the reason is that the
two objectives' losses are not commensurable at all. Any future comparison has to be against a
matched-geometry reference (e.g. the frozen-model `rect` loss above as a per-task floor), not against
the other arm's number.

### 16.3 The mechanism: one score, N conflicting demands, no difficulty ramp

From §4's gradient `dL/ds_i = Σ_j A_ji ⟨dL/do_j, v_i − o_j⟩`, count what each geometry asks of the
router at `N = 8192`:

| | rectangle (cross-replay) | causal triangle (e2e) |
|---|---|---|
| queries demanding key `i` | all **8192** | only `j > i`, ~**4096** mean, ~1 for the last key |
| candidates a query competes over | all **8192**, every row | `j+1`, so **1 … 8192** |
| rows where the choice is free | none | every `j < topk` (keep everything) |
| difficulty over the sequence | flat, maximal from step 0 | **ramps** with `j` |

Two consequences, and they both push toward concentration:

1. **Averaging.** One `s_i` must satisfy 8192 heterogeneous demands simultaneously. §4 sells this as
   the feature ("triangle→rectangle: `s_i` is forced to compromise across many unknown queries, which
   is the eviction semantics wanted") — but a compromise across 8192 demands with a
   `Σ_gated exp(g) = B` constraint is solved most cheaply by putting mass on the few keys *every*
   query wants (sinks, high-norm tokens, globally salient spans) and starving the rest. That is
   precisely a low-participation gate, and it is what all three cross-replay arms converged to
   (0.007–0.062) regardless of capacity or `B`.
2. **No curriculum.** Under the causal triangle, early rows have fewer candidates than `topk`, so
   selection is free; discrimination is only required as `j` grows past `topk`. The objective therefore
   contains a built-in difficulty ramp, and the router learns to rank on easy rows before hard ones.
   The rectangle presents the hardest setting to every row from step 0.

The evidence for "objective sets concentration" is now as strong as this setup allows — **three arms
against one, spanning 26× capacity and 2048× budget**:

| arm | objective | `mid_dim` | `B` | participation | RULER @2048 |
|---|---|---|---|---|---|
| A | cross-replay | 0 | 1 | 0.0070 | 44.75 |
| D | cross-replay | 256 | 1 | 0.0136 | 48.20 |
| B | cross-replay | 256 | 2048 | 0.0622 | 20.43 |
| **C** | **e2e** | 256 | 1 | **0.1652** | **66.24** |

Every cross-replay arm is concentrated; the e2e arm is not. Even deflating C by §15.2's worst-case
5.2× denominator correction leaves ≥0.032, above every cross-replay arm.

### 16.4 Why KVzip and NextMem work, and why that does not transfer

Both are cited as evidence that repeat-reconstruction is a sound objective. Both differ from
cross-replay in the one respect that matters here.

**KVzip does not differentiate anything.** It runs the reconstruction forward, takes `max` attention
onto `KV(C)` as a per-key **label**, and evicts by that label (§3.2, Algorithm 1). There is no gate, no
budget constraint, and no gradient. So:

* the rectangle-vs-triangle question does not arise — a `max` over queries has no averaging problem,
  because `max` is not a compromise. Each key keeps its **best** query's demand, not the mean of 8192.
* Concentration is never trained; it is read off after the fact and thresholded at eval time.
* Its own shortcuts are free. §7.2.1 established that KVzip's forward *does* include `C'`→`C'`, which
  §7.2.2 showed would destroy a differentiated objective (loss → 0.01 nats). It costs KVzip nothing.

**That points at a concrete fix**: cross-replay averages over replay queries where KVzip takes a max.
`Σ_j A_ji` in §4's gradient is the averaging; a max-like reduction (or a top-fraction of the demands)
would keep a key that *some* query needs badly, which is exactly what a retrieval task requires and
what an average destroys. This is a change to how `dL/ds` is aggregated, not to the mask.

**NextMem makes reconstruction hard on purpose.** Its memory is a handful of latent vectors
(`NextMemQwen.batch_encode`), so nothing can be copied and reconstruction must genuinely compress. It
also uses a *learned* repeat marker (a separate embedding for `<|start_of_document|>`, initialised from
the word "repeat") and a two-stage curriculum ("progressive latent substitution", with a `wo_PS`
ablation, i.e. they measured that it matters). Cross-replay has **no marker** — `C'` is the raw tokens
at shifted positions, with nothing signalling "reconstruct" rather than "continue" — and **no
curriculum**: the gate is at full strength on the full rectangle from step 0.

### 16.5 What to do next, in order of expected value per GPU-hour

Ranked by (evidence behind it) × (cost). None of these is run yet; the first two are the ones the
measurements above actually argue for.

1. **Max-style demand aggregation instead of the mean.** The KVzip contrast (§16.4) and the averaging
   argument (§16.3) both point here, and it is the only proposal that addresses the measured
   concentration directly. Concretely: weight each query's contribution to `dL/ds_i` by how *sharply*
   that query needs key `i`, or aggregate the top-m demands rather than all `N`. Cheap to prototype as
   a reweighting inside the existing gradient path; no mask change, so the §0 guards are untouched.
2. **A causal or ramped replay mask** — i.e. move cross-replay's supervision shape toward the one that
   works. `rect_causal` above is exactly this and is already measurable on the frozen model. Note this
   makes cross-replay *more* like the e2e loss, which is a legitimate outcome: the interesting question
   becomes what remains of the objective's distinctiveness once the shape is matched, and whether the
   replay-text degree of freedom (`replay_ids` ≠ `input_ids`) buys anything the triangle cannot.
3. **A learned repeat-marker token** (NextMem's, one embedding row). Cheapest of all, and cross-replay
   is currently the only one of the three methods with no reconstruction signal at all. Low expected
   effect alone, but it is nearly free and removes a confound from every later comparison.
4. **Short, non-duplicate replay queries** — §7.2.2's surviving suggestion. Real questions over `C`
   with the loss on answer tokens only, which admits self-attention without admitting induction. The
   largest change, and it converges toward an ordinary QA-supervised router, so it should be costed as
   a new objective rather than a tweak.

⚠️ **What not to do.** Do not admit `C'`→`C'` (§7.2.2: loss collapses to 0.01 nats, supervision gone),
and do not set `B = topk` (§15.3: −27.8 points isolated). Both were proposed on plausible reasoning and
both are measured failures.

**The open question this leaves.** Cross-replay's stated advantage over the e2e loss was that the
rectangle forces `s_i` to compromise across many unknown queries, matching eviction semantics (§4). The
measurements say that compromise is realised as *over-concentration* and costs 12–18 RULER points. It
is not yet established whether the rectangle can be made to work with a better aggregation (item 1) or
whether the triangle is simply the right shape and cross-replay's remaining value is only the
`replay_ids` degree of freedom.

---

## 17. §16.5 items 1 and 2, implemented (not yet trained)

Both are **off by default**, so every existing arm's configuration is bit-identical to before. Defaults
verified by the suite (80 passing, up from 70).

### 17.1 `demand_reduce` — max-style demand aggregation (item 1)

`CrossReplayTrainer.demand_reduce ∈ {"sum", "max", "mean"}`, `--demand-reduce`. `"sum"` is the default
and is plain autograd accumulation, i.e. what every measured arm trained with.

**The granularity is the query chunk, not the query, and that is a real limitation.** §4's gradient is
`dL/ds_i = Σ_j A_ji ⟨dL/do_j, v_i − o_j⟩`, and the sum over `j` happens *inside* autograd — only the
total is observable at the score leaf. A true per-query max would need one backward per query
(`N = 8192` of them). What is affordable: `leaves[idx].grad` is harvested and zeroed after each query
chunk, so with `query_chunk=1024` at `N=8192` there are **8 demand groups** competing instead of 8192
queries averaging. Raises when there is only one chunk, since the reduction is then arithmetically
inert — a knob that looks configured and does nothing is the failure this file keeps recording.

**A sign convention that is easy to get backwards, and silent if you do.** The cotangent is `dL/ds`, so
a *negative* entry marks a key the loss wants raised. "The demand of the chunk that needs this key
most" is therefore `amin` over chunks, not `amax`. Taking `amax` would keep the chunk that most wants
the key **gone** — the opposite selection, and it would still train to a plausible loss curve.

`"max"` is rescaled by the chunk count so its gradient magnitude matches `"sum"`; without that it would
double as a learning-rate ablation and the comparison would be confounded. `"mean"` is the null control
(`sum / n_chunks`): same direction, smaller magnitude, which is what separates "the reduction changed
the direction" from "the reduction changed the effective LR".

⚠️ **`demand_reduce` does not reach `gate_scale`.** It is applied inside each chunk's own graph, so its
gradient never passes through the score leaves and keeps summing regardless of the reduction. Measured:
under `"mean"` every indexer *weight* gradient scales by exactly `1/n_chunks` while `gate_scale`'s
scales by `1.0`. Left in deliberately — `gate_scale` sets the gate's magnitude, not its ranking — but it
is why the mutation test compares gradients **per parameter**: pooling a `1.0`-scaled `gate_scale` with
`0.25`-scaled weights tilts the concatenation to cos 0.9981 and reads as a direction change that is not
one. Every individual tensor is 1.000000.

### 17.2 `lookahead` — the replay horizon (item 2)

`replay_horizon_mask(...)` generalises `rectangle_mask`; `CrossReplayTrainer.lookahead`,
`--lookahead`. Row `j` sees keys `≤ j + lookahead`. `None` (default) is the unbounded rectangle and
returns `rectangle_mask` itself, tag included — the `_kvpress_all_zero` tag is load-bearing (§5,
`_attention` drops the mask rather than adding it; a broadcast add measured 73.1 GiB at 8K).

* `0` — the causal triangle the e2e loss trains on. The target `C'[j+1]` then falls outside the visible
  set, matching e2e exactly.
* `m > 0` — a ramp: the candidate set still grows with `j`, so §16.3's difficulty curriculum is
  preserved while each row sees `m` keys beyond itself.

`query_offset` is the load-bearing detail: the replay is chunked, so chunk `[start, stop)` passes
`query_offset=start` to index the same key axis the unchunked pass would. Without it every chunk's
horizon restarts at 0 and later chunks see far less than intended — a silent change to the objective.
Mutation-tested: removing the offset fails 3 tests.

⚠️ **Cost.** A bounded horizon is a real mask, so `flex_fallback_reason` refuses it and attention lands
on SDPA MATH — the 46.7 GiB retention of §6.3. Any `lookahead` run needs a smaller `query_chunk`.
Expressing the horizon inside `score_mod` is the follow-up if the ablation earns it; that would keep the
flex path and is the difference between 48 MiB and 1288 MiB per layer.

### 17.3 What is verified, and what is not

**Verified:** defaults unchanged (`lookahead=None` is bit-equal to `rectangle_mask`, tag included);
horizon visibility and offset arithmetic; `lookahead=0` equals causal; both knobs actually change the
indexer gradient; `"mean"` is exactly `1/n_chunks` of `"sum"` per parameter; invalid values rejected at
construction; both recorded in the **checkpoint config** at both save sites, which §13 established is
the only place a confound can be caught. The three §0 guards are untouched (14 guard tests pass).

Mutation-tested, i.e. the tests fail when the feature is broken:

| mutation | result |
|---|---|
| `max` reduction silently falls back to `sum` | `test_demand_reduce_max_changes_the_indexer_gradient` **fails** |
| `lookahead` ignores `query_offset` | `test_replay_horizon_mask_visibility_and_offset` **fails** (all 3 params) |

**Not verified:** that either helps. No training run has used them. The next step is one run per knob at
the standard `8192:300,16384:300`, 600 steps, `MID_DIM=256 BUDGET=1` (arm D's configuration, so the
comparison is single-variable against D's 48.20 / 72.86), then RULER at `topk=2048` and `4096`. Note
`--demand-reduce max` needs `--query-chunk` ≤ N/2, and `--lookahead` needs a smaller `--query-chunk`
for the MATH fallback; both are departures from arm D's `query_chunk=1024`, so **the chunk size becomes
a second variable** unless arm D is re-run at the matching chunk size. That is worth pricing in before
reading either result as an objective comparison — it is the same trap as §13.

---

## 18. The trivial solutions of each geometry, and which one the rectangle actually reaches

The question this section answers: what are the **degenerate solutions** of the rectangle loss and the
triangle loss, and does either objective learn one? §0/§3 answer it for one case (the flat gate) and
that answer turns out to be incomplete.

### 18.1 The degrees of freedom, and the three candidate degeneracies

The gate is `g_j = s_j − LSE(s over gated) + log B` on gated keys and `0` on the `n_sink` pinned ones.
So `s` has exactly three degrees of freedom, and only one of them carries information:

| transform | effect on the gate | information |
|---|---|---|
| `s_j + c` | none — cancels in `LSE` | pure gauge |
| `s_j → a·s_j`, `a > 0` | changes concentration, not order | scale only |
| permute `s` across keys | changes the ranking | **the only content** |

Three candidate degenerate solutions follow:

**(1) Flat, `s_j = const`.** Participation → 1.0, no ranking. §0/§3's target. Worth noting *why*
pinning works: with sinks at multiplier 1, a flat gate suppresses all history to `B/n_gated = 1.2e-04`
at 16K, which is a large perturbation of the frozen model — so the LM loss actively resists it. Flat is
reachable but no longer *free*. (Unpinned, `LSE` is a per-row constant that cancels entirely and flat
**is** free — that is the hole pinning closes.)

**(2) Position-only, `s_j = f(j)`.** e.g. "always keep the most recent `m` keys". Participation is low,
which *looks* healthy. Stronger than flat, because recency genuinely predicts the next token, and
perfectly query-agnostic — one profile serves all `N` rectangle rows with zero conflict. **Critically,
`shuffle_delta` does not catch it**: permuting a recency profile destroys it, so the control reports a
large penalty and the run looks trained.

**(3) Content-salience-only, `s_i = g(h_i)`** — "globally interesting token", independent of any query.
Also low participation, also large `shuffle_delta`, and *also* content-driven, so it passes §14.2's
cross-document Jaccard test too.

### 18.2 The asymmetry between the two geometries

**The rectangle selects for query-agnostic solutions, and (2) and (3) are exactly that.** All `N` rows
share one `s_i`. A genuinely query-relevant score must compromise across `N` heterogeneous demands
under `Σ_gated exp(g) = B`; a salience or recency score satisfies all `N` rows *with no compromise at
all*. So the degenerate solutions are not merely reachable — they are the **path of least resistance**.

**The triangle does not.** Row `t` sees only `[0, t]`, so "recent" is a different key set for every
row and no single position profile serves them all. The candidate set also grows with `t`, so rows with
`t < topk` impose no constraint at all and the objective carries a built-in difficulty ramp (§16.3).

### 18.3 MEASURED: it is (3), not (1) or (2)

**(1) and (2) are ruled out** by §14.2's dissection, on real RULER text:

| arm | `corr(score, position)` | cross-document top-k Jaccard | chance |
|---|---|---|---|
| A | 0.0049 | 0.1522 | 0.1429 |
| B | −0.0284 | 0.1490 | 0.1429 |
| C | 0.0917 | 0.1531 | 0.1429 |

A position-only score would give cross-document Jaccard ≈ 1.0 (the same positions every time);
measured at the chance floor. Participation is nowhere near 1.0 either. So neither the flat nor the
position-only degeneracy was reached — the guards in §0/§3 are doing their job, and §14.2's negative
results are re-usable here.

**(3) is confirmed by a test built for it** (`proxy_exp_budget/cross_arm_agreement.py`). If the
cross-replay arms converge on a shared salience function while the e2e arm tracks query-relevance, they
must agree **with each other** more than with the e2e arm. All arms share the frozen backbone and are
scored on *identical* hidden states, so only the learned score differs. Pairwise top-k Jaccard, same
document / layer / head, `topk=2048`, `N=7500`, 3 documents from distinct tasks, 36 layers × 8 heads:

| pair | Jaccard | |
|---|---|---|
| A (mid0, B=1) — D (mid256, B=1) | **0.4604** | cross–cross |
| B (mid256, B=2048) — D | **0.3381** | cross–cross |
| A — B | **0.3106** | cross–cross |
| A — C (e2e) | 0.2244 | cross–e2e |
| C — D | 0.2187 | cross–e2e |
| B — C | 0.1714 | cross–e2e |
| **mean cross–cross** | **0.3697** | |
| **mean cross–e2e** | **0.2048** | |
| chance floor `topk/(2N−topk)` | 0.1581 | |

**Above-chance agreement is 4.5× higher within the cross-replay family than across the objective
boundary** (+0.2116 vs +0.0467). And it is not an outlier effect: the *lowest* cross–cross pair (0.3106)
still exceeds the *highest* cross–e2e pair (0.2244). Three arms spanning 26× capacity and 2048× budget
converge on substantially the same key set, and that set is not the e2e arm's.

A detail worth keeping: **A—D is the highest pair of all (0.4604)**, and those two differ only in
`mid_dim` (0 vs 256, i.e. 26× the parameters). Changing capacity moves the selected set *less* than
changing the objective does — the same ordering §15.3 found in RULER points (+3.45 for capacity against
+18.04 for the objective), now visible directly in the selections.

### 18.4 Why this explains the 18-point gap, and what it does not explain

Salience is exactly the failure RULER punishes: **"globally salient" ≠ "the key this query needs"**. It
keeps rare/high-norm/entity-like tokens, and a needle is unremarkable until something asks for it. That
matches §14.3's needle coverage being lower for cross-replay arms at every budget while their scores
were fully content-driven and well-ranked — *a salience score ranks the right kind of token, not the
right one*.

⚠️ **One measurement does not fit the tidy version of this story, and it should not be smoothed over.**
Arm A's needle coverage (0.4796) is *higher* than arm C's (0.4459) at `topk=2048`, yet A scores 44.75
against C's 66.24. So needle coverage alone does not order the arms, and "cross-replay covers fewer
needles" is too simple. The cross-arm agreement result stands on its own evidence; the mechanism linking
it to the RULER gap is supported but not established.

**What §18 does not claim.** It does not show the salience solution is *optimal* for the rectangle, only
that three independently trained cross-replay arms land near each other and away from e2e. A stronger
test would characterise *what* the shared set is (are they high-norm tokens? rare tokens? named
entities?) — that is a direct follow-up and would turn "salience" from a label into a measurement.

### 18.5 Consequence for the two §17 interventions

This reframes both, and it is why `demand_reduce` was the weaker idea:

* **`lookahead=0` attacks the actual mechanism.** It removes the query-agnostic pressure by making the
  candidate set differ per row, so no single salience profile can serve every row without conflict.
  §18.2 is the reason to expect it to help.
* **`demand_reduce="max"` does not.** It changes *how* demands combine but leaves one score serving all
  rows, so a salience solution still satisfies every demand group equally well — `amin` over chunks does
  not penalise a key that every chunk likes. It was killed at step 480 with five flat shuffle controls
  (0.016 / 0.181 / 0.036 / 0.098 / 0.091 against arm D's +3.51 at step 200) and participation *rising*
  to 0.97, i.e. it reached degeneracy **(1)**, the flat gate — the one failure mode the guards were
  built for, arrived at by starving the gradient rather than by flattening the score.

---

## 17.4 RESULT: both §16.5 interventions failed, and one of them refutes §18.2

Both trained at arm D's exact configuration (`MID_DIM=256 BUDGET=1`, `query_chunk=1024`,
`8192:300,16384:300`, 600 steps, seed 0), so each is single-variable against D.

| arm | intervention | RULER @2048 | @4096 |
|---|---|---|---|
| A | — (`mid_dim=0`) | 44.75 | 70.88 |
| **D** | **— (the baseline)** | **48.20** | **72.86** |
| `drmax` | `demand_reduce="max"` | *killed, not evaluated* | — |
| **E = `la0`** | **`lookahead=0`** | **40.46** | **66.57** |
| C | e2e LM loss | 66.24 | 85.30 |

**Neither closed any of the 18-point objective gap. `lookahead=0` made it worse — −7.7 @2048 and
−6.3 @4096 against D.** The gap to C widened from 18.0 to 25.8 points.

### 17.4.1 `demand_reduce="max"`: reached the flat-gate degeneracy, killed at step 520

Six shuffle controls, all flat: **0.016 / 0.181 / 0.036 / 0.098 / 0.091 / 0.094**, against arm D's
+3.51 at step 200 and +6.00 final. Participation *rose* to **0.96** (D fell to 0.014) and the loss never
improved after step 100 (~6.7–7.2 throughout, D reached 2.28). This is degeneracy **(1)** of §18.1 — the
flat gate — reached not by flattening the score but by **starving the gradient**: `amin` over 8–16 chunks
keeps one chunk's demand per key and discards the rest.

Not evaluated, deliberately: a gate whose permutation costs 0.09 nats has no ranking to measure.
`step400.pt` remains if it is ever wanted.

Two candidate mechanisms were flagged during the run and **remain unseparated**, which is a real gap:
information loss (`amin` discards most of the signal) versus step size (the `× n_chunks` rescale
multiplies a single chunk's demand by 8–16, possibly too large at this LR). Both predict what was
observed. Separating them is one short run each (`max` without the rescale; `max` at LR/10) and it
decides the fix — softer reduction versus keep `amin` and fix the scaling. §18.5 argues the idea was
misdirected regardless, since changing *how* demands combine leaves one score serving all rows.

### 17.4.2 `lookahead=0`: trained cleanly, scored worse, and did not move the selection

This one is the informative failure, because every training-side diagnostic said it was working:

* `shuffle_delta` rose monotonically **0.014 → 0.653 → 1.934 → 2.424 → 3.535 → 4.265**, tracking D's
  trajectory (+3.51 @200, +4.98 @400, +6.00 final). A real ranking was learned.
* participation fell 0.127 → 0.071 → 0.055 → 0.031 → **0.0215**, i.e. it converged into the same
  concentration band as A (0.0070) and D (0.0136).
* loss 4.09 — not comparable to D's 2.28, since §16.2 established the triangle is a strictly harder
  task. Correctly ignored.

**And yet it scored 40.46.** §14's lesson holds with full force: on this objective the training curves do
not order the arms, in either direction.

**The mechanistic test refutes §18.2's explanation.** §18.2 predicted `lookahead=0` removes the
rectangle's query-agnostic pressure, so its selected set should move *off* the cross-replay cluster and
toward e2e's. Measured (`proxy_exp_budget/cross_arm_agreement.py`, same protocol as §18.3):

| | mean top-k Jaccard | above chance |
|---|---|---|
| rectangle cluster, internal (A/B/D) | 0.3697 | +0.2116 |
| **`la0` vs the rectangle arms** | **0.3214** | **+0.1633** |
| `la0` vs e2e | 0.2156 | +0.0574 |
| rectangle arms vs e2e | 0.2048 | +0.0467 |
| chance | 0.1581 | — |

**`la0` still sits with the rectangle arms** (0.321, and its single closest neighbour of any arm is
D at 0.370) and barely closer to e2e than the rectangle arms already were (0.2156 vs 0.2048). Breaking
the mask shape moved the selection **hardly at all**.

So §18.2's asymmetry argument — that the rectangle uniquely selects for query-agnostic solutions because
no single position/salience profile can serve a triangle's varying candidate sets — is **wrong, or at
least not the operative mechanism**. The causal triangle produced essentially the same solution. What
distinguishes the e2e arm must therefore be something other than the visibility geometry.

⚠️ **§18.2 is retracted as an explanation; §18.3's measurement stands.** The cross-replay family really
does converge on a shared selection distinct from e2e's (4.5× above-chance agreement, every cross–cross
pair above every cross–e2e pair). What is now open is *why*, since it survives changing the mask to the
e2e loss's own shape.

### 17.4.3 What is left, given both failures

The three variables that differ between cross-replay and e2e have now been tested individually:

| variable | tested by | result |
|---|---|---|
| capacity (`mid_dim`) | A → D | +3.45 / +1.98. Nearly inert. |
| budget (`B`) | D → B | −27.8 / −8.7. Harmful; `B=1` is best. |
| supervision **shape** | D → E (`lookahead`) | **−7.7 / −6.3. Harmful, and does not move the selection.** |
| demand aggregation | D → `drmax` | Degenerate (flat gate). |

**Everything cheap has been tried and the 18-point gap is intact.** What remains untested is the part of
the objective that is *not* a knob: cross-replay's loss is on `C'` against `KV(C)` with `C' == C` — a
reconstruction task — whereas e2e's loss is the model's own next-token prediction on `C` with the gate
in-path. §7.2.2 already showed the reconstruction framing is fragile (admitting `C'`'s own keys collapses
it to a copy at 0.01 nats). The remaining hypothesis worth the GPU time is that **reconstruction of
already-visible text is simply a weaker training signal for retrieval than in-path next-token
prediction**, in which case the objective does not have a fix and the finding is that e2e wins.

The two follow-ups that would test that, rather than another knob:

1. **`replay_ids ≠ input_ids`** — replay a *different* document against `KV(C)`. This is the one degree of
   freedom cross-replay has that e2e does not, and it is already supported (`cross_replay_training_step`
   takes `replay_ids`). If the objective's value is real, this is where it lives.
2. **Short non-duplicate replay queries with the loss on answer tokens only** (§7.2.2's surviving
   suggestion) — which converges toward QA-supervised routing and should be costed as a new objective.

---

## 19. The cross-document control, promoted to a training condition

**What it tests.** Not "does this make cross-replay better" — it almost certainly does not. It tests
**whether the reconstruction relation is teaching the router anything at all**, which is the question
left open by §17.4 and §18.

The setup today: prefill `C`, replay the *same* tokens as `C'` against `KV(C)`, predict `C'[j+1]`.
The story is "reconstruct the context from its compressed KV". The control replaces `C'` with an
**unrelated document** — everything else identical. The next token of an unrelated document *cannot*
be predicted from `C`'s keys, so:

**If it collapses** (much worse than D, selection moves) — the reconstruction relation is
load-bearing, and the router really does learn "which keys reconstruct *this* text".

**If it trains to a comparable score and still clusters with A/B/D (§18.3)** — reconstruction was
never the teacher. The router is learning a document-independent salience, which would explain in one
stroke why 26× capacity (+3.4), 2048× budget (−27.8) and the causal mask shape (−7.7) all failed to
move it.

The second outcome is the one worth having, because it explains the pattern rather than adding to it.
§18.3's measurement — three arms spanning 26× capacity and 2048× budget converging on the same
selection, 4.5× above-chance agreement — is currently unexplained; §18.2's geometric explanation was
refuted by `lookahead=0` (§17.4.2). "The score does not depend on the replay text" would account for
all of it.

**Implementation** (`--cross-doc-replay`, `CROSS_DOC_REPLAY=1`, run tag `_xdoc`). The donor is the
**next batch from the same loader**, so it is real corpus text at identical length, identical subset and
identical distribution — the only thing that changes is whether `C'` is related to `C`. Noise or a
token shuffle would confound "unrelated" with "not natural text". `replay_ids` was already supported by
`cross_replay_training_step` (it was designed as an *eval-time* null); this promotes it to a training
condition and threads it through the driver, launcher and checkpoint config.

**Verified before launching**, since a control that is not actually a control would waste the run:
consecutive loader batches are genuinely different documents — position-wise token agreement **0.68%**,
distinct-token Jaccard 0.139 (i.e. only the common English vocabulary is shared). Not the same document,
not a shifted window of it.

Trained at arm D's exact configuration (`MID_DIM=256 BUDGET=1`, `query_chunk=1024`,
`8192:300,16384:300`, 600 steps, seed 0), so it is single-variable against D's 48.20 / 72.86.

**Early signal, recorded before the result so it cannot be reinterpreted after:** the loss starts at
**7.57** against D's 6.97 at step 0, which is the expected direction — an unrelated document is harder to
predict from `C`'s keys. What matters is not the loss level (§16.2: losses across geometries are not
commensurable) but whether the *router* still converges to the same selection. If it does, the loss
being harder while the score is unchanged is precisely the "reconstruction is not the teacher" result.

⚠️ **Interpretation limit, stated up front.** A null result here (xdoc ≈ D, same cluster) shows the
*replay text* does not determine the score. It does **not** show what the score does depend on. Naming
that — high-norm tokens? rare tokens? entities? — needs the follow-up §18.4 already flags: characterise
*what* the shared selected set is, rather than only that it is shared.

### 19.1 RESULT: reconstruction IS load-bearing — the hypothesis is refuted

The prediction in §19 was the null: that `xdoc` would score like arm D and stay in the rectangle
cluster, showing the replay text does not determine the score. **Both halves came out the other way.**

**RULER**, trained at arm D's exact configuration so this is single-variable:

| arm | replay text | @2048 | @4096 |
|---|---|---|---|
| **D** | `C'` = `C` (baseline) | **48.20** | **72.86** |
| **F = `xdoc`** | `C'` = unrelated document | **35.33** | **67.32** |
| | | **−12.9** | **−5.5** |

**Selection agreement** (`cross_arm_agreement.py`, same protocol as §18.3; chance 0.1581):

| | Jaccard | above chance |
|---|---|---|
| rectangle cluster, internal (A/B/D) | 0.3697 | +0.2116 |
| `la0` vs the rectangle arms | 0.3214 | +0.1633 |
| **`xdoc` vs the rectangle arms** | **0.2431** | **+0.0850** |
| `xdoc` vs e2e | 0.2066 | +0.0485 |
| rectangle arms vs e2e | 0.2048 | +0.0467 |

**`xdoc` left the cluster.** Its above-chance agreement with A/B/D fell to **40%** of what those arms
share with each other, and it now agrees with the rectangle arms (0.2431) barely more than with e2e
(0.2066). Contrast `la0`, which stayed at 77% of the cluster's internal agreement despite having its
mask geometry replaced.

**So the reconstruction relation is what the router learns from.** Removing it — while holding the
rectangle, the capacity, the budget, the schedule, the seed and the data distribution fixed — both
moves the selected keys and costs 12.9 RULER points. The score genuinely depends on *which text is
being reconstructed against the cache*, not on a document-independent salience.

⚠️ **§18.3's cluster finding therefore needs a different explanation, and §18.4's salience label is
withdrawn.** The three rectangle arms really do converge on a shared selection (4.5× above-chance),
and it is *not* because the score ignores the replay text — `xdoc` proves the score does not ignore it.
The remaining explanation is the mundane one: A, B and D all reconstruct **the same corpus** with the
same objective, so they learn the same *reconstruction-relevant* function. That is a much weaker claim
than "query-agnostic salience", and it removes the mechanism §18 offered for the 18-point gap. The gap
is again unexplained.

**A caveat on the training-side signals, recorded because I misread them during the run.** `xdoc`'s
diagnostics all looked healthy — shuffle_delta rose 0.011 → 0.664 → 1.005 → 1.803 → 1.436 → 2.248 and
participation fell to 0.0126, i.e. arm-D-like — and I read that mid-run as evidence for the null ("the
router trains normally on text it cannot reconstruct"). It was not. The router *did* learn a ranking;
it learned a **worse** one. The one signal that pointed the right way was the *magnitude*:
`xdoc`'s final shuffle_delta (+2.25 @500) against D's +6.00 and `la0`'s +4.27 — about a third of D's.
Reading "a ranking was learned" as "the same ranking was learned" was the error, and it is the same
class of mistake as §14.4's: a diagnostic that is directionally right and quantitatively ignored.

**Where this leaves the four interventions.** Every one has now failed, and §17.4.3's table extends:

| variable | change | @2048 | @4096 |
|---|---|---|---|
| capacity | A → D | +3.45 | +1.98 |
| budget | D → B | −27.77 | −8.73 |
| supervision shape | D → E (`lookahead=0`) | −7.72 | −6.29 |
| **reconstruction relation** | **D → F (`xdoc`)** | **−12.87** | **−5.54** |
| objective | D → C (e2e) | **+18.04** | **+12.44** |

Read together: **the cross-replay objective is not mis-tuned, it is at a local optimum.** Every
perturbation tried — more capacity aside, which buys almost nothing — makes it *worse*, and the one
change that helps is replacing the objective with the e2e LM loss. That is now four failed
interventions across capacity, budget, mask geometry and replay content, so the honest conclusion is
that **the 18-point gap is a property of reconstruction-against-a-cache as a training signal**, not of
any knob. §16.5's remaining item 4 (short non-duplicate queries with the loss on answer tokens only)
is a different objective rather than a fix to this one, and should be costed as such — including
against the simpler baseline of training e2e on RULER-like synthetic data.
