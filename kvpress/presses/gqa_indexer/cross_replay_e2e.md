# Cross-replay LM loss for a query-independent indexer: design notes

Working notes for the `[C ; C']` cross-replay objective sketched in
`query_independent_indexer_cross_replay.md`, written while auditing whether it can be built on the
existing e2e path (`e2e_trainer.py`, `gated_attention.py`, `gate_pin.py`).

**Provenance of the numbers.** Every figure below was produced on a **CPU box with no GPU**, on a
randomly-initialised 2–3 layer Qwen3 (`hidden_size=64`, `head_dim=16`) unless stated otherwise.
They are therefore **mechanism and invariant checks only** — "does this tensor flow", "are these two
layouts equal", "does this constant cancel". They are **not** performance evidence and do not
predict anything about Qwen3-8B. Claims that were not verified are labelled as such.

Following `ROUTER_LEARNABILITY.md`, corrections are kept visible rather than edited away. Three
conclusions reached during this session were **wrong and are retracted below** (§2, §7.1, §7.2); one
of them had already been used to make a recommendation.

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
        + NO budget constant                 (retracted recommendation; see §2)
        + dense scope
        + chunk the replay queries           (exact, O(chunk) memory; see §6.1)
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

## 2. RETRACTED: "you need an explicit `log K` budget constant"

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

**Decision: keep dense scope + sink pin, no budget constant.**

### 2.4 The reusable lesson

Total gated mass was treated as a proxy for learnability. It is not: **the only deliverable is a
ranking**, and this constant is invisible to `TopK`. For any scalar introduced on the gate, ask first
whether it survives into the final `TopK`.

*(A first version of the K-invariance probe re-drew `randn` on every call and so compared different
samples — it printed "diff ~1e-9" while actually outputting 4.6e-02, a self-contradiction that went
unnoticed for one turn. This is the paired-vs-unpaired error `proxy_exp/HANDOFF.md` §9.4 warns
about, reproduced inside this session.)*

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
