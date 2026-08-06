# GQA Lightning Indexer

A learned KV-cache scorer for grouped-query attention, adapted from DeepSeek's lightning
indexer (DSA). Produces **one importance score per KV head per token**, so each KV head
evicts independently.

## Why per-KV-head

DSA collapses its indexer heads into a single selection with a learned `weights_proj`
pooling. That is forced by MLA, not chosen: MLA keeps one shared latent KV cache, so
"keep this token for head 3 but not head 7" is physically meaningless, and per-head
selection would also destroy the latent-reuse that makes the sparse-MLA kernel fast.

GQA has `num_key_value_heads` physically independent caches. Per-KV-head eviction is
therefore free capacity, and kvpress's `ScorerPress.compress` already does per-head
`topk` + `gather`, so no custom compress path is needed.

MiniMax M3 — the one production GQA indexer among the references — confirms the shape:
`index_n_heads == num_key_value_heads`, one independent selection per group, and **no**
activation and **no** `weights_proj`.

## Geometry

For Llama-3.1-8B (32 attention heads, 8 KV heads, `head_dim` 128):

| | shape | note |
|---|---|---|
| `w_q` | `hidden -> n_heads x head_dim` = 8 heads | one query head per KV head |
| `w_k` | `hidden -> head_dim` = **1 head (MQA)** | shared across heads; cache cost is `head_dim`/token |
| output | `(B, 8, Sq, Sk)` | one score per KV head |

`n_heads` defaults to `num_key_value_heads`, **not** `num_attention_heads`. Mirroring the
full query-head count costs ~4× the parameters and score-GEMM FLOPs while still producing
only `num_key_value_heads` usable scores:

| | params/layer | GEMM heads | indexer k-cache |
|---|---|---|---|
| 32 q heads, 8 k heads | 21.0M | 32 | 2048 B/token |
| **8 q heads, 1 k head** | **4.7M** | **8** | **256 B/token** |

Queries come straight from `hidden_states`. MLA feeds its indexer the already-computed
`q_lora`, which is free reuse rather than a design requirement — GQA has no such tensor,
and adding a bottleneck purely for the indexer would cost parameters and lose information.

## No activation, no cross-head reduction, no per-head weights

DSA's ReLU and `weights_proj` exist to make a **cross-head sum** well-behaved: ReLU keeps
per-head contributions non-negative so they cannot cancel, and `weights_proj` learns how to
weight them. With one score per KV head there is no sum, and both become inert or harmful:

- **An activation cannot change a per-head top-k.** Top-k is invariant to strictly
  increasing maps, so `softplus`/`exp` are pure overhead.
- **ReLU is worse than inert.** It is not strictly increasing — it flattens every negative
  score to exactly 0. At moderate compression the keep boundary falls *inside* that
  negative region, so ReLU decides part of the selection by arbitrary tie-break. It also
  caps how strongly the student can reject a key (floor at `exp(0)`), halves the gradient
  paths, and forces the backward pass to recompute `z` for the `1[z>0]` mask.
- **A per-head scalar weight cannot reorder a row.** It is constant along the key axis:
  `topk(w·s) == topk(s)` for `w > 0`, and reverses the ranking for `w < 0`. No-op or bug.

This is also why the three reference implementations disagree: DeepSeek-V3.2 sums heads and
needs ReLU; GLM5 and M3 do not sum and do not use it.

`tests/presses/test_gqa_indexer_press.py` pins each of these arguments so they are not
silently re-added.

## Pipeline

```
hidden_states
  -> w_q (n_heads) / w_k (1, MQA)  + LayerNorm  + RoPE on the leading rope_dim channels
  -> scores          (B, h, Sq, Sk)   fp32, einsum('bhqd,bkd->bhqk')
  -> + causal/padding mask
  -> reduce_queries  (B, h, Sk)       mean | max | last | recency
  -> [optional] chunk pooling         mean | max
  -> sink/local protection
  -> ScorerPress topk + gather (per KV head)
```

Chunk aggregation deliberately runs **after** token-level scoring, so the indexer stays a
pure token scorer and chunking remains a swappable policy.

## Training

Two stages, both training only the indexer (backbone frozen so the teacher is fixed):

- **Stage 1, `stage="dense"`** — teacher is the true attention grouped per KV head;
  student is `softmax(indexer_logits)` over all valid keys. Teaches *where to look*.
- **Stage 2, `stage="sparse"`** — both sides restricted to the indexer's own top-k support
  and renormalized there. Sharpens ranking *within* the kept set. Mirrors DSA's
  `sparse_loss=True`.

The KL helpers in `loss.py` mirror Megatron's `dsa_indexer_loss.py` / `dsa_masking.py`
(`normalize_indexer_target`, `masked_log_softmax`, `indexer_kl_per_row`).

Set `keep_ratio = 1 - compression_ratio` so stage 2 trains at the eviction budget used at
eval.

```python
from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    IndexerTrainConfig, compute_indexer_loss, freeze_all_but_indexer,
    get_attention_modules, get_input_layernorms, indexer_state_dict,
)

press = GQAIndexerPress(compression_ratio=0.5)
press.post_init_from_model(model)
params = freeze_all_but_indexer(model)
optimizer = torch.optim.AdamW(params, lr=1e-4)

out = model(input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)
loss, per_layer = compute_indexer_loss(
    press, get_attention_modules(model), out.hidden_states, out.attentions,
    IndexerTrainConfig(stage="dense"),
    attention_mask=attention_mask, labels=labels,
    model=model,                     # derives the layernorms and RoPE tables; see below
)
loss.backward(); optimizer.step()
torch.save(indexer_state_dict(model), "indexer.pt")
```

**The student must be the one the press actually runs.** Three ways to get this wrong, none of
which fails loudly — the loss falls, gradients flow, and eval is just quietly worse:

1. `compute_indexer_loss` scores layer `i` from `hidden_states[i]` (its **input**), not
   `hidden_states[i + 1]` — the latter would hand the indexer the layer's own output.
2. The decoder block applies `input_layernorm` *before* calling `self_attn`, and kvpress hooks
   `self_attn`. So at inference the indexer sees the **post-layernorm** tensor, while
   `output_hidden_states[i]` is the **pre**-layernorm one.
3. The indexer is RoPE-aware unless `rope_dim=0`, and `output_hidden_states` carries no
   `position_embeddings` — so a caller who omits them trains a **NoPE** student.

Passing `model=` derives (2) and (3) automatically. Omitting the layernorms warns; omitting the
position embeddings now **raises** rather than silently dropping the positional signal, since at
inference the press is hooked onto `self_attn` and always has them.

`FusedIndexerTrainer` has none of these problems — it hooks `self_attn` itself, so it reads
exactly what the press reads. That is why the fused and dense losses differ by more than `H(p̄)`
whenever the dense path's student drifts, which is what
`test_agrees_with_the_dense_loss_up_to_the_entropy_offset` detects.

The teacher is grouped **per KV group**, not averaged over all heads — matching MiniMax M3
Eq. 9. Averaging across groups would give every indexer head an identical target and waste
the per-head capacity that motivates the whole design.

## Correctness notes

**RoPE table narrowing** is the subtle part. HF builds `cos = cat([freqs, freqs], -1)`, so
a width-`W` table holds `W/2` frequencies twice. `rotate_half` pairs channel `j` with
`j + r/2`, and that pair must be driven by `f[j]`. The narrowed table therefore must be
the first `r/2` entries of *each half* (`slice_rope_tables`):

- a contiguous prefix `cos[..., :r]` gives `[f[0..r-1]]`, rotating the two halves of each
  pair by *different* angles;
- striding `cos[..., ::2]` samples every other frequency.

Both are silently wrong — no crash, just a degraded position signal. `test_rope_*` pins
this against an independently computed ground truth, plus norm preservation and
relative-position invariance.

**Causal masking** is mandatory and applied inside `indexer_logits`, which both the press
and the training code call. Without it, queries score future keys and those scores leak
into the query reduction, while the training target is causally masked — student and
teacher would disagree on a growing fraction of entries.

**`masked_log_softmax` convention** (inherited from Megatron): invalid entries are set to
`0.0` *in log space*, not `-inf`. Log-probs over valid entries are correct, but `exp()`
reads `1.0` at masked slots. Always pass the same `valid_mask` when consuming it —
`indexer_kl_per_row` does.

## Status

Three loss implementations:

| | `train.indexer_layer_loss` | `fused_loss.fused_indexer_loss` | `fused_sparse_loss` |
|---|---|---|---|
| stage | 1 and 2 | 1 | 2 |
| objective | full KL | cross-entropy (`KL + H(p̄)`) | **full KL** |
| memory | `O(L²)` | `O(L·h)` | `O(L·h)` |
| compute | `O(L²)` | `O(L²)` | **`O(L·topk)`** |
| teacher | `output_attentions=True` (forces eager) | recomputed per tile from Q/K + `lse` | recomputed at the support |
| passes | autograd | 1 fwd (loss + `dQ`), 1 transposed (`dK`) | + 1 `no_grad` selection pass |

Use the dense one as the readable reference and for exact-KL numbers; use the fused ones for
anything long. All three are exercised against each other — with `topk == k_len` the sparse
path reproduces the dense stage-1 KL exactly, which is a strong end-to-end check since the
two share no code.

Prefill-time compression only — `score` raises if the cache is longer than the scored
hidden states. `GQAIndexer.forward` already accepts separate `key_hidden_states` for a
decode-time path; caching indexer keys across steps is not wired up yet.

### Tiled loss

The fused path exists because the dense one caps out around a few thousand tokens: at
L=32K a single layer's `(B, h, Sq, Sk)` fp32 logits are 32 GiB. Streaming key tiles brings
persistent state down to `3·H·L + 2·h·L` floats — 14 MiB at L=32K, 56 MiB at L=128K.

Three things make one pass possible, and all three are consequences of the simplified
indexer:

1. `I[j,t,s]` depends only on that `(t,s)` pair (no activation, no cross-head reduction), so
   `Σ_s pbar·I` is **linear** in the teacher probabilities and accumulates exactly like
   FlashAttention's `Σ_s p·V`.
2. `p_bar = exp(alpha − lse)` is exact, so the teacher needs no running state at all — just
   its `lse`, which flash-attention already computes.
3. `dloss/dI = phat − pbar` depends only on its own row, so the forward pass accumulates a
   unit-weight `dQ` and backward just scales it.

`dK` still needs a second, transposed pass: `dK[s] = Σ_j Σ_t (phat−pbar)·q[j,t]` sums over
*queries* while the forward streams *keys*, and its student half carries `1/ell[j,t]`, final
only after every tile. One key tile's contribution mixes many per-query normalizers, so no
scalar can fix it up afterwards. This is a layout constraint, not an activation artifact —
StreamKL splits the same way.

**The `lse`/mask contract is the sharp edge.** `teacher_lse` must be computed under the same
mask the loss applies. Masking `p_bar` after the fact does *not* work: the rows stop summing
to one (the masked mass is simply gone), which quietly down-weights exactly the rows with
the most padding. flash-attention's `lse` covers causal masking only, so
`assert_lse_mask_compatible` **raises** when the batch also has padding, and
`teacher_lse_from_qk` is the fallback that folds any mask in before the logsumexp.

```python
from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    FusedIndexerTrainer, freeze_all_but_indexer, fused_indexer_training_step,
    indexer_state_dict,
)

press = GQAIndexerPress(compression_ratio=0.5)
press.post_init_from_model(model)
optimizer = torch.optim.AdamW(freeze_all_but_indexer(model), lr=1e-4)

trainer = FusedIndexerTrainer(press=press, key_tile=512)
loss, per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)
loss.backward(); optimizer.step()
torch.save(indexer_state_dict(model), "indexer.pt")
```

`FusedIndexerTrainer` computes each layer's loss **inside** that layer's forward, via a
hook, and keeps only the resulting scalar. That is not an optimization detail — storing
every layer's teacher Q/K would cost 10 GiB at L=32K and 40 GiB at L=128K on
Llama-3.1-8B, which would undo the whole point of the tiled loss. One layer's teacher
tensors are live at a time.

The teacher needs no extra forward pass: post-RoPE keys are already in the KV cache when
the hook fires, and queries are rebuilt from `hidden_states` with the layer's own `q_proj`
and the same `position_embeddings` it just used. So `use_cache=True` is required, and
`output_attentions` is deliberately never set — the base model keeps its fast kernel.

`key_tile` and `query_tile` trade peak memory against launch overhead; the result is
bit-comparable across every combination (tested at 1–128 on both axes).

**Both axes must be tiled.** With key tiling alone each tile is `(B, H, Sq, key_tile)`,
which grows with sequence length — 14 GiB at L=128K. That is the same `O(L·tile)` trap
FlashKL's own reference implementation falls into. Tiling queries too makes the per-tile
footprint `O(query_tile · key_tile)`:

| L | key-only | both axes |
|---|---|---|
| 4K | 0.44 GiB | **0.055 GiB** |
| 32K | 3.5 GiB | **0.055 GiB** |
| 128K | 14 GiB | **0.055 GiB** |

Query tiles are independent for the loss and `dQ` (each query row owns its own running
max/sumexp/accumulator), so that loop is parallelizable. `dK` is the exception — it sums
over queries, so every query tile accumulates into it.

### Free teacher lse

`teacher_lse_from_qk` recomputes the teacher's logsumexp, costing a second `H·L²·d` pass.
`capture_teacher_lse` avoids that by taking the value flash-attention already computes:

```python
from kvpress.presses.gqa_indexer import capture_teacher_lse

with capture_teacher_lse(model) as lse_by_layer:
    model(input_ids, use_cache=True)
```

Verified against the flash-attention source: `softmax_lse` is `(batch, nheads, seqlen)`,
`causal=True` aligns bottom-right (matching `build_indexer_mask`'s default
`query_offset = k_len - q_len`), and query head `i` reads KV head `i // group_size`.

Only valid for causal-only masking — flash-attention's `lse` does not know about padding,
and `exp(alpha - lse)` would then not sum to one over the kept keys.
`assert_lse_mask_compatible` raises on that combination rather than letting it train
slightly wrong.

**fp16/bf16 only.** flash-attention has no fp32 kernel, and `capture_teacher_lse` refuses rather
than casting: `flash_attn_func` returns the attention *output* as well as the `lse`, so a cast
would change the model's own forward, not just the value being captured. A reduced-precision
`lse` paired with a wider `alpha` would also leave the teacher rows not summing to one — the same
class of quiet mis-normalization as the padding case. Load the model in bf16, or use
`teacher_lse_from_qk`, which is exact at any dtype.

## Stage 2: sparse

Stage 1 is `O(L)` in *memory* but still `O(L²)` in *compute* — every key has to be visited
to normalize the softmax. Stage 2 restricts the objective to each query row's own top-`topk`
support, which turns the teacher recompute into `H·L·topk·d`. This is AngelPTM's central
stage-2 optimization: its `lighting_indexer` returns `(topk_scores, topk_indices)` and the
teacher is only ever recovered *at those positions*.

| `L` | `topk` | dense | sparse | |
|---|---|---|---|---|
| 32K | 512 | 8.80 TFLOP | 0.14 | **64×** |
| 128K | 512 | 140.74 TFLOP | 0.55 | **256×** |
| 128K | 2048 | 140.74 TFLOP | 2.20 | **64×** |

```python
trainer = FusedIndexerTrainer(
    press=press, stage="sparse", keep_ratio=1 - press.compression_ratio,
    query_tile=512, topk_tile=512, force_local=64,
)
loss, per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)
print(trainer.mean_recall())   # teacher mass the support captured
```

Two passes. Pass 1 (`sparse_support.py`) runs under `no_grad` and emits only int64 indices,
so it adds nothing to the autograd graph — 8 MiB at `L=32K, topk=512, h=8`. Pass 2
recomputes the indexer's logits at the support *with* gradients. Top-k is not
differentiable, so treating the support as a constant is not an approximation; it is the
only option, and it is what DSA and AngelPTM do.

The support selection streams as a tournament merge against each key tile, so no `(Sq, Sk)`
score matrix is ever built. `force_sink`/`force_local` reserve per-row slots (MSA's
always-selected local block) by **excluding** those positions from the top-k pool and
concatenating them back, rather than by biasing their logits — exclusion is magnitude-free,
so no large logit can defeat it and no sentinel can overflow. These are per-query-row, unlike
the press's `n_sink`/`n_local`, which protect keys globally after the query axis is reduced.

### Full KL, not cross-entropy

Stage 1 optimizes CE because `KL = CE − H(p̄)` and a *fixed* teacher makes the entropy a
constant with identical gradients. **Stage 2 must not take that shortcut.** Its teacher is
restricted to the student's own support, so `H(p̄)` moves as the support moves — measured
drift of 1.166 → 1.232 nats across supports on one fixed teacher. A CE curve would mix
objective progress with support churn. The entropy is also cheap here: the support is `topk`
wide, not `L` wide, so `Σ p log p` streams as easily as the cross term.

That makes `KL ≥ 0` an available assertion, which stage 1 cannot make. It is the check that
catches a mis-derived normalizer: a missing `log Z` shows up immediately as a negative KL.

### `teacher_mode`

Two defensible teachers, and they are genuinely different objectives — measured max
elementwise gap 0.238:

| | normalizes over | needs dense `lse` | `Z` |
|---|---|---|---|
| `global` (default) | full key axis, then renormalized on the support | yes | teacher recall |
| `support` | the support only, per head | **no** | `1` identically |

They coincide only when every head in a group captures the same support mass; measured
per-head mass within one group spread from 0.005 to 0.995, so not in practice. `global` is
the default because it keeps the teacher fixed across steps, so the loss curve means the
same thing at step 1 and step 10000. `support` matches sparse-MLA (whose `lse` is by
construction over the selected keys) and makes stage 2 `O(L·topk)` end to end.

Both run through one code path: the mode only chooses which `lse` feeds `exp(α − lse)`. In
support mode `Z == 1` identically (verified to 2.2e-16), so the renormalization is a no-op
rather than a special case.

### `mean_recall`

`global` mode's normalizer `Z` is the teacher probability mass the support actually
captured, so it comes for free and is exposed via `trainer.mean_recall()`. Worth watching:
a low value means `topk` is too small for how spread out the teacher really is, and **the
loss alone will not say so** — it can look healthy while the objective ignores most of the
teacher's mass.

### Accumulators

Five per-row scalars, streamed over `(query_tile, topk_tile)` blocks:

| | |
|---|---|
| `m, ell` | student online-softmax max and sumexp |
| `Z` | teacher mass on the support |
| `A` | `Σ p̄_raw log p̄_raw` — the entropy term |
| `C` | `Σ p̄_raw · I` — the cross term |

then `KL = A/Z − log Z − C/Z + lse`. Every accumulator is linear in the teacher weights,
which is exactly what lets the row sum `Z` be divided out *after* the fact instead of being
needed up front — the same property that makes stage 1's CE streamable.

As in stage 1, `dQ` accumulates during the forward pass (the row-wise gradient `q̂ − p̄` is
separable, so backward only scales it) and `dK` needs a second transposed pass. Here `dK`
scatters with `index_add` rather than a dense einsum, since each query row touches only its
own `topk` keys.



## Triton kernels (stage 1)

`fused_loss.py` is already `O(L)` in memory, but every tile round-trips through HBM: the
student logits, `exp_logits`, the teacher's `alpha` and `p_bar` are each a separate
`(B, h, query_tile, key_tile)` tensor written and read back. `triton_fused_loss.py` keeps all
of that in registers and shared memory.

```python
trainer = FusedIndexerTrainer(press=press, backend="auto")    # kernels when possible
trainer = FusedIndexerTrainer(press=press, backend="triton")  # force; raises if it can't run
trainer = FusedIndexerTrainer(press=press, backend="torch")   # reference path
print(trainer.backend_used)   # which one actually ran
```

`backend="triton"` deliberately **raises** instead of falling back. A silent fallback would
let a benchmark measure the PyTorch path while reporting a kernel number — the failure mode
that makes performance work untrustworthy. `auto` falls back and logs at debug level; it also
declines under `TRITON_INTERPRET=1`, where the kernels are correct but far slower than
PyTorch, so choosing them would be a pessimization dressed as an optimization.

Two wins here are structural, not just bandwidth:

**No dense mask.** The PyTorch path takes an additive `(B, 1, Sq, Sk)` mask — itself `O(L²)`,
64 GiB of fp32 at `L=128K`, dwarfing everything the tiling saved. The kernels derive causality
from `query_offset` arithmetic and take padding as a `(B, Sk)` keep vector. `decompose_mask`
splits a mask into `(causal, keep)` when it can, and reports failure when it cannot:

| mask | decomposes | note |
|---|---|---|
| causal | ✅ | `keep=None`, no load at all |
| causal + padding | ✅ | keep vector recovered exactly |
| causal + sink skip | ✅ | masking the first N keys *is* per-key |
| sliding window | ❌ | per-row, cannot factor |
| arbitrary per-pair bias | ❌ | |
| built with a different `query_offset` | ❌ | |

Rejection is the *correct* outcome, not a limitation: a wrong decomposition would train
against a mask the student never sees, with nothing downstream to catch it. Verified by an
exhaustive random sweep — every accepted mask rebuilds exactly from `(causal, keep)`, and every
non-decomposable one is rejected.

Pass `bsz` to `decompose_mask` whenever the result feeds a kernel. `build_indexer_mask` emits
batch 1 when there is no `attention_mask`, and the kernel indexes `KEEP` with raw pointer
arithmetic that **cannot broadcast** — a `(1, Sk)` vector is read out of bounds for every batch
after the first, which shows up as batch 0 being perfectly correct and the rest being garbage.
`triton_indexer_ce_rows` rejects a short keep batch rather than trusting the caller.

**Causal early exit.** A query block starting at `m` sees keys only up to
`m + BLOCK_M − 1 + query_offset`, so the key loop stops there rather than running to `Sk`.
That halves the work on a square causal problem. Verified never to skip a needed key across
every shape × block-size combination, including `Sq > Sk` (where a leading block's bound goes
negative and the loop correctly runs zero times).

`dK` parallelizes over **key** blocks so each program's output range is disjoint — no atomics,
even though the reduction runs over queries.

`tl.dot` is pinned to `input_precision="ieee"`, so results match the PyTorch path to fp32
rounding rather than TF32's ~1e-3. That costs throughput on Ampere+; it is deliberate for a
reference kernel whose job is to be trusted, and it is the first knob to turn once it is.

fp64 is declined rather than demoted — `tl.dot` has no fp64 path, and quietly dropping to fp32
would break the gradient tests that rely on fp64 to reach 1e-10.

### Dead rows diverge, by design

A query row with no visible key (padding + causality can produce one) gets a **different**
per-row value from the two paths. PyTorch *adds* a finite `MASK_NEG = -1e4`, so the row's
`lse` lands near `−9997`; the kernels use a true `−inf` and clamp, landing near `−23`.

Both are finite — the property that matters, since a NaN would poison every gradient in the
batch — and both are meaningless. The difference is unobservable because both loss functions
weight rows by `row_valid`, making `d(loss)/d(row)` exactly zero there: scalar losses and all
gradients agree (verified) even though the raw rows do not. **Compare raw `rows` between the
two implementations only on live rows.**

Stage 2 is torch-only for now; `backend="triton"` with `stage="sparse"` raises rather than
pretending. Fusing it needs top-k inside the kernel, which is a different problem — the
selection has to be complete before the loss pass can start.


## Memory optimizations

Three changes, all bit-exact (verified to 0.0 error), measured against the pre-optimization
prediction on an H20 (95 GiB, 15.6 GiB weights, B=1, bf16):

| config | before | after | max L before | max L after | |
|---|---|---|---|---|---|
| dense | 1518 KiB/tok | 1086 | 54.9K | **76.7K** | 1.4× |
| sparse topk=512 | 2814 KiB/tok | 1662 | 27.7K | **48.8K** | 1.8× |
| sparse topk=2048 | 6702 KiB/tok | 3390 | 11.6K | **23.9K** | 2.1× |

**Group-broadcast instead of `repeat_interleave`.** `make_recompute_teacher` used to expand
KV-head keys to `H` heads with `repeat_interleave`, which materializes a real `group_size`-fold
copy — and because the autograd graph holds the closure until `backward()`, that copy lived for
*every layer at once*. It now views the query as `(B, h, g, Sq, d)` and lets the einsum broadcast
over the group axis: same arithmetic, `group_size`× less memory. 432 KiB/token on Qwen3-8B.

The same change applies to stage 2's gather, where it matters more: a KV group shares one
support, so the teacher gathers `(B, h, dq, tk, D)` instead of `(B, H, ...)`. That is the largest
*transient* in stage 2, so the tile scratch drops 4× as well (5.0 → 2.5 GiB at 512×512×128).

**`support` stored as int32.** It only has to address `Sk`, and int32 reaches 2.1e9 — far past
any real sequence. `gather`/`index_add_` require int64, so the cast happens per *tile*
(`O(query_tile · topk_tile)` transient) rather than on the resident `(B, h, L, topk)` tensor.
`sort_support` raises rather than wrapping if `k_len` ever exceeds int32.

**`valid` no longer saved for backward.** It is exactly `support >= 0`, so retaining it stored a
full-size bool for zero information — 576 KiB/token at `topk=2048`. Backward recomputes it per
tile.

**No fp32 teacher copy.** `make_recompute_teacher` used to do `.to(float32)` on entry, so
`ctx.teacher_alpha` retained an fp32 copy of the whole teacher **per layer** — 720 KiB/token
across 36 layers, the largest single term. The upcast now happens per *tile*, so what the closure
holds is the caller's own bf16 tensors. For the keys that means a reference to the **KV cache**,
which is resident anyway — the copy disappears rather than merely shrinking.

This is bit-identical, not approximately equal: bf16 and fp16 have no more mantissa bits and no
more exponent bits than fp32, so widening is lossless, and `.to()` is elementwise so it commutes
with slicing. `test_per_tile_upcast_is_bit_identical` asserts `torch.equal`, not a tolerance.

| | before | after |
|---|---|---|
| teacher Q | 576 KiB/tok (fp32) | **288** (bf16) |
| teacher K | 144 KiB/tok (fp32) | **0** (aliases the cache) |

### Cumulative

| config | original | after group-broadcast + int32 | after teacher retention | max L then → now |
|---|---|---|---|---|
| dense | 1518 KiB/tok | 1086 | **654** | 54.8K → **127.4K** (2.3×) |
| sparse topk=512 | 2814 KiB/tok | 1662 | **1230** | 27.7K → **66.0K** (2.4×) |
| sparse topk=2048 | 6702 KiB/tok | 3390 | **2958** | 11.6K → **27.4K** (2.4×) |

### Still on the table

`teacher_q` (288 KiB/token, 44% of the dense total) could go to zero: it is derivable from
`hidden_states`, which the student already retains for its own `dQ`. Recomputing it in backward
costs one `q_proj` GEMM plus RoPE per tile and would give another ~1.8×.
`predict_bytes_per_token(retain_teacher=False)` prices it. The trade is compute for memory, so it
is worth doing only once the length is the binding constraint rather than the step time.


## Precision

Every reference implementation keeps the **score** in fp32 while storing weights and activations
in bf16. That is not incidental — the score is the quantity the objective is defined on.

| | score GEMM | norm | KL / softmax | storage |
|---|---|---|---|---|
| Megatron DSA | fp32 (`q.float()`) | fp32 (`dsa_indexer_k_norm_fp32`) | fp32 | bf16 |
| AngelPTM | fp32 | — | fp32 | bf16 |
| tilelang DSA kernel | `accum_dtype="float"` | — | `accum_dtype` | `bfloat16` |
| FlashKL | fp32 accumulate | — | fp32 | input dtype |
| MiniMax M3 | fp32 (`idx_q.float()`) | fp32, built in | — | bf16 |
| **this module** | fp32 | fp32 (`IndexerNorm`) | fp32 | bf16 |

`IndexerNorm` exists because `nn.LayerNorm` on a bf16 module reduces in bf16, and a mean/variance
over `head_dim` channels with 8 significant bits carries ~7e-2 median relative error (1e-1 worst
case, measured over 200 draws). That lands on `q`/`k` and the `head_dim`-long dot product then
amplifies it, so the score would inherit the error before its own fp32 GEMM begins. M3 builds this
in unconditionally; Megatron exposes it as a flag. Parameter names and shapes match
`nn.LayerNorm`, so checkpoints load unchanged.

### Quantization

DeepSeek-V4 does quantize the indexer — §5.2.1 is titled *FP4 Quantization-Aware Training*, and
the paper states it applies "during the post-training stage … for MoE expert weights and the
indexer QK path". Two things are worth separating before copying it:

- It is **QAT for a deployment format**, run as a post-training stage, not the precision the
  indexer's KL distillation is performed in. The paper does not describe the distillation
  precision at all.
- The serving path (`sglang/srt/.../dsv4/indexer.py`, which contains no `backward`/`autograd`)
  earns FP4 with three mechanisms, not by casting: a **Hadamard rotation** before quantization
  (`fused_q_indexer_rope_hadamard_fp4_quant`), **per-block scales** (`head_dim_with_sf = 68`
  = 64 packed bytes + one fp32 scale), and **fp32 accumulation and output** regardless.

Measured on synthetic activations with channel outliers (L=4096, d=128), MXFP4 scores against an
fp32 reference:

| | relative error | top-512 recall | top-2048 recall |
|---|---|---|---|
| no rotation | 0.114 | 93.6% | 97.2% |
| + Hadamard | 0.063 | **97.1%** | 98.8% |

At inference the score is an **argsort key** — a few percent of recall is absorbable. As a **KL
target** the same error is a systematic bias in the function being learned, and it does not
average out across steps. So: fp32 for distillation; quantization is a separate, later,
inference-side project that needs the rotation and block scales to be worth attempting.
