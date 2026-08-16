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

Two objectives ship. They are alternatives, not stages of one recipe:

- **Distillation** (`FusedIndexerTrainer`) — match the frozen model's own attention weights.
  The score never enters the forward pass.
- **End-to-end** (`E2EIndexerTrainer`) — add the score inside the attention softmax so the LM
  loss trains it directly. See [End-to-end training](#end-to-end-training-gated-attention).

Both expose a full-scope stage and a top-k-scope stage under the same `stage` names, so they
can be compared at matched budget.

### Distillation

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

## End-to-end training (gated attention)

> **Design analysis:** [`ROUTER_LEARNABILITY.md`](ROUTER_LEARNABILITY.md) works out *why* a gated
> router is learnable at all — the no-op loophole, what normalization and pinning each contribute,
> why DMA/SparseK/STE never face the problem, and HSA's two-level decomposition. It also records
> the current implementation's known gap (see "Known gap" below) and two retracted conclusions.
> Read it before changing the gate form.

Distillation's supervision is a surrogate: it teaches the indexer to rank keys by *where the
dense model attends*, which is not the same as *which keys the prediction needs* under a fixed
budget. SAS measures the gap directly — their distillation baseline covers **more** attention
mass yet scores **lower** downstream (96.8% mass / 79.5% acc against 79.5% / 79.5% at K=64), so
`mean_recall` is a biased objective, not merely a noisy one.

`E2EIndexerTrainer` removes the surrogate by putting the score inside the softmax:

```
out = softmax(scale · q·kᵀ + gate_scale · qi·kiᵀ) V
```

No teacher, no KL, no second forward pass — the loss is the model's own, and `dL/dscore` comes
from the ordinary attention backward.

```python
from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import E2EIndexerTrainer, e2e_indexer_training_step

press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)   # gate_scale=True is required
trainer = E2EIndexerTrainer(press=press, stage="dense")           # full scope
press.post_init_from_model(model)
optimizer = torch.optim.AdamW(trainer.indexer_parameters(model), lr=1e-3)

loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
loss.backward(); optimizer.step()

# Then, at the eviction budget used at eval:
trainer = E2EIndexerTrainer(press=press, stage="sparse", keep_ratio=1 - press.compression_ratio)
```

### Inside the softmax, not outside

`softmax(q·kᵀ) · g` looks equivalent and is not. Writing `p` for the ungated probability and `p̃`
for the gated one:

```
outer:  dg_m = Σ_{i∈m} p_i · doᵀ v_i
inner:  dg_m = Σ_{i∈m} (p̃_i / g_m) · doᵀ (v_i − out)
```

Only the inner form carries `(v_i − out)` — a *relative* signal saying whether key `i` pulls the
output somewhere better than where it already is, which is what lets the gate reallocate
attention mass. The outer form scales a fixed probability and can never reorder one key against
another. SAS ablates both; outer is markedly worse (42.0 vs 55.6 on MATH500).

### The no-op hole, and why pinning closes it

Adding the **same** number to every key of a row cancels in the softmax. So a gate that is *flat
along the key axis* does nothing, and the model falls back to the frozen dense backbone — which is
already strong. The router reaches that point at zero cost (`qi = 0`, or `gate_scale → 0`),
satisfying the LM loss having learned **no ranking at all**.

Every failure row of SAS's Table 1 is this one hole, reached differently:

| gate | how it flattens | distance to dense | SAS 1-epoch |
|---|---|---|---|
| `s` (raw) | `s = 0` | **0.0** | 18.8 |
| `log sigmoid(s)` | `s → +∞` (saturates) | 2e-16 | 17.0 |
| `log softmax(s)`, no pin | `s = const` | 2e-16 | — |
| `log softmax(s)` **+ pin** | **unreachable** | 1.2 | **54.4** |

(dense baseline 56.1)

Two conditions are needed **together**:

1. **Normalize** over the key axis, fixing the total multiplier the gated keys may spend at 1.
2. **Exempt some keys** from that normalizer, pinning their gate to 1 (log-space 0).

With both, a flat gate is arithmetically impossible: pinned keys sit at multiplier 1, and matching
them on all `N` gated keys would need the gated multipliers to sum to `N`, not 1. The router can
only choose **which** keys get the fixed budget — and that choice is the ranking.

Normalizing *without* pinning is **inert, not merely weaker**: `logsumexp(s)` is one constant per
row, so it cancels too, and `log softmax(s)` is *exactly* interchangeable with raw `s` (forward and
`d/ds`, to 1e-15; the mechanism is `Σ_k dS_k = 0`). The pin is what makes the normalizer
load-bearing — it is not something SAS added unnecessarily. Both facts are asserted by
`test_flat_gate_variants_all_reach_the_no_op` and `test_pin_closes_the_no_op_hole`, the latter
checking that a pin blocks *both* routes (flat `qi` **and** `gate_scale → 0`).

`pin_mode="none"` keeps the un-pinned behaviour as an ablation baseline, and warns.

### Which pins need a kernel

The only reason `self` and `sink` take different code paths.

| pin | query-dependent? | folds into concat? | cost |
|---|---|---|---|
| `sink` (first `n_sink` keys) | no | **yes**, 1 extra dim | one SDPA call |
| `self` (each query's diagonal) | **yes** | **no** | second attention path |

Folding needs the indexer key zeroed at pinned positions. `sink` pins the same keys for every
query, so a static `K` expresses it and the rank-one `−LSE` term rides in one extra dimension
(verified 1.1e-15). `self` pins a *different* column per row, which no shared `K` can represent —
the naive attempt is wrong by 2.5, not by rounding.

So `self` takes a two-branch route: history-only attention (which *does* fold — inside that branch
`−LSE` is a per-row constant and cancels, so no extra dimension is needed) plus the pinned keys,
merged by their log-sum-exps. The `−LSE` re-enters only in the merge weight, which is exactly where
the budget acts. Exact to 6.7e-16, checked against both the shared reference and a separately
written two-branch reference.

**Current limitation:** the `self` path builds explicit logits, so it is `O(Sq·Sk)` in memory. SDPA
returns no log-sum-exp and recovering one would cost a third pass, so `self` is the
correctness / short-sequence path today; a fused kernel (gate applied per tile, single pass) is what
would make it viable at 32K. **`sink` needs none of this** — it stays one SDPA call at any length.

Both pins need the history `logsumexp`. It streams over key tiles **and recomputes them in the
backward pass**, so retention is `O(L)` per layer — measured 1.27 GiB across 36 layers at
`L=8192`, against ~259 GiB for the first implementation, which OOM'd on step 1. The extra cost is
one recomputed `Di`-wide GEMM per tile, not memory.

> The first version was a plain loop of differentiable tile ops. That bounds the *forward* peak,
> which is what its docstring claimed, but autograd retained every tile's intermediates until
> backward: 3.6× the full score matrix, and **worse as the tile shrank** — the knob meant to save
> memory made it worse. The fix uses the closed-form gradient `d(lse)/d(s_k) = softmax(s)_k`, so
> only `lse` needs saving. `test_history_lse_retains_only_o_l` and
> `test_history_lse_retention_does_not_grow_as_the_tile_shrinks` are the regression tests whose
> absence let it ship; both fail on the old implementation.

A row whose only visible
key is pinned (query 0 under `self`) has *no* history; its `logsumexp` would be `−inf` and the gate
`+inf`, so those rows get an inert gate instead of a NaN that would spread through the model.

### Pinning does not apply to the sparse scope

Under `stage="sparse"` the forward is already restricted to the router's own top-k, so a flat gate
does **not** recover dense attention — there is no no-op to block. Same structural reason DMA (hard
`-inf` on unselected keys) and SparseK (`Σp = k` solved, not learned) need no pin: their training
forward is sparse, so "dense" was never a fallback.

The combination is **rejected**, not ignored. `pin_mode` left unset resolves to `"self"` for the
dense stage and `"none"` for the sparse one, so `stage="sparse"` needs no second flag.

One untested risk remains: the scarcity is `log(1/N_history)`, which deepens from −1.8 nats at 6
history keys to −10.4 at 32K. Whether that over-suppresses history at long context is not measured.
See [`ROUTER_LEARNABILITY.md`](ROUTER_LEARNABILITY.md) §9.

### The concat identity — why token-wise scoring is not the hard part

The gate is *bilinear*, and that is what makes token-level full scope affordable:

```
scale·(q·k) + λ·(qi·ki)  ≡  scale·([q, (λ/scale)·qi] · [k, ki])
```

So gated attention **is** ordinary attention at `head_dim = D + Di`, and SDPA computes it with
the gradient reaching `qi`/`ki` for free. Verified to ~1e-15 across shapes by
`test_concat_matches_explicit`.

What decides whether folding is available is **not** block-vs-token granularity. A *block* gate
folds too, provided it is bilinear — broadcast the pooled block key back to its tokens on the key
side and the identity holds unchanged (verified, 2e-16). Two other properties decide it:

1. **The score must be bilinear.** `qi·ki` folds. A score with a nonlinearity or a projection
   *after* the Q–K interaction does not, and has to be materialized.
2. **The gate must be uniform over keys.** SAS pins its self-block gate to 1; that alone breaks
   the fold (verified: 6e-1 error), for the same reason it revives the normalizer below.

This indexer satisfies both, so no gate table and no kernel. SAS satisfies neither, so it needs
one.

Granularity *does* decide how expensive that table is once you're forced to build it. At
`L=32K, H=4, fp32`:

| gate table | size |
|---|---|
| `(query, block)`, B=128 | 0.12 GiB |
| `(query, block)`, B=64 | 0.25 GiB |
| `(query, token)` | **16 GiB** |

So a materialize-the-gate design is affordable at block granularity and not at token
granularity — which is why, had this indexer needed a table, token-wise scoring would have been
the harder case, not the easier one. Folding sidesteps the question entirely.

**V must be padded to the concatenated width.** Flash requires `Q.size(-1) == V.size(-1)`, and the
concat widens Q/K to `D + Di` while V stays at `Dv`. That mismatch makes SDPA fall back to the
**math** backend, which materializes the `(B, H, Sq, Sk)` attention weights *and retains them for
backward* — 4.0 GiB per layer at `L=8192, Hq=32` in bf16, so **144 GiB across 36 layers**.
`pad_value_to_width` widens V and slices the output back, which is exact (~1e-15 in forward and
every gradient) and restores `O(L)` retention. Measured growth per doubling of `L`: **~4× unpadded,
~2× padded**.

> This shipped as an OOM. An earlier version of this note claimed `Dqk=256 / Dv=128` merely
> "steers SDPA to its memory-efficient backend — still `O(n)` memory, somewhat slower". That was
> wrong in the way that mattered: the check performed was whether SDPA *accepted* the shapes, not
> which backend it *chose*. SDPA returns correct numbers on the math backend, so nothing but the
> memory reveals it. `test_value_is_padded_so_flash_stays_eligible` and
> `test_padding_v_makes_retention_linear_not_quadratic` are the guards; the latter runs on CPU,
> since torch ships a fused CPU kernel and the backend choice reproduces there.

Cost of the pad: 2× the V bandwidth and 2× the `P @ V` GEMM at `Di == D`, plus roughly 2× what
flash retains (11.3 GiB vs 5.7 GiB across 36 layers at `L=8192`) — all `O(L)`. A fused Triton kernel
(two `tl.dot`s per tile, `Dv` at its true width) would avoid the padding entirely, and is the
remaining reason to write one.

### Why `gate_scale` starts where it does

`IndexerNorm` leaves q/k at unit variance per channel, so the raw score has std `~√head_dim`:
measured **11.4** against **1.0** for a real `q·k/√head_dim` attention logit at `head_dim=128`.
Added unscaled the gate would swamp the attention it is meant to modulate. `gate_scale` is a
learnable per-layer scalar initialized to `head_dim**-0.5`, which brings it to std ≈ 1 — and when
the indexer's `head_dim` equals the model's (the default), that is exactly the attention softmax
scale, so the concatenated form needs no rescale at all.

**It must not be initialized to 0.** Zero is tempting — training would start from exactly the
frozen dense model, unperturbed — but `dL/dscore ∝ gate_scale`, so zero is a fixed point the run
never leaves. Pinned by `test_zero_gate_scale_severs_the_router_gradient`. The price of having a
gradient is a std-1 perturbation of the logits at step 0.

Being learnable also makes it a diagnostic: `trainer.gate_scales` per layer says how hard each
layer leans on its router. One that collapses toward 0 is a router not earning its place, which no
loss curve would tell you.

### Full vs sparse scope

Under **full** scope every key is gated, so every key's score gets a content-dependent gradient.
Under **sparse** scope an unselected key contributes nothing to the output, so it has no gradient
of its own and moves only through the softmax normalizer — the whole unselected set is dragged
together rather than judged individually. SAS shows this as a perfect line (`R² = 1.00`) through
the unselected points in their Figure 5; `test_full_scope_gradients_are_independent` asserts the
stronger discrete form: the sparse-scope gradient at unselected slots is *identically zero*.

That is what full scope buys for its `O(n²)`, and it is the largest single effect in SAS's
ablation (47.4 → 55.6).

### The train/inference mismatch is deliberate

Under `stage="dense"`, training gates all keys while inference hard-selects top-k. That looks like
a bug and is the design. SAS tests the consistent alternative (STE, hard top-k in the forward) and
it is consistently **worse** — 61.30 → 51.48 on AIME25 at budget 4096 — because a hard forward
collapses the score into a selected/unselected bit and discards the ranking information near the
top-k boundary. `stage="sparse"` is the consistent variant, available for exactly that comparison.

### What is trained

Only the indexers; `freeze_backbone` puts every other parameter at `requires_grad=False`, matching
SAS and matching distillation — otherwise the comparison would confound "end-to-end gradient" with
"more trainable parameters". The gradient still *flows through* the frozen backbone to reach the
router, which is the point.

`gate_scale=False` (the default) keeps the parameter out of the state dict entirely, so
distillation checkpoints stay byte-compatible with what they were before this feature existed.

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

**At inference it is turned.** `SparseAttentionContext` defaults to `precision="tf32"`, because
`"ieee"` fp32 does not use tensor cores at all — it falls to FMA, which makes `M` (`BLOCK_G`,
padded to 16 wherever `min_dot_m()` requires it) real work rather than lanes the MMA would have
occupied regardless. Measured on an H20 at `L=8192, topk=2048`: **67.0 s per prefill under
`"ieee"` against 9.4 s under `"tf32"`**, with `M` scaling the kernel ~linearly under `"ieee"`
(1.89× for 2× `M`) and barely at all under `"tf32"` (1.09×). At 650 RULER samples that is the
difference between ~1 h and ~12 h.

For a bf16/fp16 model this costs *nothing*: tf32 keeps 10 mantissa bits against bf16's 8, so
every upcast operand is exactly representable and `Q @ K^T` is bit-identical. Only the softmax
weights in `P @ V` are genuinely fp32, and rounding them costs ~2e-4 relative — ~30× below the
bf16 epsilon the output is stored at. Both paths measure the same 7.52e-3 against the fp32
reference, i.e. bf16 store-rounding alone. `"ieee"` remains right for an fp32 caller and for the
tests, which is why the kernel's own default is unchanged.

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


## Sparse attention at inference (GQA DSA)

Everything above trains the indexer, and `GQAIndexerPress` spends it on **eviction**: keys
leave the cache, so every later query sees the same reduced cache. `sparse_attention.py` and
`triton_sparse_attention.py` do the other thing the indexer enables, and the thing DSA
actually ships — keep the cache whole and let **each query attend to its own top-k keys**.
Nothing is discarded, so a key one query ignored is still there for the next one; the saving
is in attention FLOPs and score-matrix bandwidth rather than in cache residency.

That is why this needs a kernel at all. Eviction ends with a smaller *dense* cache, which
ordinary dense attention handles unchanged. Per-query selection is a **gather** — every
`(query, KV head)` row reads a different `topk` slice — and no dense kernel expresses it.

```python
from kvpress.presses.gqa_indexer import streaming_topk_support, sparse_gqa_attention

# indices: (B, Hkv, Sq, topk) int32, ascending, -1 in empty slots
indices, _ = streaming_topk_support(q_idx, k_idx, topk=512, force_sink=4, force_local=64)
out, lse = sparse_gqa_attention(q, k, v, indices)   # (B, H, Sq, Dv), (B, H, Sq)
```

The index convention is exactly what `sort_support` already emits, so the selector feeds the
kernel with no adapter. `-1` (Megatron/sglang's choice) rather than `Sk`-as-sentinel
(tilelang's) because it is checkable without knowing `Sk` and cannot be mistaken for a real
position.

#### Selection, not attention, is the cost at length

Measured per prefill on an H20 (`topk=2048`, bf16, `precision="tf32"`):

| L | select | attend |
|---|---|---|
| 8K | 6.75 s | 1.68 s |
| 16K | **26.67 s** (3.95× — `O(L²)`) | 3.35 s (2.00× — `O(L)`) |

The Triton kernel is exactly linear in `L` at fixed `topk`, so it stops being the bottleneck
almost immediately: at 16K selection is **89%** of the prefill. `streaming_topk_support` is
`O(L²)` by nature — every query scores every key — but most of what it *spent* was avoidable.

The running buffer is re-sorted against each key tile, so total work is
`Sq · Sk · (1 + take/key_tile)`. Note that **`query_tile` cancels**: it costs scratch, never
work. A fixed `key_tile = 512` against `take = 1980` therefore carried a 4.87× redundancy
factor. `topk_tiles()` now sizes `key_tile` at ~2× `take` (~1.5× redundancy) and pays for the
scratch by lowering `query_tile`, which is free:

| `key_tile` | redundancy | select @16K |
|---|---|---|
| 512 (old fixed default) | 4.87× | 26.67 s |
| 2048 | 1.97× | 9.22 s |
| **4096 (adaptive default)** | **1.48×** | **6.54 s** (4.1×) |

The measured 4.1× beats the 3.3× the work model predicts, because fewer and larger calls also
amortize per-call overhead. Past 4096 the curve flattens (8192 buys a predicted 1.2× more) while
scratch keeps growing, so that is the knee rather than the extreme. Tiling is result-invariant —
verified identical support across every `(key_tile, query_tile)` combination — which is what
licenses tuning it for speed.

Beyond this, the remaining `O(L²)` is intrinsic to scoring every key from every query; removing
it needs a blocked/hierarchical selector, not a better tiling.

| | batched | varlen |
|---|---|---|
| entry point | `triton_sparse_gqa_attention` | `triton_sparse_gqa_attention_varlen` |
| `q` | `(B, H, Sq, D)` | `(total_q, H, D)` |
| `k`, `v` | `(B, Hkv, Sk, D)` | `(total_k, Hkv, D)` |
| `indices` | `(B, Hkv, Sq, topk)` | `(total_q, Hkv, topk)`, **sequence-local** |
| metadata | — | `cu_seqlens_q`, `cu_seqlens_k` |

Varlen indices are sequence-local (slot `j` means row `cu_seqlens_k[s] + j`), which keeps the
indexer's output usable unchanged and lets a sequence move in the buffer without rewriting its
indices. No page tables, no block tables — deliberately simpler than sglang. Mixed
prefill+decode batches and empty sequences both work; each sequence is bottom-right aligned
within itself. Both layouts share one kernel body, so there is one thing to be correct rather
than two that agree — and each packed sequence is verified **bit-identical** to the batched
kernel run on that sequence alone.

### Why the tiles are shaped this way

MLA's sparse kernels (tilelang `sparse_mla_fwd`, FlashMLA) put 64–128 heads in the GEMM's `M`
dimension, because one shared latent cache means *all* heads share one index list and so one
gathered tile. Under GQA the selection is per KV head, so only `group_size = H // Hkv` query
heads — typically 4–8 — share a list. That cannot be widened by also tiling over query tokens,
the usual fix for a short `M`: adjacent queries hold *different* index lists, so they cannot
share a gathered tile. It is intrinsic to per-query selection under GQA, not an artifact of
this implementation. The QK GEMM is therefore bandwidth-bound on the gather, which is the
regime that makes the approach worthwhile — but a configuration where `topk` approaches `Sk`
will lose to a dense kernel.

### `tl.dot` shape floors

Triton's per-backend `min_dot_size` gives the `(M, N, K)` lower bounds, and it is **not** stable
across releases: Triton 3.4+ on NVIDIA reports `(1, 1, 16)`, constraining only the contraction
dim, while Triton 3.3 reports `(16, 16, 16)`. Two consequences, and guessing wrong either way
costs something:

- `M` is `BLOCK_G`, the group's query heads, and `group_size` is 4 or 8 on real GQA models —
  below 16 either way. So the floor is *asked of the backend* at runtime by `min_dot_m()`:
  hardcoding 1 fails to compile on Triton 3.3 (`Input shapes should have M >= 16`), and
  hardcoding 16 quadruples the `[BLOCK_G, DV]` fp32 accumulator on every version that does not
  need it. When there is no driver to ask, it errs high — a floor that is too large only wastes
  padded lanes, one that is too small does not compile. Padding is correctness-neutral because
  the extra lanes load `q` under `g_valid` and are masked out of both stores.
- `block_k` **must** be ≥ 16, since `P @ V` contracts over it. This is validated eagerly
  because `TRITON_INTERPRET=1` does *not* apply the floor: `block_k=8` runs green on CPU and
  then fails to compile on the first GPU. `test_dot_shapes_are_legal_on_hardware` restates the
  rule over every tile shape the launcher computes, for both `M` floors, which is what keeps an
  interpreter-only test run honest about hardware.


### Correctness

Two independent torch references: one gathers `topk` keys, one scatters a dense `(Sq, Sk)`
mask. They share no index arithmetic — the part easiest to get subtly wrong — so their
agreement is evidence about the *operation*, not about one implementation of it. With
`topk == Sk` both reduce to `scaled_dot_product_attention` to 3e-7, which pins the scale, the
softmax and the bottom-right causal alignment at once. The kernel matches the gather reference
to ~5e-7 in fp32 across group sizes 1–16, non-power-of-two head dims, and `Dv != D`.

Empty rows are defined as `out = 0`, `lse = -inf` rather than `0/0 = NaN`: `-inf` is the honest
log of an empty sum and is what a downstream combine needs in order to give the row zero
weight. Rows can genuinely be empty — a short causal row plus padding produces one — and a NaN
there would propagate through the whole model instead of staying local. Malformed indices
(out of range, wrong negative, duplicated, past the diagonal) are masked, never dereferenced;
every gather address is clamped first.

**Not yet wired into a press.** These are the op and its kernel, tested standalone. Using them
end-to-end also needs an attention-module hook that calls the indexer, selects, and dispatches
here instead of to SDPA — plus a decode path that caches indexer keys across steps, which the
eviction press does not need either (see *Status*).


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

**It must not retain the upcast.** The obvious form, `layer_norm(x.to(fp32), …)`, makes autograd
save the *widened* tensor — so a bf16 input is held at fp32 width for the whole backward. On the
pairwise indexer that is invisible: both norms sit after `w_k`/`w_q`, so the tensor is `head_dim`
wide. On `ScalarIndexer` it is fatal: `in_norm` is the **first** op, so it runs at full
`hidden_size`, and the extra fp32 copy is 256 MiB per layer at `L=16384` — **4.5 GiB over 36
layers**, which is why `sp=8` at 16K OOM'd on the scalar arm at a length the pairwise arm trains at
comfortably. (Attention is not sharded under FFN-SP, so the indexer always sees the full `L`.)

So the statistics are taken from an fp32 view without giving that view to autograd: `_Fp32LayerNorm`
saves `x` in its own dtype plus the two `(…, 1)` statistics and recomputes the rest in its backward.
Forward arithmetic is unchanged, and both are equally far from an fp64 reference (forward 1.560e-2,
`d/dx` 1.558e-2, `d/dweight` 2.224e-1 — identical to three digits), so this is a pure memory win,
not a precision trade. Measured retained bytes for `project_k` at `L=4096`, bf16:

| | before | after |
|---|---|---|
| scalar, `mid_dim=0` | 96.1 MiB | **32.1 MiB** |
| scalar, `mid_dim=256` | 106.1 MiB | **38.0 MiB** |
| scalar, `mid_dim=1152` | 141.1 MiB | **59.0 MiB** |
| pairwise | 34.0 MiB | 32.0 MiB |

The residual scalar-over-pairwise gap is the `mid_dim` chain, which is intrinsic to having an MLP.

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
