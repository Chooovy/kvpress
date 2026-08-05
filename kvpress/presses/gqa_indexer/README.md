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
    get_attention_modules, indexer_state_dict,
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
)
loss.backward(); optimizer.step()
torch.save(indexer_state_dict(model), "indexer.pt")
```

Note `compute_indexer_loss` scores layer `i` from `hidden_states[i]` (its **input**), not
`hidden_states[i + 1]`, which is what the attention layer actually consumes.

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

Two loss implementations, same objective up to a constant:

| | `train.indexer_layer_loss` | `fused_loss.fused_indexer_loss` |
|---|---|---|
| objective | full KL | cross-entropy (`KL + H(pbar)`) |
| gradients | identical | identical |
| memory | `O(L²)` | **`O(L·h)`** |
| teacher | `output_attentions=True` (forces eager) | logits recomputed per tile from Q/K + `lse` |
| passes | autograd | 1 fwd (loss + `dQ`), 1 transposed (`dK`) |

Use the dense one as the readable reference and for exact-KL numbers; use the fused one for
anything long. Both are exercised against each other in
`tests/presses/test_gqa_indexer_fused_loss.py`.

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

