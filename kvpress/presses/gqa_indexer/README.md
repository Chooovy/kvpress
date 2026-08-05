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
`weights_proj` at all.

## Geometry

For Llama-3.1-8B (32 attention heads, 8 KV heads, `head_dim` 128) with `h=8`, `g=4`:

| | shape | note |
|---|---|---|
| `w_q` | `hidden -> (h*g) x head_dim` = 32 heads | one query per attention head |
| `w_k` | `hidden -> h x head_dim` = 8 heads | one key per KV head; cache cost is `h x head_dim`/token |
| output | `(B, 8, Sq, Sk)` | one score per KV head |

Queries come straight from `hidden_states`. MLA feeds its indexer the already-computed
`q_lora`, which is a free-reuse optimization, not a design requirement — GQA has no such
tensor, and adding a bottleneck purely for the indexer would cost parameters and lose
information.

## Pipeline

```
hidden_states
  -> w_q / w_k  (+ LayerNorm, + RoPE on the leading rope_dim channels)
  -> logits        (B, h, g, Sq, Sk)   fp32
  -> activation    relu | softplus | leaky_relu | none
  -> group_reduce  (B, h, Sq, Sk)      weights_proj | sum | mean | amax
  -> + causal/padding mask
  -> reduce_queries (B, h, Sk)         mean | max | last | recency
  -> [optional] chunk pooling          mean | max
  -> sink/local protection
  -> ScorerPress topk + gather (per KV head)
```

Chunk aggregation deliberately runs **after** token-level scoring, so the indexer stays a
pure token scorer and chunking remains a swappable policy.

`group_reduce` note: with per-head `topk`, a per-head scalar weight cannot change any
ranking (`topk(w_h * s) == topk(s)` for `w_h > 0`), so `weights_proj` is only
load-bearing for the *within-group* reduction — which is exactly where it is applied here.

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

Written for clarity, not throughput: the full `(B, h, Sq, Sk)` logits and dense attention
target are materialized, which is fine for warmup-scale sequences but needs a
fused/chunked kernel for long context. The seams for that are `indexer_layer_loss` and the
target builders.

Prefill-time compression only — `score` raises if the cache is longer than the scored
hidden states. `GQAIndexer.forward` already accepts separate `key_hidden_states` for a
decode-time path; caching indexer keys across steps is not wired up yet.
