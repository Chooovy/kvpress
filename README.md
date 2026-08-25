## IndexMem++

Learned KV routing for Qwen3-8B: a small per-layer *indexer* scores KV entries so the model can
either attend sparsely (keep the whole cache, pick top-k per query) or **evict** (free the KV
outright). Built on [kvpress](https://github.com/NVIDIA/kvpress).

Everything below is measured on RULER at `topk=2048`, `fraction=0.100`, Qwen3-8B
(`evaluation/results_*`). Dense baseline: **94.66** @4K, **93.69** @8K, **93.22** @16K.

### Where we are

**1. Query-dependent indexer, sparse attention — works.**
Score `q_i · k_j` per (query, key). Trained two ways:

| arm | RULER @8K | @16K | results dir |
|---|---|---|---|
| distillation from teacher attention | — | 92.10 | `results_sparse` (93.51 @8K, different host) |
| **end-to-end from the LM loss** | **87.06** | **79.04** | `results_sparse_e2e` |
| dense reference | 93.69 | 93.22 | `results_dense` |

The point of the e2e arm is not that it beats distillation here — it does not — but that
**training the router straight from the LM loss is viable at all**, without an auxiliary
per-key target. That matters because a distillation label collapses the sum over queries before
the set-level decision is visible (see `proxy_exp/HANDOFF.md` §1: the LM-loss gradient on a gate
is `Σ_i A_ij ⟨∂L/∂o_i, v_j − o_i⟩`, which carries value geometry *and* is implicitly conditioned
on what the other keys are doing). Retrieval SFT should close some of the remaining gap —
**TODO, deliberately last**, since it is a data change and everything else should be settled first.

**2. But query-dependent scores cannot evict.** Key `j`'s score changes with every new query, so
a freed entry may be needed again (verified: a frozen scalar score has **0** re-entries into the
top-k; a query-dependent one does not have that property). To evict, you must hand the indexer a
*proxy query* at eviction time — and simple proxies (mean of past queries, last query) score
poorly. Feeding it expected-attention-style queries is a possible fix but **low priority**: it
patches the symptom, not the mismatch.

**3. So: query-independent indexer.** Score `s_j = f(h_j)` from the key's own hidden state only.
The decisive property is complexity, not accuracy: **query-independent is `O(L)` — one score per
token at creation, then frozen** — whereas a query-dependent score must revisit past keys for
every new query. Current implementation is `ScalarIndexer`
(`kvpress/presses/gqa_indexer/scalar_indexer.py`), an MLP over `h_j` emitting one score **per KV
head** (`w_out: (n_kv_heads, mid_dim)`; per-head is the default and costs +0.17% params).

Related work, split by how the score is *trained* rather than how it is shaped:

| | score form | training signal |
|---|---|---|
| SparseK, DMA | linear / MLP on `h` | end-to-end |
| KVzip | attention during context reconstruction | — (inference-time, 2–3× prefill) |
| Fast-KVzip | `q_i · k_i` + sink margin | **distills KVzip's score** |
| this repo (`scalar`) | MLP on `h` | end-to-end LM loss |

Scored as sparse attention, the scalar arm is much worse than pairwise
(`results_sparse_scalar`): **66.24 @8K, 49.56 @16K** — i.e. −20.8 / −29.5. That comparison is
pairwise's home turf (whole cache retained, per-query top-k) and does not score the one thing
scalar uniquely buys, so it is a lower bound on the approach, not a verdict on it.

### The current thesis: the bottleneck is the loss, not the architecture

Proxy experiments on the real model (`proxy_exp/`, summarized in `HANDOFF.md`) point one way, and
**the architecture is not it**:

- **Function class is saturated.** A 4k-parameter linear score matches a 1M-parameter MLP on
  eviction damage (0.0360 vs 0.0356 — a gap ~0.04× the measurement noise floor). Bilinear
  (Fast-KVzip's `q_i·k_i` form) ties it too. So *KVzip's MLP vs Fast-KVzip's sink-attention shape
  is not the interesting axis.*
- **The query distribution is the largest lever measured (2.2×).** The achievable ceiling for a
  query-independent score moves by that much depending on what the future queries are — and it is
  worst on exactly the query-agnostic surrogates eviction papers use.
- Two attempts to feed the router extra history (a running state, a delta-rule reconstruction
  residual) both failed against their own shuffle controls.

Hence **repeat / cross-replay loss** (design notes:
`kvpress/presses/gqa_indexer/cross_replay_e2e.md`, motivation:
`query_independent_indexer_cross_replay.md`): keep the
LM loss, change the *query distribution it is taken over*. Replay the context against its own
first-pass KV (`C' → KV(C)` but `C' ↛ KV(C')`), so supervision
goes from a causal triangle to a full cross-context rectangle and every score is trained on the
question "how valuable is this token to a shared cache serving many unknown queries?" Unlike
Fast-KVzip this keeps the LM loss instead of regressing a per-key label. **This is the next thing
to build.** Note that KVzip's own supervision is *block-diagonal*, not a rectangle — its
`chunk_size` bounds the materialised scoring matrix, not the replay distance
(`cross_replay_e2e.md` §7).

> ⚠️ `proxy_exp/HANDOFF.md` is kept for its record of *which conclusions survived audit and which
> instrument mistakes recurred*, not as a source of reusable numbers — adversarial audits (§10–§13)
> found six bugs in those diagnostics, reversed one headline, and caught the proxy measuring
> something the shipped module does not do (four times). Read §10–§13 before reusing any figure
> from §8. The scripts themselves are gitignored.

### Caveats on the numbers above

- The two `distill` runs and the `@8K` e2e run were produced on different hosts/checkpoints
  (`step600` vs `final`), so the distill-vs-e2e comparison is not perfectly controlled.
- `results_dense` contains two all-zero runs at 4K/8K and one at 16K alongside the good ones;
  the figures quoted here are the non-degenerate ones.
