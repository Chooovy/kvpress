# Task: chunk-wise exact-K subset routing for the GQA indexer, trained end-to-end on LM loss

You have a GPU. Implement and validate an exact-K subset router (ProbMoE / SIMPLE style) for
sparse attention in the `kvpress` repo, trained directly on the language-modelling loss.

Repo: `/apdcephfs_tj5/share_300719894/user/guhao/kvpress`, branch `feat/gqa-indexer`.
Reference implementation to read: `/apdcephfs_tj5/share_300719894/user/guhao/ProbMoE` (already cloned).

---

## 0. Read this section before trusting any number below

The analysis handed to you was produced **on a CPU-only box** (`torch 2.13.0+cpu`,
`torch.cuda.is_available() == False`). Consequences you must respect:

- **Every timing/feasibility number below is CPU-measured.** They establish *relative* cost and
  the *shape* of the bottleneck. They are **not** valid GPU budgets. Re-measure on your GPU
  before accepting or rejecting any configuration. It is entirely possible that a configuration
  ruled "not viable" on CPU is fine on GPU, or that a "viable" one is still too slow in a real
  training loop.
- Correctness facts (exactness, gradient structure) were verified in **fp64** and do transfer.
  Performance facts do not.
- Where this document says "verified", it means a script was run and the number reproduced.
  Where it says "unverified" or "assumed", treat it as a hypothesis to test, not a premise.

**Do not report a configuration as viable without your own GPU measurement inside a real
training step** (forward + backward + optimizer), not just the marginal computation in isolation.

---

## 1. Background: the problem this solves

The indexer is a small learned scorer that decides, per KV head, which cached tokens matter.
Currently it is trained two ways, both already in the repo:

- **Distillation** (`fused_trainer.py`): match the frozen model's attention weights. The score
  never enters the forward pass.
- **End-to-end gated attention** (`e2e_trainer.py`, `gated_attention.py`, `gate_pin.py`): add the
  score inside the attention softmax, `softmax(scale·qkᵀ + g)V`, so the LM loss reaches it.

Read `kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md` in full before writing code. It is the
design analysis behind this task and it contains two explicitly retracted conclusions — read those
too, they are instructive about how this went wrong twice.

The one-line summary of that document: **a router only learns if it cannot score well without
learning a ranking.** An additive gate can go *flat* along the key axis, and a flat gate is
mathematically inert (softmax is shift-invariant), so the model reverts to the frozen pretrained
backbone and gets a good loss having learned nothing. The gated path patches this by normalizing
and pinning some keys (`pin_mode`).

**Exact-K subset routing removes the loophole structurally instead of patching it.** That is why
it is worth building.

---

## 2. What ProbMoE actually does

Paper: *ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts*, arXiv:2606.01509,
ICML 2026 (Heng Zhao, Zilei Shao, Guy Van den Broeck, Zhe Zeng). It adopts **SIMPLE**
(Ahmed et al., ICLR 2023, arXiv:2210.01941) as its gradient estimator.

⚠️ **A secondary source described ProbMoE as using Gumbel-softmax relaxation. That is wrong.**
The README and source confirm: discrete sampling forward, **exact marginals** backward, no
relaxation anywhere. Trust the source.

Read `train/olmoe/open_instruct/OLMoE/v1/ProbMoE_V1_olmoe_exact.py`. The whole method is:

```python
# lines ~158-165
g_full = (samples - marginals).detach() + marginals   # straight-through
w_full = g_full * softmax_probs                       # MULTIPLY the softmax weights
_, selected_experts = torch.topk(samples, self.top_k, dim=-1)
router_weights = torch.gather(w_full, 1, selected_experts)
```

Three parts:

1. **Forward**: sample a genuine discrete k-subset by ancestral sampling (`sample_k_subset`, O(n)).
2. **Backward**: gradient flows through the **exact marginal** `μ_i = P(z_i=1 | Σz=k)`, obtained
   from an O(nk) log-domain DP plus a "probe" autograd trick (`log_pr_exactly_k`,
   `compute_marginals`).
3. **Combination**: `g` **multiplies** the existing softmax weights. It is *not* added to logits.

Inference (`deterministic_routing`) is plain top-k. Stochasticity is training-only.

### Two numerical details that are load-bearing, not cosmetic

Both are deliberate in their source; I broke the code by omitting the first.

- `log_sigmoid` is **clamped to `max=-1e-7`** (their line 11). Without it, when `p` saturates to 1,
  `log(1-p)` becomes ±inf/NaN and `torch.bernoulli` raises `p_in >= 0 && p_in <= 1` failure.
- The DP uses **`NEG_INF = -300.0`, not `-inf`** — because `-inf - (-inf) = NaN` inside `logaddexp`.

### Dynamic-k

`ProbMoE_V1_olmoe_dynamic.py` generalizes to a cardinality *range* `[k_min, k_max]`
(`log_pr_upto_k`, `sample_band_k`). **I only skimmed this** — read it yourself if you want it.
Treat Dynamic-k as out of scope for the first milestone.

---

## 3. Verified properties (fp64, these transfer to GPU)

I implemented SIMPLE's machinery and checked it. Re-verify as your own tests; do not take my word.

| property | result |
|---|---|
| Marginals vs brute-force enumeration (n=9, k=4, all C(9,4)=126 subsets) | max err **6.66e-16** |
| `Σμ == k` | exact (`4.0000000000`) |
| Forward exactness: ST-form vs true sparse attention on the sampled subset | **1.11e-16** |
| No-op immunity: all scores equal → marginals | all `= k/n`, forward still commits to exactly k keys |
| Boundary credit: gradient on **unselected** items | **nonzero** (`1.14e-01` vs `1.06e-01` selected) |

The **forward form I verified** is the multiplicative one (matching ProbMoE, and §47 of the
design doc), *not* an additive gate:

```
α_j = g_j · exp(a_j) / Σ_i g_i · exp(a_i)     where g = (z - μ).detach() + μ
```

With `g` the straight-through mask this is *exactly* sparse attention over the sampled subset
(1.11e-16), so **there is no train/inference gap in the forward** — unlike the dense-forward gated
path.

### The result that motivates the whole thing

Adversarial setup: the needed key starts **outside** top-k (so it must be *promoted*), and the
frozen backbone is already strong (so "do nothing" is tempting).

| method | recall @ init | recall @ end |
|---|---|---|
| selected-gate proxy (score only reweights the selected set) | 0.0% | **0.0%** |
| **exact-K subset ST** | 0.0% | **93.8%** |

(random = 12.5%.) The proxy is structurally stuck: gradient on unselected keys is *identically
zero*, so a key outside the set can never be promoted. Exact-K's marginals couple every item to
every score, so credit reaches outside the selected set.

**Caveat:** this was a synthetic retrieval toy (`B=64, n=32`). It shows the *mechanism* works. It
says **nothing** about how much it helps on real 32K long-context data. ProbMoE's own evidence is
MoE finetuning (OLMoE / Qwen1.5-MoE, n=64 experts) — **there is no published attention result.**
Porting it to attention is our inference, not an established result.

---

## 4. The hard constraint that shapes the design

The DP is **sequential in n**, and with `create_graph=True` it retains all n+1 states.

CPU-measured (**re-measure on GPU**):

```
n=2048, k=64, 512 rows                            → 3312 ms
```

Row count in our setting is `B × n_kv_heads × Sq`. At `B=1, Hkv=8, Sq=32768` that is **262144
rows**, extrapolating to **~1690 s per layer per step**. Dead on arrival.

Root cause, and it is not a constant factor: ProbMoE's DP is sized for MoE — one row per *token*,
n=64 *experts*. Attention has one row per *(token, head)* and n in the thousands.

Three levers, all measured on CPU:

1. **Chunk granularity, not token.** 128K / chunk 64 = 2048 candidates instead of 131072.
2. **Gradient checkpointing on the DP** — recompute in backward, O(k) memory instead of O(nk).
3. **Query-block sharing** — one subset per *block* of queries, the big lever:

   | query block | rows | CPU time |
   |---|---|---|
   | 1 (per query) | 262144 | unmeasurable |
   | 64 | 4096 | 1211 ms |
   | **128** | **2048** | **104 ms** |
   | **256** | **1024** | **73 ms** |

   Plus candidate restriction: `M=64, k=16, 8192 rows → 183 ms`.

**All three are required; drop any one and it does not close on CPU.** Whether GPU changes this
calculus is exactly what you must determine — it may relax the constraint substantially.

⚠️ **Query-block sharing is a real modelling concession**, not just an optimization: 128–256
queries share one chunk subset, which is coarser than per-query selection. NSA and HSA do this, so
there is precedent, but it genuinely reduces selection freedom. **If per-query token-level
granularity turns out to be required, exact-K is the wrong tool** — 1690 s/layer/step is not
something optimization recovers.

**Unexplored escape hatch:** SIMPLE's paper mentions an **O(log k · log n) divide-and-conquer**
parallel DP variant (vs the O(nk) sequential one). I did **not** read or implement it. If it works,
it could restore finer granularity and would change the design shape. Worth a look early, since it
affects everything downstream.

---

## 5. What to build

### Milestone 1 — the standalone op (do this first, it is independently verifiable)

New file `kvpress/presses/gqa_indexer/exact_k_subset.py`:

```python
def exact_k_marginals(logits, k)        # O(nk) log-domain DP + probe trick, checkpointed
def sample_k_subset(logits, k)          # ancestral sampling, O(n), under no_grad
def straight_through_mask(logits, k)    # (z - mu).detach() + mu
```

Port from ProbMoE's source, **keeping the `-1e-7` clamp and `-300.0` sentinel**, but batched over
leading dims `(..., n)` rather than their `(batch, n_experts)`.

Tests in `tests/presses/test_gqa_indexer_exact_k.py`:

- marginals vs brute-force enumeration for n ≤ 12, fp64, `< 1e-14`
- `Σμ == k` exactly
- equal scores → `μ = k/n` (no-op immunity)
- unselected items receive nonzero gradient (boundary credit)
- sampled subsets always have cardinality exactly k
- numerical edge cases: extreme logits (±80), the clamp and sentinel actually needed — write a
  test that **fails** if you remove the clamp
- checkpointed and non-checkpointed paths agree

This milestone touches nothing else and is where correctness gets locked down. **Do not proceed
until these pass.**

### Milestone 2 — chunk-level subset attention

Reuse, do not reimplement:

- `aggregate.py:aggregate_chunk_scores(token_scores, chunk_size, mode)` — chunk pooling
- `aggregate.py:expand_chunk_indices(chunk_indices, chunk_size)` — chunk → token indices
- `indexer.py`: `GQAIndexer.project_q(hidden_states, cos, sin)`, `.project_k(...)`
- `sparse_attention.py:sparse_gqa_attention_reference(q, k, v, indices, ...)` — differentiable in
  q/k/v; indices are `(B, Hkv, Sq, topk)` int32, ascending, `-1` in empty slots
- `press.py:GQAIndexerPress.get_rope_tables(indexer, kwargs)` — **use this**, so the training-time
  score matches what the press computes at inference. Divergence here trains the router for a
  scoring function it never runs under, and nothing downstream flags it.

Shape:

```python
chunk_scores = pool_to_chunks(indexer_scores)       # (B, Hkv, n_qblock, n_chunk)
cand         = build_candidates(chunk_scores, M)    # candidate restriction + exploration
mask         = straight_through_mask(chunk_scores.gather(cand), K)
out          = sparse_attn_multiplicative(q, k, v, cand, mask)
loss         = lm_loss(out)                         # no distillation, no auxiliary term
```

Use the **multiplicative** normalized form (§3), not an additive gate. Verify against
`sparse_gqa_attention_reference` on the sampled subset — it should match to fp32 tolerance
(I measured 1.11e-16 in fp64).

### Milestone 3 — trainer integration

Follow `e2e_trainer.py`'s existing pattern: it swaps `ALL_ATTENTION_FUNCTIONS` and uses a forward
pre-hook to capture `hidden_states` (the attention interface never receives them). Note the
`_global_mapping` cleanup quirk documented in `teacher_lse.py` — `register()` writes to the class
mapping but `pop()` only touches the instance one, so naive cleanup leaks the entry forever.

Freeze the backbone (`E2EIndexerTrainer.freeze_backbone` shows the pattern: identify indexer
params by **module identity**, not name substring). Gradient must still *flow through* the frozen
backbone to reach the router — freeze with `requires_grad=False`, **not** `torch.no_grad()`.

---

## 6. Two decisions you must make explicitly and report

**Candidate pool needs exploration.** If the pool is just `TopM(chunk_scores)`, chunks outside it
never receive gradient — the same disease as the selected-gate proxy's 0.0%. Suggested start:
80% top-M + 10% random + 10% structural (recent / sink). **The ratio is a hyperparameter; sweep
it and report the sensitivity.** Also measure **candidate miss rate** (how often the genuinely
useful chunk is absent from the pool) — if it is high, no backward estimator can compensate.

**Sampling variance.** `torch.bernoulli` makes the forward stochastic. ProbMoE does not report
this as a problem, but they have n=64 and one row per token; our row count differs. Log
loss-curve variance and selection stability (Jaccard overlap of selected sets across adjacent
steps) and report whether it is a problem.

---

## 7. How to evaluate — do not report only final loss

Accuracy alone will not tell you whether the estimator is *correct*. Measure:

- **Router recall@K** against an oracle subset (e.g. from full attention mass)
- **Gradient vs true swap oracle**: pick boundary pairs `(i ∈ S, j ∉ S)`, compute the true
  `ΔL = L(S − i + j) − L(S)`, and correlate against the estimator's `ĝ_j − ĝ_i`. Report Pearson,
  Spearman, and **sign accuracy**. This is the single most informative diagnostic — it directly
  answers "does our gradient recover the discrete marginal utility". Consider building it first;
  it also lets you score the *existing* gated path for comparison.
- **LM-loss regret** vs an oracle subset
- **Selection stability** across steps
- **Candidate miss rate**
- **Real GPU cost**: ms/layer/step for fwd+bwd+step, and peak memory, vs the existing paths

Compare against the baselines already in the repo, at matched budget:

| route | status |
|---|---|
| distillation (`FusedIndexerTrainer`) | implemented |
| gated attention, `pin_mode="none"` (loophole open — ablation) | implemented |
| gated attention, `pin_mode="sink"` / `"self"` / `"self+sink"` | implemented |
| `stage="sparse"` (sparse-forward scope) | implemented |
| **chunk-wise exact-K subset** | **this task** |

---

## 8. Conventions

- Read `AGENTS.md` and `kvpress/presses/gqa_indexer/README.md` first.
- Match the surrounding style: these modules carry unusually dense docstrings that explain *why*,
  including failure modes that were actually hit. Preserve that. If you discover a trap, document
  it where the next person will hit it.
- fp64 for reference tests so tolerances measure floating-point noise, not approximation.
- Line length 120 (`pyproject.toml`).
- Existing suite: `python -m pytest tests/presses/ -q -p no:cacheprovider`.
  **Known pre-existing failures, not caused by you** (verified against a pristine-HEAD worktree):
  `test_gqa_indexer_fused_loss.py::test_per_tile_upcast_is_bit_identical[4-9]`,
  `test_gqa_indexer_fused_trainer.py::test_capacity_model_prices_the_stage2_tile_gather`, plus 3
  in `test_decoding_compression.py` / `test_pipeline.py`. Verify against a clean worktree before
  attributing any failure to your change.

## 9. Reporting

State plainly what you verified vs. assumed. If a milestone fails, say so with the output rather
than working around it silently. In particular:

- If GPU measurement contradicts the CPU numbers in §4 (either direction), **say so explicitly** —
  that changes the design and is the most valuable thing you can find.
- If the swap-oracle correlation is poor, that is a finding about the estimator, not a bug to hide.
- If exact-K turns out worse than the existing gated path at matched budget, report that. A
  negative result here is genuinely useful — it settles an open design question.

---

# ADDENDUM: what the GPU actually said (2026-08-22)

Written by the agent that implemented this. §0 asked for GPU re-measurement before trusting any
number above; here is what that produced. **Two of §4's conclusions are wrong, and the errors are
structural rather than constant-factor.** Read this before §4.

Implementation: `exact_k_subset.py` (estimator), `exact_k_attention.py` (chunk attention),
`exact_k_trainer.py` (trainer), `scripts/train_gqa_indexer_exact_k{.py,_gy.sh}`. Tests:
`tests/presses/test_gqa_indexer_exact_k{,_attention,_trainer}.py` — 83 tests, all passing.

## 1. §4 is wrong by ~4 orders of magnitude, because the DP is launch-bound

§4 extrapolated 262144 rows to ~1690 s/layer/step and called it "dead on arrival". Measured on an
H20 (marginals with `create_graph` + sample + backward, fp32):

| rows | M | K | ms | peak |
|---|---|---|---|---|
| 1024 | 64 | 8 | 95.1 | 27 MiB |
| 8192 | 64 | 8 | 95.6 | 217 MiB |
| 65536 | 64 | 8 | 97.4 | 1.75 GiB |
| 131072 | 64 | 8 | 114.2 | 3.56 GiB |

**Cost is independent of the row count**, and independent of `K` (81–85 ms across K=8..128 at
M=256). It is linear in `M` alone: ~310 µs per DP step, which is CUDA *launch latency* for the few
elementwise kernels each step issues. The per-step arithmetic is a `(rows, K+2)` elementwise pass,
which the GPU finishes in the noise. On CPU the per-step work is real work, so it scales with rows;
on GPU it is a launch, so it does not.

Consequences:

- **Query-block sharing is NOT required.** §4 said "all three levers are required; drop any one and
  it does not close". False. `query_block=1` — full per-query selection — is affordable. Sharing is
  kept (default 256) for the *memory* of the candidate/mask tensors and on modelling grounds, not
  for speed. So §4's warning that "if per-query token-level granularity turns out to be required,
  exact-K is the wrong tool" does not hold.
- **`M` is the only knob that costs time.** Budget it first; `K` is nearly free.
- The SIMPLE divide-and-conquer DP variant §4 flags as an escape hatch is aimed at the wrong
  problem: it would reduce the sequential depth, which is real, but the win would be launch
  reduction rather than FLOP reduction. `torch.compile` would fuse the launches and **does not
  work** — the probe trick needs a double-backward and inductor raises `element 0 of tensors does
  not require grad`. A hand-written Triton scan is the remaining option; not attempted.

## 2. What actually binds: memory, and it stops this at 16K

Not predicted by §4 at all. The frozen Qwen3-8B backbone peaks at **48.3 GiB at 8K and 89.6 GiB at
16K** of the H20's 95 (read off the gated arm's own `metrics.jsonl`), with liger fused-CE and 8-way
FFN-SP already on. So there is ~5.4 GiB of headroom at 16K, and this objective needs ~3.6 — it
OOM'd on a 256 MiB transient with the DP, the score tiles **and** the attention tiles all
recomputing in the backward.

Four separate memory bugs had to be fixed before 8K ran at all, each found by an OOM:

| what was retained | cost | fix |
|---|---|---|
| the DP's `n+1` states, `create_graph=True` | 11.13 GiB / 36 layers | hand-written double-backward (`_CheckpointedMarginals`) |
| the router's `(B,Hkv,Sq,Sk)` token logits | 8 GiB / layer | tile over queries **and** checkpoint each tile |
| the attention's `(B,Hq,Sq,M·chunk)` logits | 64.4 GiB / layer | checkpoint the tile, size the tile by **bytes** not by query count |
| fp32 upcasts of q/k/v/out | 22.5 GiB / 36 layers | upcast the gathered slice inside the tile |

Tiling without checkpointing does nothing — the retained total is unchanged. That cost one OOM to
learn.

**So: 8K fits (54.6 GiB peak), 16K does not.** The gated arm reaches 16K because its fused Triton
kernel computes the gate inside the tile loop and materializes nothing extra; a gather-based path
cannot match that without its own kernel. The run is therefore LR-matched to the gated arm but at
8K throughout — see `stage1_8k` in the launch script for exactly what that does and does not match.

## 3. Speed, at matched settings

125 s/step at 8K (GLOBAL_BATCH=8, FFN_SP=8, 8×H20). The gated arm logged **148 s/step at 8K** and
231 at 16K. So exact-K is *faster per step than the arm it is compared against*, which §4's analysis
would not have predicted.

## 4. Three numerical traps, all silent or only fatal at depth

§2 flags two "load-bearing" details. Both are real, but **the failure modes described are wrong**,
and there is a third trap §2 does not mention that is worse than either.

**(a) The `log_sigmoid` clamp.** §2 says omitting it makes `torch.bernoulli` raise. That did **not
reproduce** — the ancestral sampler's `remaining == 0` guard replaces the degenerate ratio with the
sentinel before `log1mexp` sees it, verified over 2000 draws with saturated scores. What actually
breaks is the *marginals*: at a saturated score `log p == 0`, so `log_q = -inf`, every "not
selected" transition is dead, and the marginals come out ranked by **position**, summing to `k+1`
instead of `k`, with a **NaN gradient**. No exception is raised. Same clamp, different and worse
failure — worth knowing, because the wrong-marginals failure has no traceback attached.

**(b) The `-300` sentinel.** Confirmed. Additionally, `torch.logaddexp` cannot replace the TF-style
form: it is *more* accurate on ties (2nd derivative 0.25 vs 0.0) but produces **non-finite
gradients** on saturated inputs, measured. The 2.4× speedup it offers is not worth taking.

**(c) NOT IN §2, and the one that actually broke the run: the softmax shift.** In
`alpha_j = g_j·exp(a_j)/Σ g_i·exp(a_i)`, taking the row max over *all* candidates lets it land on a
slot with `g=0`. Every surviving weight becomes `exp(a_sel − a_unsel_max)`, which underflows, so the
denominator → 0 and `p` amplifies without bound. Separately, an unselected slot's shifted exponent
is unbounded above *precisely when that chunk looks better than anything selected*, so `exp` → `inf`
and `inf·0 = NaN`. Both fixed (shift over selected slots; clamp the exponent). Note masking the
unselected slots would be **wrong**, not merely conservative — `dw_j/dg_j` for an unselected
candidate *is* the boundary credit §3 measures.

**(d) Also not in §2: the score's gradient into `hidden_states` diverges with depth.** The score is
a function of `hidden_states`, so `dL/d(hidden)` has a second path through the routing decision. It
is a genuine derivative but a **per-layer feedback loop** — gradient deposited in the residual
stream is re-amplified by every router below. Measured `grad_norm` after one backward:

| layers | attached | detached |
|---|---|---|
| 4 | 2.1e3 | — |
| 12 | 8.6e13 | — |
| 24 | **inf** | **1.1e6** |
| 36 | **nan** | — |

Per-layer amplification 10–50×, against 1.1× for the same truncated backbone running dense
attention with no router. Fixed by `detach_score_input=True` (the default); `--attach-score-input`
reproduces the divergence. What it gives up: a router no longer receives the second-order term "my
routing changed what a *lower* router sees". Distillation severs the same path, so this arm is no
worse off on that axis.

**(e) Scale.** The raw `qi·ki` score has std ≈ `sqrt(head_dim)` = 11.3. As a *Bernoulli logit* that
saturates **31% of the marginals** (vs 0% at std 1) and cuts mean gradient 6×, so 31% of candidates
lose the boundary credit the method exists for. Needs the same `head_dim**-0.5` factor
`GATE_SCALE_INIT` applies — a temperature, not a scale match. Unscaled, `grad_norm` read 6.5e4 and
the loss did not descend.

**The pattern in (a), (c), (d), (e): four layers looked perfectly healthy in every case.** Test
gradient finiteness at full depth.

## 5. Design decisions §6 asked to be reported

**Normalize over the whole candidate pool, not the selected K.** Both are numerically identical in
the forward (`g` is 0 off the subset; verified 1.1e-16 in fp64 against a masked softmax). They differ
in the backward, and it is the whole point:

| form | mean grad, selected | mean grad, unselected |
|---|---|---|
| gather the K | 7.05e-02 | 1.61e-03 |
| **full pool** | 1.22e-01 | **1.18e-01** |

Gathering only the K leaves unselected candidates with 73× less gradient. The cost: training-time
attention is over `M·chunk_size` keys rather than `K·chunk_size`, so the *pool* sets the FLOPs.

**Pad slots must be `-1`, not a repeated chunk.** §5's sketch does not mention this. Near the
diagonal a query block cannot see `M` chunks — **48% of blocks at 8K** with `chunk_size=64,
query_block=128, M=64`. Repeating a chunk to fill the pool is wrong, not harmlessly redundant: the
subset's cardinality is over *slots*, so the DP spends two of its `K` on one chunk and the row
attends to `K−1` distinct chunks while reporting a budget of `K`. Verified — the two spellings give
different outputs. `-1` slots get a saturated-but-finite score and are masked out; because
`Σμ = K` is exact, the budget then lands entirely on the real chunks (and correctly falls back to
"all visible chunks" when `V < K`, reported as `effective_topk`).

**Exploration ratio: not swept.** 80/10/10 as §6 suggests (top-M / random / structural), exposed as
`--explore-frac` with `ablate_noexplore` as the 0 arm. Candidate miss rate: not measured. Both are
open.

## 6. The swap oracle: BUILT, and the result needs care to read

§7 calls this the single most informative diagnostic. Implemented in `exact_k_diagnostics.py`
(`swap_oracle_correlation`, plus `router_recall_at_k` and `lm_loss_regret`), tested in
`test_gqa_indexer_exact_k_diagnostics.py` — 12 tests, including both halves that make it
falsifiable: a known-perfect gradient must score 1.0 and a random one must score ~0.5.

The oracle is genuine: for each boundary pair `(i ∈ S, j ∉ S)` the attention is re-run with the
swapped subset **forced**, so `ΔL = L(S − i + j) − L(S)` is a double-forward measurement rather than
a linearization. Forcing matters — the forward samples, so an unforced re-run measures sampling noise
instead of the swap.

**Result on exact-K at init, over 40 boundary pairs:**

| statistic | value | chance |
|---|---|---|
| Spearman | **+0.69** | 0 |
| Pearson | **+0.82** | 0 |
| centered sign accuracy | **0.575** | 0.5 |
| raw sign accuracy | **0.25** | 0.5 |

**The raw sign accuracy looks like a failure and is not**, and this is the part worth reading
carefully. The two gradient populations are offset — mean `g` on selected chunks measured `+2.8e-3`
against `-9.2e-4` on unselected, so `g_j − g_i` carries a systematic `-3.7e-3` shift. Meanwhile 85%
of real swaps *hurt* (the router's picks are already decent), so `ΔL_true` is mostly positive while
the shifted prediction is mostly negative: signs disagree nearly everywhere while the **ordering** is
largely right. A constant added to every score's gradient cannot change which chunk wins a
comparison, and comparisons are all the router ever makes — so the rank statistics are the ones that
bear on "does the gradient recover marginal utility". `SwapOracleResult` therefore reports the bias
and a centered sign accuracy alongside the raw one, so the two numbers reconcile instead of appearing
to contradict.

Two things I got wrong here, recorded because both were plausible:

- **The sign convention.** I first wrote `ΔL_hat = -(g_j - g_i)`, reasoning about which way descent
  moves a score. But the oracle asks how the *loss* changes, and those have opposite signs. The
  correct form is `g_j - g_i` (verifiable: for `L = -Σ_{i∈S} v_i` the truth is `v_i - v_j` and
  `g = -v`, so `g_j - g_i = v_i - v_j` identically). An inverted diagnostic reports a perfect
  estimator as 0.0 — which looks like a real result. This is exactly why the two instrument-validation
  tests assert 1.0 and 0.0 rather than "above chance".
- **The explanation for the offset.** I hypothesized marginal saturation (`dμ/ds` smaller on selected
  chunks because `μ → 1`). **Measured and rejected**: the unselected/selected self-derivative ratio
  is 0.89, i.e. the wrong direction.

### The swap oracle on the real LM loss

`scripts/exact_k_swap_oracle.py` runs the same measurement against the **model's own LM loss on real
text**, per layer, with a genuine full-model forward per pair.

| checkpoint | pairs | Spearman | mean \|ΔL_true\| |
|---|---|---|---|
| smoke (2 steps, ~untrained) | 10 | **+0.64** | 1.05e-2 |
| step200 (trained) | 4 x 64 | **+0.196** pooled, CI [+0.073, +0.313] | 1.0-1.6e-3 |

**Read this as a real effect, not noise.** The replay is bit-deterministic — 8 repeats of an identical
forced configuration gave a spread of *exactly* 0 (std 1.0e-6, which is the float32 print precision),
so signal-to-noise on ΔL_true is 755–2256×.

**Sample size matters more than it looks, and the first two runs were underpowered.** At n=12 the
95% CI on a Spearman spans roughly ±0.55 — useless. At n=48, one probed layer came out at +0.310
(CI [+0.03, +0.55], excludes 0) and the other at +0.233 (CI [−0.06, +0.49], **includes** 0), so a
per-layer claim was not supportable. The properly powered run is **4 layers × 64 pairs**:

| layer | Spearman | centered sign |
|---|---|---|
| 0 | +0.147 | 0.561 |
| 2 | +0.260 | 0.633 |
| 5 | +0.087 | 0.492 |
| 7 | +0.286 | 0.550 |

All four positive. Pooled by Fisher-z: **r = +0.196, 95% CI [+0.073, +0.313] — excludes 0.** Mean
centered sign accuracy 0.559 against a chance 0.5. The all-four-positive pattern alone is p = 0.062
by a sign test, so the pooled CI is doing the work rather than the direction count.

The claim this supports: *on a trained checkpoint the estimator still ranks boundary swaps better than
chance, weakly but measurably.* It does **not** support a per-layer effect size — layer 5 (+0.087) is
individually indistinguishable from chance.

The **drop from +0.64 to +0.27 is expected rather than a defect**: `mean |ΔL_true|` fell an order of
magnitude over the same interval, i.e. a trained router has already taken the easy wins, so the
remaining boundary swaps are near-ties that are intrinsically harder to order. A diagnostic measured
where the decisions are genuinely marginal should be closer to chance.

**Two bugs in that script, both of which produced confident-looking numbers before being caught.**
Worth recording because anyone rebuilding this diagnostic will hit them:

1. **Do not capture the gradient from a *forced* run.** The +1e4 offset that forces a subset also
   saturates the marginals, collapsing `|dμ/ds|` from 1.5e-1 to **5e-7** — so the "gradient" being
   correlated is numerical noise. The tell was `bias +0.00e+00` on every layer. Capture from the
   unforced pass; force only where the loss is needed.
2. **Force *every* layer, not just the probed one.** Otherwise the other 35 layers re-sample on each
   replay and their variation swamps the swap. Measured: `mean |dL_true|` fell **30–100×** (3.2e-1 →
   1.05e-2) once all layers were pinned, and the sign accuracy flipped from below chance to above.

What caught both was a guard that re-runs the *unforced* subset through the forced path and refuses
to proceed if the loss does not reproduce. Without it the script reports numbers about replay noise.

## 7. What is still NOT done

Stated plainly, per §9:

- **The swap oracle on the trained checkpoint is an 8-layer-truncated probe** (4 layers, 64 pairs,
  2K tokens). A truncated model's loss is not the trained model's loss, so the numbers indicate
  direction and rough magnitude, not the deployed model's behaviour. The pooled effect is significant;
  individual layers largely are not.
- **The trend is measured at step 200, not 600.** Whether Spearman keeps falling as the router commits
  further is the question this raises and does not answer.
- **The gated path was not scored on the same axis.** §7 notes the oracle also lets you score the
  *existing* arm, which is what would make it a comparison rather than a self-report. The machinery
  is arm-agnostic (pass any `score_grad`), but the gated arm was not run through it.
- **No router recall@K or LM-loss regret measured**, though both are now implemented — they need an
  attention-mass oracle wired up. **Candidate miss rate: not measured.** The diagnostics that are
  wired into the training loop are marginal entropy (has the router committed), `effective_topk` (is
  the budget reachable), and Jaccard (is the stochastic forward stable).
- **16K untested** because it does not fit; the comparison against the gated arm is LR- and
  step-matched but not context-length-matched.
- **No eval result yet** at the time of writing — training is mid-run.
- The `hard=True` and `explore_frac=0` ablations are wired but unrun.

## 8. Training signal, and the 8x wall-clock trap

**`FFN_SP=8` was copied from the gated arm and is wrong here — it costs 8x.** FFN-SP spends its ranks
on ONE sequence, so with 8 GPUs there is a single data-parallel replica and `GLOBAL_BATCH=8` has to be
reached by 8 *sequential* micro-steps. With `FFN_SP=1` there are 8 replicas and `accum=1`, so the same
8 sequences run in parallel:

| | s/step @ 8K | 600 steps | tokens/step | LR curve |
|---|---|---|---|---|
| `FFN_SP=8` | 116.6 | 19.4 h | 65536 | identical |
| **`FFN_SP=1`** | **14.6** | **2.4 h** | 65536 | identical |

The gated arm needs the sharding because it trains at 16K, where one sequence does not fit on one GPU.
This arm is capped at 8K by memory anyway (§2), and at 8K one sequence fits with room to spare —
65.8 GiB of 95. Nothing is given up: same sequences/step, same tokens/step, same WSD curve.

**Signal over the first 80 steps** (8K, `FFN_SP=1`):

| step | loss | H(μ) | eff_K | Jaccard | grad_norm |
|---|---|---|---|---|---|
| 0 | 6.42 | 0.5053 | 7.88 | — | 1.8e7 |
| 20 | 5.64 | 0.5049 | 7.88 | 0.211 | 8.2e7 |
| 50 | 3.49 | 0.4510 | 7.88 | 0.270 | 6.6e5 |
| 80 | 2.55 | 0.3860 | 7.88 | 0.362 | 7.8e3 |

All three signals move the right way together, which is the pattern that distinguishes a trained
router from a trained-looking one:

- **loss falling AND H(μ) falling** — the router is *committing*, not riding whatever random subset it
  is handed. A loss that fell with H(μ) flat would be the exact-K analogue of the flat-gate no-op, and
  is the one failure this design does not rule out structurally.
- **Jaccard rising** (0.21 → 0.36) — selections stabilise as the score sharpens, so the stochastic
  forward is not fighting the optimizer.
- **`effective_topk` 7.88 of a nominal 8** — stable, so the budget is genuinely being spent; the 0.12
  shortfall is the expected near-diagonal effect where a query block cannot yet see K chunks.
- **grad_norm 1.8e7 → 7.8e3**, finite throughout at full 36-layer depth, confirming the §4(d) fix
  holds in the real run rather than only in the depth probe.


---

# RESULT (2026-08-22): exact-K loses to the gated arm, and the reason is measured

## The numbers

RULER, fraction 0.1, `topk 2048 force_sink 4 force_local 64`, 8-way sharded. Eval configs are
byte-identical between arms apart from `indexer_ckpt`.

| | mean RULER |
|---|---|
| gated (`pin_mode=sink`) @16K | **79.04** |
| exact-K @16K — *the only like-for-like number* | **49.55** (−29.5) |
| exact-K @8K (its trained length) | 68.29 |
| exact-K @8K, topk 512 (its trained budget) | 32.61 |

Swap oracle (4 layers × 64 pairs, real LM loss, pooled Fisher-z): step200 **+0.251** [+0.131,+0.365],
step600 **+0.314** [+0.197,+0.422]. Both exclude 0, and it *improves* with training.

## Why: the score is piecewise-constant, and eval ranks tokens

Not a loading bug — the eval log records `objective: exact_k_subset, step 600`, and `w_q` moved 93%
in relative norm between step200 and step600. Not a broken estimator either — the swap oracle says
the gradient recovers discrete swap utility, better at step600 than step200.

The cause is a **granularity mismatch, and it is measurable in the weights.** Fraction of score
variance living *within* a 64-token chunk versus *between* chunks, on real text:

| layer | exact-K within/across | gated within/across |
|---|---|---|
| 0 | **0.17** | 0.70 |
| 4 | **0.16** | 0.99 |
| 7 | 0.69 | 0.74 |

exact-K learned an almost **piecewise-constant** score: at early layers the between-chunk variance is
~6× the within-chunk variance. That is exactly what its objective asked for — it was only ever
supervised on *chunk-mean* scores over *query blocks*, so within-chunk structure is unconstrained and
stays near init. The gated arm was supervised per `(query, key)` token and carries real token-level
structure (ratio 0.7–1.0).

`SparseAttentionContext` — the eval path both arms use — ranks **tokens** per **query**. So exact-K
picks the right *chunks* and then cannot order tokens inside them: the top-2048 cut falls among 64
near-tied tokens and the ordering there is close to arbitrary. It is being scored on a resolution it
was never trained to have.

This also explains the otherwise puzzling **−35.7 at `topk 512`**, its own trained budget: a tighter
token budget means *more* of the decision is within-chunk ordering, which is precisely where the score
carries no information. The method looks better the more budget it is given because budget substitutes
for the resolution it lacks.

## Decomposition of the 29.5-point gap

* **18.7 points: length.** 8K → 16K on the same checkpoint. It never saw 16K (memory, §2).
* **~10.7 points: everything else** — the objective plus the granularity mismatch, not separable here.

## What this settles, and what it does not

**Settled:** chunk-level exact-K subset routing, evaluated by a token-level per-query top-k, is worse
than the additive gated arm at matched budget. Not marginally — 29.5 points, or ~10.7 after removing
the length confound.

**Not settled:** whether the *objective* is worse, because it was never evaluated on the operator it
trains. The two cheapest experiments that would separate those, both now known affordable:

1. **Chunk-level inference.** Make the press select whole chunks (it already supports `chunk_size`;
   `SparseAttentionContext` ignores it). Then train and eval agree, and the number means something
   about the objective.
2. **`query_block=1` + `chunk_size=1`.** Token-level exact-K. §1 established the DP is launch-bound,
   so this is affordable — the original analysis wrongly ruled it out.

A third, independent of the above: a fused kernel to reach 16K so the length confound disappears.


---

# RETRACTION + chunk-wise eval result (2026-08-22, later)

## The granularity explanation above is WRONG. Measured.

The section above attributes the 29.5-point gap to a train/eval granularity mismatch: exact-K's score
is piecewise-constant (measured within/across-chunk variance ratio 0.16 vs the gated arm's 0.99), and
eval ranks tokens, so exact-K was supposedly being scored on a resolution it never learned. That
premise is real. **The conclusion drawn from it does not survive being tested.**

I built the matched evaluation — `chunk_support.py:chunk_topk_support`, whole-chunk selection with
the same token budget, same forced slots, 14 tests including "selection really is whole chunks",
causality, and budget parity — and measured **recall of the true attention-mass oracle** for both
checkpoints, token-wise vs chunk-wise.

**Chunk-wise selection is worse for BOTH arms, including exact-K.** Contested regime
(`topk=256`, only rows with ≥ 2048 visible keys; random baseline = 0.087):

| ckpt | layer | token-wise | chunk-wise | Δ | recency |
|---|---|---|---|---|---|
| exact-K | 0 | 0.298 | 0.180 | **−0.118** | 0.302 |
| exact-K | 4 | 0.174 | 0.105 | −0.069 | 0.173 |
| exact-K | 7 | 0.135 | 0.092 | −0.043 | 0.171 |
| gated | 0 | 0.237 | 0.167 | −0.071 | 0.302 |
| gated | 4 | 0.183 | 0.093 | −0.090 | 0.173 |
| gated | 7 | 0.163 | 0.094 | −0.068 | 0.171 |

So matching the eval granularity to training **does not help exact-K** — it hurts, and by a similar
margin to the gated arm. The piecewise-constant score is a real property that **does not translate
into worse key selection**: at token level the two routers score essentially the same (0.298 vs 0.237
at L0, 0.174 vs 0.183 at L4). Whatever produces the 29.5-point downstream gap, it is not this.

## A methodological trap worth more than the result

My first version of this measurement used `topk=2048` at `Sq=4096` and reported chunk-wise ≈ −0.025
for both arms — which I nearly wrote up. Adding no-router baselines killed it: **random selection
scored 0.85, beating every trained router.**

The reason: at `topk ≈ Sq/2` the average row has ~2048 visible keys and a 2048 budget, so the oracle
top-k *is* essentially the visible set and anything that takes what it can see scores high. Early rows
(visible < topk) score exactly 1.0 for any selector and dominate the mean.

**Recall@k is only informative when `visible >> k`.** Always include a random and a recency baseline;
without them a degenerate metric looks like a result.

## The finding that matters more than either

With the metric fixed, **neither router beats plain recency**: exact-K L0 0.298 vs recency 0.302,
gated L0 0.237 vs 0.302, and both are at or below recency at L4/L7. Both beat random (0.087) by ~2x,
so both learned *something* — but on attention-mass recall at a contested budget, neither arm has
learned to beat "keep the most recent k keys".

That reframes the whole comparison. The 29.5-point gap between the two arms may be much less
interesting than the fact that both arms are, on this measure, barely competitive with a heuristic
that needs no training at all. Before iterating further on the exact-K objective, the question to
settle is whether either router is earning its parameters — which the swap oracle (a *relative*
measure) cannot answer, and this (an *absolute* one, against a no-router baseline) can.

## Status of the chunk path

`chunk_topk_support` is implemented, tested (14 tests) and wired into `SparseAttentionContext`
(`chunk_size=...`) and `evaluate_sparse.py` (`--chunk_size`, with `-1` meaning "read it from the
checkpoint"). Selection is per-query rather than per-query-block, deliberately: that holds the query
axis fixed and varies only the key axis, so the comparison isolates the axis the scores actually
differ on. It found one real bug on the way in — a forced sink/local token that also falls inside a
selected chunk occupied two slots, and `sparse_gqa_attention` sums duplicate indices *with
multiplicity*, so that key would have received double softmax weight. Deduped, with a regression test.

## The full RULER run under chunk-wise eval: RUN, and it confirms the retraction

RULER @8192, fraction 0.1, `chunk_size=-1` (resolved to 64 from the checkpoint, verified in the shard
log), 8 shards, both budgets:

| task | tok/2048 | chk/2048 | tok/512 | chk/512 |
|---|---|---|---|---|
| cwe | 62.09 | 61.86 | 37.91 | 37.91 |
| fwe | 86.00 | 85.33 | 72.00 | 70.67 |
| niah_multikey_1 | 81.48 | 27.78 | 24.07 | 9.26 |
| niah_multikey_2 | 21.62 | 18.92 | 8.11 | 5.41 |
| niah_multikey_3 | 13.04 | 13.04 | 8.70 | 4.35 |
| niah_multiquery | 91.23 | 35.09 | 25.44 | 10.53 |
| niah_multivalue | 78.07 | 33.77 | 19.74 | 8.33 |
| niah_single_1 | 100.00 | 100.00 | 100.00 | 10.61 |
| niah_single_2 | 100.00 | 36.36 | 28.79 | 6.06 |
| niah_single_3 | 100.00 | 78.57 | 14.29 | 7.14 |
| qa_1 | 27.66 | 27.66 | 25.53 | 23.40 |
| qa_2 | 29.55 | 34.09 | 22.73 | 18.18 |
| vt | 97.07 | 33.17 | 36.59 | 12.20 |
| **MEAN** | **68.29** | **45.05** | **32.61** | **17.23** |

**Matching the eval granularity to training makes it much worse, at both budgets:** −23.2 at the
baseline budget, −15.4 at the trained budget. The **fully train-consistent configuration** — chunk-64
selection at the 512-token budget the router actually trained with — scores **17.23**, the worst of
the four and barely above the 8.5-ish floor these tasks give a random selector.

This is the third and strongest disconfirmation of the granularity hypothesis, after the oracle-recall
measurement and the token-level parity between the two arms. The hypothesis is dead: exact-K is not a
good chunk-router being unfairly scored at token resolution. It is simply a **worse router**, and it
happens to look least bad when given the loosest possible selection freedom (per-query, per-token, 4x
its trained budget).

Note *which* tasks collapse: the NIAH retrieval family (`niah_multiquery` 91→35, `niah_multivalue`
78→34, `vt` 97→33) while the aggregation tasks are nearly untouched (`cwe` 62.1→61.9, `qa_1`
27.7→27.7, `fwe` 86.0→85.3). Needle retrieval needs a *specific* token; forcing whole 64-token chunks
spends the budget on 63 neighbours per needle, so at a fixed token budget far fewer distinct needles
fit. Aggregation tasks read broadly and do not care. That is a budget-efficiency argument against
chunk selection at inference, independent of which router produced the score.

## What is actually worth running next

Not more exact-K variants. The two measurements that would change the picture:

1. **Recency baseline on full RULER.** Both routers were at or below plain recency on oracle-recall
   (0.298 / 0.237 vs 0.302). If recency also matches them on RULER, neither arm is earning its
   parameters and the arm-vs-arm gap is a distraction.
2. **Whether the gated arm's 79.04 survives at 8K.** It is measured at 16K; exact-K's best is 68.29 at
   8K. Those are not the same setting, and the honest gap between the arms is still unmeasured.


---

# 诊断收敛：不是 estimator，是 candidate pool（2026-08-23）

## 两个候选解释，都测了

**「sparse attention 的 n 会变，MoE 的 expert 数固定」—— 对这个实现不成立。**
`candidate restriction` 已经把它消掉了：送进 DP 的 `n` 永远是 `n_candidate=32`，与 `Sq` 无关。变的是
`n_chunk`（8K→128, 16K→256），但那不是 DP 看到的轴。而且预算基本花满 —— `effective_topk` 全程
7.88/8。真正受影响的只有近对角线那 11-22% 的 query block（可见 chunk < M），而其中可见 < K 的仅 1 个。

**「需要 chunk 内排序」—— 前提对，但不是落后的原因。** 三次独立测量都否定：chunk-wise eval（完全免除
chunk 内排序）RULER 45.05/17.23 更差；oracle recall 对两个 arm 都 −0.04~−0.12；token 级下两 arm
分数接近（L0 0.298 vs 0.237）。NIAH 的塌陷（multiquery 91→35）指向相反的机制：整块选择在固定
**token** 预算下浪费预算，每个 needle 拖 63 个邻居。

## 被数据支持的诊断：candidate miss rate

文档 §36.5 早就点名了这个指标。实测（M=32, Sq=4096, pool 覆盖 50% 的 chunk；oracle = 真实 attention
mass 按 chunk 汇总）：

| layer | miss@K | miss@2K |
|---|---|---|
| 0 | 11.2% | 20.9% |
| 4 | 14.1% | 20.9% |
| 7 | 14.9% | 23.4% |

**11–15% 的 oracle chunk 从未进入候选池。** §36.5 的原话：「如果 oracle token 根本不在 candidate
pool，任何 backward estimator 都救不了」。这是 estimator 之外的天花板。

算一下探索预算就知道为什么：`explore_frac=0.10` 于 `M=32` 是每步 3 个随机 chunk，占 8K 时 128 个
chunk 的 2.3%；要靠运气覆盖池外的 96 个 chunk 需要 ~30 步。600 步里每个池外 chunk 期望被看到 ~20
次，但每次只在那一步的那个 query block 上。

**这个诊断同时解释了之前的矛盾**：swap oracle 说梯度是好的（Spearman +0.31，且随训练**变好**），
下游却差。两者都对 —— 梯度在**池内**是正确的，瓶颈是池本身。之前把这两件事当成互相矛盾，是因为
默认了 estimator 质量决定下游质量。

## 修正后的优先级

不是"再试一个 estimator"（SparseMixer/GRIN、LapSum、Sander 都是在改 §5–§13 的 backward），而是先
把 §5.4 / §36.5 的 support 问题解决：

1. **M/K sweep**（文档 §37 第 9 条）。现在 M/K = 4；文档 §26 建议 2–4，但那是按 *token* 预算给的。
   chunk 级下 M=32 只覆盖 25%（8K）/ 12.5%（16K）的 chunk。先测 miss rate 对 M 的曲线 —— 这比训练
   任何新 estimator 都便宜，且能直接说明天花板在哪。
2. **分层选择**（§50）。粗层选 chunk、细层在选中的 chunk 内选 token，两层各自可微。这同时解决 miss
   rate（粗层看全部 chunk，无需候选池）和 NIAH 的预算浪费（细层不必整块拿）。这是 §49–§50 唯一同时
   命中两个已测问题的方案。
3. **§31 的 LM-gradient utility self-distillation**。`∂L/∂b_j = α_j (∂L/∂o)ᵀ(v_j − o)` 一次 backward
   就给**每个** key 一个 utility，不需要它先被选中 —— 正好绕过 support 问题，成本远低于 swap。可以做
   辅助 loss 直接监督 ranking。

## 不值得优先做的

- **Sander / LapSum / Fast LapSum（§12–§13）、SparseMixer / GRIN（§7）**：都是更好的 backward。
  但 swap oracle 已经说明当前 backward 在池内是有效的（+0.31，且变好），瓶颈不在这里。除非先把 miss
  rate 压下去，换 estimator 只是在优化一个不是瓶颈的环节。
- **Sparsemax / ReMoE（§14, §20）**：dynamic-K，与固定 KV 预算的推理需求不符（§34 自己也这么说）。
- **纯 token 级 exact-K（`chunk_size=1, query_block=1`）**：DP 是 launch-bound 所以算得起，但 miss
  rate 会更糟 —— 池要从 128K 个 token 里选 M 个，而不是从 128 个 chunk。除非配合分层。


---

# 实验1 结论：两个 arm 的梯度质量相同，estimator 不是差距来源（2026-08-24）

`scripts/compare_swap_oracle_arms.py`：**同一个 oracle、同一个算子、同一个探针**，只换权重。对每层的
chunk 分数加一个零偏置 `b_c`，测 `dL/db_c`（`b=0` 处前向与训练逐位相同），再与真实
`ΔL = L(S−i+j) − L(S)`（强制子集、双次前向）比较。4 层 × 64 对，Fisher-z 汇总：

| arm | Spearman | 95% CI | Pearson | centered sign |
|---|---|---|---|---|
| exact-K | **+0.309** | [+0.191, +0.417] | +0.221 | 0.636 |
| gated | **+0.324** | [+0.207, +0.431] | +0.286 | 0.614 |

**CI 大幅重叠。** gated 下游赢 18.8 分（87.06 vs 68.29），但它的梯度**并不更准**。

## 这判定了什么

- **换 estimator 不是优先项。** LapSum / Sander / SparseMixer（文档 §7, §12–13）都是"造一个更好的
  Top-K 反向"。既然赢的那个 arm 的反向并不更好，这条路线预期收益低。这也**修正了我此前的过度推断**：
  我曾说"不是 estimator"但只测了 exact-K 一个、没有对照，无从判断 +0.31 算好算坏。现在有对照了。
- 差距在别处：**候选池**（exact-K 每步只看 25% 的 chunk，11–15% 的 oracle chunk 从未进池；gated 是
  `stage=dense` → **full scope**，每个 key 每步都有梯度，miss rate = 0）与**预算效率**（整块选择在固定
  token 预算下浪费额度，NIAH 塌陷 91→35）。

**这两个 arm 之间的 support 差异是此前完全未受控的** —— 我一直按"只有目标函数不同"在比较，那不成立。

## 途中修掉一个会污染诊断的 bug

我一开始用 `gather_candidate_scores` 去 gather **梯度**。它把 `-1` pad 槽填成 `PAD_SCORE = -31`，那是
给*分数*设计的哨兵（"这个槽不是真候选，排最后"），作为*梯度*毫无意义（应为 0）。

发现方式：gated layer 3 报出 `bias = -3.10e+01` —— 恰好等于 `PAD_SCORE`。修复后 bias 降到 ~1e-18。
新增 `gather_candidate_gradient`，把陷阱写进函数名和文档。**一个被这样污染的诊断仍会产出看起来合理的
相关系数**，这是它值得单独记一笔的原因。

# 实验2 进行中：候选池开到全覆盖

`M=128`（8K 下 `n_chunk=128`，覆盖 100%，miss rate → 0），其余一切与已完成的 `M=32` run 相同，所以
唯一变量是候选池。这同时把 support 与 gated 的 full scope 对齐 —— 正是要检验的变量。

代价实测（8K，每层 fwd+bwd）：

| M | 覆盖 | keys/query | ms/layer | ×36 层 |
|---|---|---|---|---|
| 32 | 25% | 2048 | 226 | 8.2 s |
| 64 | 50% | 4096 | 455 | 16.4 s |
| **128** | **100%** | **8192** | **845** | **30.4 s** |

45 s/step 实测 → 600 步约 7.5 小时。注意 `M=128` 时 `M × chunk_size = 8192` = 全序列，**训练时
attention 实际退化成 dense**。这不是缺陷而是这个对照的要点：如果它显著变好，说明 exact-K 的瓶颈就是
support 太窄，也顺带解释 gated 为何强（它本来就是 full scope）。

`H(μ)` 起点从 0.505 降到 **0.170**，符合 `K/M` 由 0.25 变 0.0625 的预期。

**一个操作教训**：第一次启动时我让 oracle 实验和训练并发在同一批 GPU 上，NCCL 在 rank1（正是 oracle
占用的 cuda:1）报 `unhandled cuda error` 崩掉。8 卡训练要独占。
