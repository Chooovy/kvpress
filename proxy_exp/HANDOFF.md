# Scalar-indexer router: what is established, what is not, and what to run

Context for a fresh agent picking this up. §1–§7 were written on a CPU-only box, so their numbers
are either exact identities (safe) or synthetic (unsafe, and the reason this directory exists).

**§8 is the real-model run (2026-08-18, Qwen3-8B on H20) and supersedes §2. §9 refutes the
state-based router (§4's `z` input). §10–§12 are adversarial audits of both: they found six bugs and
**reversed §8's headline** — the 3–10× is ~2× end-to-end and does not grow with length, and rel_L2
correlates with LM loss at only +0.26 single-layer. **READ §10–§12 BEFORE TRUSTING ANY §8 NUMBER.**
The three cancellations (E3, bilinear, Arm C) survive; the explanations behind them did not.
**§12.1 is the one actionable positive result: a per-KV-head target beats everything §8 tested.**
Current plan: §12.7.**

---

## 0. The situation in one paragraph

The GQA lightning indexer is a router that scores KV entries so low-scoring ones can be dropped.
Two variants are implemented and both are trained end-to-end from the LM loss:

| arm | score | query-aware | can evict? | params/layer (Qwen3-8B) |
|---|---|---|---|---|
| `pairwise` (`indexer.py`) | `q_i · k_j`, per (query, key) | yes | **no** | 4.72 M |
| `scalar` (`scalar_indexer.py`) | `w_out · φ(W_in h_j) + jε` | no | **yes** | 1.06 M |

An A/B at `topk=2048` on RULER has already been run (`evaluation/results_sparse_e2e` vs
`results_sparse_scalar`) and the scalar arm is much worse:

| | pairwise | scalar | delta |
|---|---|---|---|
| RULER avg @8K | 87.06 | 66.24 | **−20.8** |
| RULER avg @16K | 79.04 | 49.56 | **−29.5** |

**That comparison is not the one that matters, for two reasons the user identified:**

1. It was run as *sparse attention* (whole KV retained, per-query top-k). That is pairwise's
   home turf. Scalar's only advantage — a frozen score, so KV can actually be **freed** — is
   not scored at all, while it pays the full cost of giving up query-awareness.
2. Parameters are not matched (4.72 M vs 1.06 M, 4.45×).

Query-dependent scores **cannot** evict: key `j`'s score changes with each new query, so a freed
entry may be needed again. The scalar score is frozen at creation (verified: **0** re-entries into
the top-k, `tests/presses/test_gqa_indexer_scalar.py::test_recency_tilt_keeps_topk_irreversible`).
So the two arms do different jobs and the real question is how scalar does **on eviction**.

---

## 1. Established — exact identities, safe to build on

These are algebra or verified-to-machine-precision, not measurements.

**The LM loss already supervises the router with future tokens.** For a gate `g` added inside the
softmax (`softmax(qk + g)`):

```
∂L/∂g_j = Σ_i A_ij · ⟨∂L/∂o_i , v_j − o_i⟩          (verified, max err 3.6e-15)
```

Decomposing by which query `i` contributed: **8.3×** more gradient magnitude arrives from queries
*after* `j` than from `j` itself. So "use future tokens to teach the score" is not a mechanism to
add — it is what the objective does. This term also contains, for free, the two things a per-key
distillation label cannot express:
- `(v_j − o_i)` — value geometry, not just attention mass;
- because `A` is computed with **all** gates present, the gradient on `g_j` is conditioned on
  what the other keys are doing, i.e. it is implicitly **set-level** (can express "these two are
  redundant, keep one").

**The user's decision — keep the LM loss, no auxiliary distillation target — is well founded.**
Any per-key regression label collapses the `Σ_i` before the set decision is visible.

**Lookahead in the score's INPUT is a trap.** `score_i = f(h_i, h_{i+1}, …)` is computable at
prefill (the whole context is present), so it trains and evaluates fine, then breaks the moment a
token arrives after the eviction decision. Measured: **566** keys re-entered the top-k versus 0
for a causal score. Future tokens may inform the *gradient*, never the *input*.

**`z_{t-1}`, not `z_t`.** If a state `z` is fed to the router and that state accumulates the keys
the router did *not* keep (the SPLA coupling, below), then `score_t → I_t → z_t` is circular.
`z_{t-1}` is the only ordering that closes. Independently, `z_{t-1}` keeps the score a genuine
prediction error: `M_{t-1} k_t` is the state's guess at `v_t` from history alone, whereas after the
delta-rule update the state has already absorbed `h_t` (at β=1 the residual is identically 0).

**Exact eviction damage.** Dropping key `j` perturbs query `i`'s output by
`A_ij (o_i − v_j) / (1 − A_ij)` (verified, 3.1e-08). Mass omits the `‖o_i − v_j‖` factor.

**SPLA's subtraction identity** (`o_rla = φ(q)S̄ − φ(q)S̃`, arXiv 2601.22379 eq. 14–16) is exact by
linearity (2.5e-07). Two consequences for this project:
- **Evicted KV need not be retained.** `S̄` is built incrementally; once token `i` is folded in,
  its KV can be freed. Eviction removes it from the sparse branch only.
- SPLA needs the subtraction because *its* unselected set is query-dependent (the paper says so).
  A **query-independent** router has a fixed evicted set, so the "unselected-only" state can be
  built directly and **the subtraction trick is unnecessary**. Rare case where
  query-independence is an advantage.
- Corollary: `S̄` must be **undecayed** (α=0) for `S̄ − S_kept` to equal the evicted contribution.

**Titans' forgetting** (arXiv 2501.00663 eq. 13–14): `M_t = (1−α_t)M_{t-1} + S_t`,
`S_t = η_t S_{t-1} − θ_t ∇ℓ`, with `ℓ = ‖M_{t-1}k_t − v_t‖²` so `∇ℓ` **is** the delta rule; α, θ, η
all data-dependent. Two hard facts measured here:
- **Momentum is a divergence trap** with fixed θ: effective LR is `θ/(1−η)`, so η=0.9 multiplies it
  10× (‖M‖ hit 1.9e7; η=0.99 → NaN). Rescaling `θ ← θ(1−η)` fixes it. Titans survives because θ_t
  is learned and absorbs the factor.
- **Delta rule requires ‖k‖=1** — unnormalized keys diverged at step 39.
- Per-token α breaks exact chunked parallelism; **per-chunk α is exact** (4e-07), which is Titans
  §3.2's own "parameters as functions of chunks" simplification.

**Cost model** (Qwen3-8B, L=64K, per layer, decode; sparse attention itself = 4.19 M MAC):

| | MAC/step | ×attn | state/cache |
|---|---|---|---|
| pairwise `head_dim=128` | 71.8 M | **17.1×** | 16.0 MiB |
| pairwise `head_dim=8` | 4.49 M | **1.1×** | 1.0 MiB |
| scalar `mid_dim=256` | 1.05 M | 0.3× | 2.0 MiB |
| shared matrix state `d_s=128` | 0.79 M | 0.19× | 1.0 MiB fp32 |

`--head-dim 8` already exists as a flag and cuts pairwise's decode cost 17.1× → 1.1× while keeping
query-awareness. It does not change the `O(L²)` asymptote, but if the real ceiling is 64–128K it may
be enough. **Cheap, untried, and possibly the highest-value experiment here.**

---

## 2. NOT established — synthetic only, do not build on

> **Superseded 2026-08-18: all five claims below were run on the real model (Qwen3-8B, 4×H20).
> Four of the five reversed. See §8 for the measured results — read that section instead of
> this one.** This section is kept because the *reasoning* behind each claim is still what §8
> is arguing against, and because the tally is now five-for-five on synthetic hidden states
> being misleading in this line of work.

Every number in this section came from synthetic hidden states. This project has already been
wrong **four** times that way in this line of work; two were user-caught, two self-caught:

- "EMA is enough, skip matrix states" — reversed on real hidden states (delta rule won 2×).
- "Decay must be aggressive (λ=0.95)" — reversed; by AUC λ=0.99 was best and 0.95 among the worst.
- "prefill vs future demand are unrelated (3.1% overlap)" — **self-reversed**: switching the
  prefill sum to a *mean* moved agreement to 92.2%, i.e. the gap was a position-visibility ramp,
  not a distributional difference. Same error shape as the fixed `reduce_queries` bug
  (`MASK_NEG · t / Sq`). C1 below exists to settle this on real data.
- An early probe measured a hand-picked scalar readout instead of a learned one; the user caught
  it, and redoing it with a learned readout raised the score 0.19 → 0.52.

Open synthetic claims, each mapped to an experiment below:

| claim | synthetic result | experiment |
|---|---|---|
| prefill/future gap is a normalization artifact | sum 10.2% → mean 92.2% | **C1** |
| bilinear beats MLP per parameter | R² 0.9966 @4k vs 0.980 @67k | **C2** |
| `z` helps beyond position | h 71.1% → h+z 89.1%, but h+pos 99.2% | **C3** |
| value geometry is negligible | 94.5% agreement with mass | **C4** |
| `mid_dim` is saturated | 64→1024 changed R² by 0.001 | **C2** |

---

## 3. Why bilinear is worth testing (C2)

Fast-KVzip scores `exp(q_i·k_i) / (exp(q_i·k_i) + Σ_sink exp(q_i·k_s))`. Two exact rewrites:

1. `= sigmoid(q_i·k_i − logsumexp_sink(q_i·k_s))` (1.7e-16). It is a **margin against the sinks** —
   "is this token more attractive than the attention garbage-bin", which gives the score a semantic
   calibration point. The current `in_norm` solves the same scale problem statistically (without
   it, score std tracked hidden-state norm: 0.009 vs 0.887 across a 100× norm range, while
   attention logits stay at std ~1).
2. `q_i·k_i = h_i^T (W_q^T W_k) h_i = h_i^T sym(W_q^T W_k) h_i` (2.1e-14) — an **indefinite**
   quadratic form (34 negative / 30 positive eigenvalues measured). A *linear* score cannot
   represent it at all (R² ≈ 0). The current `ScalarIndexer` MLP can approximate it, but a
   bilinear `(Ah)·(Bh)` matches it structurally.

**Caveat found while building the script:** a bilinear probe needs training rows far in excess of
its parameter count. On a synthetic rank-8 quadratic at `d_in=512`, a rank-32 bilinear scored
R² −0.82 at n=3k, −0.50 at 12k, **+0.999 at 50k**. At Qwen3's `d=4096` a rank-32 bilinear has
262,144 parameters and would need ~370 documents. So C2 is posed as **parameter-matched small
pairs** (`--probe-ranks 2,4`: rank-2 bilinear = 16,384 params ≈ MLP width-4 = 16,393), which are
answerable with ~6–12 documents at 32K. The script flags any probe that is underdetermined; **a
negative R² under that flag means "not enough data", not "wrong function class"**.

---

## 4. The co-design being considered

> **Partly refuted by §9.** The `W_z·norm(S̄_{t-1} k_t)` term in the router — and the stronger idea
> that the router's score should *be* the compensation residual — failed its own shuffle control
> and lost to unlimited-capacity ceilings. **Drop the router's `z` input.** The compensation branch
> below survives (its basis is the SPLA identity in §1, not this coupling); build it with undecayed
> `S̄` and **no** kept-set coupling, which §9.2 measured as actively harmful.

```
one recurrence:  S̄_t = S̄_{t-1} + φ(k_t)ᵀ v_t                      (all tokens, α = 0)
router reads:    s_t = w_out·φ(W_h·norm(h_t) + W_z·norm(S̄_{t-1} k_t)) + t·ε
compensation:    o_lin = φ(q)(S̄ − S_kept)                          (evicted-only, by linearity)
shared:          φ, W_k, W_v shared with the sparse branch          (SPLA §3.2)
```

Two roles need **different contents**, so one state cannot serve both: the router's `z` must
summarize *all* history (it judges each token before knowing its fate), while the compensation
state must hold *exactly* the evicted keys (otherwise kept keys are counted twice — once exactly
in the sparse branch, once approximately in the linear one). They share *structure*, not content:
`S̄` is needed for compensation anyway, so the router gets `z` for free.

Cost: `d_s=128` costs 1.0 MiB/layer against 128 MiB/layer of KV freed at 50% eviction.
`--probe-ranks`-scale numbers aside, the compensation branch is essentially free.

Two things that will bite:
- **Double counting.** Assert `S̄ = S_kept + S_evicted` exactly. If `S_kept` drifts from the real
  retained set, the loss still falls (the model adapts) but training and inference diverge
  silently — the `key_offset` failure mode again.
- **Pinning.** `pin_mode="sink"` is required today: an unpinned flat gate is a no-op (distance
  5.6e-17 vs 0.44 with the sink pin) and the router can satisfy the LM loss by reverting to the
  frozen dense backbone (SAS ablation: 18.8 vs 54.4). Adding a linear branch **may widen that
  hole** — the router gains a second escape route (let the linear branch carry everything).
  Re-measure the no-op distance after wiring it. Note also that Fast-KVzip's sink margin and
  `pin_mode="sink"` use the same tokens; whether they interfere is untested.

---

## 5. Experiments to run

> **Revised by §8.5.** C1–C4 have been run: E3, the bilinear change and Arm C are **cancelled**,
> and E2 is promoted above E1. E0 is still a blocker for E1. The descriptions below are unedited
> so the original reasoning stays legible next to what the data said.

### E0 (blocker) — teach the eviction eval to load a scalar checkpoint

`evaluation/evaluate_indexer_press.py` hardcodes a pairwise indexer (passes `head_dim`/`rope_dim`,
no `scorer=`). The two scorers share a parameter *prefix* but no weight names:

| | weights |
|---|---|
| scalar ckpt | `in_norm`, `w_in`, `mid_norm`, `w_out` |
| pairwise expected | `w_q`, `w_k`, `q_norm`, `k_norm` |

So a scalar checkpoint fails to load today. `evaluation/evaluate_sparse.py` already has the
machinery — `detect_scorer`, `infer_scalar_mid_dim`, and the `scalar_pos_slope` handling. Port it.
**Port `scalar_pos_slope` carefully: it is not a parameter and is never stored in the weights, so a
wrong value mis-scores silently with every weight loading cleanly.**

Until E0 lands, **the scalar arm has never been evaluated on the task it was designed for.**

### E1 (highest value) — scalar eviction vs baselines

With E0 done, run `evaluate_indexer_press.py` on the scalar checkpoint at matched compression, and
compare against the `results_dense` baseline and kvpress's own query-independent evictors
(SnapKV / H2O). Pairwise is **not** a valid comparison here — it cannot evict. `chunk_size`
(0/16/64) is already a CLI flag on this path and costs nothing to sweep; it is the natural fix if
the failure mode is truncated spans.

Note on the earlier RULER failure pattern, which E1 should re-examine: the scalar arm's errors were
**not** wrong retrieval but *truncated* retrieval — `niah_single_3` returned UUIDs with characters
missing (90.5% near-miss vs 9.5% exact, against pairwise's 100% exact). A per-key score ranks the
tokens of one entity independently, so a top-k boundary can cut through it. That is what
`chunk_size` addresses, and on the eviction path it needs no kernel change.

### E2 — `--head-dim 8` pairwise

One training run. Keeps query-awareness, cuts decode 17.1× → 1.1×. May moot the whole
query-independent direction if the target length ceiling is finite.

### E3 — `matched` (parameter-matched scalar)

`scripts/train_gqa_indexer_scalar_gy.sh matched` already exists (`MID_DIM=1152` → 4.73 M vs
pairwise's 4.72 M). This is the only run that separates "query-dependence" from "capacity" in the
existing A/B. Worth it mainly if C2 says capacity is *not* saturated.

### C1–C4 — `proxy_exp/diag_real_target.py`

This is the script to run first; it is cheap and it decides whether E3 / the bilinear change / `z`
are worth doing at all.

```bash
# C1 + C4 only (no probe fitting, fast)
python proxy_exp/diag_real_target.py \
  --model /apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B \
  --tokenized /apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k \
  --seq-len 8192 --n-docs 8 --skip-fit --out proxy_exp/real_target_8k_attn.json

# full run, all four claims (probe pairs need documents -- see below)
python proxy_exp/diag_real_target.py \
  --model /apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B \
  --tokenized /apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k \
  --seq-len 32768 --n-docs 12 --probe-ranks 2,4 \
  --out proxy_exp/real_target_32k.json
```

Start with `--seq-len 8192 --n-docs 2 --layers 0,17,35` to confirm it runs and to time one layer,
then scale. Rough sizing: probe rows = `n_docs × seq_len × cut_frac`; the parameter-matched pairs
need ≳65k training rows, so **≈6 docs at 32K or ≈23 at 8K**. The script prints an
`(UNDERDETERMINED)` flag and a warning naming the worst offender when this is not met — clear that
warning before reading C2 or C3.

**How to read the output**
- **C1**: if `ov_prefillmean_future` ≫ `ov_prefillsum_future` (synthetic: 10.2% → 92.2%), the gap
  is a normalization artifact and there is nothing to redesign. The script prints an explicit
  verdict line. Also watch `corr_prefillsum_position` — synthetically it was 0.87, i.e. the prefill
  sum is mostly position.
- **C2**: compare **equal-parameter** rows only (e.g. `bilinear 16,384` vs `mlp 16,393`). The
  `mid=256` rows are an upper-capacity reference and will be flagged until they have ~370 docs.
- **C3**: `h+z` must beat `h+z_shuffled` (same width, alignment destroyed) for `z` to be real
  signal. Compare against `h+pos`: synthetically position alone beat `h+z`, and position is
  already available via `pos_slope`.
- **C4**: high `ov_futuresum_futuredamage` means mass is a fine proxy and value norms can be
  ignored.

### Not yet planned
- Arm C (`z`-augmented scalar router) — gated on C3.
- The bilinear score change — gated on C2.
- Incremental top-k heap for real `O(1)` decode (SparseK Algorithm 2, arXiv 2406.16747) — the
  `O(L)` decode claim is currently theoretical.

---

## 6. `diag_real_target.py` — verification status

Verified on this CPU box (synthetic/tiny-model, since that is all that was available):

- `layer_key_stats` matches a brute-force reference to **0.0 / 3.7e-08 / 1.0e-07**
  (prefill_sum / future_sum / future_damage), and is invariant to `query_tile ∈ {1,5,32,64,128}`.
- **`collect_layer` reproduces the real layer's attention output to 1.6e-08**, so the q/k norm and
  RoPE wiring is faithful rather than reimplemented. This is the single most important check —
  a diagnostic that silently measures a different model is worse than none.
- Probe positive control (target *is* a low-rank quadratic, sufficient n): bilinear R² 0.999.
  Negative control (random target): R² ≈ 0, top-25% ≈ 25%. Both as expected.
- **A leakage bug was found and fixed here.** Pooling documents then splitting 70/30 by row index
  lands mid-document; adjacent tokens are correlated and the target is smooth in position, so
  every probe scored R² = 1.000. The split is now **by document** (`groups=`). If you see R²
  ≈ 1.000 across the board again, suspect the split first.
- Memory: uses `AutoModel` (not `...ForCausalLM`, saving ~28 GiB of unused 152K-wide logits) and
  never materializes an `L × L` tensor. An earlier version of this diagnostic OOMed via
  `output_attentions=True`, which retains `(layers, heads, L, L)` = 144 GiB at L=8192.

**Now verified on the real model too** (§8). The first real run was indeed a debugging run: it
exposed the pooled-metric bug in §8.6 — every R² came back negative, including for probes that
were not underdetermined, while rank agreement was fine. The tiny-model smoke test could not have
caught it, because with one synthetic document there is no cross-document scale drift to expose.
The leakage fix noted above held up: nothing scored R² ≈ 1.000.

---

## 7. Files

| path | what |
|---|---|
| `proxy_exp/diag_real_target.py` | **C1–C4 on a real model.** Run this first. |
| `proxy_exp/diag_eviction_ceiling.py` | **§8's central measurement.** Exact output damage at a budget, oracle-QI vs per-query top-k. `--self-test` checks three identities. |
| `proxy_exp/diag_learned_evictor.py` | trains a router on real `h` and evicts with it on held-out docs — the achievable number |
| `proxy_exp/diag_utility_proxy.py` | earlier: per-key utility from real attention, tiled |
| `proxy_exp/diag_rnn_state_design.py` | earlier: effective rank, redundancy AUC, state saturation |
| `proxy_exp/diag_state_probe.py` | earlier: nested state probes (ridge + MLP readouts) |
| `proxy_exp/*.json` | outputs; `ceiling_{8k,32k}.json`, `real_target_*`, `learned_evictor_*` are §8's |
| `scripts/taiji_exec.sh` | run a command in the Taiji pod (`TASK=`/`INSTANCE=` to override). Written but **never successfully executed** — the local classifier blocked every `taiji_client` call. |
| `kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md` | why a router can or cannot learn; the pin/no-op analysis |
| `kvpress/presses/gqa_indexer/scalar_indexer.py` | the scalar arm |
| `tests/presses/test_gqa_indexer_scalar.py` | 32 tests incl. irreversibility and the pin hole |
| `scripts/train_gqa_indexer_scalar_gy.sh` | training launcher (`smoke\|stage1_16k\|matched\|linear\|ablate\|stage1\|stage2`) |

Papers referenced, all read in full rather than from memory: Titans 2501.00663 · Memory Caching
2602.24281 (SSC is a router over cached states — closer to this work than expected, worth reading)
· SPLA 2601.22379 · SparseK 2406.16747 · PISA 2602.01077 (block-wise Taylor, *not* a linear branch;
low relevance).

---

## 8. Real-model results (2026-08-18, Qwen3-8B on 4×H20)

Everything in §2 has now been run on the real model, plus two measurements §2 could not make.
Setup: `Qwen3-8B` (36 layers, 8 KV heads), `longmino_tokenized_64k`, layers `{0,7,14,21,28,35}`,
`cut = seq_len/2`, sinks pinned = 4. Reported over 384 (doc × layer × head) cells at each length.

**Four of the five synthetic claims reversed. That is five-for-five in this line of work — the
next synthetic result should be treated as a hypothesis generator only.**

| claim | synthetic | real | verdict |
|---|---|---|---|
| C1 prefill/future gap is a normalization artifact | 10.2% → 92.2% | **44.4% → 60.8%** (8K), 42.9% → 55.8% (32K) | **REVERSED** — real gap |
| C1b prefill sum is mostly position | corr 0.87 | **0.077** (8K), 0.057 (32K) | **REVERSED** |
| C2 bilinear beats MLP per parameter | R² 0.9966 vs 0.980 | top25 **34.9% vs 37.1%**; eviction damage 0.0364 vs 0.0356 | **REVERSED** — MLP ≥ bilinear |
| C2b `mid_dim` is saturated | ΔR² 0.001 | **linear 0.0360 ≈ bilinear 0.0364 ≈ MLP 0.0356** | **CONFIRMED**, and much more strongly |
| C3 `z` helps beyond position | h 71.1 → h+z 89.1, h+pos 99.2 | h 37.1% → **h+z 39.8%, h+z_shuffled 38.5%, h+pos 40.7%** | **REVERSED** — `z` ≈ its own control |
| C4 value geometry is negligible | 94.5% | **93.7%** (8K), 94.5% (32K) | **CONFIRMED** |

### 8.1 The measurement that changes the plan: query-independence has a hard floor

§0 asked "how does scalar do on eviction". `diag_eviction_ceiling.py` answers a stronger version
without training anything: give a query-independent scorer **the true future demand** — the thing
no causal router can see — and measure exact output damage at a fixed budget. That is an upper
bound on *every* query-independent router at *any* parameter count.

Damage = relative L2 on future-query outputs, recomputed with evicted keys removed and the
softmax renormalized (what inference does), so it is not a proxy.

| keep | oracle_qd (per-query top-k) | **oracle_qi (QI ceiling)** | prefill_mean | recency | random |
|---|---|---|---|---|---|
| 8K, 50% | 0.0020 | **0.0122 (6.1×)** | 0.0195 | 0.0594 | 0.0511 |
| 8K, 25% | 0.0066 | **0.0278 (4.2×)** | 0.0394 | 0.0837 | 0.0824 |
| 8K, 10% | 0.0156 | **0.0468 (3.0×)** | 0.0618 | 0.0966 | 0.1024 |
| 32K, 50% | 0.0009 | **0.0098 (10.4×)** | 0.0186 | 0.0593 | 0.0517 |
| 32K, 25% | 0.0035 | **0.0226 (6.4×)** | 0.0353 | 0.0918 | 0.0835 |
| 32K, 10% | 0.0091 | **0.0382 (4.2×)** | 0.0510 | 0.1051 | 0.1090 |

**A perfect, clairvoyant query-independent evictor still costs 3–10× the damage of per-query
top-k at the same budget, and the penalty grows with context length** (4.2× → 6.4× → 10.4× at
32K vs 3.0× → 4.2× → 6.1× at 8K, for 10/25/50% budgets). Query-awareness is not a nicety the
scalar arm was unlucky to lose; it is most of the achievable quality, and it matters *more* at
the lengths this project targets.

This reframes §0's −20.8 / −29.5 RULER deltas. They were read as possibly a
capacity/query-awareness confound. The ceiling says a large part of that gap is **structural**
and no scalar router — matched parameters or not — can recover it.

### 8.2 Capacity is saturated; the features are the ceiling

`diag_learned_evictor.py` trains a router on real `h_j` (held-in docs) and evicts with it on
**held-out** docs, so the number is achievable rather than an oracle. 8K, 12 docs, 7 train / 5 test:

| scorer | params | rel_L2 @keep 25% | band recovered |
|---|---|---|---|
| linear | 4,097 | 0.0360 | 73% |
| bilinear rank-8 | 65,536 | 0.0364 | 72% |
| MLP mid=256 | 1,049,857 | **0.0356** | 74% |
| — oracle_qi | — | 0.0247 | 100% |
| — recency | — | 0.0670 | 0% |

**A 4k-parameter linear score matches a 1M-parameter MLP to within noise (0.0360 vs 0.0356).**
Two consequences:
- **E3 (parameter-matched scalar, `MID_DIM=1152`) is not worth running.** 256× more parameters
  bought nothing; matching pairwise's 4.72 M will not either. The A/B's parameter mismatch was
  never the explanation.
- **The bilinear score change is not worth making.** It loses on rank agreement (34.9% vs 37.1%)
  and ties on damage at 16× the parameters. C2's synthetic win came from fitting a *synthetic
  quadratic*; the real target does not have that structure.

The trained router recovers 74% of the oracle→recency band and reaches 52.0% top-25% overlap
with the oracle (chance 25%) — it learns real signal, then stops, because `h_j` alone does not
contain more. Note `prefill_mean` (0.0341) **beats** the trained `h`-only router (0.0356): a
feature built from observed attention outperforms anything read off the hidden state, which is
the same "features, not capacity" conclusion from the other side.

### 8.3 C3: `z` does not survive its own control

`h+z` 39.8% vs `h+z_shuffled` 38.5% top-25% — a 1.3-point gap, where shuffling destroys `z`'s
alignment with its token while keeping the width. `h+pos` (40.7%) beats both, and bare `pos_only`
reaches 39.0% with **9 parameters**. So the running state is very close to a position feature
with extra steps, and position is already available via `pos_slope`.

**Arm C (`z`-augmented router) should not be built on this evidence.** This does *not* touch §4's
compensation branch, whose justification is the SPLA subtraction identity (exact, §1), not `z`.

### 8.4 C1: the prefill/future gap is real, and worse at length

Mean-normalizing moves top-25% overlap 44.4% → 60.8% at 8K, not 10.2% → 92.2%. It helps in 87.2%
of cells at 8K but only 68.8% at 32K (median gain +0.161 → +0.119), and only 10–12 of 48
(layer, head) pairs clear 65%. The position-ramp explanation also fails directly:
`corr_prefillsum_position` is **0.077**, not 0.87. Worth noting the prior self-reversal in §2 was
itself the wrong correction — the sum-vs-mean fix is real but small, not the whole gap.

C4 is confirmed (93.7% / 94.5%, min 0.799 across all cells): mass is a fine proxy, value norms
can be ignored. `oracle_qi_dmg` beats `oracle_qi` by <1% of damage, so ranking by exact damage
instead of mass buys nothing.

### 8.5 What this implies for §5's experiment list

- **E2 (`--head-dim 8` pairwise) is now the highest-value experiment**, promoted over E1. It keeps
  query-awareness — which §8.1 shows is 3–10× and growing — at 1.1× attention decode cost. The
  ceiling result is what makes this the main line rather than a cheap side-quest.
- **E1 (scalar eviction vs SnapKV/H2O) is still worth running**, but as *characterization*, not as
  a hope. E0 (the checkpoint-loading blocker) is still required for it. Expect the scalar arm to
  land near `recency`-to-`oracle_qi`, i.e. clearly worse than sparse pairwise, and interpret it
  against the §8.1 band rather than against pairwise.
- **E3 is cancelled** (§8.2). **Bilinear is cancelled** (§8.2). **Arm C is cancelled** (§8.3).
- The `chunk_size` sweep from §5/E1 is untouched by these results and remains the right response
  to the truncated-retrieval failure mode; §8's per-key metrics cannot see span truncation.
- If query-independent eviction is kept anyway (for the memory freeing it uniquely allows), §4's
  compensation branch becomes the load-bearing part rather than an optimization — it is what
  would have to repair a floor that §8.1 says is 3–10× off per-query top-k.

### 8.6 Two instrument bugs found and fixed (both would have inverted a conclusion)

Recorded because the first one nearly produced a fifth wrong conclusion, and the failure shape is
the same normalization error as `reduce_queries` and the sum-vs-mean episode in §2.

1. **`fit_probe` pooled documents when computing `r2` / `top25`.** The first real probe run
   returned **every R² negative**, including a 4,097-parameter linear probe on 196k rows that was
   not underdetermined — while per-layer `corr` sat at 0.35–0.93 and `pos_only` scored *positive*.
   Cause: target attention scale drifts between documents, and R² was taken against the *train*
   mean, so a held-out document with a different level scored negative despite good ranking. Fixed
   to per-document metrics against each document's own mean (an eviction budget is spent within
   one sequence, so that is also the operationally correct unit). Verified on a positive control
   (target genuinely a low-rank quadratic): no drift → R² 0.851 / top25 82.5%; with per-document
   scale+offset drift → R² −0.31 while top25 held at 75.8%. **Read `top25` / `corr` / `banded`
   first; they are scale-invariant. R² < 0 with top25 ≫ 25% means "scale not predicted", not "no
   signal".** `banded_corr` was striding across document boundaries too, and now respects groups.
2. **`diag_eviction_ceiling.py` printed retained mass under the label `dropped_mass`**
   (`masked_fill(drop, 0)` keeps the survivors). It made the oracle look like it discarded *more*
   mass than random while taking 4× less damage. `rel_l2` was unaffected — reporting only — but
   the column now has a self-test asserting its orientation.

`diag_eviction_ceiling.py --self-test` verifies three identities, all passing: budget=100% gives
exactly 0 damage; the mass column is oriented correctly; and a single-key drop reproduces §1's
`A_ij (o_i − v_j)/(1 − A_ij)` to **2.1e-08** by an independent path (renormalize-and-recompute vs
the closed form), which is what makes the damage numbers above trustworthy.

**Cost/runtime, for planning:** the attention pass is ~10 s/(doc × layer) at 8K and ~90 s at 32K;
probe fitting is ~5 min/layer at n≈10⁵. A full C1–C4 run at 8K/24 docs/4 layers took ~25 min on
one H20. All of §8 is ~3 GPU-hours total.

---

## 9. ①: does causally-detectable redundancy exist? — NO USABLE SIGNAL (2026-08-18)

Tests the one unverified link in §4's co-design, in its strong form: *the router's score should be
the compensation branch's residual* — score by `r_t = v_t − S_kept_{t-1} φ(k_t)`, i.e. "evict what
a linear state can already reconstruct, keep what it cannot." Script `proxy_exp/diag_redundancy.py`,
data `redundancy_8k.json` (8 docs × 6 layers), `redundancy_32k_pooled.json` (4 seeds × 2 docs × 3
layers), `redundancy_8k_{chunksweep,exists}.json`. Damage is §8's exact renormalized `rel_l2`, so
the numbers are directly comparable to §8's table.

**Verdict: not supported. `score = f(h, r)` with `r` a reconstruction residual is abandoned.**
Independently re-verified from the raw JSON (paired per-cell, not just the summary).

### 9.1 Redundancy is present in the data but is not what makes a key cheap to drop

Near-duplicate keys genuinely exist: `1 − max cos(k_t, earlier k)` p01 = 0.064 (8K) / 0.056 (32K);
1-NN value residual p01 = 0.28 / 0.21. But **every redundancy score sits at chance on future
demand** (top-25 overlap 0.264 for `res_Skept`, 0.259 for `nn_res`; chance 0.25).

Eviction damage at keep 25%, 8K (recency 0.0859 → oracle_qi 0.0280 is the band):

| arm | rel_L2 | band recovered |
|---|---|---|
| `prefill_mean` | 0.0409 | 77.7% |
| §8's trained h-only router | 0.0356 | 74% |
| `res_Skept_rel` (best residual variant) | 0.0753 | 18.2% |
| `res_Sbar` | 0.0770 | 15.3% |
| **`res_Skept`** (the design) | **0.0841** | **3.0%** |
| `vnorm` (‖v‖ control) | 0.0895 | −6.2% |
| `nn_res` (exact 1-NN, unlimited capacity) | 0.0917 | −10.1% |
| `nn_novelty` | 0.0993 | −16% |

**The deciding test is the shuffle control**: `res_Skept` vs `res_Skept_shuf` (state↔token
alignment destroyed, scale and form preserved) is geo-ratio **1.027, residual wins 16/48 cells** —
i.e. *worse* than its own control, and indistinguishable from chance. It also fails against `‖v‖`
in the only meaningful sense: `‖v‖` is itself worse than random, so beating it is not evidence.
This is now **two independent state-based features that failed their own control** (`z` in §8.3,
the residual here).

**Capacity is not the excuse.** The unlimited-capacity ceilings are *worse than random* (`nn_res`
1.068×, `nn_novelty` 1.157×, `res_ridge` 1.109×). The state is actively harmful rather than weak:
`explained_frac` = **−0.217** for the delta rule (the residual is *larger* than `‖v‖`), and the
*optimal* closed-form causal ridge reaches only **+0.011** on average (per layer: +0.086, −0.068,
+0.091, +0.089, −0.025, −0.104). So a linear state cannot predict `v_t` from history at all — this
refutes the hypothesis, not the estimator.

### 9.2 The kept-set coupling has negative value

`S̄` (all keys, α=0, no coupling, no serial dependency) **beats** `S_kept` — geo-ratio 0.952 (8K)
/ 0.815 (32K), both significant. The expensive part of the proposed design, the part that forces
per-chunk serialization, is also the part that hurts. If a state is built at all, build undecayed
`S̄` — which is what §4's compensation branch needs anyway.

### 9.3 `chunk_size` on this path buys only recency

top-25 rises 0.248 → 0.366 for chunk 32 → 2048, but position correlation rises in lockstep
(−0.04 → +0.27): a 2048-token chunk scores its first 2048 keys against an empty state, which *is*
recency. Not evidence for chunking the score. (Orthogonal to §5/E1's `chunk_size`, whose purpose is
span truncation and which remains untested.)

### 9.4 Methodological notes worth keeping

- **A capacity confound was found in the instrument and designed around.** A `D`-wide delta-rule
  state loses a *verbatim* duplicate once `n/D > 1` (precision 100% → 72% → 49% at n/D =
  0.12/1.0/2.0) and Qwen3 runs at `n/D = 4096/128 = 32`. Without the added `res_ridge` (optimal
  causal least-squares) and `nn_res` (exact 1-NN, no state) arms, the null would have been
  uninterpretable — "state too small" vs "hypothesis wrong". They show it is the hypothesis.
- **Per-cell geometric ratios, not paired differences.** Damage varies ~7× across layers, and the
  paired *difference* claimed the residual beat its shuffle at 32K (t = −2.75) while the same cells
  split 8/24 the other way — the entire effect was eight L28 cells with 5× the absolute damage.
  Same scale-sensitivity trap as §8.6's R² bug. **Third instance in this project; treat any
  unnormalized aggregate over layers as suspect by default.**
- 7-check self-test passes, including constructed-redundancy detection (97% precision), the
  `β=1` post-update-residual-is-zero identity (2.4e-07, HANDOFF §1), the `‖k‖=1` divergence guard,
  and a pre-RoPE key path reproducing `collect_layer` to 0.0e+00.

### 9.5 What this does and does not kill

- **Killed:** router-reads-the-residual, in all variants tested (`‖r‖`, `‖r‖/‖v‖`, sign-flipped,
  pre-RoPE, ridge, ReLU-lift, 1-NN). Also killed: the claim that "score = compensation residual"
  makes the two components one quantity.
- **NOT killed:** §4's compensation branch itself, whose justification is the SPLA subtraction
  identity (exact, §1) and not this coupling. If built: undecayed `S̄`, no coupling.
- **NOT tested:** set-aware selection in general. §9 refutes *one specific* set-aware signal
  (linear reconstructability). Greedy set selection against the true damage — the ② experiment —
  is still open and is the remaining way to learn whether `oracle_qi` can be beaten at all.
- `prefill_mean` still beats every cheap causal feature by 1.9–2.6×, which keeps pointing the same
  way as §8.2: **features/targets, not architecture.**
- **§8.5's promotion of E2 (`--head-dim 8`) now stands unopposed.**

---

## 10. ADVERSARIAL AUDIT of §8/§9 (2026-08-18) — four confirmed bugs, one headline reversed

Four independent adversarial auditors were run against §8/§9 (`proxy_exp/audit_*.py`,
`audit_*.json`). Task: *find reasons the results are wrong*. They found four real bugs plus two
framing errors. **Each finding below was re-verified directly from the raw JSON before being
recorded here** — the §9.4 lesson applied to the audit itself.

### 10.1 CONFIRMED BUG — `recency` keeps the OLDEST keys (label inverted)

`rec = -torch.arange(cut)` + top-k selects positions `[0..keep-1]`, i.e. the **prefix**, while the
docstring says "keep the last k". Verified directly: budget 6 of 16 keeps `[0,1,2,3,4,5]`, not
`[10..15]`. Present in `diag_eviction_ceiling.py`, `diag_learned_evictor.py`, and `diag_redundancy.py`.

It is a *weaker* control than true recency, so the oracle→recency band is overstated and every
"% of band recovered" figure is inflated: **§8.2's "recovers 74%" becomes ~65%**. §8.1's `recency`
row is mislabeled (it is "keep the prefix"). Note it is exactly pin-invariant because it already
keeps position 0 — a second tell.

### 10.2 CONFIRMED BUG — GQA group-sum corrupts `future_damage` (C4's metric)

`A` is summed over the 4 query heads *before* the `A/(1−A)` damage factor, so entries lie in [0,4].
Where `A > 1` the denominator goes negative and `clamp_min(1e-6)` multiplies by ~1e6. Measured on
the real model: only **0.014–0.02%** of entries exceed 1, but they contribute **91–100% of all
reported `future_damage`**. Also `o = einsum(A, v)` is ~4 head outputs added (`‖o_grp‖/‖o_head‖` =
3.49), so `‖o − v_j‖` is fictitious. Same bug in `diag_real_target.py:195` and
`diag_eviction_ceiling.py`.

Against **exact single-drop ground truth**, the correctly-aggregated per-head damage scores
1.0000/1.0000 (it *is* the first-order expansion), which validates the reference. Against it:
mass recovers **88.7%** of the top-25% set, not the published 93.7%. My own independent re-run:
C4 overlap **0.940 → 0.898** when fixed.
**C4's verdict survives** ("mass is a fine proxy"; `oracle_qi_dmg/oracle_qi` = 0.995/0.980/0.962,
so exact damage still buys <4%) **but the published number measured a clamp artifact.**

### 10.3 CONFIRMED — §8.1's length-scaling claim is BACKWARDS at matched absolute budget

§8.1 compared 8K vs 32K at matched `keep_frac`, which at `cut = L/2` is not a matched operating
point: `keep_frac 0.25` is 1024 keys at 8K but 4096 at 32K. **Memory budgets are absolute, and the
RULER A/B that motivates this whole comparison used `topk=2048`.** Re-verified on paired cells
(8K docs are exact prefixes of the 32K docs), 6 paired (doc, layer) cells:

| budget | 8K qi/qd | 32K qi/qd | trend |
|---|---|---|---|
| keep=512 | 2.84 | 2.43 | **shrinks (0.85)** |
| keep=1024 | 3.79 | 3.00 | **shrinks (0.79)** |
| keep=2048 | 5.42 | 3.79 | **shrinks (0.70)** |
| keep_frac 10% | 2.62 | 3.50 | grows (1.34) |
| keep_frac 25% | 3.79 | 5.01 | grows (1.32) |
| keep_frac 50% | 5.42 | 7.40 | grows (1.36) |

And the QI evictor's **absolute** damage *improves* at 32K at matched fraction (32K/8K =
0.724/0.679/0.611). The ratio grew only because `oracle_qd` improved faster. **Nothing gets worse
with length.** §8.1's "the penalty grows with context length" and §8.5's "3–10× and growing" are
**withdrawn** — under the absolute-budget convention the penalty *shrinks*.

Related (finding 2 of the ceiling audit): the ratio also grows across budgets purely because the
two arms have different budget-response slopes (log-log −1.34 vs −0.83), and at keep 50% both arms
are nearly lossless (1.2% vs 0.2% error) — the least meaningful cell, not the most alarming.
**Fourth instance of the §9.4 scale trap.**

### 10.4 CONFIRMED — `oracle_qd` is not a same-budget reference, and gets an illegal freedom

- **Memory:** `oracle_qd`'s union over future queries covers **99.5%/100%/100%** of pre-cut keys.
  At a "10% budget" it retains ~100% of the cache. Compute-budget vs memory-budget, as suspected.
- **Kernel-illegal:** it picks top-k per *query head* (32 sets/query). The real path
  (`sparse_attention.py`, `press.py:398`) allows one set per **KV head**. Adding the GQA-legal arm
  `oracle_qd_gqa`: ratio drops **3.38/4.83/7.09 → 2.57/3.36/4.52**, and at matched absolute budget
  it is **1.9–3.4×**.

**Corrected headline:** query-independence costs ≈**2–3.4×** against a GQA-legal per-query
reference at matched absolute budget, **shrinking** with length — not "3–10× and growing".
Query-awareness still wins, so §8.5's direction survives, but the magnitude was overstated ~2–3×.

### 10.5 CONFIRMED — `oracle_qi` is NOT a ceiling (§8.1's strongest claim is false)

§8.1 called it "an upper bound on **every** query-independent router at **any** parameter count".
It is top-k by true future mass — a *heuristic* on the true target, not the optimal fixed set.
Greedy local search against true damage beats it in every cell tested: **0.0259 → 0.0246**,
per-cell geo **0.952**, up to **−10.7%** at L0, touching ≤5% of the set; headline 3.78 → 3.59.

**This partially re-opens the strategic conclusion.** A better *set*-selecting query-independent
router can beat `oracle_qi`, so ② (greedy set selection) is a live way to close part of the gap
rather than a formality — and §9 refuted only *one* set-aware signal (linear reconstructability),
not set-awareness itself.

### 10.6 CONFIRMED — C1's headline is ~61% an `n_seen → 1` edge artifact

`prefill_mean = prefill_sum / n_seen` with `n_seen[cut−1] = 1`: the last key's "mean" is a
single-query estimate, and that query is the **diagonal** `A[j,j]` (self-attention, systematically
large — 31.3% of `prefill_sum` in the last 64 positions vs 14.4% in the first 64). So
`prefill_mean` mechanically promotes recent keys: it places **71.5%** of its top-25% in the last
quarter of positions, against the target's 49.2%.

Restricting to keys with ≥1024 observations: the C1 gain falls from **+16.4 pts (44.4%→60.8%)** to
**+7.6 pts (51.3%→58.9%)**. §8.4's C1 headline must be halved and labeled partly recency.
(The `n_seen` formula itself is correct — brute-forced exact at cut=8/64/4096. No off-by-one.)

### 10.7 C2 is UNRESOLVED (was reported as REVERSED)

Two structural problems:
- **Not parameter-matched.** `real_target_8k_probe.json` pools three widths per `(features, kind)`
  row and labels it with the *first* width's count. The published "34.9% vs 37.1%" averages a 16k,
  a 33k and a 262k/1.05M probe. At genuinely matched params: **36.4% vs 36.9%** (~16k) and
  **35.5% vs 36.7%** (~33k), each with a sign flip across layers — a **tie**, not "MLP ≥ bilinear".
- **29/39 probes are undertrained** by the code's own guard, including every configuration quoted.
  Bilinear degrades monotonically with capacity (36.4→35.5→32.9%; R² +0.764→+0.672→−2.541) — the
  signature of data starvation, not saturation. All probes ran at a single fixed budget (600
  epochs, lr 3e-3) and no `train_mse` was saved, so convergence cannot be separated from capacity.
  **The convergence sweep was never run.**

**C2b survives** — it rests on the *eviction* experiment (`learned_evictor_8k*.json`, held-out
docs), which is clean and is the load-bearing evidence for cancelling bilinear. So **the decision
to cancel bilinear stands; the probe-based justification for it does not.**

### 10.8 C3's verdict is right, and stronger than stated (but its stated number is wrong)

Same width-pooling bug. At matched width, `h+z` vs `h+z_shuffled` pooled over 12 (width, layer)
pairs: mean delta **+0.0132, se 0.0168, t = 0.79**, sign test **p = 0.388** — *indistinguishable
from zero*, and at L14 the shuffled control wins at every width. `pos_only` (9 params) scores
**47.1%** correctly grouped, beating every `h+z` config including the 2.1M-parameter one (41.3%).
**"Do not build Arm C" is better supported than §8.3 claimed.**

### 10.9 REFUTED suspicions (these are clean)

- **`collect_layer` fidelity on the real model**: rebuilt attention output matches `o_proj`'s real
  input to **≤1.1e-06**, and the q/k rebuild is **bit-exact (0.0e+00)**. The instrument reads the
  real model. Most important negative result here.
- **`query_tile` invariance on the damage path**: ≤6.1e-16 across tiles 128/256/512; headline
  identical to 3 decimals. The 8K-vs-32K tile difference is not a confound.
- **`pin_sink`**: headline unchanged (3.38→3.38, 4.83→4.84, 7.09→7.10) at pin 0 vs 4. Only the weak
  controls depend on it.
- **`topk_overlap` ties**: zero ties in real data (`ties_at_thr` = 1 in all 128 cells;
  random-tiebreak identical to 0.0). Real risk in principle, absent here.
- **`n_seen` off-by-one**: none.
- **Document independence**: 0 EOS per doc, pairwise Jaccard max 0.0033, 0 duplicate pairs, 24
  distinct docs. §8.2's held-out split is genuine.
- **Budget accounting**: all arms keep exactly `keep` keys per query (asserted).

### 10.10 §8.6's R² "fix" is only half-applied

The numerator still uses the **train** mean while the denominator uses the document's own mean
(`diag_real_target.py:341-342`), so a perfect predictor of a document's centered target still
scores R² < 0 under an offset (demonstrated: δ=1.0 → R² −0.006; δ=2.0 → −3.017, vs +1.000
offset-free). Only 3 of 148 rows changed sign. **All quoted C2/C3 numbers are `top25`**, which is
computed within-document and offset-invariant, so no verdict changes — but §8.6's claim to have
fixed the R² is false and that column remains uninterpretable.

### 10.11 Net effect on the plan

| §8/§9 conclusion | status after audit |
|---|---|
| C4 mass is a fine proxy | **holds** (margin 88.7%, not 93.7%) |
| C3 don't build Arm C | **holds, stronger** |
| C2b capacity saturated → cancel E3 + bilinear | **holds** (eviction evidence is clean) |
| §9 residual router refuted | **holds** (pending audit #3, running) |
| C1 mean-norm gain | **halved** (+7.6 pts, partly recency) |
| C1b "not mostly position" | **UNRESOLVED** — Pearson-vs-linear is the wrong statistic; a pure harmonic ramp scores 0.867 and a ramp with 4 sinks scores 0.100, so 0.077 is consistent with 100% position. Rank correlation never run. |
| C2 "MLP ≥ bilinear" (probe) | **UNRESOLVED** (unmatched params + undertrained) |
| §8.1 "3–10×" | **2–3.4×** GQA-legal, matched absolute budget |
| §8.1 "grows with length" | **WITHDRAWN — shrinks** at absolute budget |
| §8.1 `oracle_qi` is a hard ceiling | **FALSE** — beatable by ≥5% |
| §8.2 "recovers 74% of band" | **~65%** |

**E2 still wins on direction but by 2–3×, not 10×, and its "growing with length" rationale is
gone.** The audit's most consequential result is 10.5: since `oracle_qi` is beatable, ② (greedy set
selection) is now a substantive experiment rather than a formality.

**Still unmeasured and larger than anything above:** all damage numbers are **single-layer**
(one layer evicted, 35 intact), and **rel_L2 is uncalibrated** to LM loss or task score.

---

## 11. VALIDITY AUDIT — the two threats §8/§9 could not see (2026-08-18)

`proxy_exp/audit_validity.py` (+ `audit_multilayer_*.json`, `audit_queries_8k.json`,
`audit_cutfrac_8k.json`, `audit_layers_8k.json`, `audit_decode_8k.json`, `audit_docs.json`).
This is about **proxy validity**, not arithmetic: §10 checked whether the code computes what it
claims; §11 checks whether that quantity answers the question. Instrument validated first (8
self-tests: custom eager == library eager to 0.0, `collect_all_qkv` == `collect_layer` to 0.0,
single-layer path reproduces §8's `eviction_damage` within 1–6% bf16-vs-fp64 **and preserves the
strategy ranking**, keep-100% == baseline loss exactly).

**All three load-bearing numbers below were re-verified directly from the raw JSON.**

### 11.1 SINGLE-LAYER MEASUREMENT — this is what broke the headline

Every §8/§9 number evicts in **one** layer with the other 35 intact. Evicting all 36
simultaneously on the real forward path (8 docs @8K, 3 @32K):

| | single-layer qi/qd | **all-36-layer qi/qd** |
|---|---|---|
| 8K keep 50% | 7.16× | **1.46×** |
| 8K keep 25% | 4.63× | **1.99×** |
| 8K keep 10% | 3.18× | **2.23×** |
| 32K keep 25% | 6.68× | **2.02×** |

Compounding is **sublinear and differential**: multi/single damage is 2.28× for `oracle_qi` but
6.22× for `recency`. Residual + LayerNorm absorb error, and absorb *more* of the well-placed error,
so the oracle's single-layer advantage largely evaporates.

- **"Grows with context length" fails end-to-end too**: 1.99× (8K) → 2.02× (32K). **Flat.** This is
  now the *second independent* refutation of that claim (§10.3 was the first, via absolute budget).
- **Absolute magnitude collapses**: at keep 25%, all-36-layer eviction with true future demand makes
  LM loss *drop* (ΔNLL −0.0064 nats, 8/8 docs); `oracle_qd` also drops. Both oracles are
  indistinguishable from dense on the actual objective. Only `recency` (+0.0609) and `random`
  (+0.0343) hurt. They separate only at keep 1–2%.
- **Rankings mostly survive** (89–91% pairwise agreement with multi-layer ΔNLL), except
  `prefill_mean` vs `prefill_sum` inverts on 4–5/8 docs. So §8's *comparisons* are usable; its
  *ratio magnitudes* are not.

### 11.2 rel_L2 IS BARELY CORRELATED WITH LOSS at the resolution §8 used

720 cells, identical masks measured both ways. Re-verified from raw JSON:

- **Single-layer: Pearson +0.310, Spearman +0.264.** Within-strategy it is worse — `oracle_qi`
  **−0.073**, `prefill_mean` +0.083. At 32K, pooled Spearman **−0.067**.
- **38% of single-layer cells have NEGATIVE ΔNLL** — dropping keys *improved* loss.
- Multi-layer (both quantities large): Spearman **+0.816** / +0.900 within (doc, budget). So rel_L2
  is monotone in loss **only in the multi-layer regime**.
- Fitted: ΔNLL ≈ 0.0035 · rel_L2^0.59.

Consequence, verified numerically: §8.2's **0.0004** rel_L2 gap ≈ 3.5e-5 nats against a cross-cell
sd of **8.7e-4** nats = **0.04× the noise floor**. So "linear ties MLP" is *correct, and now for a
stated reason* — but **§8.2's bilinear-vs-MLP (0.0364 vs 0.0356) and all of §9.1's residual-variant
ladder (0.0753 / 0.0770 / 0.0841) are unresolvable by this instrument.** Only band-recovery
differences >~10 points are supportable, and band recovery itself needs **±18-point** error bars
(`prefill_mean` at keep 25%: 75.5% single-rel_L2 / 64.6% multi-rel_L2 / 68.1% multi-ΔNLL, sd 17.6).

§9's shuffle control (geo-ratio 1.027, 16/48 cells) remains valid evidence — it is a *sign* test,
not a small-magnitude comparison. **§9's verdict stands; §9.1's rel_L2 ladder does not.**

### 11.3 THE QUERY DISTRIBUTION IS THE LARGEST SINGLE LEVER (2.2×)

Four future-query distributions on identical contexts, keep 25%, 6 docs × 6 layers:

| future queries | oracle_qd | oracle_qi | **qi/qd** | `prefill_mean` band |
|---|---|---|---|---|
| natural (what §8 used) | 0.0059 | 0.0266 | **4.52×** | 78.4% |
| **repeat (KVzip-style)** | 0.0070 | 0.0688 | **9.79×** | 63.1% |
| **quote (needle/RULER-like)** | 0.0060 | 0.0352 | **5.83×** | **40.2%** |
| qa (instruction) | 0.0055 | 0.0255 | 4.61× | 77.4% |

`oracle_qd` is nearly constant (0.0055–0.0070) — **the entire 2.2× swing is the QI ceiling itself.**
Cross-transfer: natural-text demand costs 1.23–1.36× its own oracle on `repeat`/`quote` (35/36
cells worse). Top-25% demand overlap with natural: repeat 0.754, quote 0.815, qa 0.962.

Two direct consequences:
- **§8.1's ceiling is a property of one query distribution**, not intrinsic.
- **RULER is `quote`-like**, where `prefill_mean` recovers only **40%** of the band (vs 78% on
  natural). So §8.1's reframing of §0's −20.8/−29.5 as "structural" is **overstated** — a 2×
  end-to-end factor cannot carry 20–30 RULER points.
- Directly relevant to the repeat-prompt SFT idea: `repeat` is the **hardest** distribution for a
  QI router (9.79×), which is evidence *for* it being a demanding training target — but also means
  a router trained on natural text transfers to it at 1.23–1.36× cost.

### 11.4 `oracle_qi` is not achievable by anything frozen at prefill (staleness)

Score measured on the first 64 future queries, damage paid on the last 64 → **2.54×** the fresh
oracle (144/144 cells worse); even the all-window oracle is **1.87×** worse than fresh. Future
demand is genuinely non-stationary, and §8's `oracle_qi` is an oracle over the whole window *at
once*. **So the achievable bound for a frozen router is ~1.9× looser than §8.1's "ceiling"** — in
the opposite direction from §10.5 (where set optimization made it ~5% tighter).

Also: damage is 4× worse in the first 8 future queries (0.0695) than averaged over all 4096
(0.0266), and a last-8 control gives 0.0157 — so it is **position, not count**. §8 averaged away
the short-generation regime deployment actually lives in.

### 11.5 Decode feedback flatters `oracle_qi` specifically

96-step greedy decode, all layers evicted, keep 25%:

| arm | token agreement w/ dense | first divergence |
|---|---|---|
| `prefill_mean` | 0.305 | **step 27.8** |
| **`oracle_qi`** | 0.076 | **step 4.5** |
| `random` | 0.036 | 1.5 |
| `recency` | 0.021 | 0.0 |

**The "ceiling" diverges 6× sooner than an arm §8 ranks strictly below it.** Cause: its demand was
measured against teacher-forced continuation text the model stops producing once eviction perturbs
step 1 — §11.3 and §11.4 compounding under feedback. Sharpest demonstration that "give the QI
scorer the true future demand" is **not** a bound on a deployed evictor.

### 11.6 Layer subsample is biased; per-head spread is far larger than §8 reported

`{0,7,14,21,28,35}` deliberately includes both endpoints, and L0/L35 have anomalously narrow bands
(recency/oracle_qi ratio 1.9 and 2.0 vs 4.2 for the held-out 30 layers). It is fine for qi/qd (4.72
vs 5.01) but **understates `recency`/`random` by 1.7–1.9×**, which **inflates every "% of band
recovered"** figure (denominator is `recency − oracle_qi`).

Per-head qi/qd over 864 cells: geo **5.21**, p50 4.78, **p95 15.58, p99 29.1, max 94.9**;
**12.6% of head-cells exceed 10×**. §8's "0.20–0.73 per-head spread" understates this badly for the
ratio a per-head budget decision needs. A shared-budget design is bottlenecked by heads 4–20× worse
than the average — **non-uniform per-head budget is a bigger lever than §8 suggested.**

### 11.7 Clean

**Document sampling is sound** (0 duplicate pairs, cross-doc 8-gram Jaccard max 0.0033, 24/24
distinct, 0 mid-row EOS, decoded prefixes are genuinely distinct documents). Two minor caveats:
`--seq-len 8192` always reads the same first 12.5% of the same 64K rows, and the "domain" axis has
only 2 levels. Note baseline NLL ranges 0.78–2.87 across docs and ΔNLL cross-doc sd (0.0166)
**exceeds most reported effect sizes**. `cut_frac` is benign for the ratio (4.36/4.52/4.47/3.86) but
absolute damage triples with it, so cross-`cut_frac` absolute comparisons are invalid.

### 11.8 Transformers gotcha worth keeping

`ALL_MASK_ATTENTION_FUNCTIONS["x"] = ...` writes `_local_mapping`, but
`_preprocess_mask_arguments` checks `_global_mapping` and early-exits with **no mask**, silently
running Qwen3 **bidirectionally** (0.99 rel error at L14). Use `register()`. Anyone adding a custom
attention impl here will hit this.

### 11.9 Net state of the project after §10 + §11

| claim | status |
|---|---|
| §8.1 "3–10× and growing with length" | **wrong as stated.** ~2× end-to-end, flat in length (refuted twice independently) |
| §8.1 `oracle_qi` is a hard ceiling | **false in both directions**: −5% (set search, §10.5), +90% (staleness, §11.4), ±2.2× (query distribution, §11.3) |
| §8.1 RULER deltas are "structural" | **overstated** — RULER is `quote`-like and 2× cannot carry 20–30 points |
| §8.2 linear ties MLP → cancel E3 | **holds** (gap is 0.04× the noise floor) |
| §8.3 don't build Arm C | **holds, stronger** (§10.8) |
| §9 residual router refuted | **holds** — the shuffle sign test is valid; the rel_L2 ladder is not |
| §8.4 C1 mean-norm gain | **halved**, partly recency (§10.6) |
| C1b, C2 (probe half) | **unresolved** (§10.6, §10.7) |
| §8.2/§9.1 "% of band recovered" | **±18 points**, and inflated by layer-subsample bias |
| §8.5 E2 "unopposed, highest value" | **weakened** — its motivating 3–10× is ~2×; still defensible, no longer implied by a measurement |

**The two conclusions that survive everything are the cancellations** (E3, bilinear, Arm C) — all
three rest on within-instrument sign tests rather than magnitudes. **Everything framed around the
3–10× ceiling needs restating**, and the newly promoted levers are **non-uniform per-head budget**
(§11.6) and **greedy set selection** (§10.5), neither of which the original plan ranked.

---

## 12. AUDIT of §8.2 / §9 internals — the head-shared target was the real ceiling (2026-08-18)

`proxy_exp/audit_learned_and_redundancy.py` (+ `audit_{trainsweep,docssweep,damageA,recon,damageB}.json`).
Pipeline validated: reproduces §8.2's non-learned arms to ≤2.6e-05 and the MLP retrain ties the
original at ×1.002 (ns), so all deltas are on a verified-equivalent instrument.
**Every number below re-verified from raw JSON with paired per-cell tests.**

### 12.1 THE BIGGEST FINDING — §8.2 measured the wrong ceiling. Per-head targets are worth 3× any capacity change.

Every §8.2 arm trained on `future_sum.mean(0)` and broadcast **one score across 8 KV heads**.
Training on **per-head** targets (same features, same loss, same docs):

| arm | rel_L2 @keep 25% | band |
|---|---|---|
| `oracle_qi` (true per-head) | 0.0247 | 100% |
| **`oracle_qi_shared`** (true demand, head-averaged) | **0.0299** | **88%** |
| **linear, per-head** (4,097×8) | **0.0313** | **84%** |
| **MLP-256, per-head** | **0.0313** | **84%** |
| `prefill_mean` | 0.0341 | 78% |
| MLP-256, head-shared (§8.2) | 0.0357 | 74% |
| linear, head-shared (§8.2) | 0.0359 | 73% |
| `mlp1152` (E3's config, head-shared) | 0.0357 | 74% |

Paired per-cell, re-verified: **per-head vs shared ×0.871, t=−13.80, 20/20 cells** (linear) and
×0.883, t=−11.45, 20/20 (MLP). **MLP vs linear with both per-head: ×1.002, t=+0.50, ns** — the
capacity tie survives, but 11.7% lower.

Two consequences:
- **The per-head *linear* router (4k params) BEATS `prefill_mean`** (×0.917, t=−2.88, 16/20),
  which **reverses §8.2's closing argument** that "a feature built from observed attention
  outperforms anything read off the hidden state."
- **`oracle_qi_shared` = 0.0299** (×1.216 vs true oracle, t=+23.0, **0/20**) is the structural limit
  of everything §8.2 tested. §8.2's arms sat at 84% of *their own* ceiling — **which is exactly why
  capacity looked saturated.** §8.2's "it learns real signal, then stops, because `h_j` alone does
  not contain more" attributed to `h_j` what was a property of the target.

**The cheapest high-value experiment is now a per-KV-head scalar router**, which §8 never tested.
Note `scalar_indexer.py` emits one score per key; per-head means `n_kv_heads` outputs (8× the head
of the MLP, still ~1/500 of the layer). This is a small change with a measured 11.7% gain.

### 12.2 §8.2's "within noise" is false as stated (though the decisions survive)

The three §8.2 files share bit-identical non-learned arms, so cells pair exactly. Paired:
**linear/MLP ×1.014 [1.003,1.025] t=+2.66 SIG (5/20)**; **bilinear/MLP ×1.021 t=+6.52 SIG (1/20)**.
The published "within noise" compared three *unpaired* means whose across-cell sd swamps a real
1.4–2.1% effect — **the same paired-vs-unpaired error §9.4 warns about, now found inside §8.2.**
(My own re-verification: ×1.011, t=+1.81, borderline.) The tie is a *bound on the effect size*, not
an absence of one. Cancellations survive on magnitude; the stated evidence does not.

### 12.3 The tie is a SAMPLE ceiling, not undertraining (and not capacity)

Suspicion "all three arms are undertrained" is **refuted**: epochs 400 → 12,000 moves MLP −0.0006
(ns) and linear −0.0038 (*worse*); lr and cosine schedule ±0.001. Reason: the rank objective's floor
is exactly `1/T` = 2.44e-04 at T=4096, and the MLP reaches train rank-corr **1.0000** — full
interpolation of 7 documents.

But a docs sweep (fixed test set, 3 seeds) shows **both arms still rising**: linear
0.407→0.466→0.497→0.511 and MLP 0.482→0.500→0.513→0.522 at n_docs 1/2/4/7, +0.037/+0.014 top25 per
doubling, and the MLP−linear gap *shrinks* monotonically 0.075→0.034→0.016→0.011.
**So "256× more parameters bought nothing" is confounded with "7 documents bought nothing."**
E3's cancellation still holds — `mlp1152` measured directly is 0.0357, ×1.004 vs MLP-256 (ns),
re-verified — but the reason is the head-shared target, not saturation.

Also **refuted**: the rank-loss surrogate is not degenerate, it is *essential* (MSE at the same
budget scores top25 0.3454 vs 0.5217, t=−15.5); and there is no LayerNorm/standardization leakage
(`score(h[:n])` == `score(h)[:n]` to 1.3e-07).

### 12.4 §9's verdict survives, but three of its supporting arguments are wrong

- **`explained_frac = −0.217` is a SCALE artifact.** The shipped statistic `1 − mean(‖r‖/‖v‖)` is
  scale-sensitive and the delta-rule reconstruction is ~3× too large (α* = 0.31). Corrected by one
  global scalar: **−0.254 → +0.118**; an intercept ridge reaches **+0.314**. Demonstrated
  constructively: a reconstruction equal to *truth × 3* reads `−1.00` ("actively harmful") while the
  gain-corrected version reads 1.0000. So §9.1's "the state is **actively harmful** … a linear state
  **cannot predict `v_t` at all**" is wrong; the correct claim is **"the linear state explains
  12–31% of `v_t`, and that still predicts nothing about demand"** — weaker, but equally fatal.
- **`res_ridge` was not the optimal linear reconstruction**: no bias column, and unit-norm RoPE'd
  keys have no constant component, so it could not express the mean value vector (a causal running
  mean alone explains +0.098). With intercept: +0.158/+0.283 on all keys. Damage impact nil
  (0.1007 vs 0.1010), so the null holds.
- **"Unlimited-capacity ceilings are worse than random" is 8K-only.** At 32K, `res_ridge`/random =
  **×0.820, t=−3.97, significantly BETTER**, with `explained_frac_ridge` = +0.087.

### 12.5 NEW, and it belongs in §9.5's NOT-killed list: the redundancy signal exists with the OPPOSITE sign

§9 only ever measured its own asserted orientation. Flipped (24 cells, keep 25%):

| arm | rel_L2 | band | vs random | vs recency |
|---|---|---|---|---|
| `nn_novelty` (§9's sign) | 0.1046 | −16% | ×1.158 t=+5.70 | — |
| **`nn_novelty_neg`** | **0.0638** | **+42%** | ×0.806 t=−3.08 **SIG** | ×0.758 t=−5.15, 21/24 **SIG** |
| **`nn_res_rel_neg`** | **0.0653** | **+39%** | ×0.817 t=−3.08 SIG | ×0.768 t=−5.53, 23/24 SIG |
| `res_Skept` (the design) | 0.0902 | −2.5% | ×1.020 ns | — |

**Keep the keys whose earlier neighbours already point the same way; evict the novel ones** — the
opposite of the design's premise. It recovers 42% of the band and beats `res_Skept` (×0.790,
t=−2.97). Not position: `nn_novelty_neg` beats recency 21/24 while recency is itself at the random
floor. Still 1.6× worse than `prefill_mean` and `O(L²)`/non-deployable, so **it does not resurrect
the design** — but §9.1's "**every** redundancy score sits at chance" understates a t=−5.5 effect
measured only in the losing orientation.

### 12.6 §9's deciding control is position-contaminated (direction still robust)

`res_Skept_shuf` has `spear_pos` = **+0.166** while `res_Skept` has **−0.109** — the shuffle acquired
a position correlation the arm lacks, so it is not "scale-and-form matched, alignment destroyed."
It also beats the arm against random. The direction survives anyway (**only 3/48** (layer, head)
pairs have residual > shuffle, largest |t| all favour the shuffle, e.g. L21h0 t=−25.8), but the
specific "16/48, indistinguishable from chance" reading is **against a contaminated control**.

§9.4's 32K "8/24 reversal" reproduced **exactly** (paired difference t=−2.75 while the same cells
split 8/24) — that methodological warning is sound.

### 12.7 Revised plan after §10–§12

| item | status |
|---|---|
| **per-KV-head scalar router** | **NEW, highest-value cheap experiment.** 0.0313, 84% of band, beats `prefill_mean`, 20/20 cells. Never tested. |
| non-uniform per-head budget (§11.6) | promoted — per-head qi/qd p95 is 15.6×, 12.6% of heads >10× |
| greedy set selection ② (§10.5) | live — `oracle_qi` is beatable by ≥5% |
| E3 (`MID_DIM=1152`) | still cancelled (measured ×1.004, ns) — but for the target reason, not saturation |
| bilinear | still cancelled (×1.021 t=+6.52, *significantly worse*) |
| Arm C (`z`) | still cancelled, strengthened (§10.8) |
| §9 residual router | still refuted — but the *sign-reversed* ceiling is real (§12.5) |
| E2 (`--head-dim 8`) | still defensible, no longer "unopposed"; motivating ratio is ~2× not 3–10× |
| more training documents | newly indicated — both capacity curves still rising at 7 docs |

**Every §8/§9 headline number has now been either corrected or re-derived.** The three cancellations
survive; the three explanations behind them did not. The single actionable positive result of the
whole audit is §12.1.

---

## 13. CORRECTION to §12.1 — the real `ScalarIndexer` was ALREADY per-head

§12.1 concluded "the cheapest high-value experiment is now a per-KV-head scalar router, which §8
never tested" and noted "`scalar_indexer.py` emits one score per key". **That note is wrong.**
Verified directly:

```
ScalarIndexerConfig(hidden_size=4096, n_heads=8, mid_dim=256)
  score_keys(h) -> (1, 8, 16)      # (B, n_heads, Sk) — one score PER KV HEAD
  w_out.weight  -> (8, 256)
  head scores for token 0: [0.715, -0.823, -0.296, -0.261, -0.309, -0.067, 0.337, 0.343]
  params: 1,059,328 (n_heads=8) vs 1,057,536 (n_heads=1) — only +0.17%
```

And `press.py:156`: `n_heads = self.n_heads or text_config.num_key_value_heads` — **per-head is the
default**, so the trained scalar arm and §0's RULER A/B were already per-head.

**Confirmed on the actually-trained checkpoint**, not just a constructed config:
`Qwen-3-8B-gqa_indexer_scalar/stage1/step600.pt` (252 entries, `scorer='scalar'`,
`scalar_mid_dim=256`) has `model.layers.0.self_attn.indexer.w_out.weight` of shape **(8, 256)**.
The arm that produced the −20.8/−29.5 RULER deltas emitted 8 independent per-KV-head scores.

Three consequences, all corrections to earlier sections:

1. **§12.1's recommendation is already shipped.** Nothing to implement; the 11.7% is not available
   as a free win on the real arm.
2. **§8.2/§9's proxy numbers UNDERSTATE the real scalar arm**, because `diag_learned_evictor.py` and
   `diag_redundancy.py` trained on `future_sum.mean(0)` and broadcast one score across 8 heads —
   a handicap the product does not have. So the deployed arm sits nearer `oracle_qi_shared`'s
   *removal*, i.e. above 74% of band.
3. **The −20.8/−29.5 RULER gap is therefore NOT explained by head-sharing** — that escape is closed.
   Combined with §11.3 (RULER is `quote`-like, where `prefill_mean` recovers only 40% of the band),
   the remaining explanation is the training **target/query distribution**, not the score's shape.

This is the fourth "the proxy measured something the product does not do" error in this line of work
(cf. §2's hand-picked readout, §8.6, §12.2). **Before any further proxy conclusion is used to
justify a code change, check what the shipped module already does.**
