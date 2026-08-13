# Gated attention for the GQA indexer: what makes a router learnable

Working notes on end-to-end (LM-loss) training of the GQA lightning indexer. Written while
implementing `gated_attention.py` / `e2e_trainer.py`, and revised twice when experiments
contradicted earlier conclusions in this file. Corrections are kept visible rather than edited
away — two of them were load-bearing.

Every number quoted here comes from a script re-run in one batch; fp64 throughout, so tolerances
measure floating-point noise rather than approximation. Claims that were **not** verified are
labelled as such.

---

## 0. TL;DR

**One criterion decides whether a router can learn anything: can it score well *without* learning
a ranking?**

Two ways it can:
1. The gate can become a **no-op** — mathematically inert — falling back on the pretrained dense
   model that is already strong.
2. The gate's **optimum is unrelated to the ranking**.

The corollary, which took two revisions to see: **if recovering dense attention itself requires
knowing the correct ranking, there is no loophole at all.** That is HSA's situation, and it is why
HSA needs none of the machinery SAS needs.

| scheme | train-time fwd | normalization | no-op reachable | needs pinning |
|---|---|---|---|---|
| raw `s` (now `pin_mode="none"`, ablation only) | dense | none | **yes** | — loophole open |
| `log_softmax(s)`, nothing pinned | dense | all keys | **yes** | — loophole open |
| `log_softmax(s)` + pin | dense | history only | no | **yes** |
| `log_sigmoid(s)` | dense | per-key, independent | **yes** | — loophole open |
| outer gate + softmax | dense | all blocks | no | but **breaks output scale** |
| **HSA (two-level)** | sparse | within-chunk **and** across-chunk | yes, but **harmless** | **no** |
| DMA | **sparse** | none (uses `-inf`) | no | no |
| SparseK | **sparse** | hard `Σp = k` | no | no |
| STE (fwd sparse / bwd dense) | **sparse** | none | no | no |
| sparse-scope training | **sparse** | within selected set | no | no |

The deepest split in that table is not the normalization column — it is **dense vs sparse
train-time forward**. Normalization is only needed to *manufacture* scarcity when the forward is
dense. Once the forward is genuinely sparse, scarcity is structural and free.

**Status:** route A is implemented as `E2EIndexerTrainer(pin_mode=...)` with modes `none` / `sink` /
`self` / `self+sink`; route B is the existing `stage="sparse"`. See §8. The open risk in §9 — that
scarcity deepens to −10.4 nats at 32K — is still unmeasured.

---

## 1. What an inner gate actually does

We add a number `g` to each key's attention logit:

```
out = softmax(scale · q·kᵀ + g) V
```

`g` means exactly one thing: **multiply this key's attention weight by `exp(g)`.** Verified — the
`exp(gate)` ratios and the realized attention-weight ratios match exactly.

An important correction to how this is often described: the gate does **not** zero anything out.
Softmax renormalizes, so suppressing some keys *raises* the others:

```
suppressed half's mass : 0.4961 -> 0.0467
row sum                : 1.000000   (always)
```

Suppressed keys shrink but stay nonzero. **Training-time attention under full scope remains
dense.** Real sparsity appears only at inference, from a hard top-k on the score. That gap is what
the two training stages exist to probe.

---

## 2. The loophole

Softmax is invariant to a shift shared by all keys:

```
softmax(z + 5)    vs softmax(z): 2.22e-16
softmax(z - 100)  vs softmax(z): 1.25e-15
softmax(z + 1000) vs softmax(z): 1.31e-14
```

So a gate that is **flat along the key axis** is indistinguishable from no gate at all, and the
model reverts to the frozen, already-strong pretrained backbone. The router gets a good loss for
free, having learned no ranking whatsoever.

With raw `s`, flat means `s = 0` — reachable, at zero cost, from anywhere.

---

## 3. Which schemes close it

Gap to dense attention, where ~0 means "the no-op is reachable":

| gate | best inert attempt | gap to dense |
|---|---|---|
| raw `s` | `s = 0` | **0.00e+00** |
| `log_softmax(s)`, no pin | `s` = const | **1.11e-16** |
| `log_sigmoid(s)` | `s → +∞` (saturates) | **1.39e-16** |
| **`log_softmax(s)` + pin** | *none exists* | **6.85e-01** |

**Both ingredients are required.** Either alone leaks:

- **`log_softmax` alone** forces the multipliers to sum to 1, but the constraint applies to
  *everyone equally*, so all keys sharing it equally is still flat. A constraint that treats all
  parties identically creates no differences.
- **Pinning alone** (under raw `s`) fails because history can freely take `s = 0`, matching the
  pinned keys exactly.

Together they bite. With 2 keys pinned and 6 in history:

```
pinned keys  : [0.0, 0.0]                          <- fixed at log(1) = 0
history keys : [-1.7918, -1.7918, ...]             <- log(1/6) each
history multipliers sum to exactly 1               <- forced by log_softmax
```

For history to match the pinned `0`, every one of the 6 would need multiplier 1, summing to 6, not
1. **Impossible.** Six people cannot each take a whole cake when there is one cake.

**So the mechanism is not "the range of the denominator" but "somebody is excluded from the
denominator."** Pinned keys are a referee whose score is fixed and does not compete. History can
never collectively equal the referee, so the tie — the lazy way out — is gone. All the router can
then do is decide *who* gets the fixed budget. That decision *is* the ranking.

Two forces, both needed: `log_softmax` creates **scarcity** (fixed total); pinning creates
**contrast** (a reference point exempt from that scarcity).

### This unifies SAS Table 1

Rows 4/5/6 are one phenomenon, not three — all are "can the gate go flat":

| SAS row | how it goes flat | 1 epoch |
|---|---|---|
| (6) raw `s` | `s = 0` | 18.8 |
| (5) `log sigmoid(s)` | `s → +∞`, saturating to 1 | 17.0 |
| (4) `log softmax(s)` + pin | nowhere to go | **54.4** |
| (1) dense baseline | — | 56.1 |

Sigmoid is the clearest case: computed per key independently, with no "fixed total", so every key
can saturate to 1 on its own. The paper's wording — *"sigmoid gates gradually saturate toward
1... diminish the distinction between historical blocks and the unit-gated current block"* — is
describing exactly a tie with the referee.

---

## 4. Outer gate (`g` on the value)

`out = Σᵢ pᵢ · g_b(i) · vᵢ`. Two findings:

```
g = 1 everywhere (unconstrained)      : gap 0.00e+00   <- loophole still open
g = softmax(s), uniform (SAS row 3)   : gap 3.46e-01   <- closed
   output magnitude: dense 0.2284 vs gated 0.0571      (~1/C)
```

An unconstrained outer gate has the same loophole. Normalizing does close it — but **by breaking
the output scale**: the global softmax has already normalized, so multiplying by something `< 1`
only shrinks the output, by roughly `1/C`. The frozen backbone has never seen inputs at that
magnitude.

So SAS row 3 (41.6 vs 54.4) is **not collapse — it is a different failure**. Two distinct failure
modes that are easy to conflate.

---

## 5. HSA: the exception, and why

**Correction.** Earlier in this investigation I dismissed HSA as "a form of outer gate, and SAS
shows outer gating is worse." That was wrong, and the error mattered. HSA's within-chunk
renormalization makes it a different object entirely.

HSA computes `out = Σ_c w_c · softmax_within-chunk-c(qk) @ v_c`. Verified properties:

```
router's w            : [0.3637, 0.3495, 0.0739, 0.2129]
realized chunk mass   : [0.3637, 0.3495, 0.0739, 0.2129]
max|w − mass|         : 5.55e-17        <- identical
A_ij row sum          : 1.0000000000    <- still a valid distribution
```

**The router's `w_c` *is* the mass chunk `c` receives — exactly.** Because the within-chunk softmax
already sums to 1, multiplying by `w_c` makes the chunk's total mass exactly `w_c`.

This makes HSA an **exact decomposition** of dense attention:

```
HSA with w = true chunk mass : gap to dense = 3.33e-16   <- exact
HSA with w = uniform         : gap to dense = 5.09e-01   <- flat w is NOT a no-op
```

Is that reachability a loophole? **No — this is the elegant part.** Reproducing dense requires
`w_c` = the true chunk mass, and the true chunk mass *is* the correct ranking. **"Getting a good
score" and "learning the ranking" are the same point.** There is no lazy flat solution: uniform
`w` is 0.51 away from dense.

> SAS: switching the gate off scores well → must be blocked.
> HSA: scoring well requires learning the ranking → nothing to block.

---

## 6. The two-level advantage, precisely

At chunk level the structural difference is decoupling.

**HSA — multiplicative, decoupled:**

```
A_ij = w_c × p_(j|c)
       ↑      ↑
   across    within
   chunks    chunk
```

- `w_c` decides how mass is split **between** chunks — the router's job
- `p_(j|c)` decides how mass is split **within** a chunk — the backbone's job
- the final `A_ij` is the **product**; neither term contaminates the other

**Chunk-level `g` in softmax — additive, entangled.** Same chunking, `softmax(qk + g)`:

```
normalized exp(g)  : [0.039, 0.8362, 0.0099, 0.1149]   <- what the router said
realized mass      : [0.086, 0.3763, 0.1336, 0.4041]   <- what actually happened
max|exp(g) − mass| : 3.71e-01
```

The router asks for 84% on chunk 1 and gets 38%; a chunk it gave 1% receives 13%. The exact law
(verified to 1.11e-16):

```
chunk mass = softmax(g_c + LSE_c)
                     ↑      ↑
                  router   backbone's own
                           "loudness"
```

where `LSE_c = logsumexp` of chunk `c`'s internal logits. **The chunk's final mass is two forces
fighting inside one softmax, and the router controls only one of them** — while being unable to
observe the other.

### The consequence that matters

Inference does top-k **on the router's own score**. So: is the quantity the router learns the
quantity we want to rank by?

| | router must output | can its output rank directly? |
|---|---|---|
| **HSA** | `w_c = mass_c` | **yes** — the output *is* the mass |
| **SAS** | `g_c = log(mass_c) − LSE_c + const` | **no** — needs `LSE_c` too |

For a frozen backbone, `mass_c = softmax(LSE_c)` holds exactly (5.55e-17). Therefore:

> **SAS's gate is a *correction* to the backbone — optimally zero when the backbone is already
> right.** Verified: `g* = log(mass) − LSE_c` is constant to 4.44e-16.
>
> **HSA's weight is *the quantity itself* — it must always express the full ranking.**

**A correction that is optimally zero cannot carry a ranking.** This is precisely why SAS needs
pinning: to forbid the gate from taking that rank-free constant value.

This is the same fact as §2–3's no-op finding, seen from the router's side rather than the loss's.
The router-side view is arguably more fundamental, because it explains *why* the no-op exists.

**Second correction.** At one point I reported that SAS's dense-reproducing `g*` has 0.0% top-k
agreement with the true mass ranking, as evidence of mis-ranking. That number was an artifact:
`g*` is a *constant*, so its top-k is arbitrary tie-breaking. The correct statement is not "SAS
mis-ranks" but "**SAS's gate carries no ranking at its optimum**" — stronger and cleaner, but not
what I first claimed.

### HSA at token granularity

With `chunk_size = 1`, the within-chunk softmax degenerates to `softmax(single element) = 1`, so:

```
out = Σⱼ wⱼ · vⱼ
```

`w` becomes the attention weights themselves. **HSA degenerates into replacing attention with the
router** — the `q·k` term vanishes and the router must learn the whole distribution from scratch.
That loses both the decomposition and the frozen backbone's prior.

**HSA's mechanism structurally requires `chunk_size > 1`.** Not a tuning issue.

---

## 7. Why DMA / SparseK / STE never had this problem

Because **their training-time forward is already sparse.** SAS's is dense.

```
DMA (w=6, deltas equalized)       : gap to dense = 9.62e-01   <- cannot flatten
SparseK (budget k=4)              : Σp = 4.0000, 4/12 keys dropped
STE fwd-sparse, s = 0 (flat)      : gap to dense = 1.41e+00   <- cannot flatten
```

- **DMA** sets non-selected entries to `-inf`. Equalizing the surviving `δ` does not help: the
  dropped keys are gone regardless. **`-inf` does not scale with `δ`** — shrink `δ` a
  million-fold and `-inf` is still `-inf`. That is the referee, built into the formula.
- **SparseK** solves for `τ` such that `Σp = k`. All-ones would sum to `n ≠ k`. The budget is
  algebraic, not learned, so the router cannot learn its way out of it.
- **STE** keeps the hard top-k mask in the forward, for the same reason as DMA.

And the deepest layer — **DMA is pretrained from scratch** (per its paper: scaling laws, ~80M
params, 13.5K steps, SmolLMCorpus):

| | DMA / SparseK / STE | SAS / us |
|---|---|---|
| training | from scratch | post-training, backbone frozen |
| train-time attention | **sparse** throughout | **dense** (full scope) |
| switching the router off | model has never seen dense — no fallback | reverts to a strong pretrained model |
| must manufacture scarcity? | no | **yes** |

> **DMA's sparsity is load-bearing structure — remove it and the model collapses.**
> **SAS's sparsity is a bolt-on — unbolt it and the model still runs.**
>
> So SAS has to weld the bolt shut.

### STE's real cost

STE closes the loophole but trades it for variance: with flat `s`, top-k picks **arbitrary** keys
(all tied), the loss is bad, and gradients are noisy. This matches SAS row 8 —
*"substantially increases the gradient norm."* **Collapse becomes high variance, not safety.**

---

## 8. Where this leaves the implementation

`raw s` + full scope — what was built first — is the worst cell in the table: a strong frozen
backbone as fallback, a dense training forward, and an additively flattenable gate. All three
conditions for the loophole hold simultaneously. DMA/SparseK satisfy none; SAS closes the third
with pinning.

Two **orthogonal** fixes:

| route | mechanism | cost |
|---|---|---|
| **A. `log_softmax` + pin sink** | manufacture scarcity | keeps full-scope gradients; needs an LSE pass; concat width `D+Di+1`; **scarcity grows with length (untested)** |
| **B. sparse-scope training** | eliminate the fallback | loses independent gradients for unselected keys (SAS: 47.4 → 55.6) |
| C. STE | eliminate the fallback | high gradient variance; SAS reports it as worse |

A and B compose, which is probably the safest configuration: **stage 1 = full scope with
`log_softmax` + pin** (independent gradients *and* scarcity), **stage 2 = sparse scope**
(train/inference consistency).

**Implemented** (`gate_pin.py`, `E2EIndexerTrainer(pin_mode=...)`): route A, with four modes —
`none` (the un-pinned ablation), `sink`, `self`, `self+sink`. `pin_mode` left unset resolves to
`self` for the dense stage and `none` for the sparse one, since the sparse scope makes pinning
inert. Route B is the existing `stage="sparse"`, so A + B compose as described.

Whether a kernel is needed turns on **query-dependence**, not on token-vs-block granularity:

| pin | query-dependent | folds into concat | verified |
|---|---|---|---|
| `sink` | no | **yes**, width `D+Di+1` | 1.1e-15 |
| `self` | **yes** | **no** — naive attempt off by 2.5 | — |

The `-LSE` term is rank-1 (per-query scalar × per-key 0/1 indicator), so for `sink` one extra
dimension carries it and a single SDPA call still suffices. For `self` the pinned column moves with
the row, and no shared `K` can zero a per-query position — so it takes a two-branch route instead:
history-only attention (which folds; inside that branch `-LSE` is a per-row constant and cancels)
plus the pinned keys, merged by log-sum-exp. Exact to 6.7e-16 against two independently written
references.

`self` currently builds explicit logits and is therefore `O(Sq·Sk)` in memory — SDPA returns no
log-sum-exp and recovering one costs a third pass. So `self` is the correctness / short-sequence
path today and a fused kernel is what would make it viable at 32K; `sink` needs none of this at any
length.

Also verified: pinning blocks **both** routes to the no-op, `qi` flat *and* `gate_scale → 0`. The
latter matters because the earlier claim that a learnable `gate_scale` collapses to 0 turned out to
be an artifact of a flawed toy (a dense-output target, under which switching the router off is
genuinely optimal); on a prediction objective it does not collapse. `gate_scale` is nevertheless
kept fixed by default, with `gate_scales` logged per layer as a diagnostic.

### If moving to chunk level

| | needs pinning | what the router learns | cost |
|---|---|---|---|
| **HSA two-level** | **no** | the chunk mass itself | needs a within-chunk-softmax kernel; requires `chunk_size > 1` |
| SAS `g` in softmax | **yes** | mass minus `LSE`, a correction | pin + LSE pass; but folds into concat |

HSA's structural advantage is real and is exactly the decoupling point above: one kernel buys away
pinning, the normalization tricks, *and* the awkwardness of the router learning a different
quantity than the one inference ranks on.

---

## 9. Open risk: scarcity scales with sequence length

The one thing repeatedly flagged and **never measured**. Pinning's scarcity strength is
`log(1/N_history)`:

| history keys | suppression |
|---|---|
| 6 | −1.79 nats |
| 44 | −3.78 nats |
| 1024 | −6.93 nats |
| **32768** | **−10.40 nats** |

Suppression grows with context length, so **pinning may over-suppress history at 32K** — which is
the target regime. SAS is block-level, so its `C` is only 256–512 and the effect is far milder.

This is route A's sole unknown, and it lands squarely on the intended use case.

---

## 10. Honest limits of this analysis

- **The mechanism claims (§1–7) are verified**, in fp64, and I stand behind them.
- **The "how much does it matter in real training" claims are not.** The toy experiments here
  (B=4, Sk=64, synthetic retrieval) showed pinned and unpinned at 100% vs 100% router recall
  under full scope — i.e. **the toy is too easy to resolve the difference.** It does not refute
  SAS.
- **Where the toy and SAS disagree, trust SAS**: 54.4 vs 18.8 on Qwen3-4B / 32K / real reasoning
  is real-scale evidence; my toy lacks the resolution to contradict it.
- **A retracted finding:** I earlier reported that a learnable `λ` collapses to exactly 0 ("router
  suicide"). That was an artifact of a toy whose target was *reproducing dense attention* — an
  objective under which **any** gate is worse than no gate (verified: `loss(no gate) = 0`,
  `loss(boost true-top-8) = 3.4e-2`), so `λ → 0` was the correct answer, not a pathology. On an
  honest retrieval objective `λ` **rises** (0.32 → 1.45 with a weak backbone, 0.32 → 0.61 with a
  strong one). Fixed `λ` remains the prudent default, but not for the reason first given.
- **Unverified:** whether SeerAttention's AttnGate is bilinear (its score is what would decide
  whether it, too, could fold into concat); whether SparseK is also trained from scratch.

---

## Sources

- **SAS**, `16448_SAS_Simple_Attention_Spa.pdf` and `SAS_arxiv.pdf`. The two versions differ:
  the arxiv Table 1 adds the `softmax(qK⊤ + s)` row (raw-logit injection, 18.8) that the other
  omits — which is precisely the variant implemented here first. Gate-gradient derivations in
  App. D; kernel pseudocode in App. F.
- **Trainable Dynamic Mask Sparse Attention (DMA)**, Eq. 3–5 for the gate, §4.3 for the gradient
  claim, Table 4 for the from-scratch scaling-law setup.
- **SparseK**, §3.3 for the operator and its `Σp = k` constraint, §3.5 for the position slope and
  the hard-K/soft-V trick.
- **HSA**, `sparsex/model/qwen3_hsa/hils_attention.py` — `chunk_weights` / `HSA_block_M_head`.
