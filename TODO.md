# TODO

Open work items, with enough context that the *reason* survives without the conversation that
produced it. Same convention as `kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md`: claims that
were verified say so, claims that were not are labelled, and a retracted claim is kept visible
rather than edited away.

---

## The e2e router's RULER collapse is a SELECTION failure (established causally)

**Status:** diagnosed, not fixed. This is the headline finding; the budget item below is
lower priority than it was.

### The gap

`niah_multikey_3`, matched checkpoints (both step 600, same schedule/LR/seed/corpus, `topk=2048`,
`force_sink=4`, `force_local=64` — only the objective differs):

| | 8K | 16K |
|---|---|---|
| distill (step600) | 97.83 | **95.65** |
| e2e (final) | 41.30 | **4.35** |

Only `niah_multikey_2/3` regress. `niah_single_1/2/3`, `multiquery`, `multivalue`, `vt` are 100.0
for both at both lengths, so basic retrieval is intact.

### The causal test and its result

`evaluation/probe_forced_needle.py` forces the retrieval `key: value` span (61–71 tokens) into
every row's support at every layer, replacing MIDDLE slots so `topk`, sink and local are all
unchanged — equal budget, so the delta is attributable to the needle's presence alone.

| checkpoint | off | forced | delta |
|---|---|---|---|
| distill | 0.850 | 1.000 | +0.150 |
| **e2e** | **0.000** | **1.000** | **+1.000** |

**e2e goes 0/20 → 20/20.** The router's *scores* are the whole problem: given the needle, the same
weights answer perfectly, so nothing downstream of selection is damaged. The failure is that the
router does not select the needle.

Off-arm sanity: distill 17/20 (85%) against the published 95.65 and e2e 0/20 against 4.35 — the
distill gap is the script's simplified greedy decode versus the pipeline's `generate_answer`, and
0/20 is consistent with a 4.35% rate (expected 0.87 hits). Both arms share one decoder, so the
delta stands. Injection fired on all 36 layers across every decode step.

Failure mode in the predictions is diagnostic on its own: `off` answers are well-formed uuids from
elsewhere in the haystack (`3c7ed208-...` → `57968d64-...`), sometimes annotated by the model as
placeholders. It is confidently reading the wrong span, not failing to read.

### Why this is expected of the objective, and what to do

**Both runs read the SAME corpus** — `subsets=['2e16','2e17']`, verified from both checkpoints'
config blobs. Neither saw `synth_cwe`/`synth_rex`. So the difference is not the data, and
"add retrieval data" is **not** the explanation for why distillation succeeds. (An earlier version
of this entry claimed it was; that was wrong.)

The difference is what each objective supervises:

| | signal per sequence | what it says |
|---|---|---|
| distill | `36 layers x 8 heads x 33.5M causal pairs` ≈ 9.7e9 KL terms | "key 8241 should get 0.03 of this query's attention" — per-key, absolute, positional |
| e2e | 8192 scalars, shared by all 288 routers | "the prediction was off by 1.7 nats" — carries no key identity at all |

Roughly a million-to-one difference in density, but **specificity is the real gap**. Distillation
does not teach *retrieval*; it teaches the router to **reproduce the teacher's attention shape**.
Ordinary long text already contains the structure retrieval needs — induction heads, repeated-token
matching, coreference — so a router that imitates where those heads look inherits uuid matching for
free, because uuid matching runs on the same induction machinery with rarer tokens. **The teacher
already knows how to retrieve, and distillation copies its positions.**

LM loss asks only whether the next token was predicted well. On generic long text almost every
token is predictable from local context; the tokens where induction decides the answer are a
vanishing share of the total. And the frozen backbone is already strong, so a router that merely
avoids interfering gets loss to ~1.7. **There is no gradient pressure for exact retrieval to form.**

Note this is invisible to both training metrics: loss reached backbone level **and** `gate_sparsity`
fell 0.486 → 0.197, i.e. the router did learn *a* selectivity — just one tuned to what LM loss
rewards. **Two healthy training metrics, and retrieval still collapsed.**

Ranked candidate fixes:

1. **Mixed objective: LM loss + a distillation term.** Now the first choice, since the diagnosis is
   a missing *supervision signal*, not missing data. Distillation supplies the per-key positional
   target; the LM loss keeps the end-to-end adaptation that motivated e2e in the first place. The
   two trainers already share the corpus, the schedule and the checkpoint format, so the pieces
   exist.
2. **Retrieval-heavy data for the e2e arm.** Still worth trying, but on different reasoning than
   before: not "distillation had it and e2e didn't" (neither did), but "LM loss needs the retrieval
   tokens to be a large enough share of the loss to matter". A synthetic-retrieval mixture would
   raise that share. Unverified whether it is sufficient on its own.
3. **Not the gate budget.** This is a missing-signal problem, not an over-suppression one.

### Verification status

- **Verified** (real 8B model, 20 samples): the forced-needle table; injection on 36 layers; span
  self-check decodes to text containing the answer.
- **Verified** (from both checkpoints' config blobs): both objectives trained on the same
  `['2e16','2e17']` subsets, so the corpus is not the variable.
- **Not verified:** that a mixed objective fixes it, or that retrieval-heavy data alone would.
  The induction-head account of *why* distillation transfers is a mechanism story consistent with
  the evidence, not something measured here. Also unverified at 32K, where the e2e stage-1 run
  never trained.

### Probes that did NOT work, and why (do not repeat)

Two static probes were run first and both were uninformative. `evaluation/probe_router_selection.py`
and `evaluation/probe_sanity_check.py` are kept because the negative results are worth having, but
their **numbers should not be quoted**:

- Needle coverage came out at or below chance (0.048 distill, 0.072 e2e; chance = 2048/21025 =
  0.097) while distill scores 95.65 — self-contradictory, so the probe measured the wrong thing.
- `probe_sanity_check` found why the premise was wrong: **dense attention does not concentrate on
  the needle either.** Best head, best layer: 1.91x uniform weight, median rank 10722 of 21025.
  So "the router must rank the needle highly" was never a valid expectation, and retrieval evidently
  does not work by a single attention peak on the answer.
- Three bugs, all mine, all worth knowing about: `answer` is a `numpy.ndarray` so an
  `isinstance(x, (list, tuple))` check silently made the needle unfindable in all 500 rows; the
  prompt omitted `answer_prefix`, shifting every position; and RULER asks for the value *keyed by*
  another uuid 2–3 tokens earlier, so scoring only the answer span measured the wrong target.
- `probe_sanity_check` also had a fourth: attaching both checkpoints up front, when
  `attach_indexer` writes into the shared model, left both presses reading the second one's weights
  — caught because the two columns came out bit-identical. Fixed by re-attaching per use.

**Lesson for the next probe:** anchor on a quantity whose expected value is known (here, distill's
own 95.65) before interpreting anything, and prefer an intervention over a correlation.

---

## Ablate the gate budget `B` against the current `B = 1`

**Status:** not started, and **lower priority than when it was written** — the RULER regression above
turned out to be a missing-signal problem in the objective, not over-suppression, so `B` is no
longer a candidate explanation for anything observed. Nothing in the tree implements `B != 1` yet.

### What `B` is

The gate on a history key is `s_j - lse`, so `Σ_j exp(gate_j) = 1` over the history: the whole
non-pinned set is worth **one** pinned key's multiplier, because a pinned key sits at `gate = 0`
i.e. multiplier 1. Generalizing that total to `B` gives `gate_j = s_j - lse + ln(B)`, and the
suppression each history key carries becomes

```
suppression = ln(n_history / B)      (B = 1 today, so ln(n_history))
```

### Why it looked necessary

Suppression grows with context length: 9.01 nats at 8K, 10.40 at 32K — `ROUTER_LEARNABILITY.md`
§9, the file's one "repeatedly flagged and never measured" risk, and it lands on the target
regime. Setting `B = keep_ratio * n_history` makes suppression `ln(1/keep_ratio)`, a **constant in
`L`** (0.69 nats at `keep_ratio = 0.5`), which would retire §9 outright.

It also aligns train with inference. `compression_ratio = 0.5` keeps 4096 of 8192 keys at
eval, but `B = 1` lets the router fully retain only **one** key during training. Assuming a
perfectly ranked router, the gap between the training-time attention distribution and the
top-50%-eviction distribution it will actually run under:

| `B` | TV(train, inference) |
|---|---|
| 1 (today) | 0.7687 |
| 512 | 0.3883 |
| `0.5 * n_history` | **0.0991** |

### Why it is NOT clearly necessary — two findings that cut against it

**1. A retracted claim of mine.** I argued SAS avoids this because it is block-level, so its
normalizer spans only 256–512 units instead of 8192–32768. **That was a unit error.** SAS's blocks
hold ~64 tokens each, and the tokens inside a block share the block's multiplier, so at 32K:

| | competing units | B | per-unit | **token-level suppression** |
|---|---|---|---|---|
| SAS @32K (block=64, C=512) | 512 blocks | 1 | 1/512 | `ln(512*64)` = **10.40 nats** |
| here @32K | 32764 tokens | 1 | 1/32764 | `ln(32764)` = **10.40 nats** |

Total suppression is set by how many tokens the normalizer covers, **not** by how they are
grouped. SAS is under the same suppression we are, and reaches 54.4 against a 56.1 dense baseline
with `B = 1`. So "suppression is too strong" lost its main support. §9's parenthetical ("SAS is
block-level, so the effect is far milder") compares block-level against token-level numbers and
needs the same correction.

**2. Empirically fine at 8K.** First healthy run after the `gate_scale` fp32 fix
(`pin sink`, `n_sink=4`, `compression_ratio=0.5`, `ffn_sp=8`, `accum=8`):

| step | lm_loss | gate_scale | gate sparsity |
|---|---|---|---|
| 0 | 4.5167 | 0.0884 | 0.486 |
| 20 | 2.4120 | 0.0891 | 0.276 |
| 80 | 1.7093 | 0.0898 | **0.197** |

Sparsity 0.197 of history is `PR ≈ 1613` effective keys, comfortably inside the 4096 that eviction
keeps — the retained set covers essentially all of the gate's mass. Loss back to backbone level
(~1.7–2.2) **and** sparsity falling: loss alone could not show this, since a flat gate also
recovers the backbone. Under `B = 1` the router learns.

### What the ablation should answer

Does aligning the training budget to the eval budget help, hurt, or do nothing — given that
suppression is evidently survivable?

Three arms, matched on everything else, 8K, ~600 steps (this is the `B = 1` arm the title asks
for, run as the control rather than assumed):

| arm | `B` | suppression | rationale |
|---|---|---|---|
| control | `1` | 9.01 nats | today's setting, and SAS's |
| middle | `0.05 * n_history` | 3.00 nats | near SAS's block-level 5.55–6.24 |
| aligned | `0.5 * n_history` | 0.69 nats | `= compression_ratio`, train/inference consistent |

Read `gate_sparsity` and `lm_loss` together — neither alone distinguishes a learned ranking from a
flat gate. The tension to watch: larger `B` means weaker scarcity, and SAS's 18.8-vs-54.4 says
scarcity is load-bearing, so `aligned` may align the objective and still fail to force a ranking.

**Then evaluate retrieval, not just loss.** Both this and §9 are invisible to a loss curve.
Needle-in-a-haystack at 32K is what would show over-suppression of history, or the loss of local
continuity that token-level granularity risks and SAS's blocks preserve.

### Implementation notes

Cheap, and does not touch a kernel. The Triton kernel **loads** `lse` (`tl.load`,
`triton_gated_attention.py:160`) rather than computing it, so passing `lse - ln(B)` at the call
site is sufficient. Verified equivalent to adding `ln(B)` to the gate explicitly (max difference
8.88e-16, fp64) and the gradient path is unchanged because `ln(B)` is a constant. Reference
implementation, backward and `history_lse` all stay as they are.

Pinning still works at `B > 1`, which was the thing worth checking: under a flat gate
(`s_j = c`) the pinned-to-history gap is `ln(n_history / B)`, **independent of `c`**, so the router
cannot reach the no-op by emitting a constant. At `B = 0.5 * n_history` that gap is 0.69 nats — far
smaller than today's 9.01, but not zero.

Suggested interface: `--gate-budget` defaulting to `1.0`, so existing checkpoints, metrics and the
distillation comparison are untouched unless the flag is passed. Note it **changes the objective**,
so any arm other than the control is not comparable against existing `step600.pt` checkpoints.

### Verification status

- **Verified** (analytic / fp64): the suppression formula; `ln(B)`-into-`lse` equivalence; that a
  flat gate cannot reach the no-op for any `B`; SAS's token-level suppression equalling ours.
- **Verified** (real run, 8K): the loss and sparsity table above.
- **Not verified:** that `B = 1` over-suppresses anything in practice at 16K/32K — the whole point
  of the ablation. Also not verified: the TV(train, inference) table, which assumes a perfectly
  ranked router over a synthetic backbone attention profile (sink peak plus local decay), not
  Qwen3-8B's measured distribution.

---

## Fix the `pin_mode="self"` description

`scripts/train_gqa_indexer_e2e*.sh` calls `self` "closest to SAS's always-retained current block".
Misleading twice over: SAS retains a **block** (~64 tokens), not a single token, and it is not a
hard share of the mass. Measured at 8K, `self` puts **0.94** of the attention on the pinned token
(against 0.9996 across 4 sink tokens, but concentrated on the *current* position, which carries
almost no next-token signal): `lm_loss` 12.4939 versus 4.5167 for `sink` at step 0, above
`ln(151936) = 11.93` — worse than uniform guessing. `|g|` 61.19 versus 6.37.

Note the run that produced 12.49 differed from the healthy one **only** in `pin_mode`; the
`gate_scale` fp32 change was ruled out separately (step-0 loss identical to 0.00e+00, and step 0
precedes any optimizer step).

---

## Reconsider `FFN_SP=8` for the 8K stage

`FFN_SP=8` on 8 GPUs is one data-parallel replica, so `--global-batch-size 8` becomes `accum=8`:
eight serial forward+backward passes per step, 64.1 s/step, ~10.7 h for 600 steps. Peak is
**48.3 of 80 GiB**, and `ffn_sp.py`'s own table puts one-GPU 8K at 55.6 GiB — so 8K does not need
sequence parallelism at all.

`FFN_SP` is fixed at launch and does not follow the curriculum, and the schedule moves to 16K at
step 300 where one GPU needs 93.7 GiB. So the options are to split the run (8K at `FFN_SP=1`, then
`--resume-from` at higher SP) or to compromise at `FFN_SP=2` for both. `--global-batch-size 8`
keeps tokens/step at 65536 either way, so the distillation comparison is unaffected.

**Not verified:** whether 16K actually fits at `FFN_SP=2`. The 61.2 GiB figure in that table is
for 8-way SP; 2-way lands somewhere between 61.2 and 93.7 and may exceed 80. Measure before
relying on it.
