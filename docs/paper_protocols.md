# Paper protocols and known gaps

This refactor starts from `Chooovy/kvpress@03b5caf05eefd27f57c2c87a71798c43d4720a09`. Terminology and retained ablations follow manuscript revision `617e4d05e4c0e0a28fdcbb14b7ed4fcfe99c1e58`.

## Execution paths

The historical RULER and LongBench quality measurements use masking with dense KV storage. Math500/AIME25 and the decode benchmark use physical eviction. The two paths differ in their ranking reference row and occupancy at the same nominal budget. This branch preserves each path; it does not claim they produce identical outputs.

RULER budgets use the benchmark length: at compression ratio c and benchmark length L, K=(1-c)L. LongBench uses the configured context-dependent budget ratio and its original KVPress prompt processing. Preserve the dataset split, prompts, truncation, seeds, and aggregation when making comparisons.

The documented final method uses retained-mass allocation and 64 CMP slots within the total budget. Some historical experiment artifacts use other settings:

- Recorded Math500 runs use physical eviction, static head budgets, no CMP, and decode batch 4. The K=512 table uses floor 128; K=1024 uses floor 256. Recorded generation length is 16,384, while the manuscript describes 32,768. The K=512 record includes 696 resumed traces; a new run does not reproduce that interrupted sampling stream.
- Recorded AIME25 runs use physical eviction, static head budgets, no CMP, decode batch 4, and generation length 32,768. K=1024 uses floor 256; K=2048 uses floor 512.
- The evaluation presets expose those recorded settings. They are not a claim that the final-method defaults reproduce every published number.

## Retained ablations

The implementation retains linear/MLP/conv/RNN/prefix/KVzip scorers; CE, LongCE, forward KL, reverse KL, and the 0.9 RKL + 0.1 CE mixture; protected-region and retention-rate controls; uniform, permuted, offline, and mass head allocation; CMP on/off; existing joint CE training; and the standalone Linear/TTT compensation diagnostic.

The no-protected-region table entry comes from CE training. It is not a reverse-KL/no-pin experiment. Offline allocation tables must come from the matching model, training data, total budget, and floor. Regenerating a table without the original fitting inputs does not establish exact reproduction of the paper row.

## Unresolved source differences

- The manuscript reports reverse-KL joint training, but the source branch contains only a joint CE training entrypoint. This refactor does not add a new reverse-KL joint implementation.
- Joint-training token counts and some Full-KV reference values disagree between the current TeX and older result records. Existing functionality is retained; the discrepant rows are not presented as reproduced.
- Some scorer parameter counts in the paper do not match the stated pilot model. Checkpoint geometry, not the table's parameter-count column, determines model construction.
- The tracked compensation diagnostic and manuscript cosine ranges do not have a complete one-to-one provenance mapping. The diagnostic remains runnable without claiming that its default inputs recreate those ranges.

No paper text, metrics, or historical result files are changed by this branch.
