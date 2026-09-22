# Validation

The reference is 03b5caf05eefd27f57c2c87a71798c43d4720a09; paper terminology follows 617e4d05e4c0e0a28fdcbb14b7ed4fcfe99c1e58. Checks were run on September 22, 2026. [The source manifest](validation/source_manifest.json) identifies the verified code independently of this report's commit.

## Environments and automated checks

Local checks used Python 3.12, PyTorch 2.8.0, and Transformers 4.57.3. CUDA checks ran on NVIDIA H20 with Python 3.11, PyTorch 2.8.0+cu128, Transformers 4.57.3, Triton 3.4.0, FlashAttention 2.8.3, and Liger 0.6.2.

The original scalar-retention, cache, and CMP tests passed before editing: 77 passed. The final H20 suite passed all 184 tests with zero skips; local results were 160 passed and 24 CUDA skips. Full totals are recorded in [test results](validation/tests.json). Local CUDA skips are counted separately; the complete suite was also run on H20 without CUDA skips.

Wheel construction and installation into a separate directory passed. Public imports, ordinary KVPress compression, and actual Math500/AIME25 metric imports work without an evaluation-script directory on the Python path. All eleven [CLI help checks](validation/cli.json) passed. Fire-based evaluation entrypoints use the double-dash separator before their help flag.

The upstream GitHub Actions workflows are retained unchanged. The validation reported here was executed locally and on H20; it is not a GitHub Actions result.

## Numerical preservation

| Check | Result |
| --- | --- |
| Six scorer structures, float32/bfloat16, two layers | Converted weights, outputs, input gradients, and parameter gradients match exactly |
| Complete checkpoint schemas | Frozen/joint conversion, native save/load, layer counts, gate scale, and full adapted-backbone loading pass |
| Runtime mathematics and state | 426 old/new tensor comparisons are exactly equal |
| Qwen3-4B and Qwen3-8B, masking and eviction separately | Final prefill-row logits, all 16 decode logits, and generated tokens are exactly equal |
| Eviction state, both pretrained models | Retained positions and CMP cluster counts match after every decode step |
| Training normalizer and KL | Old/new values and gradients have maximum absolute difference 0 |
| Native training resume | Step 1 to step 2 equals uninterrupted training, maximum scorer-weight difference 0 |

The pretrained-model comparisons use the same trained checkpoint, seed 719 input of 1024 tokens, total budget 512, sink size 4, window size 128, CMP capacity 64, mass-allocation floor 128, and 16 greedy decode steps. Prefill compares the last context row before the first decode forward. Checkpoint hashes identify the two source weights. These comparisons are per execution path; they do not establish equality between masking and eviction.

Evidence: [scorer conversion](validation/checkpoint_parity.json), [runtime](validation/runtime_parity.json), [4B](validation/qwen3_4b_parity.json), [8B](validation/qwen3_8b_parity.json), [checkpoint hashes](validation/checkpoints.json), and [training](validation/training.json).

## Coverage

Runtime tests cover empty and short histories, cache growth and saturation, ragged head budgets, shared-context question batches of different lengths, current-token causal attention, CMP initialization and updates, budget conservation, and maintenance after attention reads. Training checks cover all retained scorer architectures and objectives, protected regions, frozen-backbone gradients, and two-rank FFN sequence-parallel gradients.

A two-rank joint CE run exercised FSDP and FFN sequence parallelism. Its adapted checkpoint loaded strictly; eight backbone tensors changed while FFN, embeddings, and the LM head stayed unchanged. Tokenization, teacher-cache creation, LongCE-weight preparation, cache reading, and a Liger-enabled LongCE step also completed.

The decode benchmark CLI completed for Full-KV and IndexMem++ on Qwen3-4B at 8192 input tokens, batch 1, one document, one warmup, two repeats, and four steps. Each method's repeated token hashes matched, with no non-finite logits; CMP populations increased by the expected amount. [This is an execution smoke](validation/benchmark_smoke.json), not a repeat of the paper's throughput measurements.

## Deliberate changes and limits

The offline budget fitter previously applied its unclipped floor again after integer allocation, which could increase the total budget for an infeasible floor. That redundant clamp is removed. Tables for valid historical configurations remain identical; the boundary case now conserves the requested budget. [The comparison](validation/offline_budget_parity.json) records both behaviors.

No full training recipe, long-context quality benchmark, or paper throughput sweep was rerun. The real pretrained-model comparisons used 4B and 8B; 0.6B shares the supported Qwen3 implementation but was not independently evaluated with pretrained weights in this pass. Missing joint reverse-KL code and inconsistent historical recipes remain listed in [paper protocols](paper_protocols.md).

The public code has no legacy API aliases or checkpoint inference. Explanatory comments and docstrings are removed from IndexMem++ code; licensing and tool directives remain. Generic KVPress components retain their existing responsibilities.
