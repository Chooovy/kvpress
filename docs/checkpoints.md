# Checkpoints

New checkpoints contain an explicit `scorer_config`, strict per-layer `scorers` state, and the training step. Saving expands the actual scorer configuration, including defaults used during construction. Training checkpoints can additionally contain optimizer and scheduler state. Joint-training checkpoints also contain the adapted backbone.

Convert a legacy checkpoint using a JSON configuration that records the original scorer geometry and constants:

```json
{
  "kind": "mlp",
  "hidden_size": 4096,
  "n_heads": 8,
  "mid_dim": 256,
  "norm_eps": 0.00001,
  "pos_slope": 0.000001,
  "gate_scale": true,
  "decay": true,
  "age_scale": 16384.0,
  "decay_init": -1.0
}
```

This example describes a Qwen3-8B MLP scorer. Use the actual model geometry and training recipe for your checkpoint; the example is not a substitute for missing metadata. Every field shown above is required, including settings disabled for that checkpoint. Alternative configurations also require these architecture fields:

| `kind` | Additional required fields |
| --- | --- |
| `mlp` | None |
| `linear` | None; `mid_dim` must be `0` |
| `conv` | `conv_kernel`, `conv_dim`, `exclude_self`, `zero_init_conv` |
| `rnn` | `state_dim`, `gate_mode`, `gate_bias`, `fixed_half_life`, `zero_init_state` |
| `prefix` | `head_dim`, `value_dim`, `zero_init_prefix` |
| `kvzip` | `kvzip_dim`, `kvzip_base`, `kvzip_ngroup` |

```bash
python -m scripts.convert_checkpoint old_final.pt checkpoints/indexmem.pt --scorer-config scorer.json
```

The converter accepts legacy `indexer` keys of the form `model.layers.N.self_attn.indexer.PARAMETER`, with contiguous layers starting at zero. It maps parameter names, normalizes gate tensors to shape `(1,)` in float32, and verifies every layer against the selected scorer. Position slope, age scale, normalization epsilon, and architecture settings must be supplied explicitly. They cannot be recovered reliably from tensor shapes.

Conversion preserves legacy configuration under `source_config`, resets the new training step to zero, and omits legacy optimizer and scheduler state. Use converted weights with `--init-from`; use `--resume` only with a checkpoint written by the new training code. Joint conversions require the full adapted backbone and remove sequence-parallel wrapper names. Loading validates the full backbone and scorer layer count against the target model. Loading only its scorer into an unmodified backbone would evaluate a different model.

The runtime does not load old formats, infer scorer types, or accept missing/unexpected parameter keys.
