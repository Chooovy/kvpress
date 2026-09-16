# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train the router **and the backbone's attention** together, from the plain LM loss.

The counterpart of :mod:`scripts.train_gqa_indexer_e2e`, which freezes the backbone and moves
only the router. This is SP-KV's setting: the language model and the utility predictor are
optimized jointly, so the model can *adapt* to being gated rather than merely tolerating it.

Why joint training at all
-------------------------
SP-KV's Appendix C.6 measures exactly the frozen-backbone configuration this repo has been using:
freeze the LLM, train only the utility predictor for 20 TPP at the same LR, and average gate
density stays **above 80%** -- sparsity barely emerges. Their sparsity is free precisely *because*
the backbone trains: closing a gate lowers the loss once the model has reorganized around it, so
no auxiliary sparsity loss and no budget are needed (their gate is an unnormalized
``log sigmoid``, which *can* go flat and simply does not).

This repo instead forces sparsity with ``log_softmax + pin + budget``. That works -- the router
this script starts from reached ``gate_sparsity`` 0.267 -- but it is pressure applied from
outside. Unfreezing the backbone is the one lever SP-KV has that a frozen run cannot get.

What is trainable, and the constraint that decides it
-----------------------------------------------------
**Attention projections (q/k/v/o) + norms + router. The FFN, embeddings and lm_head stay frozen.**

The FFN exclusion is a *correctness* requirement, not a memory tradeoff. Without activation
checkpointing this run needs FFN sequence parallelism to fit, and
:class:`~kvpress.presses.gqa_indexer.ffn_sp._ScatterSequence` is only correct while the FFN holds
no trainable parameter. Its backward all-gathers, so every rank in the SP group ends up with the
*identical* complete gradient -- which is what makes FSDP's uniform reduction right. Were the FFN
weights trainable they would need the opposite treatment: each rank sees only its 1/8 of the
sequence, so their gradients are complementary and want a SUM while every attention-path gradient
wants a MEAN. One reduction cannot serve both, and getting it wrong is silent -- the FFN gradient
would simply be ``sp_size`` times too small and the loss would still descend. (The same
per-path scale mismatch is what ``_ScatterSequence`` was written to fix in the first place;
its docstring records cosine 0.98 against the dense gradient, i.e. a wrong *direction*.)

That leaves 1.51B trainable backbone parameters of 8.19B (18%): 41.9M per layer of attention
against the MLP's 151M. Gating acts inside attention, so this is also the part of the backbone
most directly implicated in adapting to it.

Parallelism
-----------
``HYBRID_SHARD``: FSDP shards parameters/gradients/optimizer state **within** a node and
replicates **across** nodes, so the all-gather traffic stays on NVLink and only the
reduce-scatter crosses the network. Composed with ``--ffn-sp-size`` inside the node, which is a
*different axis* (sequence split) from FSDP's (parameter shard) and therefore composes.

Note what HYBRID_SHARD does not do: model state per rank does **not** fall with node count, since
sharding is intra-node only. Extra nodes buy throughput, not memory.

Two learning rates
------------------
SP-KV gives the utility predictor ``5x`` the global LR (their Table 4 baseline; the ablation reads
density 82.7% at multiplier 0.1, 37.8% at 1, 25.4% at 5). That ratio is reproduced here through
``--backbone-lr`` and ``--router-lr-mult``, but the *absolute* backbone LR must be far below
SP-KV's 5.02e-4: theirs is a pretraining LR spending 20 TPP, this is a short adaptation on a
model that is already trained. Default 2e-5.

Data
----
``--seed`` picks the data stream. The loader is deterministic in ``seed + seq_len``
(``loader_for``), so a fresh seed gives an entirely different shard order *and* row order. This
matters because the router being loaded already consumed 2400 documents of this corpus under
``--seed 0``; at 373K documents that is 0.64%, so a new seed's expected overlap is ~15 documents.
Use ``--seed 1000`` (the default here) rather than 0.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer import (  # noqa: E402
    DEFAULT_DECAY_INIT,
    DEFAULT_DECAY_REF,
    DEFAULT_POS_SLOPE,
    PIN_MODES,
    E2EIndexerTrainer,
    e2e_indexer_training_step,
    load_indexer_state_dict,
)
from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    SUBSETS,
    LengthSchedule,
    read_index,
    wsd_lr_lambda,
)
from kvpress.presses.gqa_indexer.gate_pin import DEFAULT_N_LOCAL  # noqa: E402
from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402

from scripts.train_gqa_indexer import (  # noqa: E402
    all_reduce_mean,
    build_model,
    loader_for,
    setup_distributed,
)
from scripts.train_gqa_indexer_e2e import apply_liger_fused_ce, ffn_sp_group  # noqa: E402

logger = logging.getLogger("train_gqa_indexer_joint")


def attention_modules(model):
    """The ``self_attn`` module of every layer -- where the trainable backbone weights live."""
    return [layer.self_attn for layer in get_language_model(model).layers]


def split_trainable(model, scorer_attr: str, scope: str = "attention", freeze_router: bool = False):
    """
    Partition parameters into ``(router, backbone_attention)`` and freeze everything else.

    Returns two disjoint lists. Identification is by **module identity**, never by parameter-name
    substring: a name filter would also catch a frozen parameter whose name happens to contain
    ``mlp`` or the scorer attribute, and would train it silently.

    ``freeze_router=True`` returns an EMPTY router list and leaves every router parameter frozen.
    That removes an ADVERSARIAL dynamic rather than merely one variable: router and backbone
    minimize the same LM loss, and with 1.51B trainable backbone parameters against the router's
    38M there are two ways to lower it -- the router learns to spend its budget on the right keys
    (hard), or the backbone re-spreads attention through q/k so the gate's constraint stops
    binding (easy, and 40x more parameters to do it with). The first joint run took the second
    route: gate_sparsity went 0.282 -> 0.359 (the frozen arm reached 0.267), i.e. the router got
    LESS selective while the loss fell, and RULER 8K dropped 77.62 -> 69.17 with the whole loss
    confined to needle retrieval. Freezing the router makes that route unavailable.

    ``scope="attention"`` (the default) trains q/k/v/o + norms + router and freezes the FFN,
    embeddings and lm_head. The FFN exclusion is a *correctness* requirement whenever FFN-SP is
    on -- see the module docstring on ``_ScatterSequence``.

    ``scope="all"`` additionally trains the FFN, embeddings and lm_head. It is **only valid with
    ``--ffn-sp-size 1``**, which ``main`` enforces: with FFN-SP the two groups need opposite
    gradient reductions (a sharded FFN wants SUM, every attention path wants MEAN) and one
    reduce-scatter cannot serve both. Note also that a trainable ``lm_head`` makes ``--liger``'s
    fused kernel allocate its ``(vocab, hidden)`` fp32 ``grad_weight``, which the frozen path
    skips entirely.
    """
    if scope not in ("attention", "all"):
        raise ValueError(f"scope must be 'attention' or 'all', got {scope!r}")
    model.requires_grad_(False)

    router_params, router_ids = [], set()
    frozen_gate_scales = 0
    for attn in attention_modules(model):
        indexer = getattr(attn, scorer_attr, None)
        if indexer is None:
            continue
        gate_scale = getattr(indexer, "gate_scale", None)
        for name, param in indexer.named_parameters():
            # gate_scale is claimed by the indexer but deliberately NOT trained here, and it is
            # the one parameter that cannot be. FSDP flattens each layer into one FlatParameter
            # and requires a single dtype, so a trainable gate_scale would have to be either:
            #   * fp32 while the layer is bf16 -> "Must flatten tensors with uniform dtype", or
            #   * bf16 like the layer -> the exact rounding failure upcast_gate_scales exists to
            #     prevent (bf16 spacing near the init value is ~1e-4 against an AdamW step of
            #     ~lr, so every update rounds straight back and the scalar freezes anyway).
            # Freezing resolves both honestly rather than keeping a parameter that reports as
            # trained while standing still. Its VALUE still comes from the checkpoint.
            if gate_scale is not None and param is gate_scale:
                param.requires_grad = False
                router_ids.add(id(param))
                frozen_gate_scales += 1
                continue
            # Claimed either way, so the backbone loop below cannot pick these up a second time.
            router_ids.add(id(param))
            if freeze_router:
                param.requires_grad = False
                continue
            param.requires_grad = True
            router_params.append(param)
    if not router_ids:
        raise RuntimeError(
            f"no {scorer_attr!r} modules found; call press.post_init_from_model(model) first"
        )
    if not router_params and not freeze_router:
        raise RuntimeError(
            f"no trainable {scorer_attr!r} parameters found; call "
            "press.post_init_from_model(model) first"
        )
    if frozen_gate_scales:
        logger.info(
            "froze %d gate_scale scalar(s): FSDP needs one dtype per flattened layer, and a bf16 "
            "gate_scale cannot be moved by an AdamW step at this LR anyway. Value is inherited "
            "from the checkpoint; gate_scale_mean in the metrics is therefore constant by design.",
            frozen_gate_scales,
        )

    backbone_params = []
    for attn in attention_modules(model):
        for param in attn.parameters():
            # attn.parameters() recurses into the indexer, which is already claimed.
            if id(param) in router_ids:
                continue
            param.requires_grad = True
            backbone_params.append(param)

    # Layer norms sit outside self_attn but are tiny and directly rescale what attention reads,
    # so they ride along with it rather than staying frozen mid-block.
    for layer in get_language_model(model).layers:
        for name, module in layer.named_children():
            if "norm" not in name.lower():
                continue
            for param in module.parameters():
                param.requires_grad = True
                backbone_params.append(param)

    if scope == "all":
        # Everything still frozen at this point is the FFN, the embeddings, lm_head and the final
        # norm. Claimed by identity against what is already collected rather than by name, so a
        # parameter cannot be counted twice or missed because a name pattern drifted.
        claimed = router_ids | {id(p) for p in backbone_params}
        for param in model.parameters():
            if id(param) in claimed:
                continue
            param.requires_grad = True
            backbone_params.append(param)

    return router_params, backbone_params


def build_joint_optimizer(router_params, backbone_params, args):
    """
    AdamW with two param groups and one shared WSD multiplier.

    The groups differ only in base LR, so ``LambdaLR`` -- which scales every group by the same
    factor -- keeps SP-KV's ``router = mult x backbone`` ratio fixed for the whole run rather than
    letting the two schedules drift apart.
    """
    router_lr = args.backbone_lr * args.router_lr_mult
    groups = [{"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"}]
    # Omitted entirely when the router is frozen. An empty param group is legal but makes
    # optimizer.param_groups[1]["lr"] a number that moves while nothing consumes it, which the
    # metrics would report as a live router LR.
    if router_params:
        groups.append({"params": router_params, "lr": router_lr, "name": "router"})
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    # final_lr is expressed as a fraction of peak so both groups decay to the same relative floor.
    lr_lambda = wsd_lr_lambda(
        args.total_steps,
        warmup_frac=args.warmup_frac,
        stable_frac=args.stable_frac,
        peak_lr=1.0,
        final_lr=args.final_lr_frac,
    )
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), router_lr


def wrap_fsdp(model, args, sharding_group, replicate_group):
    """
    Wrap in FSDP ``HYBRID_SHARD``, one flat parameter group per decoder layer.

    ``use_orig_params=True`` is required, not preferred: the optimizer holds two param groups
    keyed on the *original* Parameter objects, and FSDP's default flattening replaces them with
    one opaque flat tensor per unit -- which would collapse both groups into one LR.

    Mixed precision keeps parameters and reductions in bf16 to match the frozen-backbone runs'
    numerics, except that the reduce is left in fp32: gradients for 1.51B parameters accumulate
    across 8 ranks, and bf16's 8 mantissa bits lose the small contributions.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_cls = type(get_language_model(model).layers[0])
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})

    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        process_group=(sharding_group, replicate_group),
        mixed_precision=MixedPrecision(
            param_dtype=getattr(torch, args.dtype),
            reduce_dtype=torch.float32,
            buffer_dtype=getattr(torch, args.dtype),
        ),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=False,
    )


def save(path: Path, model, args, step: int, extra: dict | None = None) -> None:
    """
    Write the **entire model** plus a router-only view, under FSDP.

    Joint training moves 1.51B attention parameters, so the backbone is part of the trained
    artefact: a router evaluated against the pretrained backbone is measuring a model that no
    longer exists. ``payload["model"]`` is therefore the full state dict (~16 GB in bf16), and
    ``payload["indexer"]`` is a redundant router-only view kept so the press and
    ``load_indexer_state_dict`` keep working unchanged.

    ``offload_to_cpu=True, rank0_only=True`` means rank 0 materializes the whole model in host
    memory here -- ~16 GB at bf16. That is the cost of a single-file checkpoint; switch to
    ``StateDictType.SHARDED_STATE_DICT`` if host memory on the launcher node is tight.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        full = model.state_dict()

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # THE WHOLE MODEL, not a diff. Joint training moves 1.51B attention parameters, so a router
    # evaluated against the ORIGINAL backbone measures a model that no longer exists -- the
    # checkpoint has to carry the adapted backbone with it.
    #
    # Saved by taking `full` wholesale rather than by selecting trainable names. The previous
    # version intersected against `inner.named_parameters()` and silently produced an EMPTY
    # backbone: FSDP auto-wraps each decoder layer, so those names carry a
    # `_fsdp_wrapped_module.` prefix, while FULL_STATE_DICT hands back CLEAN names. The exact-set
    # test therefore never matched. (The indexer survived only because it was selected by
    # substring, which is prefix-insensitive.) No name matching, no such failure.
    # Normalize away FFN sequence parallelism's wrapper before writing. SequenceParallelFFN holds
    # the real module as `self.inner`, so a wrapped run saves `mlp.inner.gate_proj.weight` while
    # every consumer -- the eval, a plain from_pretrained, anything not running FFN-SP -- expects
    # `mlp.gate_proj.weight`. Keeping the prefix would make the checkpoint's shape depend on a
    # training-time memory optimization, and the failure it causes downstream ("108 keys the model
    # does not accept") names the wrapper rather than the cause. 36 layers x 3 projections.
    wrapped = [k for k in full if ".mlp.inner." in k]
    for key in wrapped:
        full[key.replace(".mlp.inner.", ".mlp.")] = full.pop(key)
    if wrapped:
        logger.info(
            "unwrapped %d FFN-SP key(s) before saving (mlp.inner.* -> mlp.*), so the checkpoint "
            "loads into a model that is not running sequence parallelism",
            len(wrapped),
        )

    scorer = f".{args.scorer_attr}."
    payload = {
        "model": full,
        # Kept alongside, and deliberately redundant: this is the layout the press and
        # `load_indexer_state_dict` expect, so an eval that only wants the router does not have to
        # know about the joint format. 38M of 8.19B, so the duplication is ~76 MB.
        "indexer": {k: v for k, v in full.items() if scorer in k},
        "step": step,
        "config": {
            "objective": "joint_lm_loss",
            "joint": True,
            "train_scope": args.train_scope,
            # Not recoverable from the weights: a frozen router's tensors are simply equal to
            # --init-from's, which is indistinguishable from a run that trained it and barely
            # moved. The two are different experiments, so the distinction is recorded.
            "freeze_router": args.freeze_router,
            # The training-time gate geometry. NOT recoverable from the weights, and it changes
            # what the backbone was adapted to, so an eval at a different topk is a different
            # experiment rather than a different measurement.
            "hard_topk": args.hard_topk or None,
            "trainable": (
                "attention+norms+router (ffn/embed/lm_head frozen)"
                if args.train_scope == "attention"
                else "ALL parameters (ffn/embed/lm_head included)"
            ),
            # A full model state dict is present under "model"; "indexer" is a redundant view of
            # the router subset. An eval MUST load "model" -- the backbone is no longer the
            # pretrained one.
            "payload": "full_model+indexer",
            "model": args.model,
            "scorer": args.scorer,
            "scalar_mid_dim": args.scalar_mid_dim,
            "scalar_pos_slope": args.scalar_pos_slope,
            "scalar_decay": args.scalar_decay,
            "scalar_decay_ref": args.scalar_decay_ref if args.scalar_decay else None,
            "scalar_decay_init": args.scalar_decay_init if args.scalar_decay else None,
            "stage": "dense",
            "pin_mode": args.pin_mode,
            "n_sink": args.n_sink,
            "n_local": args.n_local,
            "gate_budget": args.gate_budget,
            "gate_budget_ratio": args.gate_budget_ratio,
            "schedule": args.schedule,
            "subsets": list(args.subsets),
            "backbone_lr": args.backbone_lr,
            "router_lr_mult": None if args.freeze_router else args.router_lr_mult,
            "ffn_sp_size": args.ffn_sp_size,
            "sharding": "HYBRID_SHARD",
            # The data stream. Load-bearing for the no-reuse claim: the router this run starts
            # from consumed seed 0's first 2400 documents.
            "seed": args.seed,
            "take_from": args.take_from,
            "init_from": args.init_from,
        },
    }
    if extra:
        payload["metrics"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("saved %s (step %d)", path, step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True)
    data.add_argument("--tokenized", default=None)
    data.add_argument("--subsets", nargs="+", default=["2e16", "2e17"], choices=list(SUBSETS))
    data.add_argument(
        "--take-from", choices=("head", "random"), default="head",
        help="'head' matches the LongCE-trained router this starts from, so a document drawn "
        "again yields the same window; 'random' additionally decorrelates the window.",
    )
    data.add_argument("--shuffle-buffer", type=int, default=64)
    data.add_argument("--min-tokens", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--batch-size", type=int, default=1)
    data.add_argument(
        "--seed", type=int, default=1000,
        help="data stream selector AND torch seed. NOT 0: the router being loaded already "
        "consumed seed 0's first 2400 documents of this corpus (0.64%% of 373K), and the loader "
        "is deterministic in seed + seq_len, so reusing 0 would replay exactly those.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen3-8B")
    model_group.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    model_group.add_argument("--attn", default="sdpa")
    model_group.add_argument("--compression-ratio", type=float, default=0.5)
    model_group.add_argument("--scorer", choices=("pairwise", "scalar", "prefix", "dma", "kvzip", "conv", "rnn"), default="scalar")
    model_group.add_argument("--scalar-mid-dim", type=int, default=256)
    model_group.add_argument("--scalar-pos-slope", type=float, default=DEFAULT_POS_SLOPE)
    model_group.add_argument("--scalar-decay", action="store_true")
    model_group.add_argument("--scalar-decay-ref", type=float, default=DEFAULT_DECAY_REF)
    model_group.add_argument("--scalar-decay-init", type=float, default=DEFAULT_DECAY_INIT)
    model_group.add_argument("--press-n-sink", type=int, default=4)
    model_group.add_argument("--scorer-attr", default="indexer")
    model_group.add_argument(
        "--init-from", default=None,
        help="load router WEIGHTS from a frozen-backbone checkpoint. The recorded scorer must "
        "match, and --scalar-decay / --n-local / --gate-budget must match what trained it: the "
        "first is a parameter shape, the others silently change what the score means.",
    )
    model_group.add_argument(
        "--freeze-router",
        action="store_true",
        help="freeze the router and train ONLY the backbone. Removes the adversarial dynamic the "
        "first joint run hit: router and backbone minimize the same loss, and the backbone (1.51B "
        "params vs the router's 38M) can lower it by re-spreading attention through q/k instead of "
        "letting the router concentrate. Measured: gate_sparsity rose 0.282 -> 0.359 (frozen arm: "
        "0.267) while the loss fell, and RULER 8K went 77.62 -> 69.17 with every point of the loss "
        "in needle retrieval. With the router fixed, the backbone must adapt TO the gating instead "
        "of around it. router_lr_mult is ignored.",
    )
    model_group.add_argument(
        "--train-scope", choices=("attention", "all"), default="attention",
        help="which backbone parameters to train. 'attention' = q/k/v/o + norms + router (1.51B "
        "of 8.19B), and is the only scope compatible with FFN sequence parallelism. 'all' adds "
        "the FFN, embeddings and lm_head (8.19B) and REQUIRES --ffn-sp-size 1: with FFN-SP the "
        "sharded FFN's gradient wants a SUM while every attention-path gradient wants a MEAN, and "
        "a single reduce-scatter cannot do both -- the FFN gradient would come out sp_size times "
        "too small with the loss still descending.",
    )
    model_group.add_argument(
        "--ffn-sp-size", type=int, default=8,
        help="shard FFN activations across this many intra-node ranks. REQUIRED to fit at 8K "
        "without activation checkpointing, and the reason the FFN weights stay frozen -- see the "
        "module docstring.",
    )
    model_group.add_argument("--liger", action="store_true")

    gate = parser.add_argument_group("gate")
    gate.add_argument("--pin-mode", choices=list(PIN_MODES), default="local+sink")
    gate.add_argument("--n-sink", type=int, default=4)
    gate.add_argument("--n-local", type=int, default=DEFAULT_N_LOCAL)
    budget = gate.add_mutually_exclusive_group()
    budget.add_argument("--gate-budget", type=float, default=256.0)
    budget.add_argument("--gate-budget-ratio", type=float, default=None)
    gate.add_argument(
        "--hard-topk", type=int, default=0,
        help="train on HARD eviction with this per-row budget instead of the soft gate: keys "
        "outside each row's top-k are removed with -inf, not down-weighted. This is the geometry "
        "inference runs, and the gap matters -- with the router training, the backbone lowered the "
        "LM loss by re-spreading attention until the gate stopped binding (gate_sparsity 0.282 -> "
        "0.359) and RULER 8K fell 77.62 -> 69.17, while topk=8192 still matched dense (93.71 vs "
        "93.69). It routed AROUND a soft constraint that is absolute at inference. REQUIRES "
        "--freeze-router (the threshold is only equivalent to a top-k while scores are fixed) and "
        "a query-independent scorer. Match it to the eval's --topk. 0 keeps the soft gate.",
    )
    gate.add_argument("--key-tile", type=int, default=1024)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--schedule", default="8192:300")
    optim.add_argument("--max-steps", type=int, default=0)
    optim.add_argument(
        "--backbone-lr", type=float, default=2e-5,
        help="peak LR for the attention projections. Deliberately ~25x below SP-KV's 5.02e-4, "
        "which is a PRETRAINING LR spending 20 TPP; this is a short adaptation of an already "
        "trained model, and the global batch here is ~64x smaller than theirs.",
    )
    optim.add_argument(
        "--router-lr-mult", type=float, default=5.0,
        help="router LR as a multiple of --backbone-lr. 5 is SP-KV's value (Table 4 baseline); "
        "their ablation reads gate density 82.7%% at 0.1, 37.8%% at 1, 25.4%% at 5.",
    )
    optim.add_argument("--final-lr-frac", type=float, default=0.01)
    optim.add_argument("--warmup-frac", type=float, default=0.10)
    optim.add_argument("--stable-frac", type=float, default=0.60)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--accum-steps", type=int, default=1)
    optim.add_argument("--global-batch-size", type=int, default=0)

    io = parser.add_argument_group("io")
    io.add_argument("--out", default="checkpoints/gqa_indexer_joint")
    io.add_argument("--save-every", type=int, default=100)
    io.add_argument("--log-every", type=int, default=10)
    io.add_argument("--metrics-file", default=None)
    io.add_argument("--gate-sparsity", action="store_true")
    io.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        parser.error("no CUDA device")
    if args.hard_topk and not args.freeze_router:
        # Refused, not warned. A training router changes scores inside the step, so a threshold
        # computed at its start no longer marks the top-k -- and an evicted key receives no
        # gradient, so the router could not correct course anyway. hard_evict.py enforces this
        # too; caught here so it fails before a device is touched.
        parser.error(
            f"--hard-topk {args.hard_topk} needs --freeze-router. The hard mask is a per-row "
            "score THRESHOLD, which equals a per-row top-k only while the scores are fixed; with "
            "the router training the boundary goes stale within the step. An evicted key also "
            "gets zero gradient, so a training router could never learn to re-select it."
        )
    if args.train_scope == "all" and args.ffn_sp_size > 1:
        # Refused rather than warned: this combination is silently WRONG, not merely slow. See
        # --train-scope, and _ScatterSequence's docstring for the measurement (cosine 0.98 against
        # the dense gradient, with no single divisor able to repair it).
        parser.error(
            f"--train-scope all needs --ffn-sp-size 1 (got {args.ffn_sp_size}). With FFN-SP each "
            "rank sees only 1/sp_size of the sequence, so a trainable FFN's gradient wants a SUM "
            "while every attention-path gradient wants a MEAN; one reduce-scatter cannot serve "
            "both and the FFN gradient ends up sp_size times too small WITHOUT any error. Pass "
            "--ffn-sp-size 1 (expect much higher activation memory) or --train-scope attention."
        )
    if args.seed == 0:
        parser.error(
            "--seed 0 is the stream the loaded router already trained on (2400 documents). "
            "Pass a different seed, e.g. --seed 1000."
        )

    schedule = LengthSchedule.parse(args.schedule)
    args.total_steps = schedule.total_steps
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.WARNING)

    if world_size == 1:
        parser.error("this script needs torchrun: FSDP HYBRID_SHARD requires a process group")

    # Intra-node FFN sequence-parallel group, and the data-parallel identity beside it.
    sp_group, dp_rank, dp_world_size, sp_rank = ffn_sp_group(world_size, args.ffn_sp_size, rank)

    # FSDP HYBRID_SHARD takes (shard_group, replicate_group). Shard within the node so the
    # all-gather stays on NVLink; replicate across nodes so only reduce-scatter crosses the wire.
    gpus_per_node = torch.cuda.device_count()
    if world_size % gpus_per_node:
        parser.error(
            f"world_size={world_size} is not a multiple of {gpus_per_node} GPUs per node; "
            "HYBRID_SHARD needs whole nodes to shard within"
        )
    n_nodes = world_size // gpus_per_node
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (n_nodes, gpus_per_node), mesh_dim_names=("replicate", "shard"))
    shard_group = mesh.get_group("shard")
    replicate_group = mesh.get_group("replicate")
    if rank == 0:
        logger.info(
            "HYBRID_SHARD: %d node(s) x %d GPUs. Sharding intra-node (all-gather on NVLink), "
            "replicating across nodes. NOTE model state per rank does NOT fall with node count.",
            n_nodes, gpus_per_node,
        )
        if args.ffn_sp_size > 1:
            logger.info(
                "FFN sequence parallel: sp_size=%d -> %d data-parallel replica(s). Ranks in one "
                "SP group read the SAME sequence.", args.ffn_sp_size, dp_world_size,
            )

    torch.manual_seed(args.seed + dp_rank)
    out_dir = Path(args.out)

    if args.tokenized:
        index = read_index(args.tokenized)
        if rank == 0:
            logger.info(
                "pre-tokenized corpus: %d docs at seq_len<=%d, subsets %s; reading stream "
                "seed=%d (the loaded router used seed 0)",
                index["total_docs"], index["seq_len"], index["subsets"], args.seed,
            )

    model, tokenizer = build_model(args.model, getattr(torch, args.dtype), args.attn, device)
    if args.liger:
        apply_liger_fused_ce(model, args.model)

    press_kwargs = {
        "compression_ratio": args.compression_ratio,
        "scorer_attr": args.scorer_attr,
        "gate_scale": True,
        "n_sink": args.press_n_sink,
        "scorer": args.scorer,
    }
    if args.scorer in ("scalar", "prefix"):
        press_kwargs["scalar_mid_dim"] = args.scalar_mid_dim
        press_kwargs["scalar_pos_slope"] = args.scalar_pos_slope
        if args.scorer == "scalar":
            press_kwargs["scalar_decay"] = args.scalar_decay
            press_kwargs["scalar_decay_ref"] = args.scalar_decay_ref
            press_kwargs["scalar_decay_init"] = args.scalar_decay_init
    press = GQAIndexerPress(**press_kwargs)
    press.post_init_from_model(model)

    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu")
        ckpt = payload.get("config") or {}
        ckpt_scorer = ckpt.get("scorer")
        if ckpt_scorer is not None and ckpt_scorer != args.scorer:
            raise SystemExit(
                f"--init-from was trained with scorer={ckpt_scorer!r} but this run is "
                f"scorer={args.scorer!r}; the parameter names differ, so nothing would load."
            )
        # These do not change any tensor SHAPE, so a mismatch loads cleanly and silently trains a
        # router whose score means something else than the one being continued.
        for key, mine in (
            ("n_local", args.n_local),
            ("gate_budget", args.gate_budget),
            ("pin_mode", args.pin_mode),
            ("scalar_decay", args.scalar_decay),
        ):
            theirs = ckpt.get(key)
            if theirs is not None and theirs != mine:
                raise SystemExit(
                    f"--init-from recorded {key}={theirs!r} but this run passes {mine!r}. The "
                    f"gate geometry decides what the score means, and every tensor would still "
                    f"load -- pass the checkpoint's value, or start fresh without --init-from."
                )
        load_indexer_state_dict(model, payload.get("indexer", payload), args.scorer_attr)
        if rank == 0:
            logger.info(
                "loaded router from %s (step %s, gate_sparsity at save: %s)",
                args.init_from, payload.get("step"),
                (payload.get("metrics") or {}).get("gate_sparsity_mean", "n/a"),
            )

    trainer = E2EIndexerTrainer(
        press=press,
        stage="dense",
        pin_mode=args.pin_mode,
        n_sink=args.n_sink,
        n_local=args.n_local,
        gate_budget=args.gate_budget,
        gate_budget_ratio=args.gate_budget_ratio,
        hard_topk=args.hard_topk or None,
        key_tile=args.key_tile,
        # Nothing is frozen by the trainer here: split_trainable below owns that decision, and
        # letting hooks() re-freeze would undo it on every forward.
        freeze=False,
    )

    # NOT calling trainer.upcast_gate_scales(): that promotes gate_scale to fp32 so an AdamW step
    # can move it, which is right for the frozen-backbone scripts and wrong here. FSDP flattens a
    # layer into one FlatParameter and demands a uniform dtype, so an fp32 scalar among bf16
    # weights raises outright. gate_scale is frozen instead -- see split_trainable.
    router_params, backbone_params = split_trainable(
        model, args.scorer_attr, scope=args.train_scope, freeze_router=args.freeze_router
    )
    n_router = sum(p.numel() for p in router_params)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(
            "train_scope=%s: router %.2fM + backbone %.3fB = %.3fB of %.3fB (%.1f%%)",
            args.train_scope,
            n_router / 1e6, n_backbone / 1e9, (n_router + n_backbone) / 1e9,
            n_total / 1e9, 100 * (n_router + n_backbone) / n_total,
        )

    if args.hard_topk:
        # BEFORE FSDP wraps anything: inside a forward, FSDP marks its all-gathered parameters
        # requires_grad=True regardless of the user's setting, so this same assertion made from
        # the gate path fires on a correctly-frozen run (observed exactly that). Here the flags
        # are still the ones split_trainable set.
        from kvpress.presses.gqa_indexer.hard_evict import assert_router_frozen

        n_checked = assert_router_frozen(model, args.scorer_attr)
        if rank == 0:
            logger.info(
                "HARD EVICTION at topk=%d: verified all %d router parameters frozen. Keys outside "
                "each row's top-k are REMOVED (-inf), which is the geometry inference runs -- "
                "match --hard-topk to the eval's --topk.",
                args.hard_topk, n_checked,
            )

    if args.ffn_sp_size > 1:
        from kvpress.presses.gqa_indexer.ffn_sp import wrap_ffn_sequence_parallel

        wrap_ffn_sequence_parallel(model, group=sp_group)

    model = wrap_fsdp(model, args, shard_group, replicate_group)
    optimizer, lr_schedule, router_lr = build_joint_optimizer(router_params, backbone_params, args)
    if rank == 0:
        logger.info(
            "AdamW: backbone peak %.2e, router peak %.2e (mult %.1f)%s, WSD warmup %d stable %d "
            "-> floor %.0f%% of peak",
            args.backbone_lr, router_lr, args.router_lr_mult,
            " -- ROUTER FROZEN, mult ignored" if args.freeze_router else "",
            int(args.total_steps * args.warmup_frac),
            int(args.total_steps * args.stable_frac), 100 * args.final_lr_frac,
        )

    if args.global_batch_size:
        per_replica = dp_world_size * args.batch_size
        if args.global_batch_size % per_replica:
            raise SystemExit(
                f"--global-batch-size {args.global_batch_size} is not divisible by "
                f"{dp_world_size} replica(s) x --batch-size {args.batch_size}"
            )
        args.accum_steps = args.global_batch_size // per_replica
        if rank == 0:
            logger.info(
                "--global-batch-size %d / (%d replica(s) x %d) -> --accum-steps %d",
                args.global_batch_size, dp_world_size, args.batch_size, args.accum_steps,
            )

    metrics_handle = None
    if args.metrics_file and rank == 0:
        Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)
        metrics_handle = open(args.metrics_file, "a")

    current_len, loader, iterator = None, None, None
    window: list[float] = []
    started = time.time()
    step = 0

    try:
        with trainer.hooks(model):
            for step, seq_len in schedule.lengths():
                if seq_len != current_len:
                    loader = loader_for(
                        seq_len, args, tokenizer, dp_rank, dp_world_size,
                        batch_size=args.batch_size,
                    )
                    iterator = iter(loader)
                    current_len = seq_len
                    if rank == 0:
                        logger.info("step %d: seq_len=%d", step, seq_len)

                optimizer.zero_grad(set_to_none=True)
                accumulated = 0.0
                will_log = (
                    step % args.log_every == 0 or step == args.total_steps - 1
                    or (bool(args.max_steps) and step + 1 >= args.max_steps)
                )
                for micro in range(args.accum_steps):
                    trainer.measure_sparsity = (
                        args.gate_sparsity and will_log and micro == args.accum_steps - 1
                    )
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    loss = e2e_indexer_training_step(
                        model, trainer, input_ids=input_ids,
                        skip_logits=True if args.liger else None,
                    )
                    (loss / args.accum_steps).backward()
                    accumulated += float(loss) / args.accum_steps

                # NO average_gradients here: FSDP reduces gradients itself as part of
                # reduce-scatter. A second manual all-reduce would divide by world_size again and
                # scale every gradient down by that factor -- silently, with the loss still falling.
                grad_norm = model.clip_grad_norm_(args.grad_clip)
                optimizer.step()
                lr_schedule.step()

                if step % args.log_every == 0:
                    accumulated = all_reduce_mean(accumulated, device)
                window.append(accumulated)
                reached_max = bool(args.max_steps) and step + 1 >= args.max_steps

                if will_log:
                    gate_scale = trainer.mean_gate_scale()
                    gate_sparsity = trainer.mean_gate_sparsity()
                    history_mass = trainer.mean_history_attention_mass()
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    # Keyed by name, not by index: with --freeze-router there is no second
                    # group and param_groups[1] would raise.
                    lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)}
                    lr_bb = lrs.get("backbone", float("nan"))
                    lr_rt = lrs.get("router")
                    if rank == 0:
                        logger.info(
                            "step %4d/%d L=%-6d lm_loss %.4f (avg %.4f) |g| %.3f lr bb %.2e "
                            "router %.2e gate %.4f sparsity %s hist_mass %s peak %.1f GiB "
                            "%.1f s/step",
                            step, args.total_steps, seq_len, accumulated,
                            sum(window) / len(window), float(grad_norm), lr_bb,
                            lr_rt if lr_rt is not None else float("nan"),
                            gate_scale if gate_scale is not None else float("nan"),
                            f"{gate_sparsity:.3f}" if gate_sparsity is not None else "off",
                            f"{history_mass:.3f}" if history_mass is not None else "off",
                            peak, (time.time() - started) / (step + 1),
                        )
                    if metrics_handle:
                        metrics_handle.write(json.dumps({
                            "step": step,
                            "seq_len": seq_len,
                            "loss": accumulated,
                            "grad_norm": float(grad_norm),
                            "lr_backbone": lr_bb,
                            # null under --freeze-router, so a plot cannot mistake a stale
                            # number for a router that is still moving.
                            "lr_router": lr_rt,
                            "gate_scale_mean": gate_scale,
                            # THE readout for this run: SP-KV's frozen-LLM ablation sat above 80%
                            # density. If joint training is doing what it claims, this should fall
                            # BELOW the 0.267 the frozen router reached.
                            "gate_sparsity_mean": gate_sparsity,
                            "history_attention_mass_mean": history_mass,
                            "peak_gib": peak,
                            "tokens": dp_world_size * args.batch_size * args.accum_steps * seq_len,
                            "n_trainable_backbone": n_backbone,
                            "seed": args.seed,
                        }) + "\n")
                        metrics_handle.flush()
                    window = window[-50:]

                if args.save_every and (step + 1) % args.save_every == 0:
                    save(out_dir / f"step{step + 1}.pt", model, args, step + 1,
                         {"loss": accumulated})

                if reached_max:
                    if rank == 0:
                        logger.info("reached --max-steps %d", args.max_steps)
                    break
                if args.dry_run and step >= 1:
                    if rank == 0:
                        logger.info("dry run complete")
                    break
    finally:
        if metrics_handle:
            metrics_handle.close()

    save(out_dir / "final.pt", model, args, step + 1)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        logger.info("done in %.1f min", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
