# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train the memory module end to end, with the router frozen.

The forward pass replaced into every attention layer is::

    o = fuse_memory( flex_attention(q, k, v, block_mask=qi_block_mask(deadlines(s))), memory(q) )

i.e. exactly the sparse attention the press's eviction produces, plus the memory's contribution in
the same softmax. So what is trained is the *compensation for eviction*, measured against the
eviction it actually compensates for -- not against a dense reference the deployed model never sees.

Why this is built on ``flex_attention`` and not on the gather path
-----------------------------------------------------------------
:mod:`~.e2e_trainer` records that ``stage="sparse"`` backward falls through to the gather
*reference* (the Triton sparse kernel has no ``autograd.Function``), retaining
``O(Hkv Sq topk D)`` per layer -- ~39 GiB/layer at ``Sq=16384, topk=512``, hence the conclusion that
the sparse scope is unaffordable. That number is real but it is a property of the gather, not of
sparse training. Measured on H20 at this geometry, fwd+bwd peak allocated per layer:

===========================  ======  ======  ============  ============
path                         ``Sq``  topk    per layer     x36 layers
===========================  ======  ======  ============  ============
flex block-sparse (+lse)     16384   4096    1.38 GiB      **49.7 GiB**
flex block-sparse (+lse)     32768   8192    2.76 GiB      **99.4 GiB**
gather reference             4096    512     48.3 GiB      1739 GiB
gather reference             8192    512     OOM on one layer
===========================  ======  ======  ============  ============

flex retains ``O(Sq D)`` and is independent of ``topk``; the gather retains ``O(Hkv Sq topk D)`` and
dies on a single layer past 4K. At 4x the length and 8x the topk, flex uses ~35x less. So the memory
arm trains at 32K on one node, and any future sparse-forward-with-gradient work should be built here
rather than on ``streaming_topk_support`` + gather.

The other prerequisite is that ``lse`` be differentiable, because the fusion's denominator path
(``do/dd = -o/(D_S+d)``) runs through it. Verified against an fp64 masked reference under a holey
block-sparse mask: value 8.2e-7, ``d(lse)/dq`` 4.2e-7 relative, ``d(lse)/dv`` exactly 0, and
backward through ``out`` and ``lse`` together correct to ~5e-7 in fp32. Asking for ``lse`` costs
0.456 -> 0.529 ms and +0.1 MiB.

Streaming schedule
------------------
Each query block reads the memory state accumulated from the keys evicted before it -- see
:mod:`~.memory_schedule`. The alternative (compress once, supervise only the tail) gives ``psi`` one
eviction event per sequence; streaming gives it ~``L``, and sweeps ``|E|`` from 0 to ``L - budget``
inside a single sequence, which is what makes the explicit ``|E|`` normalization trainable rather
than merely correct.

What is trained, and what is deliberately not
---------------------------------------------
Only the memory modules. The router is loaded from a trained checkpoint and frozen, so this arm
answers "can a constant-size memory recover what a *fixed, already-good* router discarded" without
confounding it with the router moving. Unfreezing the router is a later stage: the moment it moves,
``E`` moves, and the memory's target becomes non-stationary.

There is nothing to pin, and that is structural rather than lucky. The indexer's degenerate point (a
flat gate) restores the frozen dense backbone, which is already strong -- so the LM loss can be
satisfied with no ranking learned, the 18.8-against-54.4 failure ``gate_pin`` exists to close. The
memory's degenerate point (``gamma -> 0``) restores the plain eviction baseline, which is the thing
being beaten. The escape hatch leads to the worst case, so the loss has no shortcut to take.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.memory import (
    DEFAULT_SCALAR_EPS,
    MemoryKernel,
    fuse_memory,
    memory_mass_share,
)
from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.qi_flex_attention import (
    FLEX_BLOCK,
    HAS_FLEX,
    _flex,
    deadlines,
    qi_block_mask,
)
from kvpress.presses.gqa_indexer.sparse_support import resolve_topk

logger = logging.getLogger(__name__)

#: The name the fused attention is registered under in ``ALL_ATTENTION_FUNCTIONS``.
IMPL_NAME = "kvpress_gqa_indexer_memory"

#: Eviction schedules. ``streaming`` gives every query block its own state (``L`` supervised
#: positions, ``|E|`` swept in-sequence); ``oneshot`` compresses once at the end and is what
#: deployment does, so it is the validation setting rather than the training one.
SCHEDULES = ("streaming", "oneshot")


@dataclass
class MemoryTrainer:
    """
    Replace every attention layer with sparse-attention-plus-memory, and train the memory.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer routers (frozen) and the per-layer :class:`~.memory.MemoryKernel`
        modules. Must have been built with ``memory=True``.
    topk : int, optional
        Retained keys per row, including the forced slots. ``None`` derives it from
        ``keep_ratio``, so a run can be configured at the ratio the press evicts at.
    keep_ratio : float
        Used when ``topk`` is ``None``. Set to ``1 - compression_ratio``.
    force_sink, force_local : int
        Rows' reserved leading and most-recent slots. Must match what evaluation uses, or the
        memory is trained against a different partition than it is read under.
    schedule : str
        ``streaming`` or ``oneshot``; see :data:`SCHEDULES`.
    block : int
        Query-block width for the streaming state. Left at :data:`~.qi_flex_attention.FLEX_BLOCK`,
        which is the granularity ``create_block_mask`` already quantizes to -- so matching it makes
        the partition exact by construction and costs nothing. A different value would need an
        intra-block correction term in the kernel.
    freeze_router : bool
        Freeze the routers as well as the backbone. Leave ``True``: with the router moving, ``E``
        moves and the memory's target is non-stationary. See "What is trained".
    """

    press: GQAIndexerPress
    topk: int | None = None
    keep_ratio: float = 0.25
    force_sink: int = 4
    force_local: int = 0
    schedule: str = "streaming"
    block: int = FLEX_BLOCK
    freeze_router: bool = True

    #: Layer index -> mean ``d/(D_S + d)``, the share of softmax mass the memory took. **The
    #: diagnostic to read before the loss** -- the analogue of ``E2EIndexerTrainer.gate_scales``.
    #: Compare it against the measured true evicted mass (rho ~ 0.21-0.23 on average, 0.41 at layer
    #: 0): a layer whose share climbs well past its own rho has stopped compensating for eviction
    #: and started degenerating into pure linear attention.
    mass_shares: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean ``gamma``, so the bootstrap out of the off state is visible. Should climb
    #: off ``exp(DEFAULT_LOG_GAMMA)`` within ~100 steps; if it does not, the scalar learning rate is
    #: too low and no loss curve will say so.
    gammas: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean ``|E_t|``, as a wiring check on the partition. Should be ~``L - topk``.
    evicted_counts: dict[int, float] = field(default_factory=dict)
    #: How many layers ran the fused attention, as a wiring check.
    layers_fused: int = field(default=0, init=False)
    #: Collect the diagnostics on this forward. They cost a reduction each, so the driver turns
    #: this on only for the steps it logs.
    measure: bool = field(default=False)

    _hidden_states: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _kwargs: dict[int, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not HAS_FLEX:
            raise RuntimeError(
                "MemoryTrainer requires torch.nn.attention.flex_attention: the fusion needs a "
                "differentiable lse from a real block-sparse kernel, and the gather fallback's "
                "backward retains O(Hkv*Sq*topk*D) per layer (~48 GiB at Sq=4096) which does not "
                "fit for even one layer past 4K."
            )
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.force_sink < 0 or self.force_local < 0:
            raise ValueError("force_sink and force_local must be non-negative")
        if self.block <= 0 or self.block % FLEX_BLOCK:
            raise ValueError(
                f"block must be a positive multiple of FLEX_BLOCK={FLEX_BLOCK}, got {self.block}: "
                "a finer granularity than the block mask's would make the memory ingest keys the "
                "attention still attends to, which double-counts their mass."
            )

    def reset(self) -> None:
        """Drop the per-pass state from the previous forward."""
        self.mass_shares = {}
        self.gammas = {}
        self.evicted_counts = {}
        self.layers_fused = 0
        self._hidden_states.clear()
        self._kwargs.clear()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def memory_modules(self, model: nn.Module) -> list[MemoryKernel]:
        """Every :class:`~.memory.MemoryKernel` on the model, in layer order."""
        mods = []
        for layer in get_language_model(model).layers:
            memory = getattr(layer.self_attn, self.press.memory_attr, None)
            if memory is not None:
                mods.append(memory)
        return mods

    def parameter_groups(
        self,
        model: nn.Module,
        *,
        kernel_lr: float,
        scalar_lr: float,
        scalar_eps: float = DEFAULT_SCALAR_EPS,
    ) -> list[dict]:
        """
        Two optimizer groups: the ``phi``/``psi`` weights, and the fp32 scalars.

        Both the separate learning rate **and** the separate ``eps`` are load-bearing rather than
        tidy, and for related reasons.

        ``log_gamma`` has to travel ~8 units of log space to switch the memory on at all (see
        :data:`~.memory.DEFAULT_LOG_GAMMA`), so at the MLP's 1e-3 the memory would arrive thousands
        of steps after the run ended, with a healthy-looking loss curve throughout.

        And ``dL/dlog_gamma`` is proportional to ``gamma``, hence ~2e-9 at initialization -- under
        AdamW's default ``eps=1e-8``, which makes the denominator eps instead of the gradient scale
        and throttles the step to ``lr * g / eps``. See :data:`~.memory.DEFAULT_SCALAR_EPS`; the
        6-step smoke run moved ``gamma`` 5x slower than ``lr`` implied before this was fixed. The
        ``eps`` therefore has to travel with the group, not be left to the caller's optimizer
        defaults, which is why this returns it in the group dict.

        :meth:`upcast_scalars` is called here, before the groups are handed out, because the fp32
        conversion rebinds the parameters and an optimizer built on the old tensors would keep
        stepping those.
        """
        converted = self.upcast_scalars(model)
        if converted:
            logger.info(
                "upcast %d memory scalar(s) to fp32: at bf16's ~3e-4 spacing a warmup learning "
                "rate rounds every step back and they would stay frozen at initialization.",
                converted,
            )
        modules = self.memory_modules(model)
        if not modules:
            raise RuntimeError(
                f"no {self.press.memory_attr!r} modules found; build the press with memory=True "
                "and call press.post_init_from_model(model) first"
            )
        kernel_params: list[nn.Parameter] = []
        scalar_params: list[nn.Parameter] = []
        for memory in modules:
            kernel_params.extend(memory.kernel_parameters())
            scalar_params.extend(memory.scalar_parameters())
        groups = [{"params": kernel_params, "lr": kernel_lr, "name": "memory_kernels"}]
        if scalar_params:
            groups.append(
                {
                    "params": scalar_params,
                    "lr": scalar_lr,
                    "eps": scalar_eps,
                    "name": "memory_scalars",
                }
            )
        return groups

    def upcast_scalars(self, model: nn.Module) -> int:
        """Force every memory scalar to fp32. Must precede optimizer construction."""
        return sum(memory.upcast_scalars() for memory in self.memory_modules(model))

    def memory_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """Every memory parameter -- the trainable set."""
        return [p for memory in self.memory_modules(model) for p in memory.parameters()]

    def freeze_all_but_memory(self, model: nn.Module) -> None:
        """
        Put every parameter except the memory modules' at ``requires_grad=False``.

        Selected by **module identity**, not by name substring, for the reason
        :meth:`~.e2e_trainer.E2EIndexerTrainer.freeze_backbone` gives: a name filter would also
        catch any backbone parameter whose name happens to contain the attribute string and train it
        silently. The routers are frozen by the same pass when :attr:`freeze_router` is set --
        they are simply not in the memory set.
        """
        trainable = {id(p) for p in self.memory_parameters(model)}
        if not trainable:
            raise RuntimeError(
                f"no {self.press.memory_attr!r} parameters found; build the press with memory=True "
                "and call press.post_init_from_model(model) first"
            )
        if not self.freeze_router:
            for layer in get_language_model(model).layers:
                indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
                if indexer is not None:
                    trainable |= {id(p) for p in indexer.parameters()}
        for param in model.parameters():
            param.requires_grad = id(param) in trainable

    # ------------------------------------------------------------------
    # The fused forward
    # ------------------------------------------------------------------
    def fused_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float | None,
    ) -> torch.Tensor:
        """
        One layer: sparse attention over the retained keys, fused with the memory's contribution.

        ``query``/``key``/``value`` arrive post-RoPE from the layer, so ``psi`` and ``phi`` see the
        same tensors that are in the cache at inference -- which is the point of feeding them the
        post-RoPE ``k`` rather than recomputing a pre-RoPE one.
        """
        layer_idx = int(module.layer_idx)
        # Popped, not read: leaving it in the dict pins this layer's (B, L, hidden) tensor for the
        # whole forward AND backward, which also stops autograd releasing per-layer activations as
        # it unwinds. E2EIndexerTrainer.gated_forward documents the same choice.
        hidden_states = self._hidden_states.pop(layer_idx, None)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached the fused attention without its hidden_states being "
                "captured. Register through MemoryTrainer.hooks(); a second call for the same "
                "layer in one forward (gradient checkpointing recomputes a block) also lands here, "
                "because the entry is consumed on first use."
            )
        self._kwargs.pop(layer_idx, None)

        bsz, n_q_heads, q_len, head_dim = query.shape
        n_kv_heads, k_len = key.shape[1], key.shape[2]
        group = n_q_heads // n_kv_heads
        if bsz != 1:
            # The block mask is built with B=None, so a per-sequence deadline would be ignored
            # rather than applied -- every sequence would attend over sequence 0's support.
            raise NotImplementedError(
                f"MemoryTrainer supports batch 1, got {bsz}. qi_block_mask is built with B=None, "
                "so per-sequence routing would be silently discarded."
            )

        indexer = self.press.get_indexer(module)
        memory = self.press.get_memory(module)

        # The router is frozen and top-k is not differentiable anyway, so the whole selection runs
        # under no_grad -- deadlines() argsorts, which would retain a sort graph for nothing.
        with torch.no_grad():
            scores = indexer.score_keys(hidden_states)  # (B, Hkv, Sk) fp32
            topk = resolve_topk(k_len, self.topk, self.keep_ratio)
            dl = deadlines(
                scores[0], topk, force_sink=self.force_sink, force_local=self.force_local
            )

        block_mask = qi_block_mask(
            dl,
            q_len=q_len,
            k_len=k_len,
            n_q_heads=n_q_heads,
            force_sink=self.force_sink,
            force_local=self.force_local,
            device=query.device,
        )
        o_s, lse_s = _flex()(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            block_mask=block_mask,
            scale=scaling,
            return_lse=True,
        )

        n, d = self.memory_terms(
            memory, query, key, value, dl, scores=scores[0], q_len=q_len, group=group
        )
        out = fuse_memory(o_s, lse_s, n, d, group=group)

        self.layers_fused += 1
        if self.measure:
            with torch.no_grad():
                self.mass_shares[layer_idx] = float(
                    memory_mass_share(lse_s, d, group=group).mean()
                )
                self.gammas[layer_idx] = float(memory.gamma.detach().mean())
        # The attention interface contract is (B, Sq, H, D); flex returns (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    def memory_terms(
        self,
        memory: MemoryKernel,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        deadline: torch.Tensor,
        *,
        scores: torch.Tensor,
        q_len: int,
        group: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ``(n, d)`` for every query row, under the configured schedule.

        The GQA reduction happens here rather than in the kernel: the memory state is per KV head,
        so ``phi`` must be fed a per-KV-head query. The group's queries are **averaged**, which is
        the cheap choice and the one consistent with a per-KV-head state -- a per-query-head
        ``phi`` would be ``group`` times the work and could not be read against a shared ``H``
        anyway. Since ``phi`` is nonlinear this is an approximation on the query side; the exact
        branch is untouched by it, and the memory is a low-rank summary to begin with.
        """
        k_len = key.shape[2]
        # (B, H, Sq, D) -> (B, Hkv, Sq, D) by averaging each group's queries.
        q_kv = query.view(query.shape[0], -1, group, q_len, query.shape[-1]).mean(2)

        if self.schedule == "oneshot":
            if q_len > 1 and q_len == k_len:
                # A single state read by every row is only causal when the rows all sit at or after
                # the last ingested key -- i.e. decode (q_len == 1), or a suffix of a longer cache.
                # Over a full prefill it is a **future leak**, and a severe one: measured on layer 18
                # at L=8192, query row 0 read a state built from 6208 keys, every one of them at a
                # position it cannot see. That is not a small approximation -- it collapsed RULER 8K
                # from 73.71 to 4.00 while training (which uses the streaming schedule and is
                # causal) reported a healthy mass share the whole way.
                raise ValueError(
                    f"schedule='oneshot' with q_len == k_len == {q_len} would let every query row "
                    "read a memory state built from keys in its own future. Use "
                    "schedule='streaming' for a full-sequence forward; 'oneshot' is for decode "
                    "rows, where the query sits after everything the state contains."
                )
            # One state, read by rows that all sit after it: what decode does.
            enter = deadline.to(torch.int64) + 1
            weights = memory.ingest_weights(enter, k_len, scores=scores)
            state = memory.ingest(key, value, weights)
            counts = (enter <= k_len - 1).sum(-1)  # (Hkv,)
            n_evicted = counts.view(1, -1, 1).expand(query.shape[0], -1, q_len).float()
            return memory.read(q_kv, state, n_evicted)

        H, z, W, counts = block_memory_states(
            memory,
            key,
            value,
            deadline,
            q_len=q_len,
            block=self.block,
            n_local=self.force_local,
            scores=scores,
        )
        if self.measure:
            with torch.no_grad():
                self.evicted_counts[int(len(self.evicted_counts))] = float(counts.float().mean())
        return self._read_per_block(memory, q_kv, H, z, W, counts, block=self.block)

    @staticmethod
    def _read_per_block(
        memory: MemoryKernel,
        q_kv: torch.Tensor,
        H: torch.Tensor,
        z: torch.Tensor,
        W: torch.Tensor,
        counts: torch.Tensor,
        *,
        block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :meth:`~.memory.MemoryKernel.read` against a per-query-block state.

        The contraction is done **per block**, on a reshaped ``phi(q)``, rather than by broadcasting
        the state out to one entry per row. That is not a micro-optimization: ``H`` expanded to rows
        is ``(B, Hkv, Sq, R, D)``, which at ``Sq=8192`` is 512 MiB per layer and **18 GiB across 36
        layers** -- 72 GiB at 32K -- for a tensor holding only ``n_blocks`` distinct values. It is
        what OOM'd the first 8K run. Reshaping the *query* axis to ``(n_blocks, block)`` instead
        keeps the state at its natural ``n_blocks`` resolution (4 MiB per layer at 8K, a 128x
        reduction) and contracts ``R`` against it directly.

        Kept here rather than as a second signature on the kernel: the kernel's contract is one
        state, which is what inference has, and the streaming schedule is a training-time
        elaboration.
        """
        bsz, n_kv_heads, q_len, _ = q_kv.shape
        n_blocks = H.shape[2]
        pad = n_blocks * block - q_len
        if memory.rank_zero:
            phi_q = torch.ones(bsz, n_kv_heads, q_len, 1, device=q_kv.device, dtype=torch.float32)
        else:
            phi_q = memory.phi(q_kv).float()  # (B, Hkv, Sq, R)
        rank = phi_q.shape[-1]
        if pad:
            # The final block is ragged whenever q_len is not a multiple of `block` (every RULER
            # context). Padded with zeros so the reshape is legal, then trimmed off the result --
            # the padding rows contribute nothing because phi is zero there.
            phi_q = torch.nn.functional.pad(phi_q, (0, 0, 0, pad))
        phi_blocks = phi_q.view(bsz, n_kv_heads, n_blocks, block, rank)

        # Same L1-normalized mass as MemoryKernel.read -- see memory.memory_terms for why /W leaves
        # d proportional to the kernels' own magnitude, which measured a 3e4 spread across layers.
        # Written out rather than routed through memory_terms because the state carries a block axis
        # here and the query axis is split (n_blocks, block); the arithmetic is identical.
        gamma = memory.gamma.view(1, -1, 1, 1)  # (1, Hkv, 1, 1)
        phi_hat = phi_blocks / phi_blocks.sum(-1, keepdim=True).clamp(min=1e-20)
        z_f = z.float()
        z_sum = z_f.sum(-1, keepdim=True).clamp(min=1e-20)  # (B, Hkv, n_blocks, 1)
        z_hat = z_f / z_sum

        counts_f = counts.unsqueeze(0).unsqueeze(-1).float()  # (1, Hkv, n_blocks, 1)
        d = gamma * counts_f * torch.einsum("bhnjr,bhnr->bhnj", phi_hat, z_hat)
        num = torch.einsum("bhnjr,bhnrd->bhnjd", phi_hat, H.float())
        den = torch.einsum("bhnjr,bhnr->bhnj", phi_hat, z_f).clamp(min=1e-20)
        n = d.unsqueeze(-1) * num / den.unsqueeze(-1)
        n = n.reshape(bsz, n_kv_heads, n_blocks * block, -1)
        d = d.reshape(bsz, n_kv_heads, n_blocks * block)
        if pad:
            n, d = n[:, :, :q_len], d[:, :, :q_len]
        return n, d

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """Stash this layer's ``hidden_states`` before its attention runs.

        The attention interface receives q/k/v only, never ``hidden_states`` -- which is what the
        router scores from. A forward pre-hook on the attention module is the earliest point both
        are available.
        """
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        self._hidden_states[int(layer_idx)] = hidden_states
        self._kwargs[int(layer_idx)] = kwargs
        return None

    @contextmanager
    def hooks(self, model: nn.Module):
        """
        Fuse every attention layer for the block's duration.

        A pre-hook per attention module captures ``hidden_states``, and a temporary
        ``ALL_ATTENTION_FUNCTIONS`` entry that ``config._attn_implementation`` is pointed at
        replaces the attention. Both are removed on exit, including on exception.

        The registry cleanup goes through ``_global_mapping`` directly for the reason
        :func:`~.teacher_lse.capture_teacher_lse` documents: ``register()`` writes there while
        ``pop()`` only touches the instance mapping, so the naive removal leaks the entry forever.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        self.freeze_all_but_memory(model)

        def memory_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self.fused_forward(
                module, query, key, value, attention_mask, scaling
            ), None

        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        previous_impls = [cfg._attn_implementation for cfg in configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        had_previous = IMPL_NAME in global_mapping
        previous_fn = global_mapping.get(IMPL_NAME)
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, memory_attention_impl)

        handles = []
        try:
            for layer in get_language_model(model).layers:
                handles.append(
                    layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
                )
            for cfg in configs:
                cfg._attn_implementation = IMPL_NAME
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for cfg, previous in zip(configs, previous_impls):
                cfg._attn_implementation = previous
            if had_previous:
                global_mapping[IMPL_NAME] = previous_fn
            else:
                global_mapping.pop(IMPL_NAME, None)
            self._hidden_states.clear()
            self._kwargs.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def mean_mass_share(self) -> float | None:
        """Mean ``d/(D_S+d)`` over the layers that measured it. Read this before the loss."""
        values = list(self.mass_shares.values())
        return sum(values) / len(values) if values else None

    def mean_gamma(self) -> float | None:
        """Mean ``gamma`` over the layers that measured it -- the bootstrap readout."""
        values = list(self.gammas.values())
        return sum(values) / len(values) if values else None


def memory_longce_step(
    model: nn.Module,
    trainer: MemoryTrainer,
    *,
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    scored: torch.Tensor | None = None,
    logit_chunk: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    One step of the LongCE-weighted objective through the fused forward.

    The same weighting :func:`~.e2e_trainer.e2e_indexer_longce_step` applies, for the same reason:
    ``longce_weights`` records the delta arm collapsing RULER 66.24 -> ~35.2 by concentrating on the
    high-loss tail where irreducible entropy lives, while LongCE's offline ``spearman(w, L_long)``
    is -0.001 to -0.029, i.e. the weight is decorrelated from the loss it multiplies. Weights come
    from the cache, so this is **one** forward pass.

    Returns
    -------
    (loss, stats)
        ``stats`` carries ``weight_participation`` (judge the weighting by this, directly comparable
        to the failed delta run's 0.13-0.18 and the offline 0.66-0.87), ``sparse_loss`` (the plain
        mean, so the curve is comparable to the router-only arm's), and the memory diagnostics
        ``mass_share`` / ``gamma``.
    """
    from kvpress.presses.gqa_indexer.delta_loss import (
        DEFAULT_LOGIT_CHUNK,
        per_token_ce,
        valid_mask,
    )
    from kvpress.presses.gqa_indexer.e2e_trainer import _final_hidden_states
    from kvpress.presses.gqa_indexer.longce_weights import longce_weighted_loss

    chunk = DEFAULT_LOGIT_CHUNK if logit_chunk is None else logit_chunk
    target = input_ids if labels is None else labels
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model exposes no output embeddings, so per-token CE cannot be formed")

    with trainer.hooks(model):
        hidden = _final_hidden_states(
            model, input_ids=input_ids, attention_mask=attention_mask
        )
        if trainer.layers_fused == 0:
            raise RuntimeError(
                "no layer ran the fused attention: the model kept its own attention "
                "implementation. This usually means the model's config is not the one "
                f"MemoryTrainer pointed at {model.config._attn_implementation!r}."
            )
        loss_per_token = per_token_ce(lm_head, hidden, target, chunk_size=chunk)

    flat_weights = weights.reshape(-1).to(loss_per_token.device, dtype=torch.float32)
    if flat_weights.shape != loss_per_token.shape:
        # Explicit rather than left to broadcasting: a weight vector off by one position still
        # multiplies elementwise without complaint, and the run would train the wrong tokens while
        # every logged number stayed plausible.
        raise ValueError(
            f"cached weights flatten to {tuple(flat_weights.shape)} but the per-token loss is "
            f"{tuple(loss_per_token.shape)}. The cache was built at a different sequence length "
            "than this stage draws, so the weights do not line up with the tokens."
        )
    flat_scored = None
    if scored is not None:
        flat_scored = scored.reshape(-1).to(loss_per_token.device, dtype=torch.bool)

    mask = valid_mask(target)
    loss, stats = longce_weighted_loss(
        loss_per_token, flat_weights, mask=mask, scored=flat_scored
    )
    with torch.no_grad():
        n_valid = int(mask.sum())
        stats["sparse_loss"] = (
            float(loss_per_token.detach()[mask].mean()) if n_valid else 0.0
        )
        share, gamma = trainer.mean_mass_share(), trainer.mean_gamma()
        if share is not None:
            stats["mass_share"] = share
        if gamma is not None:
            stats["gamma"] = gamma
    return loss, stats


def memory_lm_step(
    model: nn.Module,
    trainer: MemoryTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    One step of the plain (unweighted) LM loss through the fused forward.

    The baseline the LongCE arm is compared against, so the weighting can be attributed rather than
    assumed. ``use_cache=False`` -- nothing here reads a cache and building one only costs memory.
    """
    with trainer.hooks(model):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids if labels is None else labels,
            use_cache=False,
        )
        if trainer.layers_fused == 0:
            raise RuntimeError(
                "no layer ran the fused attention: the model kept its own attention "
                f"implementation ({model.config._attn_implementation!r})."
            )
    stats: dict = {"sparse_loss": float(out.loss.detach())}
    share, gamma = trainer.mean_mass_share(), trainer.mean_gamma()
    if share is not None:
        stats["mass_share"] = share
    if gamma is not None:
        stats["gamma"] = gamma
    return out.loss, stats
