# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constant-size linear **memory** for the keys the router evicts, fused into one softmax.

Eviction throws mass away, and measurably a lot of it. Measured with the trained LongCE scalar
router at ``keep_ratio=0.25`` on Qwen3-8B, the evicted share of each row's softmax mass

    ``rho = 1 - exp(lse_S - lse_dense)``

is **0.213 / 0.219 / 0.234** at ``L = 4K / 8K / 32K`` (mean over layers 0, 9, 18, 27, 35; 2
samples). It is essentially flat in length, so this is not a short-context artifact. The
distribution is heavily skewed -- median 0.03-0.17 against p90 0.37-0.92 -- so most rows lose
little and a tail loses nearly everything, which is where compensation pays. Layer 0 is the
outlier at ``rho ~ 0.41``.

This module keeps a rank-``R`` summary of the evicted keys instead of dropping them, following
LESS (ICML'24, arXiv 2402.09398), and folds it into the *same* softmax as the retained keys::

    o_t = (N_S + n) / (D_S + d)

    N_S = sum_{j in S} exp(s_j) v_j        D_S = sum_{j in S} exp(s_j)      s_j = q_t.k_j/sqrt(D)
    n   = d * (phi_hat.H)/(phi_hat.z)      d   = gamma_h |E_t| <phi_hat(q_t), z_hat>

with the per-KV-head state accumulated over the evicted set ``E``::

    H = sum_{j in E} w_j psi(k_j)^T v_j    in R^{R x D}
    z = sum_{j in E} w_j psi(k_j)          in R^{R}
    W = sum_{j in E} w_j                   scalar, the normalizer
    w_j = lambda_h^(L-1-j) exp(a s~_j + b)

Three invariants, each of which silently degrades the model rather than crashing if broken
----------------------------------------------------------------------------------------
1. **``gamma``, ``lambda`` and ``w_j`` must scale H and z together.** Applied to the numerator
   only, ``H/z`` stops being a weighted average of value vectors, ``d`` loses its meaning as
   softmax mass, and "one softmax" is no longer true -- the thing becomes a learned gate between
   two unrelated quantities.
2. **``|E|`` must be factored out explicitly**, and the mass must be made **scale-free**. ``H``
   and ``z`` grow linearly in ``|E|`` -- at 128K a sum over ~10^5 terms -- while ``D_S`` is bounded
   by the budget, so without normalization ``gamma`` learns a ``1/|E|`` that is correct only at the
   training length. ``phi_hat`` and ``z_hat`` are ``phi`` and ``z`` divided by their own sums, which
   removes the ``|E|`` growth *and* the kernels' own learned magnitude; ``|E_t|`` is then multiplied
   back in by hand. Both halves matter and the second is not optional -- dividing by ``W`` alone
   left ``d`` at 0.21 in layer 0 and 6.3e3 in layer 35, a 3e4 spread no single ``gamma`` can
   straddle. See :func:`memory_terms`. Either way ``n/d`` is untouched: this corrects the *mass*,
   not the direction.
3. **DeltaNet-style matrix updates (``S <- S(I - beta psi psi^T) + beta psi v^T``) are not
   usable here.** That recurrence has no companion denominator, so fusion could only degrade to a
   learned gate (what NSA does) and the single-softmax property would be gone. It is a separate
   arm, not a drop-in.

Zero initialization has three independent dead points
-----------------------------------------------------
The training start should be the plain eviction baseline, bit-identical. Naive zeroing fails
three separate ways, and only the third is specific to this design:

======  ==========================================================  ==================
 #      dead point                                                   symptom
======  ==========================================================  ==================
 1      ``z = 0`` makes ``H/z`` a ``0/0``                             NaN
 2      writing it as ``o = w o_S + (1-w) o_E`` with                  gradient identically 0
        ``w = sigmoid(lse_S - lse_E)``: at ``w = 1``,
        ``dw/dtheta = w(1-w)(...) = 0``
 3      ``psi = |A W_3|`` with ``W_3 = 0``: ``torch.abs`` has         ``dpsi/dW_3 = 0``
        subgradient **0** at 0
======  ==========================================================  ==================

Point 3 is a structural conflict between the non-negativity ``abs`` provides and zero
initialization, independent of which fusion form is chosen. The fix is two things together:

* **The additive form above, not the sigmoid form.** At ``H = z = 0`` it is already exactly
  ``N_S/D_S`` -- no ``0/0`` -- and its gradients at that point are alive:
  ``do/dn = 1/(D_S+d) != 0`` and ``do/dd = -o/(D_S+d) != 0``. The sigmoid form is algebraically
  identical but its gradient is annihilated by ``w(1-w)``, so it is provided here **only** as a
  diagnostic readout (:func:`memory_mass_share`) and must never be in the compute graph.
* **Random ``phi``/``psi``, with the "off" state carried by ``gamma_h = exp(beta_h)``,
  ``beta_h`` initialized to** :data:`DEFAULT_LOG_GAMMA`. The memory contributes ~1e-4 of the
  softmax mass at the start, which is a no-op in effect, but ``do/dbeta_h != 0``, so ``beta`` moves
  first; once ``gamma`` leaves zero, the ``phi``/``psi`` gradients (which are proportional to it)
  open up on their own. **Self-bootstrapping -- no hand-written warmup schedule.** The size of that
  initial value has to be picked against ``|E|`` rather than against 1, for the same reason
  invariant 2 exists; :data:`DEFAULT_LOG_GAMMA` records the measurement.

``a`` and ``g(s~)`` have exactly the same structure: ``dw_j/da = w_j s~_j != 0`` even at ``a = 0``
(so ``a`` can move first), while ``dw_j/ds~_j = a w_j = 0`` until it does. Defaulting ``a = 0``
therefore does not prevent the feature from switching itself on.

Why this needs the scalar (query-independent) scorer
---------------------------------------------------
``E_t`` must be a function of ``t`` alone for a single accumulated state to represent it. Under a
pairwise scorer the evicted set varies per query and no constant-size state can express it. So
this is a structural argument for the scalar arm rather than a limitation of it: the scalar
scorer is not merely cheaper at decode, it is the only one that admits a constant-size memory.

The other half of that is :func:`~.qi_flex_attention.deadlines`, which already computes the
ingestion time. Its contract is that a row with horizon ``hi_t`` keeps key ``j`` iff
``hi_t <= deadline[j]``, so

    ``j in E_t  <=>  horizon_t > deadline[j]``  i.e. ``j`` enters the memory at
    ``horizon = deadline[j] + 1``

which makes sink and local fall out for free: a sink has ``deadline = k_len - 1`` and so is never
ingested, and a local key is outside the horizon. **No special cases.**

Why the degenerate point is not a free lunch here
-------------------------------------------------
Worth stating because it is the opposite of the indexer's situation. A flat gate recovers the
frozen *dense* model, which is already strong -- so the router can satisfy the LM loss having
learned nothing, which is the 18.8-against-54.4 failure SAS measures and why ``pin_mode`` exists.
The memory's degenerate point (``gamma -> 0``) recovers the plain *eviction* baseline, which is
precisely the thing being beaten. The escape hatch leads to the worst case rather than to a good
one, so there is no shortcut for the loss to take and nothing to pin. Correspondingly
``stage="sparse"`` requires ``pin_mode="none"`` anyway.

The reverse degeneracy is the one to watch: the memory *dominating*. Log
:func:`memory_mass_share` (``d/(D_S+d)``) per layer -- the analogue of ``gate_scales``. Large
values mean the layer is drifting towards pure linear attention.

What this can and cannot do
---------------------------
A rank-16 state cannot hold "the uuid is at position 41022" -- exact retrieval is not what a
low-rank summary of 10^5 vectors is for. What it can restore is roughly *what the distant context
was about*, since the evicted mass is a long, smooth, low-weight tail. So the honest prediction is
that RULER's niah tasks barely move while vt/qa/cwe and perplexity improve. Measured support for
the rank claim: fitting one learned vector per head to the exact evicted output
``o_E* = (o_dense - (1-rho) o_S)/rho`` leaves **78%** relative residual (mean cosine to the
per-head mean only 0.55-0.77), so ``R > 0`` is doing real work and the rank-0 ablation is expected
to lose -- by a measured margin rather than an assumed one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import IndexerNorm

logger = logging.getLogger(__name__)

#: ``beta_h`` initialization, giving ``gamma = exp(-7) = 9.1e-4``.
#:
#: The value has to be chosen against the *whole* mass expression, not against 1, and getting it
#: right took two corrections. What reaches the softmax is ``d = gamma |E| <phi_hat, z_hat>``, so
#: both the evicted count and the kernels' normalization enter. Measured on the real model, the
#: memory's share of each row's softmax mass at initialization is ``1.7e-6 * gamma * |E|``:
#:
#:     log_gamma    |E|=6K     |E|=31K    |E|=123K
#:     -18          1.6e-10    8.1e-10    3.3e-9     <- off, but 21 log units from useful
#:     -10          4.7e-7     2.4e-6     9.7e-6
#:      -7          9.5e-6     4.9e-5     1.9e-4     <- off at every length, 10 units to travel
#:      -6          2.6e-5     1.3e-4     5.3e-4
#:
#: Both ends are real constraints. Too large and the run does not start at the eviction baseline it
#: exists to beat -- under the pre-normalization mass, ``-10`` reached ``d = 2.0`` at 128K, i.e. the
#: memory taking most of the mass while still being described as "off". Too small and the bootstrap
#: has further to climb than the run is long: from ``-18`` the journey to a useful mass share is 21
#: log units, against 10 from ``-7``. ``-7`` keeps the initial share under 2e-4 at every length
#: tested while halving the distance.
#:
#: Still far above fp32's floor, so ``do/dbeta`` is representable and the parameter can move -- the
#: whole point of carrying the off state here rather than in a zeroed weight. See "Zero
#: initialization has three independent dead points".
#:
#: The bootstrap needs no schedule, which is worth recording because "start it at 1e-4" invites the
#: worry that it can never climb back. AdamW's step is ~``lr`` per iteration almost independently of
#: the gradient's size, so what matters is the *scalar* learning rate rather than how small ``gamma``
#: starts -- and since ``dL/dlog_gamma`` grows with ``gamma``, the climb accelerates. Observed on the
#: real 8K run: 1.5e-8 -> 2.4e-5, a factor of 1600 over 240 steps, each decade faster than the last.
DEFAULT_LOG_GAMMA = -7.0

#: Learning rate for the fp32 scalars. Two orders of magnitude above the MLP's 1e-3, and that gap is
#: deliberate: ``log_gamma`` has to travel ~8 units of log space to switch the memory on (see
#: :data:`DEFAULT_LOG_GAMMA`), while the ``phi``/``psi`` weights sit near ``mid_dim**-0.5`` and want
#: ordinary steps. At 1e-3 the scalars would need ~8000 steps to make that trip; at 0.05 they take
#: ~100. Because AdamW normalizes by the gradient's own scale, giving them the MLP's rate does not
#: make them "slower but fine", it makes the memory arrive after the run has ended.
DEFAULT_SCALAR_LR = 0.05

#: Learning rate for the ``phi``/``psi`` weights. LESS trains its kernels at 1e-3.
DEFAULT_KERNEL_LR = 1e-3

#: AdamW ``eps`` for the scalar group. **Not** the default 1e-8, and this is the third instance in
#: this package of "a scalar parameter silently fails to move".
#:
#: AdamW's update is ``lr * m / (sqrt(v) + eps)``, which is ~``lr`` only while ``sqrt(v) >> eps``.
#: Here ``dL/dlog_gamma`` is proportional to ``gamma`` itself, so at initialization it is ~2e-9 --
#: *below* the default eps, and the denominator becomes eps rather than the gradient's own scale.
#: The step degrades from ``lr`` to ``lr * g / eps``, i.e. the normalization AdamW exists to provide
#: stops working exactly where it is needed. Measured over 20 steps at ``lr=0.05`` from -18:
#:
#:     eps=1e-8   ->  -17.66     (ideal: -17.00)
#:     eps=1e-12  ->  -16.83
#:
#: and in the real 6-step smoke run ``gamma`` crept 1.52e-8 -> 1.59e-8, about 5x slower than the
#: learning rate implies. Nothing about that looks wrong from the outside: the loss is flat because
#: the memory is off, and the memory is off because the parameter that turns it on is being
#: throttled. Same class of bug as ``upcast_gate_scales``' bf16 freeze, different mechanism.
DEFAULT_SCALAR_EPS = 1e-16

#: Decay time constant at initialization, in tokens. ``lambda = exp(-1/tau)``, so this is
#: ``lambda ~ 0.9999``. Parameterized as ``log tau`` rather than as ``lambda`` directly because at
#: 128K the useful ``lambda`` values crowd against 1.0 (``lambda = 0.99999``) where fp32 spacing
#: leaves nothing to optimize over.
DEFAULT_TAU = 1e4

#: Upper clamp on ``a * s~_j`` before ``exp``. The router's raw scores have arbitrary per-layer
#: scale, so an unclamped exponent can overflow the moment ``a`` leaves zero.
INGEST_LOGIT_CLAMP = 10.0

#: Upper clamp on ``-lse_S`` inside :func:`fuse_memory`. Below fp32's ``exp`` limit of 88.7 with
#: margin, and the bound is unreachable in any regime that matters: it engages only where the memory
#: outweighs the retained branch by ``e^80``, at which point the fused output equals ``n/d`` to far
#: beyond fp32 precision either way. Without it, low-mass rows (a query retaining only a sink key
#: has an lse of a single logit) produce ``inf``, and ``inf * 0`` -- the empty-memory case -- is NaN.
_MAX_FUSE_EXP = 80.0


@dataclass
class MemoryConfig:
    """
    Geometry and initialization for :class:`MemoryKernel`.

    Attributes
    ----------
    n_kv_heads : int
        KV heads, which is the granularity the state is held at. Eviction is per-KV-head in GQA,
        so the memory has to be too.
    head_dim : int
        Value width ``D``. The ``psi``/``phi`` inputs are post-RoPE ``k`` and ``q``, both
        ``head_dim`` wide.
    rank : int
        State rank ``R``. The state costs ``n_kv_heads * R * (head_dim + 1)`` floats per layer:
        at ``R=16`` that is 66 KB/layer in fp32 and **2.4 MB** for all 36 layers, i.e. about what
        8 tokens of KV cache cost. Sweep ``{8, 16, 64}``; ``0`` is the rank-0 ablation
        (:attr:`MemoryKernel.rank_zero`), one learned vector per head times the mass, which
        measured 78% residual against the exact evicted output and is expected to lose.
    mid_dim : int
        Trunk width ``R'``. 256 rather than LESS's 512: the input is 128-wide, so 512 is
        over-parameterized, and memory was never the binding constraint here.
    per_head_readout : bool
        Give each KV head its own final ``R' -> R`` layer, sharing only the trunk. On by default
        because the KV heads genuinely disagree about which keys matter -- the eight heads of this
        geometry overlap on only 14-17% of their top-k (see :mod:`~.scalar_indexer`) -- so what
        they need summarized differs. ``False`` shares the readout across heads too, which is the
        cheap ablation.
    log_gamma_init : float
        Initial ``beta_h`` where ``gamma_h = exp(beta_h)``. This is what carries the "off" state;
        see the module docstring. Raising it towards 0 starts the memory switched on, which is a
        deliberate ablation rather than a faster start -- the first steps then perturb a model
        that was working.
    tau_init : float
        Initial decay constant in tokens. ``inf`` disables decay (``lambda = 1``, pure
        accumulation, which is what LESS does).
    ingest_scale, ingest_bias : float
        ``a`` and ``b`` in ``w_j = lambda^(...) exp(a s~_j + b)``. ``a = 0`` (the default) reduces
        the ingestion weight to pure decay and leaves the router's score unused, which is the
        first-version configuration. ``a`` is still a live parameter at 0 -- ``dw/da = w s~ != 0``
        -- so it can turn itself on.
    learn_decay : bool
        Whether ``log tau`` is trained.
    """

    n_kv_heads: int
    head_dim: int
    rank: int = 16
    mid_dim: int = 256
    per_head_readout: bool = True
    log_gamma_init: float = DEFAULT_LOG_GAMMA
    tau_init: float = DEFAULT_TAU
    ingest_scale: float = 0.0
    ingest_bias: float = 0.0
    learn_decay: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("n_kv_heads", "head_dim", "mid_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if self.tau_init <= 0:
            raise ValueError(f"tau_init must be positive, got {self.tau_init}")


class MemoryKernel(nn.Module):
    """
    The per-layer ``phi``/``psi``/``gamma``/``lambda`` kernel and its accumulated state.

    A standalone module rather than extra heads on the indexer, deliberately: it consumes the
    *post-RoPE* ``k`` out of the cache while the indexer consumes ``hidden_states``, and the two
    are trained at different stages with different learning rates. Sharing projections would
    couple them for no measured benefit.

    Asymmetry between the two kernels, kept from LESS: ``psi`` is one layer deeper than ``phi``.
    The key side decides *what gets stored* and runs once per token; the query side runs at every
    decode step, so it has to stay cheap.

    Parameter count at this geometry (``n_kv_heads=8``, ``head_dim=128``, ``R'=256``, ``R=16``):

    ==========  =================  ====================  ==============
    kernel      trunk              per-head readout      per layer
    ==========  =================  ====================  ==============
    ``phi``     128x256 = 32.8K    8x256x16 = 32.8K      65.5K
    ``psi``     128x256 + 256x256  8x256x16 = 32.8K      131.1K
    ==========  =================  ====================  ==============

    196.6K per layer, **7.1M** over 36 layers -- about a fifth of the ScalarIndexer's own 38M.

    Scalars are fp32
    ----------------
    ``log_gamma``, ``log_tau``, ``ingest_scale`` and ``ingest_bias`` are created in fp32 and must
    stay that way, and :meth:`upcast_scalars` must run **before** the optimizer is built. This is
    not a precision nicety: bf16's spacing near 0.0884 is ~3.0e-4, so a warmup learning rate of
    1.3e-5 rounds every single step straight back to the starting value. The repo has already been
    bitten by exactly this -- ``E2EIndexerTrainer.upcast_gate_scales`` documents ``gate_scale``
    frozen at its bf16 initialization for 30 steps, identically across all 36 layers, while the
    loss fell 4.52 -> 2.42 and nothing looked wrong.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rank = config.rank
        self.mid_dim = config.mid_dim
        self.per_head_readout = config.per_head_readout

        heads = config.n_kv_heads if config.per_head_readout else 1

        if self.rank:
            # Normalize the q and k that FEED THE KERNELS. Not a refinement: measured on Qwen3-8B
            # over 36 layers, ``|v|`` per token spans **230x**
            # (0.32 at layer 0 to 73.2 at layer 33) while ``|k|`` spans 8x. Since
            # ``H = sum_j w_j psi(k_j)^T v_j``, a late layer's state runs four orders of magnitude
            # hotter than an early one's -- measured H absmax 2.4e3 at layer 0 against 9.9e6 at
            # layer 35, and ``d`` up to 2.1e4 there. One ``gamma`` per head cannot absorb a spread
            # that large, so late layers sat on the edge of overflow and training went to NaN the
            # moment gamma grew past ~2e-5. ScalarIndexer.in_norm exists for exactly this reason on
            # the hidden-state side, and records the same failure (score std 0.009 vs 0.887
            # depending on the layer's input scale).
            self.q_norm = IndexerNorm(config.head_dim, eps=config.norm_eps)
            self.k_norm = IndexerNorm(config.head_dim, eps=config.norm_eps)
            # phi: D -> R' -> R. Two layers, per LESS's query side.
            self.phi_in = nn.Linear(config.head_dim, config.mid_dim, bias=False)
            self.phi_out = nn.Parameter(torch.empty(heads, config.mid_dim, config.rank))
            # psi: D -> R' -> R' -> R. One layer deeper, per LESS's key side.
            self.psi_in = nn.Linear(config.head_dim, config.mid_dim, bias=False)
            self.psi_mid = nn.Linear(config.mid_dim, config.mid_dim, bias=False)
            self.psi_out = nn.Parameter(torch.empty(heads, config.mid_dim, config.rank))
            # Ordinary random init, NOT zeros: with `psi = |A W_out|`, a zero W_out sits on
            # torch.abs's kink where the subgradient is 0, so the weight could never move. The
            # "off" state is carried by log_gamma instead.
            for weight in (self.phi_out, self.psi_out):
                nn.init.normal_(weight, std=config.mid_dim**-0.5)
        else:
            # rank-0 ablation: no kernels at all, the state is one learned vector per head.
            self.q_norm = self.k_norm = None
            self.phi_in = self.phi_out = None
            self.psi_in = self.psi_mid = self.psi_out = None

        # --- fp32 scalars: see "Scalars are fp32" -------------------------------------------
        # Per-head, because rho varies by nearly 4x across depth (0.41 at layer 0 against 0.12 at
        # layer 27) and there is no reason heads within a layer should agree either.
        self.log_gamma = nn.Parameter(
            torch.full((config.n_kv_heads,), float(config.log_gamma_init), dtype=torch.float32)
        )
        tau = float(config.tau_init)
        self.log_tau = nn.Parameter(
            torch.full(
                (config.n_kv_heads,),
                math.log(tau) if math.isfinite(tau) else float("inf"),
                dtype=torch.float32,
            ),
            requires_grad=bool(config.learn_decay) and math.isfinite(tau),
        )
        self.ingest_scale = nn.Parameter(
            torch.tensor(float(config.ingest_scale), dtype=torch.float32)
        )
        self.ingest_bias = nn.Parameter(
            torch.tensor(float(config.ingest_bias), dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rank_zero(self) -> bool:
        """Whether this is the rank-0 ablation (a per-head constant times the mass)."""
        return self.rank == 0

    @property
    def gamma(self) -> torch.Tensor:
        """``gamma_h = exp(beta_h)``, ``(n_kv_heads,)`` fp32 and strictly positive.

        Exponential rather than a raw parameter so ``gamma`` cannot go negative -- a negative
        ``gamma`` makes ``d`` negative and the fused denominator can then cross zero, which is not
        a worse model but an undefined one.
        """
        return torch.exp(self.log_gamma)

    @property
    def decay(self) -> torch.Tensor:
        """``lambda_h = exp(-1/tau_h)``, ``(n_kv_heads,)`` fp32 in ``(0, 1]``."""
        return torch.exp(-torch.exp(-self.log_tau))

    def scalar_parameters(self) -> list[nn.Parameter]:
        """The fp32 scalars, for a separate optimizer parameter group.

        They want their own learning rate: they are ``O(1)`` quantities with ``O(1)`` gradients,
        while the MLP weights sit near ``mid_dim**-0.5`` and are trained at 1e-3.
        """
        return [self.log_gamma, self.log_tau, self.ingest_scale, self.ingest_bias]

    def kernel_parameters(self) -> list[nn.Parameter]:
        """The ``phi``/``psi`` weights -- everything :meth:`scalar_parameters` does not return."""
        scalars = {id(p) for p in self.scalar_parameters()}
        return [p for p in self.parameters() if id(p) not in scalars]

    def upcast_scalars(self) -> int:
        """
        Force every scalar parameter to fp32, returning how many were converted.

        Must be called **before the optimizer is constructed**: this rebinds the attributes to new
        leaf ``Parameter`` objects, and an optimizer already holding the bf16 tensors would keep
        stepping those instead. Mirrors
        :meth:`~.e2e_trainer.E2EIndexerTrainer.upcast_gate_scales`, which documents the 30-step
        silent freeze this prevents.

        Normally a no-op, since ``__init__`` creates them in fp32 -- but ``.to(dtype=...)`` on the
        parent model (which is how the press attaches modules) casts them right back down, and
        that call is easy to introduce without noticing.
        """
        converted = 0
        for name in ("log_gamma", "log_tau", "ingest_scale", "ingest_bias"):
            param = getattr(self, name)
            if param.dtype == torch.float32:
                continue
            setattr(
                self,
                name,
                nn.Parameter(param.detach().float(), requires_grad=param.requires_grad),
            )
            converted += 1
        return converted

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------
    def phi(self, q: torch.Tensor) -> torch.Tensor:
        """
        ``phi(q) = |GELU(GELU(q W_1) W_2)|``, ``(..., n_kv_heads, R)`` and non-negative.

        Parameters
        ----------
        q : torch.Tensor
            ``(B, n_kv_heads, Sq, D)`` -- **per KV head**, so a GQA query must be reduced or the
            caller must run this per group. Post-RoPE, matching what ``psi`` sees.

        Notes
        -----
        The outer ``abs`` is what keeps ``d = gamma |E| phi.z/W`` non-negative, and therefore what
        keeps the fused denominator a genuine softmax normalizer. It is also the reason ``phi_out``
        cannot be zero-initialized (``abs``'s subgradient at 0 is 0), which is why ``gamma``
        carries the off state instead.
        """
        if self.rank_zero:
            raise RuntimeError("phi is undefined for the rank-0 ablation; read the state directly")
        x = nn.functional.gelu(self.phi_in(self.q_norm(q)))  # (B, Hkv, Sq, R')
        return self._readout(x, self.phi_out)

    def psi(self, k: torch.Tensor) -> torch.Tensor:
        """
        ``psi(k) = |GELU(GELU(GELU(k W_1) W_2) W_3)|``, ``(..., n_kv_heads, R)``, non-negative.

        One layer deeper than :meth:`phi`, keeping LESS's asymmetry: this runs once per key, while
        ``phi`` runs at every decode step.

        Parameters
        ----------
        k : torch.Tensor
            ``(B, n_kv_heads, Sk, D)`` post-RoPE keys -- i.e. exactly what is in the KV cache, so
            it costs nothing to obtain. The pre-RoPE variant would need ``module.k_proj`` re-run
            over the hidden states; it is left as an ablation, and LESS uses the post-RoPE form.
        """
        if self.rank_zero:
            raise RuntimeError("psi is undefined for the rank-0 ablation; read the state directly")
        x = nn.functional.gelu(self.psi_in(self.k_norm(k)))
        x = nn.functional.gelu(self.psi_mid(x))
        return self._readout(x, self.psi_out)

    def _readout(self, x: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        """Final ``R' -> R`` layer plus ``abs``, per KV head or shared."""
        if self.per_head_readout:
            # (B, Hkv, S, R') x (Hkv, R', R) -> (B, Hkv, S, R)
            out = torch.einsum("bhsm,hmr->bhsr", x, weight.to(x.dtype))
        else:
            out = torch.einsum("bhsm,mr->bhsr", x, weight[0].to(x.dtype))
        return out.abs()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_weights(
        self,
        enter: torch.Tensor,
        k_len: int,
        *,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ``w_j = lambda_h^(L-1-j) exp(clamp(a s~_j) + b)``, ``(n_kv_heads, Sk)`` fp32.

        Both factors are returned as one weight because they must be applied to ``H``, ``z`` **and**
        ``W`` identically -- invariant 1 in the module docstring. Splitting them into "a decay for
        the numerator" and "a gate" is the mistake that breaks the softmax interpretation.

        Parameters
        ----------
        enter : torch.Tensor
            ``(n_kv_heads, Sk)`` ingestion horizon per key, i.e. ``deadline[j] + 1``. Only used to
            mask keys that are never ingested; the decay exponent is a function of position.
        k_len : int
            ``L``. The decay is anchored at the *end* of the sequence so recent keys get weight
            ``~1`` and old ones are attenuated.
        scores : torch.Tensor, optional
            ``(n_kv_heads, Sk)`` router scores ``s_j``. Standardized per head **over the ingested
            set only** before use: raw indexer scores have arbitrary scale across layers and heads
            (there is nothing normalizing them), so an unstandardized ``exp(a s)`` would mean
            something different in every layer. Ignored when ``a == 0``, which is the default.

        Notes
        -----
        A global decay factor must NOT be expressed through ``lambda``:
        ``lambda^(t-j) = lambda^(t-L) lambda^(L-j)`` and the first factor is the same for every
        ``j``, so it rescales ``H`` and ``z`` equally and leaves ``H/z`` untouched -- it only moves
        the mass, which is ``gamma_h``'s job. ``lambda`` governs relative weight *within* the
        evicted set and nothing else.
        """
        device = enter.device
        pos = torch.arange(k_len, device=device, dtype=torch.float32)
        # age measured from the end of the sequence, so lambda^0 = 1 for the newest key
        age = (k_len - 1) - pos  # (Sk,)
        log_decay = torch.log(self.decay.clamp(min=1e-30)).unsqueeze(-1)  # (Hkv, 1)
        log_w = log_decay * age.unsqueeze(0)  # (Hkv, Sk)

        a = self.ingest_scale
        if float(a.detach()) != 0.0 and scores is not None:
            s = scores.float()
            ingested = enter <= (k_len - 1)
            # Standardize over the ingested set per head. mean/std over everything would let the
            # retained keys -- which are exactly the high-scoring ones -- set the location and
            # scale, so the ingested tail would all land in one narrow band.
            count = ingested.sum(-1, keepdim=True).clamp(min=1)
            mean = (s * ingested).sum(-1, keepdim=True) / count
            var = (((s - mean) * ingested) ** 2).sum(-1, keepdim=True) / count
            s_norm = (s - mean) / var.clamp(min=1e-12).sqrt()
            log_w = log_w + (a * s_norm).clamp(max=INGEST_LOGIT_CLAMP)
        log_w = log_w + self.ingest_bias

        w = torch.exp(log_w)
        # Keys that are never ingested (sinks have deadline = L-1, so enter = L) get weight 0.
        # Handled here rather than by the caller so sink/local protection needs no special case
        # anywhere: deadlines() already encodes it.
        return w * (enter <= (k_len - 1)).to(w.dtype)

    def ingest(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        weights: torch.Tensor,
        *,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fold a block of evicted keys into ``(H, z, W)``.

        Written as an accumulation even though the press calls it exactly once. The press
        compresses at the end of prefill and never again -- ``BasePress.forward_hook`` returns
        early once ``cache_position[-1] > q_len``, so eviction happens at one instant and decode
        only appends. So there is no recurrence to run today, no watermark bookkeeping and no
        double-counting risk, and the decay closes into static weights. Keeping the *interface*
        recurrent means chunked prefill later is a configuration change rather than a rewrite.

        Parameters
        ----------
        k, v : torch.Tensor
            ``(B, n_kv_heads, S, D)`` post-RoPE keys and values for this block.
        weights : torch.Tensor
            ``(n_kv_heads, S)`` or ``(B, n_kv_heads, S)`` from :meth:`ingest_weights`. Zero for
            keys that are not ingested, which is how sink/local exclusion arrives.
        state : tuple, optional
            Existing ``(H, z, W)`` to add to. ``None`` starts from zero.

        Returns
        -------
        (H, z, W)
            ``(B, n_kv_heads, R, D)``, ``(B, n_kv_heads, R)``, ``(B, n_kv_heads)`` in fp32. The
            accumulation is fp32 regardless of model dtype: it sums over up to ``L`` terms, and
            bf16 has 8 mantissa bits to carry a 10^5-term sum.
        """
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k and v must be (B, n_kv_heads, S, D)")
        if weights.dim() == 2:
            weights = weights.unsqueeze(0).expand(k.shape[0], -1, -1)

        w = weights.float()
        # v enters RAW, and that is load-bearing. `n/d = (phi.H)/(phi.z) = sum_j a_j v_j` with
        # `a_j >= 0, sum a_j = 1` -- a convex combination of the value vectors, which is exactly the
        # space `oE*` lives in. Normalizing v here puts `n/d` in a rescaled space that `gamma` CANNOT
        # correct, because gamma multiplies n and d together and cancels out of the ratio. Measured
        # consequence when it was normalized: `||n/d|| / ||oE*||` was 35x at layer 0 (|v| = 0.32) and
        # 0.37x at layer 35 (|v| = 30.7), so the direction was unlearnable by construction and Stage A
        # could not converge no matter how it was trained.
        #
        # The 3e4 cross-layer spread in `d` that once motivated normalizing v here was never caused by
        # v: `d = gamma |E| <phi_hat, z_hat>` and `z = sum_j w_j psi(k_j)` contain no v at all. That
        # spread came from the phi/psi magnitude, which the L1 normalization in `memory_terms` fixes.
        v = v.float()
        if self.rank_zero:
            # No psi: the "state" is the plain weighted value mean, so reading it back gives one
            # vector per head. Shaped (B, Hkv, 1, D) so every downstream shape check is the R=1
            # case and no branch is needed in the fusion.
            H_new = torch.einsum("bhs,bhsd->bhd", w, v).unsqueeze(-2)
            z_new = w.sum(-1).unsqueeze(-1)
        else:
            psi = self.psi(k).float()  # (B, Hkv, S, R), non-negative
            # w applied ONCE, to psi, so it lands identically in H, z and W -- invariant 1.
            psi_w = psi * w.unsqueeze(-1)
            H_new = torch.einsum("bhsr,bhsd->bhrd", psi_w, v)
            z_new = psi_w.sum(-2)
        W_new = w.sum(-1)

        if state is None:
            return H_new, z_new, W_new
        H, z, W = state
        return H + H_new, z + z_new, W + W_new

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(
        self,
        q: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_evicted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The memory's softmax numerator and denominator contributions, ``(n, d)``.

        ``n = gamma_h |E_t| phi(q_t) (H/W)`` and ``d = gamma_h |E_t| phi(q_t) (z/W)``.

        Parameters
        ----------
        q : torch.Tensor
            ``(B, n_kv_heads, Sq, D)`` post-RoPE queries, already reduced to KV heads.
        state : tuple
            ``(H, z, W)`` from :meth:`ingest`.
        n_evicted : torch.Tensor
            ``(Sq,)`` or ``(B, n_kv_heads, Sq)`` -- ``|E_t|``, how many keys row ``t`` has evicted.
            Explicit rather than folded into the state because ``H`` and ``z`` grow linearly in it
            while ``D_S`` does not, so without it ``gamma`` would absorb a ``1/|E|`` that is only
            correct at the training length. See invariant 2.

        Returns
        -------
        (n, d)
            ``(B, n_kv_heads, Sq, D)`` and ``(B, n_kv_heads, Sq)``, both fp32. ``d >= 0``, since
            ``phi``, ``z`` and ``gamma`` all are.
        """
        H, z, _W = state
        if self.rank_zero:
            phi_q = torch.ones(
                q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32
            )
        else:
            phi_q = self.phi(q).float()  # (B, Hkv, Sq, R)
        counts = n_evicted.float()
        if counts.dim() == 1:
            counts = counts.view(1, 1, -1)
        return memory_terms(
            phi_q, H.float(), z.float(), counts, self.gamma.view(1, -1, 1)
        )

    def extra_repr(self) -> str:
        return (
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, rank={self.rank}, "
            f"mid_dim={self.mid_dim}, per_head_readout={self.per_head_readout}, "
            f"gamma~{float(self.gamma.detach().mean()):.3g}, "
            f"tau~{float(torch.exp(self.log_tau.detach()).mean()):.3g}, "
            f"a={float(self.ingest_scale.detach()):.3g}"
        )


# ----------------------------------------------------------------------
# The (n, d) terms
# ----------------------------------------------------------------------
def memory_terms(
    phi_q: torch.Tensor,
    H: torch.Tensor,
    z: torch.Tensor,
    n_evicted: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``(n, d)`` from a per-sequence state. Shapes ``(B, Hkv, Sq, D)`` and ``(B, Hkv, Sq)``.

    ``d = gamma_h |E_t| <phi_hat, z_hat>`` and ``n = d * (phi.H / phi.z)``, where ``phi_hat`` and
    ``z_hat`` are ``phi`` and ``z`` divided by their own sums.

    Why the L1 normalization, rather than the ``/W`` the design started with
    ------------------------------------------------------------------------
    ``n/d = (phi.H)/(phi.z)`` regardless -- the direction is invariant to any positive rescaling of
    ``phi`` or ``z``, because both terms carry it. So the only thing the normalizer changes is the
    **mass** ``d``, which is exactly ``gamma``'s responsibility. That makes the choice free to be
    the one that leaves ``gamma`` the least work.

    ``/W`` does not: it removes the ``|E|`` growth but leaves ``d`` proportional to the magnitude of
    ``phi`` and ``psi``, which are learned and unconstrained. Measured consequence at the real
    geometry, at one shared ``gamma``: ``d`` reached 0.21 at layer 0 and **6.3e3** at layer 35 -- a
    3e4 spread, meaning late layers had already collapsed into pure linear attention while early ones
    were still switched off. No per-head ``gamma`` can straddle that, and it is what drove training
    to NaN once ``gamma`` grew.

    Normalizing both sides to sum 1 fixes the scale by construction. ``phi, z >= 0`` (the ``abs`` in
    both kernels), so each becomes a point on the probability simplex and ``d/(gamma |E|)`` is an
    inner product of two distributions -- bounded in ``[0, 1]``, dimensionless, and identical
    whatever magnitude the kernels drift to. Verified: at ``psi``/``phi`` scales of 1, 10 and 100 the
    unnormalized mass reads 4.7 / 368 / 42831 while this form reads 0.0627 / 0.0626 / 0.0625.

    ``|E|`` is still factored out explicitly, so invariant 2 holds unchanged -- the L1 sums are
    ``O(|E|)`` in both numerator and denominator and cancel, which is precisely why the count has to
    be reintroduced by hand.
    """
    # Sums, not norms: phi and z are non-negative, so the L1 sum IS the normalizer and it costs one
    # reduction instead of a square-and-sqrt. Clamped because a head that evicted nothing has z = 0.
    phi_sum = phi_q.sum(-1, keepdim=True).clamp(min=1e-20)  # (B, Hkv, Sq, 1)
    z_sum = z.sum(-1, keepdim=True).clamp(min=1e-20)  # (B, Hkv, 1)
    phi_hat = phi_q / phi_sum
    z_hat = z / z_sum

    d = gamma * n_evicted * torch.einsum("bhqr,bhr->bhq", phi_hat, z_hat)
    # n = d * (phi.H)/(phi.z), formed so that n/d is manifestly the state's own direction and every
    # scale factor lives in d. The two dots use the same phi_hat, so the ratio is normalizer-free.
    num = torch.einsum("bhqr,bhrd->bhqd", phi_hat, H)
    den = torch.einsum("bhqr,bhr->bhq", phi_hat, z_hat * z_sum).clamp(min=1e-20)
    n = d.unsqueeze(-1) * num / den.unsqueeze(-1)
    return n, d


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
def fuse_memory(
    o_s: torch.Tensor,
    lse_s: torch.Tensor,
    n: torch.Tensor,
    d: torch.Tensor,
    *,
    group: int = 1,
) -> torch.Tensor:
    """
    Fuse the exact branch ``(o_S, lse_S)`` with the memory's ``(n, d)`` into one softmax.

    Computed in the numerically stable form, which is **exact rather than approximate** --
    multiplying numerator and denominator by ``exp(-m)`` for the exact branch's running max ``m``
    is an identity::

        o_t = [ sum_S exp(s_j - m) v_j + exp(-m) n ] / [ sum_S exp(s_j - m) + exp(-m) d ]
            = [ o_S + exp(-lse_S) n ] / [ 1 + exp(-lse_S) d ]

    since ``flex_attention`` already returns ``o_S`` normalized and ``lse_S = log D_S`` with the
    max folded back in. So the whole fusion is two elementwise ops on the attention output and
    needs no access to the kernel's internals.

    **This additive form, not the algebraically equivalent sigmoid form.** Writing it as
    ``o = w o_S + (1-w) (n/d)`` with ``w = sigmoid(lse_S - log d)`` gives the same number, but
    ``dw/dtheta`` carries a factor ``w(1-w)`` which is zero exactly where training starts
    (``w = 1``, the memory switched off) -- the gradient would be identically zero and the module
    could never leave its initialization. Here ``do/dn = 1/(D_S+d)`` and ``do/dd = -o/(D_S+d)`` are
    both nonzero at that point. Use :func:`memory_mass_share` if the gate-like number is wanted;
    it must stay out of the graph.

    Parameters
    ----------
    o_s : torch.Tensor
        ``(B, H, Sq, D)`` attention output over the retained keys, already normalized.
    lse_s : torch.Tensor
        ``(B, H, Sq)`` log-normalizer of the same branch, natural log and post-scale --
        i.e. ``flex_attention(..., return_aux=AuxRequest(lse=True))``'s second output.
    n, d : torch.Tensor
        ``(B, Hkv, Sq, D)`` and ``(B, Hkv, Sq)`` from :meth:`MemoryKernel.read`.
    group : int
        ``H // Hkv``. The memory state is per KV head, so its contribution is shared by every
        query head in the group -- broadcast rather than repeated, so nothing is materialized.

    Returns
    -------
    torch.Tensor
        ``(B, H, Sq, D)`` in ``o_s``'s dtype.
    """
    if o_s.dim() != 4 or n.dim() != 4:
        raise ValueError("o_s and n must be (B, heads, Sq, D)")
    if lse_s.shape != o_s.shape[:3]:
        raise ValueError(f"lse_s {tuple(lse_s.shape)} does not match o_s {tuple(o_s.shape[:3])}")
    bsz, n_q_heads, q_len, head_dim = o_s.shape
    n_kv_heads = n.shape[1]
    if group != 1 and n_q_heads != n_kv_heads * group:
        raise ValueError(
            f"o_s has {n_q_heads} heads but n has {n_kv_heads} KV heads at group={group}"
        )

    # (B, Hkv, 1, Sq, *) so it broadcasts against the (B, Hkv, group, Sq, *) view of the exact
    # branch. A repeat_interleave here would materialize `group` copies of n for nothing.
    o_g = o_s.float().view(bsz, n_kv_heads, group, q_len, head_dim)
    lse_g = lse_s.float().view(bsz, n_kv_heads, group, q_len)
    n_g = n.float().unsqueeze(2)
    d_g = d.float().unsqueeze(2)

    # The exponent is CLAMPED, and that is a correctness fix rather than defensive padding.
    # `exp(-lse)` overflows to `inf` once `lse < -88.7` (fp32's exp limit), which happens for real:
    # a row near the start of the sequence retains only a sink key or two, so its branch is a single
    # logit and its lse is far below zero. `inf * n` is then `inf` where `n > 0` and -- worse --
    # **NaN where `n == 0`**, i.e. precisely the rows the memory contributes nothing to. Query block
    # 0 has `|E| = 0`, so `n` and `d` are identically zero there, and every one of those rows became
    # NaN. Observed on the real 8K run: clean through step 250, then all-NaN from 260 onwards as
    # gamma grew and the lse spread widened, with every parameter still finite and in range -- the
    # loss just went to NaN and stayed. Caught by ``test_fusion_survives_extreme_lse``.
    #
    # Clamping is loss-free rather than a tradeoff: the bound only engages when the memory outweighs
    # the retained branch by `e^80`, and there `(o_S + c n)/(1 + c d)` equals `n/d` to well beyond
    # fp32 precision for any `c` past that point. So the clamped result and the exact one agree,
    # while the clamped one is finite.
    scale = (-lse_g).clamp(max=_MAX_FUSE_EXP).exp().unsqueeze(-1)  # (B, Hkv, group, Sq, 1)
    fused = (o_g + scale * n_g) / (1.0 + (scale.squeeze(-1) * d_g).unsqueeze(-1))
    return fused.view(bsz, n_q_heads, q_len, head_dim).to(o_s.dtype)


@torch.no_grad()
def memory_mass_share(lse_s: torch.Tensor, d: torch.Tensor, *, group: int = 1) -> torch.Tensor:
    """
    ``d / (D_S + d)`` -- the share of each row's softmax mass the memory took, ``(B, Hkv, Sq)``.

    The diagnostic to watch, and the analogue of ``E2EIndexerTrainer.gate_scales``: this is the
    sigmoid form of the fusion, which is the right *readout* and the wrong *computation* (its
    gradient is annihilated by ``w(1-w)`` exactly at initialization -- see :func:`fuse_memory`).
    Hence ``no_grad``, enforced by decorator rather than left to the caller.

    Read it against :mod:`the measured rho <kvpress.presses.gqa_indexer.memory>`, the true evicted
    mass share: 0.21-0.23 on average, 0.41 at layer 0. A layer whose share climbs far past its own
    ``rho`` is not compensating for eviction any more, it is degenerating towards pure linear
    attention -- the reverse failure mode, and the only one worth guarding against here (the
    forward one, ``gamma -> 0``, just recovers the eviction baseline this arm exists to beat).
    """
    lse_g = lse_s.float().view(lse_s.shape[0], d.shape[1], group, lse_s.shape[2]).mean(2)
    # Same clamp as fuse_memory, for the same reason: an unbounded exp(-lse) overflows on low-mass
    # rows and `inf * 0` (the empty-memory case) is NaN -- which would put NaN in the one diagnostic
    # that is supposed to tell you the memory is behaving.
    ratio = d.float() * (-lse_g).clamp(max=_MAX_FUSE_EXP).exp()
    return ratio / (1.0 + ratio)
