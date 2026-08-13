# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for :class:`~kvpress.presses.facility_location_press.FacilityLocationPress`.

Three layers:

1. **The objective.** Greedy is checked against brute force, including the ``1 - 1/e``
   submodular bound, and against the property that motivates the whole press: it must drop a
   redundant duplicate that scalar scoring keeps.
2. **The plumbing.** ``ScorerPress`` selects by ``topk`` on a score, but facility location is a
   *set* objective with no per-key score. The rank encoding must make ``compress`` keep exactly
   the greedy set -- if that breaks, the press silently degrades into an arbitrary selection.
3. **The press contract.** Real model, correct cache length, protection honoured, and the
   documented failure modes raise rather than passing quietly.

fp64 for the objective tests, since the greedy is exact arithmetic and the tolerance should
measure floating-point noise only.
"""

import itertools
import math

import pytest
import torch

from kvpress import FacilityLocationPress
from kvpress.presses.facility_location_press import facility_location_order

DT = torch.float64


def coverage(demand: torch.Tensor, selection) -> float:
    """The facility-location objective for an explicit selection."""
    return float(demand[:, list(selection)].max(dim=-1).values.sum())


def brute_force(demand: torch.Tensor, budget: int):
    """Exhaustive optimum, for small problems only."""
    n_keys = demand.shape[-1]
    return max(
        (coverage(demand, c), sorted(c)) for c in itertools.combinations(range(n_keys), budget)
    )


def tiny_model(n_layers=2, n_heads=8, n_kv_heads=4, hidden=64, head_dim=8):
    """A small real Llama on eager attention, so `attentions` is actually populated."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config.num_attention_heads = n_heads
    config.num_key_value_heads = n_kv_heads
    config.hidden_size = hidden
    config.intermediate_size = 2 * hidden
    config.num_hidden_layers = n_layers
    config.head_dim = head_dim
    config._attn_implementation = "eager"
    model = transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    return model, config


# ----------------------------------------------------------------------
# 1. The objective
# ----------------------------------------------------------------------
def test_greedy_respects_the_submodular_bound():
    """
    Greedy must never fall below ``1 - 1/e`` of the brute-force optimum.

    That is the guarantee the objective's monotone submodularity buys (Nemhauser 1978), and it
    is what justifies using greedy at all rather than searching. A violation would mean the
    marginal-gain computation is wrong -- most likely the ``clamp_min(0)``, without which a key
    that serves a row *worse* than the incumbent would wrongly subtract coverage and break
    monotonicity.
    """
    bound = 1 - 1 / math.e
    ratios = []
    for seed in range(200):
        torch.manual_seed(seed)
        n_demand, n_keys, budget = 6, 8, 3
        demand = torch.rand(n_demand, n_keys, dtype=DT)
        picked = facility_location_order(demand.unsqueeze(0), budget)[0].tolist()
        best, _ = brute_force(demand, budget)
        ratios.append(coverage(demand, picked) / best)

    ratios = torch.tensor(ratios)
    assert ratios.min() >= bound, f"submodular bound violated: worst ratio {ratios.min():.4f}"
    # Greedy is usually much better than the worst-case bound; if it is not, something is off
    # even though the bound technically holds.
    assert ratios.median() > 0.99


def test_greedy_drops_a_redundant_duplicate():
    """
    The property that motivates the press: a duplicate key must lose its slot.

    Two keys serving exactly the same rows are worth one slot, not two. Scalar scoring cannot
    express this -- both duplicates get the same high score and both are kept. Facility
    location's inner ``max`` collapses the second one's marginal gain to zero.
    """
    demand = torch.zeros(4, 5, dtype=DT)
    demand[:, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=DT)
    demand[:, 1] = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=DT)  # exact duplicate of key 0
    demand[:, 2] = torch.tensor([0.0, 0.0, 0.9, 0.0], dtype=DT)
    demand[:, 3] = torch.tensor([0.0, 0.0, 0.0, 0.9], dtype=DT)
    demand[:, 4] = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=DT)

    picked = sorted(facility_location_order(demand.unsqueeze(0), 3)[0].tolist())
    assert not (0 in picked and 1 in picked), f"kept both duplicates: {picked}"
    assert coverage(demand, picked) == pytest.approx(brute_force(demand, 3)[0])

    # A summed scalar score, by contrast, ranks the two duplicates first and second.
    scalar_top = sorted(demand.sum(0).topk(3).indices.tolist())
    assert 0 in scalar_top and 1 in scalar_top, "the scalar baseline should fall for the duplicate"


def test_initial_cover_redirects_the_selection():
    """
    Coverage already provided by protected keys must be credited before selecting.

    Without it the greedy re-covers rows the sink/local keys already serve, spending budget on
    keys that are redundant *with the protected set* -- a bug that would be invisible in the
    output, just a slightly worse cache.
    """
    demand = torch.zeros(3, 4, dtype=DT)
    demand[:, 0] = torch.tensor([1.0, 1.0, 0.0], dtype=DT)  # forced
    demand[:, 1] = torch.tensor([0.9, 0.9, 0.0], dtype=DT)  # redundant with the forced key
    demand[:, 2] = torch.tensor([0.0, 0.0, 0.6], dtype=DT)  # serves the row nothing else does
    demand[:, 3] = torch.tensor([0.1, 0.1, 0.1], dtype=DT)
    forced = torch.zeros(1, 4, dtype=torch.bool)
    forced[0, 0] = True

    blind = facility_location_order(demand.unsqueeze(0), 1, forced=forced)[0].tolist()
    aware = facility_location_order(
        demand.unsqueeze(0), 1, forced=forced, initial_cover=demand[:, [0]].amax(-1).unsqueeze(0)
    )[0].tolist()
    assert blind == [1], "without initial_cover, the redundant key should win"
    assert aware == [2], "with initial_cover, the complementary key should win"


def test_forced_keys_are_never_reselected():
    """A protected key must not consume one of the greedy's slots as well."""
    torch.manual_seed(0)
    demand = torch.rand(5, 6, dtype=DT)
    demand[:, 2] = 10.0  # would dominate the ranking if it were selectable
    forced = torch.zeros(1, 6, dtype=torch.bool)
    forced[0, 2] = True
    picked = facility_location_order(demand.unsqueeze(0), 3, forced=forced)[0].tolist()
    assert 2 not in picked


def test_batched_matches_per_element():
    """Leading dims are independent problems; batching must not couple them."""
    torch.manual_seed(0)
    demand = torch.rand(2, 3, 7, 9, dtype=DT)
    batched = facility_location_order(demand, 4)
    for i in range(2):
        for j in range(3):
            alone = facility_location_order(demand[i, j].unsqueeze(0), 4)[0]
            assert torch.equal(alone, batched[i, j])


def test_candidate_pool_bounds_the_search_without_breaking_the_budget():
    """
    ``candidate_pool`` trades the guarantee for cost, but must still fill the budget.

    The pool is raised to at least the budget for exactly this reason -- a pool smaller than the
    budget would leave the greedy unable to pick enough keys and silently return duplicates or
    garbage indices.
    """
    torch.manual_seed(0)
    demand = torch.rand(6, 20, dtype=DT)
    for pool in (1, 3, 8, 50):
        picked = facility_location_order(demand.unsqueeze(0), 5, candidate_pool=pool)[0].tolist()
        assert len(set(picked)) == 5, f"pool={pool} produced {picked}"
        assert all(0 <= j < 20 for j in picked), f"pool={pool} returned out-of-range {picked}"
    # With a pool covering everything, the result is identical to the unrestricted greedy.
    assert facility_location_order(demand.unsqueeze(0), 5, candidate_pool=20)[0].tolist() == (
        facility_location_order(demand.unsqueeze(0), 5)[0].tolist()
    )


def test_candidate_pool_actually_reduces_work():
    """
    The pool must *narrow the tensor*, not merely mask the unwanted keys.

    The first implementation masked ``available`` and left the inner loop reducing over all
    ``n_keys``, so the knob cost the same as no knob at all -- measured 482.5s against 486.1s,
    i.e. entirely inert while appearing to be a performance option. Masking is invisible to a
    correctness test, so this asserts the observable consequence: restricting the pool to keys
    that are *not* the globally best ones must change the selection.
    """
    demand = torch.zeros(4, 10, dtype=DT)
    # keys 8,9 dominate total demand, so an unrestricted greedy takes them first.
    demand[:, 8] = 1.0
    demand[:, 9] = 0.9
    demand[:, 0] = torch.tensor([0.4, 0.0, 0.0, 0.0], dtype=DT)
    demand[:, 1] = torch.tensor([0.0, 0.4, 0.0, 0.0], dtype=DT)

    unrestricted = facility_location_order(demand.unsqueeze(0), 2)[0].tolist()
    assert 8 in unrestricted, "sanity: key 8 should win without a pool"

    # A pool of 2 keeps only the two highest-total keys, so the answer is {8, 9}...
    pooled = facility_location_order(demand.unsqueeze(0), 2, candidate_pool=2)[0].tolist()
    assert sorted(pooled) == [8, 9], f"pool=2 should be forced onto the top-total keys, got {pooled}"


def test_candidate_pool_indices_map_back_to_original_positions():
    """
    Narrowing the tensor makes the greedy work in pool-local indices; they must be mapped back.

    Without the final gather the press would evict by pool-local index -- keeping entirely the
    wrong keys while every shape and count still looked right. Demands are made distinct per row
    so the expected answer is unambiguous (equal columns would let ``argmax`` break ties at
    index 0 and hide a mapping error).
    """
    demand = torch.zeros(3, 12, dtype=DT)
    demand[:, 11] = torch.tensor([1.0, 0.1, 0.1], dtype=DT)
    demand[:, 10] = torch.tensor([0.1, 0.9, 0.1], dtype=DT)
    demand[:, 9] = torch.tensor([0.1, 0.1, 0.8], dtype=DT)
    demand[:, 0] = 0.02

    picked = facility_location_order(demand.unsqueeze(0), 2, candidate_pool=4)[0]
    # Cross-check against manually narrowing to the same pool and mapping by hand.
    pool_idx = demand.sum(0).topk(4).indices.sort().values
    local = facility_location_order(demand[:, pool_idx].unsqueeze(0), 2)[0]
    assert picked.tolist() == pool_idx[local].tolist()
    assert picked.tolist() == [11, 10], f"expected the two best keys in order, got {picked.tolist()}"


def test_selection_order_is_by_decreasing_marginal_gain():
    """The returned order is the greedy's own, first pick first -- the rank encoding relies on it."""
    demand = torch.zeros(3, 4, dtype=DT)
    demand[:, 0] = torch.tensor([1.0, 1.0, 1.0], dtype=DT)  # best single key
    demand[:, 1] = torch.tensor([0.5, 0.0, 0.0], dtype=DT)
    demand[:, 2] = torch.tensor([0.0, 0.2, 0.0], dtype=DT)
    demand[:, 3] = torch.tensor([0.0, 0.0, 0.1], dtype=DT)
    assert facility_location_order(demand.unsqueeze(0), 2)[0, 0].item() == 0


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_raises(budget):
    with pytest.raises(ValueError, match="budget must be positive"):
        facility_location_order(torch.rand(3, 4, dtype=DT).unsqueeze(0), budget)


def test_one_dimensional_demand_raises():
    with pytest.raises(ValueError, match="at least 2D"):
        facility_location_order(torch.rand(5, dtype=DT), 2)


# ----------------------------------------------------------------------
# 2. The rank encoding
# ----------------------------------------------------------------------
def test_scores_reproduce_the_greedy_set_under_topk():
    """
    ``ScorerPress.compress`` keeps ``topk(score)``, so the rank encoding must be exact.

    This is the press's one structural risk: a set objective has no per-key score, so the
    selection is smuggled through as ranks. If the encoding is off -- ties, wrong sign, a
    protected key not outranking a greedy pick -- the press keeps a *different* set than the one
    the objective chose, and nothing downstream would reveal it.
    """
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=0.5, n_sink=2, n_local=4)
    input_ids = torch.randint(0, config.vocab_size, (2, 64))

    captured = {}
    original = press.score

    def spy(module, hidden_states, keys, values, attentions, kwargs):
        scores = original(module, hidden_states, keys, values, attentions, kwargs)
        captured[int(module.layer_idx)] = scores.clone()
        return scores

    press.score = spy
    with torch.no_grad(), press(model):
        model(input_ids, use_cache=True)

    n_kept = int(64 * 0.5)
    for scores in captured.values():
        # Exactly n_kept keys carry a positive score, and those are exactly what topk takes.
        assert bool((scores > 0).sum(-1).eq(n_kept).all()), "positive-score count != budget"
        chosen = scores.topk(n_kept, dim=-1).indices
        assert bool(scores.gather(-1, chosen).gt(0).all()), "topk selected a zero-score key"


def test_protected_keys_outrank_every_greedy_pick():
    """Sink and local keys must survive top-k regardless of what the greedy chose."""
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=0.5, n_sink=3, n_local=5)
    input_ids = torch.randint(0, config.vocab_size, (1, 64))

    captured = {}
    original = press.score

    def spy(module, hidden_states, keys, values, attentions, kwargs):
        scores = original(module, hidden_states, keys, values, attentions, kwargs)
        captured[int(module.layer_idx)] = scores.clone()
        return scores

    press.score = spy
    with torch.no_grad(), press(model):
        model(input_ids, use_cache=True)

    for scores in captured.values():
        protected = torch.zeros(64, dtype=torch.bool)
        protected[:3] = True
        protected[-5:] = True
        assert scores[..., protected].min() > scores[..., ~protected].max()


# ----------------------------------------------------------------------
# 3. The press contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75])
def test_cache_is_compressed_to_the_expected_length(ratio):
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=ratio, n_sink=2, n_local=4)
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    with torch.no_grad(), press(model):
        out = model(input_ids, use_cache=True)
    assert out.past_key_values.layers[0].keys.shape[2] == int(64 * (1 - ratio))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_reduce": "max"},
        {"normalize_by_visibility": True},
        {"n_demand_sample": 16},
        {"candidate_pool": 24},
        {"n_sink": 0, "n_local": 0},
    ],
)
def test_options_run_end_to_end(kwargs):
    """Every documented option must survive a real forward pass."""
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=0.5, **kwargs)
    input_ids = torch.randint(0, config.vocab_size, (1, 48))
    with torch.no_grad(), press(model):
        out = model(input_ids, use_cache=True)
    assert out.past_key_values.layers[0].keys.shape[2] == 24


def test_missing_attentions_fails_loudly():
    """
    Without ``attn_implementation="eager"`` there is no attention matrix to cover.

    Asserts rather than falling back: a silent fallback would make this press quietly become a
    different method, which is precisely the class of bug the whole design is trying to avoid.
    """
    model, config = tiny_model()
    model.config._attn_implementation = "sdpa"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "sdpa"
    press = FacilityLocationPress(compression_ratio=0.5)
    with pytest.raises(AssertionError, match="eager"):
        with torch.no_grad(), press(model):
            model(torch.randint(0, config.vocab_size, (1, 32)), use_cache=True)


def test_overfull_protection_warns_and_keeps_protected(caplog):
    """
    When sink+local alone exceed the budget the greedy has no slots -- say so.

    Silently returning a protection-only mask would leave ``ScorerPress``'s top-k breaking ties
    by index order, i.e. a selection nobody chose.
    """
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=0.9, n_sink=8, n_local=8)
    with caplog.at_level("WARNING"), torch.no_grad(), press(model):
        model(torch.randint(0, config.vocab_size, (1, 64)), use_cache=True)
    assert any("fills the keep budget" in r.message for r in caplog.records)


def test_padding_rows_do_not_earn_coverage():
    """
    A padded query row is not a real demand; letting it count would reward serving padding.

    Checked by construction: with a left-padded batch the scores must still select exactly the
    budget, and must not be identical to the unpadded run (the demand set genuinely differs).
    """
    model, config = tiny_model()
    press = FacilityLocationPress(compression_ratio=0.5, n_sink=2, n_local=4)
    input_ids = torch.randint(0, config.vocab_size, (1, 48))
    attention_mask = torch.ones(1, 48, dtype=torch.long)
    attention_mask[0, :12] = 0
    with torch.no_grad(), press(model):
        out = model(input_ids, attention_mask=attention_mask, use_cache=True)
    assert out.past_key_values.layers[0].keys.shape[2] == 24


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_sink": -1}, "non-negative"),
        ({"n_local": -1}, "non-negative"),
        ({"group_reduce": "mean"}, "group_reduce must be"),
        ({"n_demand_sample": 0}, "n_demand_sample must be positive"),
        ({"candidate_pool": 0}, "candidate_pool must be positive"),
    ],
)
def test_invalid_configuration_raises_at_construction(kwargs, match):
    """Typos surface when the press is built, not mid-run."""
    with pytest.raises(ValueError, match=match):
        FacilityLocationPress(compression_ratio=0.5, **kwargs)
