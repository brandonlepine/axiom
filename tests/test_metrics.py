"""Metric formula invariants -- the math the whole pipeline trusts (CLAUDE.md testing)."""
import math

import pytest

from axiom.metrics import (
    avg_logp,
    bias_effect,
    bias_fraction,
    bias_score,
    continuation_logp,
    fraction_bias_removed,
    jaccard,
    normalized_restoration,
    rank_biased_overlap,
    safe_norm,
    selectivity,
    spearman_layer_correlation,
    topk_set,
)


def test_continuation_logp_sum_and_avg():
    lps = [-1.0, -2.0, -3.0]
    assert continuation_logp(lps) == -6.0
    assert avg_logp(lps) == -2.0


def test_empty_continuation_is_nan():
    assert math.isnan(continuation_logp([]))
    assert math.isnan(avg_logp([]))


def test_bias_score_direction():
    # stereotype more probable for target (higher avg logp) -> positive
    assert bias_score(-1.0, -3.0) == 2.0
    assert bias_score(-3.0, -1.0) == -2.0


def test_bias_effect_is_patched_minus_reference():
    assert bias_effect(-1.5, -2.0) == pytest.approx(0.5)


def test_normalized_restoration_full_and_none():
    # patched == target -> fully restored (1.0); patched == reference -> 0.0
    assert normalized_restoration(-1.0, -3.0, -1.0) == pytest.approx(1.0)
    assert normalized_restoration(-3.0, -3.0, -1.0) == pytest.approx(0.0)


def test_normalized_restoration_zero_gap_is_nan():
    assert math.isnan(normalized_restoration(-2.0, -2.0, -2.0))


def test_safe_norm_guards_zero_denominator():
    assert math.isnan(safe_norm(1.0, 0.0))
    assert safe_norm(1.0, 2.0) == 0.5


def test_bias_fraction_ignores_nans():
    assert bias_fraction([1.0, -1.0, 2.0, float("nan")]) == pytest.approx(2 / 3)
    assert math.isnan(bias_fraction([float("nan")]))


def test_fraction_bias_removed():
    assert fraction_bias_removed(2.0, 0.0) == pytest.approx(1.0)
    assert fraction_bias_removed(2.0, 2.0) == pytest.approx(0.0)
    assert fraction_bias_removed(2.0, 1.0) == pytest.approx(0.5)


def test_jaccard():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0


def test_topk_set_deterministic_ties():
    scores = {"a": 1.0, "b": 1.0, "c": 0.5}
    assert topk_set(scores, 2) == {"a", "b"}
    assert topk_set(scores, 0) == set()


def test_spearman_perfect_and_anti():
    assert spearman_layer_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_layer_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_constant_is_nan():
    assert math.isnan(spearman_layer_correlation([1, 1, 1], [1, 2, 3]))


def test_selectivity_concentrated_vs_diffuse():
    # one component carries everything -> selectivity 1.0
    assert selectivity([0.0, 0.0, 5.0]) == pytest.approx(1.0)
    # perfectly diffuse over n=4 -> 1/n
    assert selectivity([1, 1, 1, 1]) == pytest.approx(0.25)


def test_rank_biased_overlap_identical_and_disjoint():
    a = ["x", "y", "z"]
    assert rank_biased_overlap(a, a, p=0.9) == pytest.approx(1.0, abs=0.05)
    assert rank_biased_overlap(["x"], ["y"], p=0.9) == 0.0
