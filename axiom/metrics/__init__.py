"""Centralized metric definitions.

Every metric formula used anywhere in the pipeline is defined here exactly once and
imported by the analysis classes -- never re-derived in a runner (CLAUDE.md, "Metrics
must be defined once"). Raw log-probability effects and normalized effects are kept
distinct because they answer different questions.
"""
from axiom.metrics.bias import (
    bias_effect,
    bias_fraction,
    bias_score,
    fraction_bias_removed,
    normalized_restoration,
    safe_norm,
)
from axiom.metrics.continuation import avg_logp, continuation_logp
from axiom.metrics.overlap import (
    jaccard,
    rank_biased_overlap,
    selectivity,
    spearman_layer_correlation,
    topk_set,
)

__all__ = [
    "avg_logp",
    "bias_effect",
    "bias_fraction",
    "bias_score",
    "continuation_logp",
    "fraction_bias_removed",
    "jaccard",
    "normalized_restoration",
    "rank_biased_overlap",
    "safe_norm",
    "selectivity",
    "spearman_layer_correlation",
    "topk_set",
]
