"""Cross-dataset / cross-identity comparison metrics.

These quantify how similar two circuits are: which components (heads, neurons, layers)
they share, and how correlated their layer profiles are. Used by the segmented and
cross-dataset analyses to compare *mechanistic profiles*, not just effect sizes
(planning doc, "Treat the Datasets as Matched Bias Axes").
"""
from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence


def topk_set(scores: dict[Hashable, float], k: int) -> set[Hashable]:
    """The ``k`` keys with the highest scores (ties broken by sorted key for determinism)."""
    if k <= 0:
        return set()
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], repr(kv[0])))
    return {key for key, _ in ordered[:k]}


def jaccard(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    """Jaccard overlap ``|A ∩ B| / |A ∪ B|``. Two empty sets overlap perfectly (1.0)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank transform (ties share the mean of their ranks), like scipy's default."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman_layer_correlation(profile_a: Sequence[float], profile_b: Sequence[float]) -> float:
    """Spearman rank correlation between two per-layer effect profiles.

    Measures *layer stability*: whether two datasets/identities localize bias to the
    same layers in the same order, independent of absolute magnitude. ``nan`` if either
    profile is constant or empty.
    """
    if len(profile_a) != len(profile_b) or len(profile_a) < 2:
        return float("nan")
    return _pearson(_rankdata(profile_a), _rankdata(profile_b))


def selectivity(values: Iterable[float]) -> float:
    """Herfindahl-style concentration of (non-negative) effect mass, in ``[0, 1]``.

    ``sum(p_i^2)`` over the normalized magnitudes. Near ``1`` => a few components carry
    the effect (a concentrated circuit); near ``1/n`` => diffuse. Magnitudes are used so
    sign does not cancel concentration.
    """
    mags = [abs(v) for v in values]
    total = sum(mags)
    if total <= 0:
        return float("nan")
    return float(sum((m / total) ** 2 for m in mags))


def rank_biased_overlap(a: Sequence[Hashable], b: Sequence[Hashable], p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap of two ranked lists (top-weighted, ``[0, 1]``).

    Unlike Jaccard, RBO rewards agreement near the top of the ranking. ``p`` controls
    top-weighting (smaller => more top-heavy). This is the RBO_ext form (Webber et al.,
    2010), so identical lists score 1.0 and disjoint lists 0.0; evaluated to depth
    ``k = min(len(a), len(b))`` (exact for equal-length rankings -- the usual top-k head
    or neuron comparison).
    """
    if not a or not b:
        return 0.0
    k = min(len(a), len(b))
    sa: set[Hashable] = set()
    sb: set[Hashable] = set()
    weighted = 0.0  # sum_{d=1}^{k} (X_d / d) * p^d
    x_k = 0  # intersection size at depth k
    for d in range(1, k + 1):
        sa.add(a[d - 1])
        sb.add(b[d - 1])
        x_d = len(sa & sb)
        if d == k:
            x_k = x_d
        weighted += (x_d / d) * (p ** d)
    return (x_k / k) * (p ** k) + ((1 - p) / p) * weighted
