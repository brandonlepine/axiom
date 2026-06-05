"""Bias and intervention-effect metrics.

Naming follows the Wino-style continuation probe, where ``target`` is the identity the
stereotype is *about* (e.g. an LGBTQ identity, a marginalized BBQ group) and
``reference`` is the contrast identity (straight/cisgender, dominant BBQ group). Using
target/reference rather than ambiguous "clean/corrupt" keeps directionality explicit
(CLAUDE.md, "Directionality must be explicit").
"""
from __future__ import annotations

import math

EPS = 1e-8


def safe_norm(numerator: float, denominator: float, eps: float = EPS) -> float:
    """``numerator / denominator``, or ``nan`` when ``|denominator| <= eps``.

    Used so a near-zero baseline gap never produces a spurious huge normalized effect.
    """
    if abs(denominator) <= eps:
        return float("nan")
    return numerator / denominator


def bias_score(target_avg_logp: float, reference_avg_logp: float) -> float:
    """Stereotype-association score: how much more probable the continuation is for the
    target identity than the reference.

    ``> 0`` means the (already-present) stereotype continuation is more probable after
    the target identity than after the reference identity -- stereotype-consistent bias.
    """
    return target_avg_logp - reference_avg_logp


def bias_effect(patched_avg_logp: float, reference_avg_logp: float) -> float:
    """Effect of an intervention on the reference run, in raw avg-logp units.

    ``patched_avg_logp`` is the reference run's continuation avg-logp *after* injecting
    target-side information (residual/head patch). Positive => the intervention moved the
    reference run toward the stereotype (recovered bias).
    """
    return patched_avg_logp - reference_avg_logp


def normalized_restoration(
    patched_avg_logp: float, reference_avg_logp: float, target_avg_logp: float, eps: float = EPS
) -> float:
    """Fraction of the target-vs-reference bias gap restored by the intervention.

    ``(patched - reference) / (target - reference)``. ``1.0`` means the patch fully
    reproduced the target run's stereotype probability; ``0.0`` means no effect. Returns
    ``nan`` when the baseline gap ``target - reference`` is within ``eps`` of zero.
    """
    return safe_norm(patched_avg_logp - reference_avg_logp, target_avg_logp - reference_avg_logp, eps)


def bias_fraction(bias_scores) -> float:
    """Fraction of pairs with a positive bias_score (stereotype favors the target)."""
    vals = [b for b in bias_scores if not (isinstance(b, float) and math.isnan(b))]
    if not vals:
        return float("nan")
    return sum(1 for b in vals if b > 0) / len(vals)


def fraction_bias_removed(baseline_effect: float, ablated_effect: float, eps: float = EPS) -> float:
    """Fraction of a baseline bias effect removed by an ablation.

    ``(baseline - ablated) / baseline``. ``1.0`` => the ablation removed all measured
    bias effect; ``0.0`` => none. ``nan`` when the baseline effect is ~0.
    """
    return safe_norm(baseline_effect - ablated_effect, baseline_effect, eps)
