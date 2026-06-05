"""Continuation log-probability metrics (the core Wino-style probe quantity).

For a sentence ``prefix + continuation``, the model assigns each continuation token a
conditional log-probability ``log P(tok_t | tok_<t)``. The continuation score is the
sum of those, and the *average* (per-token) is the length-normalized version used for
comparing the two identity variants. These functions take an already-extracted list of
per-token log-probabilities so they are pure and unit-testable without a model.
"""
from __future__ import annotations

import math
from typing import Sequence


def continuation_logp(token_logprobs: Sequence[float]) -> float:
    """Sum of per-token continuation log-probabilities.

    Args:
        token_logprobs: ``log P(tok_t | tok_<t)`` for each continuation token, in order.

    Returns:
        The summed log-probability. ``nan`` for an empty continuation (no tokens to score).
    """
    if len(token_logprobs) == 0:
        return float("nan")
    return float(sum(token_logprobs))


def avg_logp(token_logprobs: Sequence[float]) -> float:
    """Length-normalized continuation log-probability (mean over continuation tokens).

    Length normalization is what makes the two identity variants comparable even when a
    BPE merge changes the continuation's token count between them.
    """
    if len(token_logprobs) == 0:
        return float("nan")
    return float(sum(token_logprobs) / len(token_logprobs))


def is_scoreable(prefix: str, continuation: str) -> bool:
    """A pair is scoreable iff prefix and continuation are both non-empty after strip.

    (Both prefixes must also be non-empty; callers check the second prefix separately.)
    """
    return bool(prefix.strip()) and bool(continuation.strip())


def perplexity(avg_token_logp: float) -> float:
    """Convert an average log-probability (natural log) to perplexity, for reporting."""
    if math.isnan(avg_token_logp):
        return float("nan")
    return math.exp(-avg_token_logp)
