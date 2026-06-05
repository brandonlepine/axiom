"""Token alignment and continuation-span detection (pure, no model required).

Wino-style scoring compares two minimally different sentences that share a trailing
stereotype *continuation* and differ only in the named identity. To score the
continuation we must find, in token space, where the continuation begins -- robust to
BPE boundary merges between the prefix and the continuation. These helpers operate on
already-tokenized id lists so they are deterministic and unit-testable without a model.
"""
from __future__ import annotations

from typing import Sequence


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest shared leading run of token ids between ``a`` and ``b``."""
    k = 0
    n = min(len(a), len(b))
    while k < n and a[k] == b[k]:
        k += 1
    return k


def continuation_start(full_ids: Sequence[int], prefix_ids: Sequence[int]) -> int:
    """Index in ``full_ids`` where the continuation begins.

    Defined as the longest shared token prefix between the full sentence and the prefix
    (so BPE merges across the boundary are handled), clamped to >= 1 so there is always
    at least one conditioning token for the autoregressive score.

    Args:
        full_ids: token ids of ``prefix + continuation`` (with any special tokens).
        prefix_ids: token ids of the prefix alone (tokenized the same way).

    Returns:
        The 0-based start index of the continuation within ``full_ids``.
    """
    return max(common_prefix_len(full_ids, prefix_ids), 1)


def continuation_span(full_ids: Sequence[int], prefix_ids: Sequence[int], real_len: int) -> tuple[int, int]:
    """Half-open ``[start, end)`` token span of the continuation within ``full_ids``.

    Args:
        full_ids: token ids of the full sentence.
        prefix_ids: token ids of the prefix alone.
        real_len: number of non-pad tokens in ``full_ids`` (end of the real sequence).

    Returns:
        ``(start, end)`` with ``end == real_len``. ``end - start`` may be 0 for an empty
        continuation -- the caller is expected to record that as a diagnostic, not score it.
    """
    start = continuation_start(full_ids, prefix_ids)
    return start, real_len
