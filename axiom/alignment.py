"""Token alignment and span detection (pure, no model required).

Wino-style scoring compares two minimally different sentences that differ only in the
named identity. Two alignment views are needed and both live here so they are
deterministic and unit-testable without a model:

  * :func:`shared_token_spans` -- the canonical WinoQueer view: the aligned token
    positions the two sentences *share* (everything except the swapped identity), found
    with difflib. The bias scorer sums log-probs over these shared tokens.
  * :func:`continuation_start` / :func:`continuation_span` -- the prefix+continuation
    view used by the patching steps, which read a single readout/continuation span.
"""
from __future__ import annotations

import difflib
from typing import Sequence


def shared_token_spans(ids_a: Sequence[int], ids_b: Sequence[int]) -> tuple[list[int], list[int]]:
    """Aligned positions of the tokens two sequences *share*, via difflib (WinoQueer `get_span`).

    Returns ``(positions_a, positions_b)`` -- equal-length lists where ``positions_a[k]``
    in ``ids_a`` and ``positions_b[k]`` in ``ids_b`` are the same token. Only ``equal``
    opcodes contribute, so the swapped identity tokens (the only difference between a
    minimal pair) are excluded; everything else (context + stereotype) is shared.

    This is the exact alignment the published WinoQueer autoregressive metric uses to
    decide which tokens to score. Operating on ids (cast to str for the matcher) makes it
    deterministic and model-free.
    """
    sa = [str(x) for x in ids_a]
    sb = [str(x) for x in ids_b]
    matcher = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    pos_a: list[int] = []
    pos_b: list[int] = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            pos_a.extend(range(a0, a1))
            pos_b.extend(range(b0, b1))
    assert len(pos_a) == len(pos_b), "shared spans must align one-to-one"
    return pos_a, pos_b


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
