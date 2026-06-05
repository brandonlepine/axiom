"""Token alignment and span detection (pure, no model required).

Wino-style scoring compares two minimally different sentences that differ only in the
named identity. Two alignment views are needed and both live here so they are
deterministic and unit-testable without a model:

  * :func:`shared_token_spans` -- the canonical WinoQueer view: the aligned token
    positions the two sentences *share* (everything except the swapped identity), found
    with difflib. The bias scorer sums log-probs over these shared tokens.
  * :func:`continuation_start` / :func:`continuation_span` -- the prefix+continuation
    view used by the patching steps, which read a single readout/continuation span.
  * :func:`align_minimal_pair` -- the stringent prefix/suffix alignment the activation
    patching steps use: it locates the single contiguous identity span and maps each
    target-run position to its aligned source-run position, end-aligning the shared
    post-identity scaffold and continuation exactly.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
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


def common_suffix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest shared trailing run of token ids between ``a`` and ``b``."""
    k = 0
    n = min(len(a), len(b))
    while k < n and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


# Span labels for an aligned minimal pair, in prompt order.
PATCH_SPANS = ("shared_pre", "identity", "shared_post", "continuation")


@dataclass(frozen=True)
class PairAlignment:
    """Token-level alignment of a minimal pair for activation patching.

    ``source`` is the stereotype-associated variant (sent_x: the activations injected);
    ``target`` is the reference variant (sent_y: the run patched into). For each target
    position, :attr:`source_pos` gives the aligned source position to copy from and
    :attr:`spans` gives its span label (one of :data:`PATCH_SPANS`).
    """

    source_ids: list[int]
    target_ids: list[int]
    source_pos: list[int]
    spans: list[str]
    cont_start: int               # continuation start index within the target sequence
    cont_count: int               # number of continuation (readout) tokens
    prefix_len: int               # P: shared leading tokens
    suffix_len: int               # S: shared trailing tokens
    source_identity_len: int      # Lx: identity-span length in the source
    target_identity_len: int      # Ly: identity-span length in the target
    identity_target_positions: list[int]
    identity_source_positions: list[int]


def align_minimal_pair(
    source_ids: Sequence[int], target_ids: Sequence[int], target_prefix_ids: Sequence[int]
) -> tuple[PairAlignment | None, str]:
    """Align a minimal pair for activation patching; return ``(alignment, reason)``.

    The two prompts must differ only in a single contiguous identity span (a clean
    minimal pair). Let ``P`` = longest common token prefix and ``S`` = longest common
    token suffix of (source, target). ``[0, P)`` is identical 1:1; ``[P, end)`` is
    shifted by ``delta = len_source - len_target`` so the shared post-identity scaffold
    and the continuation end-align exactly, and each target identity token maps to the
    end-aligned source identity token (clamped into the source identity span).

    Args:
        source_ids: token ids of the source (stereotype) sentence.
        target_ids: token ids of the target (reference) sentence.
        target_prefix_ids: token ids of the target's prefix (everything before the
            scored continuation), used to locate the readout span.

    Returns:
        ``(PairAlignment, "")`` on success, or ``(None, reason)`` if the pair is not a
        clean single-identity-span difference (an expected, recorded skip -- not an error).
    """
    s = list(source_ids)
    t = list(target_ids)
    len_s, len_t = len(s), len(t)

    p = common_prefix_len(s, t)
    suf = common_suffix_len(s, t)
    suf = min(suf, len_s - p, len_t - p)  # keep prefix/suffix from overlapping
    lx = len_s - p - suf                  # source identity-span length
    ly = len_t - p - suf                  # target identity-span length
    if lx <= 0 or ly <= 0:
        return None, "no contiguous identity span (lengths)"

    cont_start = common_prefix_len(t, list(target_prefix_ids))
    cont_count = len_t - cont_start
    if cont_count <= 0:
        return None, "empty continuation in target"
    # The continuation must lie within the shared post-identity suffix (identical in both).
    if cont_start < p + ly or cont_start < len_t - suf:
        return None, "continuation not within shared suffix"

    delta = len_s - len_t  # = lx - ly
    source_pos: list[int] = []
    spans: list[str] = []
    for c in range(len_t):
        if c < p:
            source_pos.append(c)
            spans.append("shared_pre")
            continue
        source_pos.append(min(max(c + delta, p), len_s - 1))
        if c < p + ly:
            spans.append("identity")
        elif c < cont_start:
            spans.append("shared_post")
        else:
            spans.append("continuation")

    id_target = list(range(p, p + ly))
    id_source = [min(max(c + delta, p), len_s - 1) for c in id_target]
    return (
        PairAlignment(
            source_ids=s, target_ids=t, source_pos=source_pos, spans=spans,
            cont_start=cont_start, cont_count=cont_count, prefix_len=p, suffix_len=suf,
            source_identity_len=lx, target_identity_len=ly,
            identity_target_positions=id_target, identity_source_positions=id_source,
        ),
        "",
    )


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
