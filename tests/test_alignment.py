"""Token-alignment / continuation-span invariants (CLAUDE.md: test token alignment)."""
from axiom.alignment import common_prefix_len, continuation_span, continuation_start


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([5, 6], [7, 8]) == 0


def test_continuation_start_clamped_to_one():
    # identical leading tokens -> continuation begins right after the shared prefix
    full = [1, 2, 3, 4, 5]      # prefix(1,2,3) + continuation(4,5)
    prefix = [1, 2, 3]
    assert continuation_start(full, prefix) == 3


def test_continuation_start_never_zero():
    # no shared prefix would give 0; clamp to 1 so there's a conditioning token
    assert continuation_start([9, 8, 7], [1, 2]) == 1


def test_continuation_span_returns_real_len_as_end():
    full = [1, 2, 3, 4, 5, 0, 0]   # real_len excludes padding
    prefix = [1, 2, 3]
    start, end = continuation_span(full, prefix, real_len=5)
    assert (start, end) == (3, 5)
    assert end - start == 2  # two continuation tokens
