"""align_minimal_pair invariants -- the stringent prefix/suffix patching alignment.

Worked example (ids abstract; BOS=0):
  source "X is  a lesbian  and autistic" -> [0, 10, 11, 20, 21, 12, 13]
  target "X is  straight   and autistic" -> [0, 10, 11, 22,     12, 13]
  identity = source[20,21] vs target[22]; shared post = [12]; continuation = [13]
"""
from axiom.alignment import align_minimal_pair, common_suffix_len

SOURCE = [0, 10, 11, 20, 21, 12, 13]
TARGET = [0, 10, 11, 22, 12, 13]
TARGET_PREFIX = [0, 10, 11, 22, 12]  # everything before the continuation token 13


def test_common_suffix_len():
    assert common_suffix_len([1, 2, 9, 8], [5, 9, 8]) == 2
    assert common_suffix_len([1, 2, 3], [4, 5, 6]) == 0


def test_clean_pair_spans_and_source_map():
    aln, reason = align_minimal_pair(SOURCE, TARGET, TARGET_PREFIX)
    assert reason == "" and aln is not None
    assert aln.prefix_len == 3 and aln.suffix_len == 2
    assert aln.source_identity_len == 2 and aln.target_identity_len == 1
    assert aln.cont_start == 5 and aln.cont_count == 1
    assert aln.spans == ["shared_pre", "shared_pre", "shared_pre", "identity", "shared_post", "continuation"]
    # each target position maps to its aligned source position (end-aligned identity)
    assert aln.source_pos == [0, 1, 2, 4, 5, 6]
    # the single target identity token maps to the LAST source identity token
    assert aln.identity_target_positions == [3]
    assert aln.identity_source_positions == [4]


def test_continuation_token_aligns_to_same_id():
    aln, _ = align_minimal_pair(SOURCE, TARGET, TARGET_PREFIX)
    # continuation position in target is 5 (token 13); its source position must hold the same token
    cont_pos = aln.cont_start
    assert TARGET[cont_pos] == SOURCE[aln.source_pos[cont_pos]] == 13


def test_identical_sentences_have_no_identity_span():
    aln, reason = align_minimal_pair([0, 5, 6, 7], [0, 5, 6, 7], [0, 5, 6])
    assert aln is None and "identity span" in reason


def test_empty_continuation_rejected():
    # prefix == full target -> no continuation tokens
    aln, reason = align_minimal_pair(SOURCE, TARGET, TARGET)
    assert aln is None and "continuation" in reason
