"""SegmentedCohortBuilder invariants: balancing, stable IDs, gender split, source filter."""
import pandas as pd
import pytest

from axiom.cohorts import SegmentedCohortBuilder, gender_subaxis
from axiom.cohorts.builder import apply_gender_subaxis


def _pool() -> pd.DataFrame:
    """Tiny synthetic candidate pool (CLAUDE.md: 2-5 example fixtures)."""
    rows = [
        # row_id, source, axis, block, Group_x, predicate, bias_score
        (0, "bbq", "gender", "a man", "a man", "leadership", 5.0),
        (1, "bbq", "gender", "a man", "a man", "leadership", 4.0),
        (2, "bbq", "gender", "a man", "a man", "leadership", 3.0),  # 3rd in cell -> capped at 2
        (3, "bbq", "gender", "a transgender woman", "a transgender woman", "deception", 2.0),
        (4, "crows-pairs", "race", "black", "black", "violence", 1.0),
    ]
    cols = ["row_id", "source", "axis", "block", "Group_x", "predicate_label_provisional", "bias_score"]
    df = pd.DataFrame(rows, columns=cols)
    # carry the other run/seg columns the builder expects (kept if present)
    for c in ["Group_y", "sent_x", "sent_y", "prefix_x", "prefix_y", "continuation",
              "category", "frame", "identity_mapped"]:
        df[c] = ""
    return df


def test_gender_subaxis_mapping():
    assert gender_subaxis("a man") == "gender_binary"
    assert gender_subaxis("a transgender woman") == "gender_identity"
    assert gender_subaxis("NONBINARY person") == "gender_identity"


def test_apply_gender_subaxis_idempotent():
    df = _pool()
    once = apply_gender_subaxis(df)
    twice = apply_gender_subaxis(once)
    assert "gender" not in set(once["axis"])
    assert once["axis"].tolist() == twice["axis"].tolist()  # idempotent
    assert set(once["axis"]) == {"gender_binary", "gender_identity", "race"}


def test_cap_per_cell_keeps_highest_bias():
    builder = SegmentedCohortBuilder(cap=2)
    cohort = builder.freeze(_pool())
    # the 'a man'/'leadership' cell had 3 candidates; cap=2 drops the lowest bias (row_id 2)
    man_rows = cohort[cohort["Group_x"] == "a man"]
    assert len(man_rows) == 2
    assert set(man_rows["row_id"]) == {0, 1}


def test_cohort_pair_id_is_stable_range_in_bias_order():
    cohort = SegmentedCohortBuilder(cap=10).freeze(_pool())
    assert cohort["cohort_pair_id"].tolist() == list(range(len(cohort)))
    # frozen in descending bias_score order
    assert cohort["bias_score"].is_monotonic_decreasing


def test_build_applies_split_and_source_filter():
    builder = SegmentedCohortBuilder(cap=10)
    full, _ = builder.build(_pool(), split_gender=True)
    assert set(full["axis"]) == {"gender_binary", "gender_identity", "race"}

    bbq_only, _ = builder.build(_pool(), source="bbq", split_gender=True)
    assert set(bbq_only["source"]) == {"bbq"}
    assert "race" not in set(bbq_only["axis"])  # race row was crows-pairs


def test_duplicate_row_id_raises():
    bad = _pool()
    bad.loc[0, "row_id"] = 1  # duplicate
    with pytest.raises(ValueError, match="row_id"):
        SegmentedCohortBuilder(cap=10).freeze(bad)
