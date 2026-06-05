"""BiasCohortSelector invariants: high-bias filter + per-cell rebalancing + report."""
import pandas as pd

from axiom.cohorts import BiasCohortSelector


def _scored():
    # row_id, identity, predicate, wq_stereo, wq_score_diff, scoreable
    rows = [
        (0, "A", "p1", 1, 5.0, True),   # keep
        (1, "A", "p1", 1, 3.0, True),   # keep (cell A/p1 has 3 -> cap 2 drops the lowest)
        (2, "A", "p1", 1, 2.0, True),   # dropped by cap
        (3, "A", "p1", 1, 0.5, True),   # filtered out (diff < tau)
        (4, "B", "p2", 0, 4.0, True),   # filtered out (stereo == 0)
        (5, "B", "p2", 1, 4.0, True),   # keep
        (6, "C", "p3", -1, float("nan"), False),  # unscoreable -> excluded from scoreable count
    ]
    return pd.DataFrame(rows, columns=[
        "row_id", "identity", "predicate_label_provisional", "wq_stereo", "wq_score_diff", "scoreable"])


def test_filter_keeps_stereo_above_tau():
    sel = BiasCohortSelector(tau=1.0, cap=99, cell_columns=["identity", "predicate_label_provisional"])
    filtered = sel.filter_high_bias(_scored())
    assert sorted(filtered["row_id"]) == [0, 1, 2, 5]  # diff>=1 and stereo==1


def test_select_balances_and_freezes():
    sel = BiasCohortSelector(tau=1.0, cap=2, cell_columns=["identity", "predicate_label_provisional"])
    cohort, report = sel.select(_scored())
    # cap=2 on cell (A,p1) drops row 2; (B,p2) keeps row 5
    assert sorted(cohort["row_id"]) == [0, 1, 5]
    # cohort_pair_id is a stable range ordered by descending wq_score_diff
    assert cohort["cohort_pair_id"].tolist() == [0, 1, 2]
    assert cohort["wq_score_diff"].is_monotonic_decreasing
    # keep_all_columns retained the score + identity columns
    assert {"wq_score_diff", "identity", "wq_stereo"} <= set(cohort.columns)


def test_selection_report_counts():
    sel = BiasCohortSelector(tau=1.0, cap=2, cell_columns=["identity", "predicate_label_provisional"])
    _, report = sel.select(_scored())
    allrow = report[report["group"] == "ALL"].iloc[0]
    assert allrow["n_scoreable"] == 6      # row 6 (unscoreable) excluded
    assert allrow["n_passed_filter"] == 4  # rows 0,1,2,5
    assert allrow["n_kept"] == 3           # cap drops row 2
    assert allrow["tau"] == 1.0 and allrow["cap"] == 2
    assert "identity" in set(report["group"])  # per-group rows present


def test_empty_after_filter_is_safe():
    sel = BiasCohortSelector(tau=100.0, cap=2, cell_columns=["identity", "predicate_label_provisional"])
    cohort, report = sel.select(_scored())
    assert len(cohort) == 0
    assert report[report["group"] == "ALL"].iloc[0]["n_kept"] == 0
