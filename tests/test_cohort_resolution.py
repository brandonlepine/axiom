"""Cohort auto-resolution: latest selected analysis cohort + precedence."""
import pytest

from axiom.datasets import latest_analysis_cohort, resolve_analysis_cohort


def _make_selection_run(root, run_id):
    d = root / "outputs" / "gpt2" / "winoqueer" / "selection" / run_id
    d.mkdir(parents=True)
    f = d / "winoqueer_gpt2_analysis_cohort.csv"
    f.write_text("cohort_pair_id,row_id\n0,1\n")
    return f


def test_latest_picks_most_recent_run(tmp_path):
    _make_selection_run(tmp_path, "20260101T000000Z-aaaa")
    newer = _make_selection_run(tmp_path, "20260605T120000Z-bbbb")
    assert latest_analysis_cohort("gpt2", "winoqueer", tmp_path) == newer


def test_latest_none_when_absent(tmp_path):
    assert latest_analysis_cohort("gpt2", "winoqueer", tmp_path) is None


def test_resolve_precedence(tmp_path):
    selected = _make_selection_run(tmp_path, "20260605T120000Z-bbbb")

    # explicit override wins
    override = tmp_path / "custom.csv"
    path, src = resolve_analysis_cohort("winoqueer", "gpt2", tmp_path, cohort_override=override)
    assert path == override and src == "override"

    # use_raw -> the frozen pod cohort path
    path, src = resolve_analysis_cohort("winoqueer", "gpt2", tmp_path, use_raw=True)
    assert path == tmp_path / "data/cohorts/winoqueer/cohort.csv" and src == "raw_frozen_cohort"

    # default -> the selected analysis cohort
    path, src = resolve_analysis_cohort("winoqueer", "gpt2", tmp_path)
    assert path == selected and src == "selected_analysis_cohort"


def test_resolve_errors_when_no_selection(tmp_path):
    with pytest.raises(SystemExit, match="No selected analysis cohort"):
        resolve_analysis_cohort("winoqueer", "gpt2", tmp_path)
