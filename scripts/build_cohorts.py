#!/usr/bin/env python3
"""Freeze the combined BBQ+CrowS pod cohorts from the candidate pool, with the gender split.

Thin orchestration over :class:`axiom.cohorts.SegmentedCohortBuilder` (CLAUDE.md: runners
are thin; analysis logic lives in classes). Produces the three combined pod inputs:

  data/cohorts/combined_bbq_crows/cohort.csv          pooled BBQ+CrowS (head/MLP/resid)
  data/cohorts/residual_bbq_crows/bbq/cohort.csv      BBQ-only   (per-source resid)
  data/cohorts/residual_bbq_crows/crows/cohort.csv    CrowS-only (per-source resid)

The candidate pool already carries the gender_identity/gender_binary split; the builder
re-applies it idempotently so the output is correct regardless of pool state. The frozen
output is byte-for-byte the same membership the pod already validated, with the *only*
change being ``gender -> {gender_binary, gender_identity}`` -- this script asserts exactly
that against the pre-split cohorts when ``--verify-against`` is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `axiom` importable when run directly

import pandas as pd

from axiom.cohorts import SegmentedCohortBuilder
from axiom.cohorts.builder import gender_subaxis
from axiom.data.schemas import COMBINED_SCHEMA, validate_cohort

COMBINED_CAP = 80  # reproduces the pod cohorts exactly (verified empirically)

# (output subdir under --cohorts-root, source filter) for each combined cohort.
TARGETS = [
    ("combined_bbq_crows", None),
    ("residual_bbq_crows/bbq", "bbq"),
    ("residual_bbq_crows/crows", "crows-pairs"),
]


def _assert_only_gender_relabelled(new: pd.DataFrame, old: pd.DataFrame, name: str) -> None:
    """Assert ``new`` equals ``old`` except that coarse ``gender`` rows are now sub-axed.

    Confirms identical row membership (by row_id) and that every non-axis field matches;
    the only permitted difference is ``axis`` changing ``gender -> gender_subaxis(Group_x)``.
    """
    if set(new["row_id"]) != set(old["row_id"]):
        only_new = set(new["row_id"]) - set(old["row_id"])
        only_old = set(old["row_id"]) - set(new["row_id"])
        raise AssertionError(
            f"[{name}] row_id membership changed (only_new={len(only_new)}, only_old={len(only_old)})."
        )
    n = new.set_index("row_id").sort_index()
    o = old.set_index("row_id").sort_index()
    shared = [c for c in o.columns if c in n.columns and c not in ("cohort_pair_id", "axis")]
    mism = [c for c in shared if not n[c].fillna("").astype(str).equals(o[c].fillna("").astype(str))]
    if mism:
        raise AssertionError(f"[{name}] non-axis columns changed: {mism}")
    expected = o["axis"].where(o["axis"] != "gender", o["Group_x"].map(gender_subaxis))
    bad = n["axis"][n["axis"] != expected]
    if len(bad):
        raise AssertionError(f"[{name}] axis relabel mismatch on {len(bad)} rows.")
    n_split = int((o["axis"] == "gender").sum())
    print(f"  [{name}] verified: {len(new)} rows identical to pre-split; {n_split} gender rows sub-axed.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze combined BBQ+CrowS pod cohorts with the gender split.")
    ap.add_argument("--pool", type=Path, default=Path("data/processed/combined/bbq_crows_candidates.csv"))
    ap.add_argument("--cohorts-root", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--cap", type=int, default=COMBINED_CAP)
    ap.add_argument("--verify-against", type=Path, default=None,
                    help="Dir of pre-split cohorts to assert only-gender-relabelled against.")
    args = ap.parse_args()

    pool = pd.read_csv(args.pool)
    print(f"Candidate pool: {len(pool)} rows | sources: {dict(pool['source'].value_counts())}")
    builder = SegmentedCohortBuilder(cap=args.cap)

    for subdir, source in TARGETS:
        out_dir = args.cohorts_root / subdir
        cohort, _cov = builder.build(pool, out_dir=out_dir, source=source, split_gender=True)
        validate_cohort(cohort, COMBINED_SCHEMA)
        axes = dict(cohort["axis"].value_counts())
        print(f"\n[{subdir}] {len(cohort)} pairs -> {out_dir}/cohort.csv")
        print(f"  axes: {axes}")
        if args.verify_against is not None:
            old_path = args.verify_against / subdir / "cohort.csv"
            if old_path.exists():
                _assert_only_gender_relabelled(cohort, pd.read_csv(old_path), subdir)


if __name__ == "__main__":
    main()
