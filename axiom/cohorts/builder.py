"""SegmentedCohortBuilder: freeze a balanced, stable cohort from a candidate pool.

A *candidate pool* is the scored, source-tagged set of identity-contrast pairs. The
intervention run-scripts, however, need a single **frozen** cohort that:

  * is balanced (no identity/predicate cell dominates the effect estimates),
  * is consumed in file order (so a positional index is stable across scripts), and
  * carries a stable ``cohort_pair_id`` plus the ``row_id`` every segmentation join keys on.

This class is the one place that freezing happens. It generalizes the prior repo's two
separate builders (WinoQueer and BBQ) into one config-driven component:

  * balancing caps each ``(block, predicate_label_provisional)`` cell at ``cap``, keeping
    the highest-``bias_score`` pairs;
  * the survivors are re-sorted globally by ``bias_score`` (descending, ``row_id`` as a
    deterministic tiebreak) and frozen with a fresh ``cohort_pair_id``.

Because the cap keys on ``block`` (the per-identity grouping), and the gender sub-axis is
a pure function of the identity (man/woman blocks vs transgender blocks are disjoint),
splitting ``gender`` into ``gender_binary`` / ``gender_identity`` is *balance-neutral*:
it relabels ``axis`` without moving any row between cells. See ADR 0002.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

# Tokens marking the trans/nonbinary identity construct, used to split the coarse
# `gender` axis. WinoQueer's gender_identity axis is trans/NB IDENTITY bias; BBQ/CrowS
# `gender` is mostly cis man/woman ROLE bias with a trans minority. Routing them to
# separate axes lets the trans/NB cells compare directly to WinoQueer (gender_identity).
GENDER_IDENTITY_TOKENS: tuple[str, ...] = ("trans", "nonbinary", "non-binary", "enby")

# The pair-level columns the run-scripts read (align_pair needs sent_x/sent_y/prefix_y;
# raws re-emit the rest) plus the segmentation columns that downstream analyses join on.
_RUN_COLS = (
    "row_id", "Group_x", "Group_y", "sent_x", "sent_y", "prefix_x", "prefix_y",
    "continuation", "predicate_label_provisional", "bias_score",
)
_SEG_COLS = ("category", "axis", "block", "frame", "source", "identity_mapped")


def gender_subaxis(group: str) -> str:
    """Map an identity string to ``gender_identity`` (trans/NB) or ``gender_binary``."""
    g = str(group).lower()
    return "gender_identity" if any(t in g for t in GENDER_IDENTITY_TOKENS) else "gender_binary"


def apply_gender_subaxis(df: pd.DataFrame, axis_col: str = "axis", group_col: str = "Group_x") -> pd.DataFrame:
    """Return a copy with any coarse ``gender`` axis rows split by identity.

    Idempotent: rows already labelled ``gender_binary`` / ``gender_identity`` are left
    untouched, so running this on an already-split pool is a no-op. Rows on other axes
    are never affected.
    """
    out = df.copy()
    mask = out[axis_col] == "gender"
    if mask.any():
        out.loc[mask, axis_col] = out.loc[mask, group_col].map(gender_subaxis)
    return out


@dataclass
class SegmentedCohortBuilder:
    """Freeze a balanced cohort from a candidate pool.

    Args:
        cap: maximum pairs kept per ``cell_columns`` cell (highest bias_score wins).
        cell_columns: the balancing cell (default ``(block, predicate_label_provisional)``).
        sort_by: descending sort keys defining "highest bias first" + a stable tiebreak.
        run_columns / seg_columns: columns carried into the frozen cohort if present.
    """

    cap: int
    cell_columns: Sequence[str] = ("block", "predicate_label_provisional")
    sort_by: Sequence[str] = ("bias_score", "row_id")
    run_columns: Sequence[str] = _RUN_COLS
    seg_columns: Sequence[str] = _SEG_COLS
    cell_for_coverage: str = "predicate_label_provisional"

    def _validate_pool(self, df: pd.DataFrame) -> None:
        if "row_id" not in df.columns:
            raise ValueError("candidate pool must have a row_id column (the stable join key).")
        if df["row_id"].isna().any() or df["row_id"].duplicated().any():
            raise ValueError("row_id has nulls or duplicates; it is the join key and must be unique.")
        missing = [c for c in self.cell_columns if c not in df.columns]
        if missing:
            raise ValueError(f"candidate pool missing balancing-cell columns: {missing}")

    def freeze(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Balance + freeze the pool into a cohort with a fresh stable ``cohort_pair_id``.

        Deterministic: ``mergesort`` (stable) on descending ``sort_by``, cap per cell,
        re-sort survivors, then assign ``cohort_pair_id = 0..n-1`` in frozen order.
        """
        self._validate_pool(candidates)
        by = list(self.sort_by)
        df = candidates.sort_values(by, ascending=False, kind="mergesort")
        kept = df.groupby(list(self.cell_columns), sort=False, group_keys=False).head(self.cap)
        kept = kept.sort_values(by, ascending=False, kind="mergesort").reset_index(drop=True)
        kept.insert(0, "cohort_pair_id", range(len(kept)))
        cols = ["cohort_pair_id"] + [c for c in (*self.run_columns, *self.seg_columns) if c in kept.columns]
        return kept[cols].copy()

    def coverage(self, candidates: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        """Per ``(axis, block, Group_x, predicate)`` availability vs kept counts."""
        keys = [k for k in ["axis", "block", "Group_x", self.cell_for_coverage] if k in cohort.columns]
        avail = candidates.groupby(keys).size().rename("n_available")
        kept = cohort.groupby(keys).size().rename("n_kept")
        cov = pd.concat([avail, kept], axis=1).fillna(0).astype(int).reset_index()
        cov["capped"] = cov["n_available"] > self.cap
        return cov

    def build(
        self,
        candidates: pd.DataFrame,
        out_dir: Path | None = None,
        source: str | None = None,
        split_gender: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build (and optionally write) a frozen cohort + its coverage report.

        Args:
            candidates: the candidate pool.
            out_dir: if given, writes ``cohort.csv`` + ``cohort_coverage.csv`` there.
            source: if given, restricts to ``candidates['source'] == source`` first
                (used to build the per-source BBQ-only / CrowS-only cohorts).
            split_gender: apply the (idempotent) gender sub-axis split before freezing.

        Returns:
            ``(cohort, coverage)`` DataFrames.
        """
        pool = candidates
        if source is not None:
            pool = pool[pool["source"] == source].copy()
        if split_gender and "axis" in pool.columns:
            pool = apply_gender_subaxis(pool)
        cohort = self.freeze(pool)
        cov = self.coverage(pool, cohort)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cohort.to_csv(out_dir / "cohort.csv", index=False)
            cov.to_csv(out_dir / "cohort_coverage.csv", index=False)
        return cohort, cov
