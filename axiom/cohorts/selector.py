"""BiasCohortSelector: freeze a model-specific high-bias analysis cohort.

The mechanistic analyses (patching, ablation, steering, ...) should run only on instances
where the target model actually exhibits the bias -- otherwise the circuit estimate is
diluted by no-bias pairs. This selector turns a WinoQueer-scored candidate pool into a
balanced, model-specific cohort:

  1. filter to high-bias pairs: the model prefers the stereotype variant
     (``wq_stereo == 1``) by at least ``tau`` (``wq_score_diff >= tau``);
  2. re-balance: cap each ``(identity/block, predicate)`` cell so no group dominates,
     keeping the highest-``wq_score_diff`` pairs (reuses :class:`SegmentedCohortBuilder`);
  3. freeze a stable ``cohort_pair_id`` ordered by descending bias.

The result is what downstream steps consume, and it is regenerated per model (each model
has its own bias profile).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from axiom.cohorts.builder import SegmentedCohortBuilder

STEREO_COL = "wq_stereo"
DIFF_COL = "wq_score_diff"


@dataclass
class BiasCohortSelector:
    """Select + balance a high-bias analysis cohort from a scored pool.

    Args:
        tau: magnitude floor on ``wq_score_diff`` (the model must prefer the stereotype by
            at least this much). ``wq_score_diff`` is a sum of log-probs over the shared
            span; ``tau`` is in those (nats) units.
        cap: max pairs kept per balancing cell.
        cell_columns: the balancing cell (e.g. ``(identity, predicate_label_provisional)``
            for WinoQueer, ``(block, predicate_label_provisional)`` for combined).
    """

    tau: float
    cap: int
    cell_columns: Sequence[str]
    sort_by: Sequence[str] = (DIFF_COL, "row_id")

    def filter_high_bias(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Keep pairs the model prefers the stereotype for, by margin >= tau."""
        for col in (STEREO_COL, DIFF_COL):
            if col not in scored.columns:
                raise ValueError(f"scored pool missing {col!r}; run the BiasScorer first.")
        keep = (scored[STEREO_COL] == 1) & (scored[DIFF_COL] >= self.tau)
        return scored[keep].copy()

    def select(self, scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(analysis_cohort, selection_report)``.

        ``analysis_cohort`` is filtered, balanced, and frozen with a fresh
        ``cohort_pair_id``. ``selection_report`` records, overall and per primary group,
        how many pairs were scoreable / passed the filter / were kept.
        """
        filtered = self.filter_high_bias(scored)
        builder = SegmentedCohortBuilder(
            cap=self.cap, cell_columns=self.cell_columns, sort_by=self.sort_by, keep_all_columns=True
        )
        cohort = builder.freeze(filtered) if len(filtered) else filtered.assign(cohort_pair_id=[])
        report = self._report(scored, filtered, cohort)
        return cohort, report

    def _report(self, scored: pd.DataFrame, filtered: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        scoreable = scored[scored.get("scoreable", True).astype(bool)] if "scoreable" in scored else scored
        group_col = self.cell_columns[0]
        rows = [self._report_row("ALL", "", scoreable, filtered, cohort)]
        if group_col in scored.columns:
            for key in sorted(scoreable[group_col].dropna().unique()):
                rows.append(self._report_row(
                    group_col, str(key),
                    scoreable[scoreable[group_col] == key],
                    filtered[filtered[group_col] == key],
                    cohort[cohort[group_col] == key] if group_col in cohort.columns else cohort.iloc[0:0],
                ))
        return pd.DataFrame(rows)

    def _report_row(self, group, key, scoreable, filtered, cohort) -> dict:
        n_scoreable = int(len(scoreable))
        n_passed = int(len(filtered))
        n_kept = int(len(cohort))
        return {
            "group": group, "key": key, "tau": self.tau, "cap": self.cap,
            "n_scoreable": n_scoreable, "n_passed_filter": n_passed, "n_kept": n_kept,
            "pass_rate": round(n_passed / n_scoreable, 4) if n_scoreable else float("nan"),
            "mean_kept_score_diff": float(cohort[DIFF_COL].mean()) if n_kept and DIFF_COL in cohort else float("nan"),
        }
