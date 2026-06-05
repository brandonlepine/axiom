"""Explicit cohort schemas + validation (CLAUDE.md, "All dataset schemas are explicit").

A *cohort* is a frozen, balanced table of minimal identity-contrast pairs that the
intervention run-scripts consume in file order. Two cohort shapes exist; both carry the
same scoring core (the columns ``align_pair`` + continuation scoring read), and differ
only in their identity/segmentation columns:

  * WinoQueer cohort: identity columns ``Gender_ID_x/Gender_ID_y`` + ``identity`` /
    ``is_umbrella``.
  * Combined / per-source BBQ+CrowS cohort: identity columns ``Group_x/Group_y`` +
    ``category`` / ``block`` / ``frame`` / ``source`` / ``identity_mapped``.

Validation enforces the invariants downstream joins depend on: required columns present,
stable keys unique and non-null (CLAUDE.md, "Stable IDs are mandatory"). Row-level
tabular data is validated with these utilities rather than a per-row pydantic model
(too slow for 5k-row cohorts); pydantic is reserved for configs and artifact metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns every cohort must carry: the stable keys + the exact fields the scoring /
# alignment code reads (sent_x/sent_y to align, prefix_y for the reference run, the
# continuation to score, and the precomputed bias_score that ordered the cohort).
SCORING_CORE_COLUMNS: tuple[str, ...] = (
    "cohort_pair_id",
    "row_id",
    "sent_x",
    "sent_y",
    "prefix_x",
    "prefix_y",
    "continuation",
    "bias_score",
)

WINOQUEER_COHORT_COLUMNS: tuple[str, ...] = SCORING_CORE_COLUMNS + (
    "Gender_ID_x",
    "Gender_ID_y",
    "predicate",
    "predicate_label_provisional",
    "axis",
    "identity",
    "is_umbrella",
)

COMBINED_COHORT_COLUMNS: tuple[str, ...] = SCORING_CORE_COLUMNS + (
    "Group_x",
    "Group_y",
    "predicate_label_provisional",
    "category",
    "axis",
    "block",
    "frame",
    "source",
    "identity_mapped",
)


@dataclass(frozen=True)
class CohortSchema:
    """A named cohort schema: its required columns and stable-key columns.

    ``stable_keys`` must be unique and non-null across the cohort -- they are the join
    keys every segmented/cross-dataset analysis relies on.
    """

    name: str
    required_columns: tuple[str, ...]
    stable_keys: tuple[str, ...] = ("cohort_pair_id", "row_id")
    identity_x_col: str = field(default="")
    identity_y_col: str = field(default="")


WINOQUEER_SCHEMA = CohortSchema(
    name="winoqueer",
    required_columns=WINOQUEER_COHORT_COLUMNS,
    identity_x_col="Gender_ID_x",
    identity_y_col="Gender_ID_y",
)

COMBINED_SCHEMA = CohortSchema(
    name="combined_bbq_crows",
    required_columns=COMBINED_COHORT_COLUMNS,
    identity_x_col="Group_x",
    identity_y_col="Group_y",
)


def validate_cohort(df: pd.DataFrame, schema: CohortSchema) -> None:
    """Raise ``ValueError`` if ``df`` violates ``schema``.

    Checks, in order: required columns present; stable keys non-null; stable keys unique.
    A schema mismatch is a catastrophic failure (CLAUDE.md) -- it raises rather than
    emitting a diagnostic, because every downstream join would otherwise be corrupt.
    """
    missing = [c for c in schema.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"[{schema.name}] cohort missing required columns: {missing}")

    for key in schema.stable_keys:
        if df[key].isna().any():
            raise ValueError(f"[{schema.name}] stable key {key!r} has null values.")
        if df[key].duplicated().any():
            n = int(df[key].duplicated().sum())
            raise ValueError(f"[{schema.name}] stable key {key!r} has {n} duplicate values.")
