"""Central registry of the frozen pod-input datasets.

One place that maps a dataset slug to its cohort path, provenance manifest, and schema,
so every runner (scoring, patching, ...) resolves datasets identically instead of
re-hardcoding paths (CLAUDE.md, "Interfaces for extensibility"; avoid duplication).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from axiom.data.adapters import CohortFileAdapter
from axiom.data.schemas import COMBINED_SCHEMA, WINOQUEER_SCHEMA, CohortSchema
from axiom.paths import REPO_ROOT, DataLayout, latest_run_dir


@dataclass(frozen=True)
class DatasetSpec:
    """How a dataset enters the pipeline: cohort + candidate-pool location, schema, and
    the rules for building a model-specific high-bias analysis cohort from the pool.

    ``pool_relpath`` is the full candidate universe the selector scores + filters;
    ``pool_source_filter`` restricts a shared pool to one source (the per-source cohorts
    share the combined pool); ``cell_columns`` is the balancing cell for re-balancing;
    ``winoqueer_annotate`` marks pools that need the identity-taxonomy annotation.
    """

    dataset_id: str
    cohort_relpath: str
    manifest_dataset: str
    schema: CohortSchema
    pool_relpath: str
    cell_columns: tuple[str, ...]
    pool_source_filter: str | None = None
    winoqueer_annotate: bool = False


_WQ_POOL = "data/processed/winoqueer/results/patching_candidates/winoqueer_patching_candidates_all.csv"
_COMBINED_POOL = "data/processed/combined/bbq_crows_candidates.csv"

DATASETS: dict[str, DatasetSpec] = {
    "winoqueer": DatasetSpec(
        "winoqueer", "data/cohorts/winoqueer/cohort.csv", "winoqueer", WINOQUEER_SCHEMA,
        pool_relpath=_WQ_POOL, cell_columns=("identity", "predicate_label_provisional"),
        winoqueer_annotate=True),
    "combined_bbq_crows": DatasetSpec(
        "combined_bbq_crows", "data/cohorts/combined_bbq_crows/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA,
        pool_relpath=_COMBINED_POOL, cell_columns=("block", "predicate_label_provisional")),
    "bbq": DatasetSpec(
        "bbq", "data/cohorts/residual_bbq_crows/bbq/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA,
        pool_relpath=_COMBINED_POOL, cell_columns=("block", "predicate_label_provisional"),
        pool_source_filter="bbq"),
    "crows": DatasetSpec(
        "crows", "data/cohorts/residual_bbq_crows/crows/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA,
        pool_relpath=_COMBINED_POOL, cell_columns=("block", "predicate_label_provisional"),
        pool_source_filter="crows-pairs"),
}

# All four frozen pod cohorts, in run order.
ALL_POD_DATASETS: tuple[str, ...] = ("winoqueer", "combined_bbq_crows", "bbq", "crows")


def cohort_path(dataset: str, root: Path = REPO_ROOT) -> Path:
    """Resolve a dataset's cohort CSV (registry first, else data/cohorts/<dataset>/cohort.csv)."""
    if dataset in DATASETS:
        return root / DATASETS[dataset].cohort_relpath
    return DataLayout(root).cohorts(dataset) / "cohort.csv"


def manifest_path(dataset: str, root: Path = REPO_ROOT) -> Path:
    """Resolve a dataset's provenance manifest (per-source cohorts share the combined one)."""
    manifest_ds = DATASETS[dataset].manifest_dataset if dataset in DATASETS else dataset
    return DataLayout(root).manifest(manifest_ds)


def pool_path(dataset: str, root: Path = REPO_ROOT) -> Path:
    """Resolve a dataset's candidate-pool CSV (the universe the selector scores+filters)."""
    return root / DATASETS[dataset].pool_relpath


def load_pool(dataset: str, root: Path = REPO_ROOT) -> "pd.DataFrame":
    """Load a dataset's candidate pool, applying its source filter + WinoQueer annotation.

    Per-source datasets (bbq/crows) share the combined pool and are filtered by ``source``;
    WinoQueer pools are annotated with axis/identity/is_umbrella so the frozen analysis
    cohort satisfies the schema.
    """
    spec = DATASETS[dataset]
    df = pd.read_csv(pool_path(dataset, root))
    if spec.pool_source_filter is not None:
        df = df[df["source"] == spec.pool_source_filter].copy()
    if spec.winoqueer_annotate:
        from axiom.data.winoqueer_taxonomy import annotate
        df = annotate(df, strict=True)
    return df.reset_index(drop=True)


def make_adapter(dataset: str, root: Path = REPO_ROOT, cohort_override: Path | None = None) -> CohortFileAdapter:
    """Construct a :class:`CohortFileAdapter` for a dataset slug."""
    if dataset not in DATASETS and cohort_override is None:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)} (or pass a cohort override)")
    schema = DATASETS[dataset].schema if dataset in DATASETS else COMBINED_SCHEMA
    path = cohort_override or cohort_path(dataset, root)
    return CohortFileAdapter(dataset, path, schema)


def latest_analysis_cohort(model_name: str, dataset: str, root: Path = REPO_ROOT) -> Path | None:
    """Path to the most recent SELECTED analysis cohort for (model, dataset), or ``None``.

    Globs the latest ``outputs/<model>/<dataset>/selection/<run_id>/`` for the frozen
    ``*_analysis_cohort.csv`` written by ``run_select_cohort.py``.
    """
    run_dir = latest_run_dir(model_name, dataset, "selection", root)
    if run_dir is None:
        return None
    matches = sorted(run_dir.glob("*_analysis_cohort.csv"))
    return matches[-1] if matches else None


def resolve_analysis_cohort(
    dataset: str,
    model_name: str,
    root: Path = REPO_ROOT,
    *,
    cohort_override: Path | None = None,
    use_raw: bool = False,
) -> tuple[Path, str]:
    """Resolve which cohort a mechanistic step should consume, with explicit precedence.

    1. ``cohort_override`` (an explicit ``--cohort`` path) wins.
    2. ``use_raw=True`` deliberately selects the raw frozen pod cohort (the candidate
       universe, NOT bias-selected for this model) -- escape hatch only.
    3. Otherwise the latest model-specific SELECTED analysis cohort
       (``run_select_cohort.py`` output). This is the methodological default (ADR 0003):
       mechanistic analyses run on instances where THIS model exhibits the bias.

    Returns ``(path, source_label)``. Raises ``SystemExit`` with guidance if no selected
    cohort exists and neither override nor ``use_raw`` was given.
    """
    if cohort_override is not None:
        return cohort_override, "override"
    if use_raw:
        return cohort_path(dataset, root), "raw_frozen_cohort"
    latest = latest_analysis_cohort(model_name, dataset, root)
    if latest is None:
        raise SystemExit(
            f"No selected analysis cohort for model={model_name!r} dataset={dataset!r}.\n"
            f"Run:  python scripts/run_select_cohort.py --config <cfg> --datasets {dataset}\n"
            f"or pass --cohort <path>, or --raw-cohort to use the (non-selected) frozen cohort."
        )
    return latest, "selected_analysis_cohort"
