"""Central registry of the frozen pod-input datasets.

One place that maps a dataset slug to its cohort path, provenance manifest, and schema,
so every runner (scoring, patching, ...) resolves datasets identically instead of
re-hardcoding paths (CLAUDE.md, "Interfaces for extensibility"; avoid duplication).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from axiom.data.adapters import CohortFileAdapter
from axiom.data.schemas import COMBINED_SCHEMA, WINOQUEER_SCHEMA, CohortSchema
from axiom.paths import REPO_ROOT, DataLayout


@dataclass(frozen=True)
class DatasetSpec:
    """How a dataset enters the pipeline: cohort location, manifest, and schema."""

    dataset_id: str
    cohort_relpath: str
    manifest_dataset: str
    schema: CohortSchema


DATASETS: dict[str, DatasetSpec] = {
    "winoqueer": DatasetSpec(
        "winoqueer", "data/cohorts/winoqueer/cohort.csv", "winoqueer", WINOQUEER_SCHEMA),
    "combined_bbq_crows": DatasetSpec(
        "combined_bbq_crows", "data/cohorts/combined_bbq_crows/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA),
    "bbq": DatasetSpec(
        "bbq", "data/cohorts/residual_bbq_crows/bbq/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA),
    "crows": DatasetSpec(
        "crows", "data/cohorts/residual_bbq_crows/crows/cohort.csv", "combined_bbq_crows", COMBINED_SCHEMA),
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


def make_adapter(dataset: str, root: Path = REPO_ROOT, cohort_override: Path | None = None) -> CohortFileAdapter:
    """Construct a :class:`CohortFileAdapter` for a dataset slug."""
    if dataset not in DATASETS and cohort_override is None:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)} (or pass a cohort override)")
    schema = DATASETS[dataset].schema if dataset in DATASETS else COMBINED_SCHEMA
    path = cohort_override or cohort_path(dataset, root)
    return CohortFileAdapter(dataset, path, schema)
