"""Canonical on-disk layout for inputs and outputs.

Outputs are organized first by model, then dataset, then analysis step, then run:

    outputs/<model_slug>/<dataset_slug>/<step>/<run_id>/<artifact>

This makes the tree human-navigable ("show me everything WinoQueer on Llama-3.1-8B")
while ``run_id`` (carried in every sidecar) still uniquely identifies a run. Local
GPT-2 smoke tests land under ``outputs/gpt2/...`` and never collide with pod results.

Inputs follow CLAUDE.md's data layout:

    data/raw/<dataset>/        upstream sources (gitignored; large)
    data/processed/<dataset>/  parsed / scored intermediates
    data/cohorts/<dataset>/    frozen, balanced cohorts -- the pod inputs (tracked)
    data/manifests/            per-dataset MANIFEST.json (tracked)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def slugify(name: str) -> str:
    """Filesystem-safe, stable slug for model/dataset identifiers.

    ``meta-llama/Llama-3.1-8B`` -> ``llama-3.1-8b``; ``gpt2`` -> ``gpt2``. Keeps dots
    (version numbers) but lowercases and collapses other separators to single dashes.
    """
    base = name.split("/")[-1].lower()
    base = re.sub(r"[^a-z0-9.]+", "-", base)
    return re.sub(r"-+", "-", base).strip("-")


@dataclass(frozen=True)
class DataLayout:
    """Resolver for input data directories under a repo root."""

    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    def raw(self, dataset: str) -> Path:
        return self.data / "raw" / dataset

    def processed(self, dataset: str) -> Path:
        return self.data / "processed" / dataset

    def cohorts(self, dataset: str) -> Path:
        return self.data / "cohorts" / dataset

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    def manifest(self, dataset: str) -> Path:
        return self.manifests / f"{dataset}.MANIFEST.json"


@dataclass(frozen=True)
class OutputLayout:
    """Resolver for ``outputs/<model>/<dataset>/<step>/<run_id>/``.

    Args:
        model: model identifier (slugified internally), e.g. ``meta-llama/Llama-3.1-8B``.
        dataset: dataset identifier (slugified internally), e.g. ``winoqueer``.
        step: pipeline step name, e.g. ``residual_patching`` (see :data:`STEPS`).
        run_id: the run id from :func:`axiom.provenance.new_run_id`.
        root: repo root (override in tests).
    """

    model: str
    dataset: str
    step: str
    run_id: str
    root: Path = REPO_ROOT

    @property
    def dir(self) -> Path:
        """The run's output directory (created on demand by :meth:`ensure`)."""
        return self.root / "outputs" / slugify(self.model) / slugify(self.dataset) / self.step / self.run_id

    def ensure(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def artifact(self, filename: str) -> Path:
        """Path to a named artifact within this run's directory."""
        return self.dir / filename


# Canonical step names used in output paths and runner registration.
STEPS = (
    "scoring",
    "selection",
    "residual_patching",
    "head_patching",
    "head_ablation",
    "greedy_knockout",
    "steering",
    "mlp_attribution",
    "segmented",
    "cross_dataset",
)


def outputs_root(root: Path = REPO_ROOT) -> Path:
    return root / "outputs"


def step_dir(model: str, dataset: str, step: str, root: Path = REPO_ROOT) -> Path:
    """``outputs/<model>/<dataset>/<step>`` (the parent of all runs of that step)."""
    return outputs_root(root) / slugify(model) / slugify(dataset) / step


def latest_run_dir(model: str, dataset: str, step: str, root: Path = REPO_ROOT) -> Path | None:
    """The most recent run directory for a (model, dataset, step), or ``None``.

    Run ids are timestamp-prefixed (``YYYYMMDDTHHMMSSZ-...``) so the lexicographic max is
    the most recent run.
    """
    base = step_dir(model, dataset, step, root)
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    return runs[-1] if runs else None
