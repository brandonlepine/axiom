"""The InterventionRunner contract shared by every mechanistic-analysis step.

Each step (residual patching, head patching, head ablation, greedy knockout, steering,
MLP attribution) is a class that declares its required inputs and expected outputs, can
load its inputs from disk (steps never assume a previous step ran in the same process),
checkpoints long runs atomically, and emits a provenance sidecar for every artifact
(CLAUDE.md, "Pipelines should be restartable and composable").

This module defines only the abstract contract. Concrete runners live in sibling
modules and are ported from the validated upstream scripts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.config import RunConfig
from axiom.data.adapters import DatasetAdapter
from axiom.models.loader import LoadedModel
from axiom.paths import OutputLayout
from axiom.provenance import ArtifactMetadata, InputArtifact, atomic_write_bytes


@dataclass
class StepResult:
    """What a runner returns: the artifacts it wrote and a short summary for logs."""

    artifacts: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class InterventionRunner(ABC):
    """Abstract base for a single mechanistic-interpretability pipeline step.

    Args:
        config: validated :class:`RunConfig` for this step.
        adapter: the dataset adapter providing the frozen cohort.
        layout: resolved output layout (model/dataset/step/run_id).
        loaded: the loaded model bundle (None for steps that don't need a model).

    Subclasses implement :meth:`run`. They should write artifacts via
    :meth:`write_dataframe` so naming, atomicity, and provenance are handled uniformly.
    """

    #: Canonical step name, used in output paths (must be one of axiom.paths.STEPS).
    step_name: str = ""

    def __init__(
        self,
        config: RunConfig,
        adapter: DatasetAdapter,
        layout: OutputLayout,
        loaded: LoadedModel | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.layout = layout
        self.loaded = loaded

    @property
    @abstractmethod
    def required_inputs(self) -> list[Path]:
        """Input artifacts this step reads; checked to exist before the step runs."""
        raise NotImplementedError

    @abstractmethod
    def run(self) -> StepResult:
        """Execute the step, writing artifacts and their sidecars; return a summary."""
        raise NotImplementedError

    def check_inputs(self) -> None:
        """Raise ``FileNotFoundError`` if any required input artifact is missing."""
        missing = [p for p in self.required_inputs if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"[{self.step_name}] missing required inputs: {missing}")

    def write_dataframe(
        self, df: pd.DataFrame, filename: str, produced_by: str, extra: dict[str, Any] | None = None
    ) -> Path:
        """Atomically write a CSV artifact to this run's dir and emit its sidecar.

        Args:
            df: the table to write.
            filename: artifact name; should encode dataset/model/step/raw|summary per
                CLAUDE.md's naming rules (the caller composes it).
            produced_by: module or class name that produced the artifact.
            extra: optional extra metadata (e.g. layer list, cap, n_pairs).

        Returns:
            The path written.
        """
        out_path = self.layout.ensure() / filename
        atomic_write_bytes(out_path, df.to_csv(index=False).encode("utf-8"))
        ArtifactMetadata(
            artifact=filename,
            produced_by=produced_by,
            run_id=self.layout.run_id,
            dataset=self.config.dataset,
            model=self.config.model.name,
            input_artifacts=[InputArtifact.of(p) for p in self.required_inputs if Path(p).exists()],
            config=self.config.model_dump(mode="json"),
            extra=extra or {},
        ).write(out_path)
        return out_path
