"""Pydantic configuration models -- the single source of run parameters.

Config is data, not code (CLAUDE.md): model names, dataset paths, caps, layer lists,
alpha grids, top-k values and filtering rules live in YAML/JSON and are validated here
before any run touches a model. Loaders raise on unknown fields so a typo in a config
fails loudly instead of silently using a default.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Device = Literal["auto", "cuda", "mps", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]


class _Strict(BaseModel):
    """Base: forbid unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    """How to load a model through the centralized loader.

    ``hf_path`` is what HuggingFace ``from_pretrained`` reads (a local dir or a hub id);
    ``tl_name`` is the TransformerLens architecture name used to wrap it. For GPT-2 the
    two are typically the same hub id (``gpt2``); for a locally-downloaded Llama the
    ``hf_path`` is a local directory while ``tl_name`` stays the hub id TL recognizes.
    """

    name: str = Field(..., description="Human/identifier name, also used to slug output paths.")
    hf_path: str = Field(..., description="HF from_pretrained source: local dir or hub id.")
    tl_name: str = Field(..., description="TransformerLens model name used to wrap the HF model.")
    family: Literal["gpt2", "llama", "other"] = "other"
    device: Device = "auto"
    dtype: DType = "float16"
    prepend_bos: bool = True
    padding_side: Literal["left", "right"] = "right"


class ScoringConfig(_Strict):
    """Continuation/bias scoring step parameters."""

    batch_size: int = Field(16, ge=1)
    max_pairs: int | None = Field(None, ge=1, description="Smoke-test cap on scored rows.")
    add_special_tokens: bool = True


class CohortConfig(_Strict):
    """SegmentedCohortBuilder parameters."""

    cap: int = Field(..., ge=1, description="Max pairs per (block, predicate) cell.")
    cell_columns: list[str] = Field(default_factory=lambda: ["block", "predicate_label_provisional"])
    sort_by: list[str] = Field(default_factory=lambda: ["bias_score", "row_id"])


class SelectionConfig(_Strict):
    """High-bias analysis-cohort selection parameters.

    ``tau`` is the magnitude floor on ``wq_score_diff`` (sum of log-probs over the shared
    span); a pair is kept iff the model prefers the stereotype (``wq_stereo == 1``) by at
    least ``tau``. ``cap`` re-balances by capping each cell after filtering.
    """

    tau: float = Field(0.5, description="Min wq_score_diff to keep (model prefers stereotype by >= tau).")
    cap: int = Field(100, ge=1, description="Max pairs per (cell) after filtering, highest wq_score_diff kept.")
    max_pool_pairs: int | None = Field(None, ge=1, description="Smoke-test cap on scored pool rows.")


class PatchingConfig(_Strict):
    """Activation-patching step parameters (residual / head patching)."""

    patch_batch_size: int = Field(32, ge=1, description="Target positions patched per forward batch.")
    max_pairs: int | None = Field(None, ge=1, description="Smoke-test cap on cohort pairs.")
    layers: list[int] | None = Field(None, description="Restrict to these layers (default: all).")
    resume: bool = Field(True, description="Resume from an existing raw CSV, redoing the last pair.")


class RunConfig(_Strict):
    """Top-level config for a single pipeline step invocation.

    ``run_id`` is optional in the file; runners mint one if absent and stamp it into
    every artifact and sidecar.
    """

    step: str
    model: ModelConfig
    dataset: str = Field(..., description="Dataset slug, e.g. winoqueer / combined_bbq_crows.")
    seed: int = 0
    run_id: str | None = None
    scoring: ScoringConfig | None = None
    cohort: CohortConfig | None = None
    selection: SelectionConfig | None = None
    patching: PatchingConfig | None = None

    @field_validator("dataset")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset must be a non-empty slug")
        return v


def _read_structured(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format {path.suffix!r}; use .yaml/.yml/.json")


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a :class:`RunConfig` from YAML/JSON, raising on any error."""
    return RunConfig.model_validate(_read_structured(path))


def load_model_registry(path: str | Path) -> dict[str, ModelConfig]:
    """Load ``configs/models.yaml`` (a mapping ``key -> ModelConfig``)."""
    raw = _read_structured(path)
    if not isinstance(raw, dict):
        raise ValueError("model registry must be a mapping of name -> model spec")
    return {k: ModelConfig.model_validate(v) for k, v in raw.items()}
