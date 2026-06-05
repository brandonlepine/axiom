"""axiom: mechanistic-interpretability pipelines for social-bias analysis.

This package localizes and intervenes on the circuits an LLM uses to express social
bias, across matched Wino-style datasets (WinoQueer, BBQ-derived WinoGender/WinoRace,
CrowS-derived pairs, and an identity-only probe set).

Design principles (see CLAUDE.md):
  * Object-oriented pipeline components carry their own config, model/tokenizer
    handles, and paths; runner scripts under ``scripts/`` are thin orchestration.
  * Metrics, model loading, token alignment, and provenance are centralized here and
    reused everywhere -- never reimplemented per script.
  * Every artifact is reproducible: it carries a ``run_id``, a provenance sidecar, and
    a deterministic config.

Submodules:
  axiom.config         Pydantic config models (the only source of run parameters).
  axiom.device         Device / dtype resolution (cuda / mps / cpu).
  axiom.models         Centralized HuggingFace + TransformerLens model loading.
  axiom.metrics        Pure metric functions (continuation logp, bias, overlap).
  axiom.alignment      Token-alignment + continuation-span detection (pure).
  axiom.data           Dataset schemas + adapter interface.
  axiom.cohorts        SegmentedCohortBuilder: freeze balanced, stable cohorts.
  axiom.interventions  InterventionRunner ABC for patching/ablation/steering steps.
  axiom.paths          Output-layout resolver (outputs/<model>/<dataset>/<step>/<run_id>).
  axiom.provenance     run_id, git sha, atomic writes, checksums, metadata sidecars.
  axiom.figures        Publication / colorblind-safe figure styling.
"""

__version__ = "0.1.0"
