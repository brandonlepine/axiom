#!/usr/bin/env python3
"""Score continuation/bias for a dataset cohort (thin orchestration over BiasScorer).

Loads a validated config, picks a model (config's model, or a registry override such as
``--model gpt2`` for a local smoke test), constructs the centralized loader + BiasScorer,
scores the cohort, and writes a raw per-pair CSV + a summary, each with a provenance
sidecar under ``outputs/<model>/<dataset>/scoring/<run_id>/``.

Examples:
    # Local GPT-2 smoke test on 5 WinoQueer pairs (MPS/CPU):
    python scripts/run_scoring.py --config configs/scoring/winoqueer_llama31_8b.yaml \\
        --model gpt2 --max-pairs 5

    # Full pod run on Llama-3.1-8B (uses the model in the config):
    python scripts/run_scoring.py --config configs/scoring/winoqueer_llama31_8b.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from axiom.config import load_model_registry, load_run_config
from axiom.models import ModelLoader
from axiom.paths import DataLayout, OutputLayout, slugify
from axiom.provenance import ArtifactMetadata, InputArtifact, atomic_write_bytes, new_run_id
from axiom.scoring import BiasScorer

REPO = Path(__file__).resolve().parent.parent

# Known dataset slugs -> (cohort path relative to repo, manifest dataset). The two
# per-source cohorts share the combined manifest. Any other dataset falls back to
# data/cohorts/<dataset>/cohort.csv.
DATASET_COHORTS: dict[str, tuple[str, str]] = {
    "winoqueer": ("data/cohorts/winoqueer/cohort.csv", "winoqueer"),
    "combined_bbq_crows": ("data/cohorts/combined_bbq_crows/cohort.csv", "combined_bbq_crows"),
    "bbq": ("data/cohorts/residual_bbq_crows/bbq/cohort.csv", "combined_bbq_crows"),
    "crows": ("data/cohorts/residual_bbq_crows/crows/cohort.csv", "combined_bbq_crows"),
}


def _resolve_cohort(dataset: str, override: Path | None) -> Path:
    if override is not None:
        return override
    if dataset in DATASET_COHORTS:
        return REPO / DATASET_COHORTS[dataset][0]
    return DataLayout(REPO).cohorts(dataset) / "cohort.csv"


def _resolve_manifest(dataset: str) -> Path:
    manifest_ds = DATASET_COHORTS.get(dataset, (None, dataset))[1]
    return DataLayout(REPO).manifest(manifest_ds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score continuation/bias for a dataset cohort.")
    ap.add_argument("--config", type=Path, required=True, help="RunConfig YAML/JSON.")
    ap.add_argument("--models-config", type=Path, default=REPO / "configs" / "models.yaml")
    ap.add_argument("--model", type=str, default=None, help="Registry key override (e.g. gpt2).")
    ap.add_argument("--cohort", type=Path, default=None, help="Cohort CSV override (default: by dataset).")
    ap.add_argument("--max-pairs", type=int, default=None, help="Smoke-test cap (overrides config).")
    args = ap.parse_args()

    from axiom.config import ScoringConfig

    cfg = load_run_config(args.config)
    if args.model is not None:
        registry = load_model_registry(args.models_config)
        if args.model not in registry:
            raise SystemExit(f"--model {args.model!r} not in registry {sorted(registry)}")
        cfg = cfg.model_copy(update={"model": registry[args.model]})
    scoring = cfg.scoring or ScoringConfig()
    if args.max_pairs is not None:
        scoring = scoring.model_copy(update={"max_pairs": args.max_pairs})
    cfg = cfg.model_copy(update={"scoring": scoring})

    cohort_path = _resolve_cohort(cfg.dataset, args.cohort)
    if not cohort_path.exists():
        raise SystemExit(f"cohort not found: {cohort_path}")
    df = pd.read_csv(cohort_path)

    run_id = cfg.run_id or new_run_id()
    layout = OutputLayout(model=cfg.model.name, dataset=cfg.dataset, step="scoring", run_id=run_id, root=REPO)
    layout.ensure()
    print(f"[run_scoring] dataset={cfg.dataset} model={cfg.model.name} run_id={run_id}")
    print(f"[run_scoring] cohort={cohort_path} ({len(df)} rows) -> {layout.dir}")

    loaded = ModelLoader(cfg.model).load()
    scorer = BiasScorer(loaded, cfg.scoring)
    scored = scorer.score(df)
    summary = scorer.summarize(scored)

    model_slug = slugify(cfg.model.name)
    manifest = _resolve_manifest(cfg.dataset)
    config_payload = cfg.model_dump(mode="json")
    config_payload["scoring_metric"] = BiasScorer.metric
    config_payload["model_runtime"] = loaded.provenance()

    for frame, kind in [(scored, "raw"), (summary, "summary")]:
        name = f"{cfg.dataset}_{model_slug}_scoring_{kind}.csv"
        path = layout.artifact(name)
        atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        ArtifactMetadata(
            artifact=name,
            produced_by=BiasScorer.produced_by,
            run_id=run_id,
            dataset=cfg.dataset,
            model=cfg.model.name,
            dataset_manifest=str(manifest.relative_to(REPO)) if manifest.exists() else None,
            input_artifacts=[InputArtifact.of(cohort_path)],
            config=config_payload,
        ).write(path)
        print(f"  wrote {path.name}")

    overall = summary[summary["group"] == "ALL"].iloc[0]
    print(f"\n[summary] WinoQueer score (ALL): {overall['winoqueer_score']}% "
          f"| n={int(overall['n'])} | neutral={overall['pct_neutral']}% "
          f"| mean score_diff={overall['mean_score_diff']:.3f}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
