#!/usr/bin/env python3
"""Score Wino-style bias for one or more cohorts (thin orchestration over BiasScorer).

Loads a validated config, picks a model (config's model, or a registry override such as
``--model gpt2`` for a local smoke test), and scores each requested dataset with the
canonical WinoQueer autoregressive metric, writing a raw per-pair CSV + a win-rate
summary with provenance sidecars under ``outputs/<model>/<dataset>/scoring/<run_id>/``.

When several datasets are requested (``--datasets`` / ``--all``) the model is loaded
ONCE and reused, and all datasets share a single ``run_id`` so a scoring sweep is
grouped (each lands under its own dataset subtree at the same run_id leaf).

Examples:
    # Local GPT-2 smoke test on 5 WinoQueer pairs:
    python scripts/run_scoring.py --config configs/scoring/winoqueer_llama31_8b.yaml \\
        --model gpt2 --max-pairs 5

    # Full pod run on Llama-3.1-8B, all four pod cohorts, one model load:
    python scripts/run_scoring.py --config configs/scoring/winoqueer_llama31_8b.yaml --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from axiom.config import RunConfig, ScoringConfig, load_model_registry, load_run_config
from axiom.models import ModelLoader
from axiom.models.loader import LoadedModel
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
ALL_POD_DATASETS = ("winoqueer", "combined_bbq_crows", "bbq", "crows")


def _resolve_cohort(dataset: str, override: Path | None) -> Path:
    if override is not None:
        return override
    if dataset in DATASET_COHORTS:
        return REPO / DATASET_COHORTS[dataset][0]
    return DataLayout(REPO).cohorts(dataset) / "cohort.csv"


def _resolve_manifest(dataset: str) -> Path:
    manifest_ds = DATASET_COHORTS.get(dataset, (None, dataset))[1]
    return DataLayout(REPO).manifest(manifest_ds)


def score_dataset(
    cfg: RunConfig, dataset: str, loaded: LoadedModel, run_id: str, cohort_override: Path | None
) -> None:
    """Score one dataset and write its raw + summary artifacts with provenance."""
    cohort_path = _resolve_cohort(dataset, cohort_override)
    if not cohort_path.exists():
        raise SystemExit(f"cohort not found: {cohort_path}")
    df = pd.read_csv(cohort_path)

    layout = OutputLayout(model=cfg.model.name, dataset=dataset, step="scoring", run_id=run_id, root=REPO)
    layout.ensure()
    print(f"\n[run_scoring] dataset={dataset} ({len(df)} rows) -> {layout.dir}")

    scorer = BiasScorer(loaded, cfg.scoring)
    scored = scorer.score(df)
    summary = scorer.summarize(scored)

    model_slug = slugify(cfg.model.name)
    manifest = _resolve_manifest(dataset)
    config_payload = cfg.model_dump(mode="json")
    config_payload["dataset"] = dataset
    config_payload["scoring_metric"] = BiasScorer.metric
    config_payload["model_runtime"] = loaded.provenance()

    for frame, kind in [(scored, "raw"), (summary, "summary")]:
        name = f"{dataset}_{model_slug}_scoring_{kind}.csv"
        path = layout.artifact(name)
        atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        ArtifactMetadata(
            artifact=name,
            produced_by=BiasScorer.produced_by,
            run_id=run_id,
            dataset=dataset,
            model=cfg.model.name,
            dataset_manifest=str(manifest.relative_to(REPO)) if manifest.exists() else None,
            input_artifacts=[InputArtifact.of(cohort_path)],
            config=config_payload,
        ).write(path)
        print(f"  wrote {path.name}")

    overall = summary[summary["group"] == "ALL"].iloc[0]
    print(f"  WinoQueer score (ALL): {overall['winoqueer_score']}% | n={int(overall['n'])} "
          f"| neutral={overall['pct_neutral']}% | mean score_diff={overall['mean_score_diff']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Wino-style bias for one or more cohorts.")
    ap.add_argument("--config", type=Path, required=True, help="RunConfig YAML/JSON.")
    ap.add_argument("--models-config", type=Path, default=REPO / "configs" / "models.yaml")
    ap.add_argument("--model", type=str, default=None, help="Registry key override (e.g. gpt2).")
    ap.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset slugs (overrides config).")
    ap.add_argument("--all", action="store_true", help=f"Score all pod cohorts: {', '.join(ALL_POD_DATASETS)}.")
    ap.add_argument("--cohort", type=Path, default=None, help="Cohort CSV override (single dataset only).")
    ap.add_argument("--max-pairs", type=int, default=None, help="Smoke-test cap (overrides config).")
    args = ap.parse_args()

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

    if args.all:
        datasets = list(ALL_POD_DATASETS)
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [cfg.dataset]
    if args.cohort is not None and len(datasets) != 1:
        raise SystemExit("--cohort override is only valid when scoring a single dataset.")

    run_id = cfg.run_id or new_run_id()
    print(f"[run_scoring] model={cfg.model.name} run_id={run_id} datasets={datasets}")
    loaded = ModelLoader(cfg.model).load()  # load ONCE, reuse across datasets
    for dataset in datasets:
        score_dataset(cfg, dataset, loaded, run_id, args.cohort)


if __name__ == "__main__":
    main()
