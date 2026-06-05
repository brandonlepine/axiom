#!/usr/bin/env python3
"""Build a model-specific high-bias analysis cohort (score pool -> filter -> balance -> freeze).

For each dataset: load the full candidate pool, score it with the WinoQueer autoregressive
metric on the target model, keep the pairs the model is biased on (``wq_stereo == 1`` and
``wq_score_diff >= tau``), re-balance per cell, and freeze a stable ``cohort_pair_id``. The
frozen analysis cohort is what the mechanistic steps (patching, ...) consume via
``--cohort``. Outputs land under ``outputs/<model>/<dataset>/selection/<run_id>/`` with
provenance sidecars. The model is loaded ONCE and reused across datasets.

Examples:
    # Local GPT-2 smoke test: select from a 200-pair slice of each pool
    python scripts/run_select_cohort.py --config configs/scoring/winoqueer_llama31_8b.yaml \\
        --model gpt2 --all --max-pool-pairs 200 --cap 20

    # Full pod run on Llama-3.1-8B, all datasets
    python scripts/run_select_cohort.py --config configs/scoring/winoqueer_llama31_8b.yaml --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from axiom.cohorts import BiasCohortSelector
from axiom.config import RunConfig, ScoringConfig, SelectionConfig, load_model_registry, load_run_config
from axiom.datasets import ALL_POD_DATASETS, DATASETS, load_pool, manifest_path
from axiom.data.schemas import validate_cohort
from axiom.models import ModelLoader
from axiom.models.loader import LoadedModel
from axiom.paths import OutputLayout, slugify
from axiom.provenance import ArtifactMetadata, InputArtifact, atomic_write_bytes, new_run_id
from axiom.scoring import BiasScorer

REPO = Path(__file__).resolve().parent.parent


def select_dataset(cfg: RunConfig, dataset: str, loaded: LoadedModel, run_id: str) -> None:
    spec = DATASETS[dataset]
    sel = cfg.selection
    pool = load_pool(dataset, REPO)
    if sel.max_pool_pairs is not None:
        pool = pool.head(sel.max_pool_pairs).copy()

    layout = OutputLayout(model=cfg.model.name, dataset=dataset, step="selection", run_id=run_id, root=REPO)
    layout.ensure()
    print(f"\n[select] dataset={dataset} pool={len(pool)} rows -> {layout.dir}")

    scored = BiasScorer(loaded, ScoringConfig(batch_size=cfg.scoring.batch_size if cfg.scoring else 16)).score(pool)
    selector = BiasCohortSelector(tau=sel.tau, cap=sel.cap, cell_columns=list(spec.cell_columns))
    cohort, report = selector.select(scored)
    if len(cohort):
        validate_cohort(cohort, spec.schema)

    model_slug = slugify(cfg.model.name)
    manifest = manifest_path(dataset, REPO)
    cfg_payload = cfg.model_dump(mode="json")
    cfg_payload["dataset"] = dataset
    cfg_payload["scoring_metric"] = BiasScorer.metric
    cfg_payload["model_runtime"] = loaded.provenance()
    cfg_payload["pool"] = str(spec.pool_relpath)

    inputs = [InputArtifact.of(REPO / spec.pool_relpath)]
    artifacts = {
        f"{dataset}_{model_slug}_scored_pool.csv": scored,
        f"{dataset}_{model_slug}_analysis_cohort.csv": cohort,
        f"{dataset}_{model_slug}_selection_report.csv": report,
    }
    for name, frame in artifacts.items():
        path = layout.artifact(name)
        atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        ArtifactMetadata(
            artifact=name, produced_by="BiasCohortSelector", run_id=run_id,
            dataset=dataset, model=cfg.model.name,
            dataset_manifest=str(manifest.relative_to(REPO)) if manifest.exists() else None,
            input_artifacts=inputs, config=cfg_payload,
            extra={"tau": sel.tau, "cap": sel.cap, "cell_columns": list(spec.cell_columns)},
        ).write(path)

    overall = report[report["group"] == "ALL"].iloc[0]
    print(f"  kept {int(overall['n_kept'])} / {int(overall['n_passed_filter'])} passed "
          f"/ {int(overall['n_scoreable'])} scoreable (pass_rate {overall['pass_rate']}) "
          f"| tau={sel.tau} cap={sel.cap}")
    print(f"  analysis cohort -> {layout.artifact(f'{dataset}_{model_slug}_analysis_cohort.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build model-specific high-bias analysis cohorts.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--models-config", type=Path, default=REPO / "configs" / "models.yaml")
    ap.add_argument("--model", type=str, default=None, help="Registry key override (e.g. gpt2).")
    ap.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset slugs.")
    ap.add_argument("--all", action="store_true", help=f"All datasets: {', '.join(ALL_POD_DATASETS)}.")
    ap.add_argument("--tau", type=float, default=None, help="Min wq_score_diff to keep (override).")
    ap.add_argument("--cap", type=int, default=None, help="Max pairs per cell after filtering (override).")
    ap.add_argument("--max-pool-pairs", type=int, default=None, help="Smoke-test cap on pool rows.")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.model is not None:
        registry = load_model_registry(args.models_config)
        if args.model not in registry:
            raise SystemExit(f"--model {args.model!r} not in registry {sorted(registry)}")
        cfg = cfg.model_copy(update={"model": registry[args.model]})
    sel = cfg.selection or SelectionConfig()
    updates = {k: v for k, v in
               {"tau": args.tau, "cap": args.cap, "max_pool_pairs": args.max_pool_pairs}.items() if v is not None}
    cfg = cfg.model_copy(update={"selection": sel.model_copy(update=updates),
                                 "scoring": cfg.scoring or ScoringConfig(), "step": "selection"})

    if args.all:
        datasets = list(ALL_POD_DATASETS)
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [cfg.dataset]

    run_id = cfg.run_id or new_run_id()
    print(f"[select] model={cfg.model.name} run_id={run_id} datasets={datasets} "
          f"tau={cfg.selection.tau} cap={cfg.selection.cap}")
    loaded = ModelLoader(cfg.model).load()
    for dataset in datasets:
        select_dataset(cfg, dataset, loaded, run_id)


if __name__ == "__main__":
    main()
