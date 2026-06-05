#!/usr/bin/env python3
"""Run residual-stream activation patching for a cohort (thin orchestration).

Loads a validated config, picks a model (config or ``--model`` registry override),
constructs the dataset adapter + :class:`axiom.interventions.ResidualPatcher`, and runs
it. Outputs (raw per-(pair, layer, token) CSV, span + identity-by-layer summaries, a
colorblind-safe span heatmap) land under
``outputs/<model>/<dataset>/residual_patching/<run_id>/`` with provenance sidecars.

Examples:
    # Local GPT-2 smoke test: 2 pairs, layers 0-3 of WinoQueer
    python scripts/run_residual_patching.py --config configs/scoring/winoqueer_llama31_8b.yaml \\
        --model gpt2 --max-pairs 2 --layers 0,1,2,3

    # Full pod run on Llama-3.1-8B (uses the model + frozen cohort in file order)
    python scripts/run_residual_patching.py --config configs/scoring/winoqueer_llama31_8b.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom.config import PatchingConfig, load_model_registry, load_run_config
from axiom.datasets import make_adapter
from axiom.interventions import ResidualPatcher
from axiom.models import ModelLoader
from axiom.paths import OutputLayout
from axiom.provenance import new_run_id

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Residual-stream activation patching for a cohort.")
    ap.add_argument("--config", type=Path, required=True, help="RunConfig YAML/JSON.")
    ap.add_argument("--models-config", type=Path, default=REPO / "configs" / "models.yaml")
    ap.add_argument("--model", type=str, default=None, help="Registry key override (e.g. gpt2).")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset slug (overrides config.dataset).")
    ap.add_argument("--cohort", type=Path, default=None, help="Cohort CSV override.")
    ap.add_argument("--max-pairs", type=int, default=None, help="Smoke-test cap on pairs.")
    ap.add_argument("--layers", type=str, default=None, help="Comma-separated layer subset (default: all).")
    ap.add_argument("--patch-batch-size", type=int, default=None, help="Target positions per forward batch.")
    ap.add_argument("--overwrite", action="store_true", help="Ignore any existing raw CSV (no resume).")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.model is not None:
        registry = load_model_registry(args.models_config)
        if args.model not in registry:
            raise SystemExit(f"--model {args.model!r} not in registry {sorted(registry)}")
        cfg = cfg.model_copy(update={"model": registry[args.model]})
    dataset = args.dataset or cfg.dataset
    cfg = cfg.model_copy(update={"dataset": dataset, "step": "residual_patching"})

    pc = cfg.patching or PatchingConfig()
    updates: dict = {}
    if args.max_pairs is not None:
        updates["max_pairs"] = args.max_pairs
    if args.layers is not None:
        updates["layers"] = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    if args.patch_batch_size is not None:
        updates["patch_batch_size"] = args.patch_batch_size
    if args.overwrite:
        updates["resume"] = False
    cfg = cfg.model_copy(update={"patching": pc.model_copy(update=updates)})

    adapter = make_adapter(dataset, REPO, cohort_override=args.cohort)
    run_id = cfg.run_id or new_run_id()
    layout = OutputLayout(model=cfg.model.name, dataset=dataset, step="residual_patching", run_id=run_id, root=REPO)
    print(f"[run_residual_patching] dataset={dataset} model={cfg.model.name} run_id={run_id}")
    print(f"[run_residual_patching] cohort={adapter.cohort_path()} layers={cfg.patching.layers or 'all'} "
          f"max_pairs={cfg.patching.max_pairs}")

    loaded = ModelLoader(cfg.model).load()
    result = ResidualPatcher(cfg, adapter, layout, loaded).run()

    print(f"\n[done] {result.summary}")
    print("artifacts:")
    for p in result.artifacts:
        print(f"  {p}")


if __name__ == "__main__":
    main()
