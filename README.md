# axiom

Mechanistic-interpretability pipelines for **social-bias analysis** in LLMs, across
matched Wino-style datasets: **WinoQueer**, **BBQ-derived WinoGender / WinoRace**,
**CrowS-derived** pairs, and an **identity-only** probe set.

The goal is not just to produce figures — it is a repository where every figure, table,
and claim traces back to a dataset, model, configuration, and exact analysis run. See
[`CLAUDE.md`](CLAUDE.md) for the full engineering agreements.

## What this measures

For each dataset we localize and intervene on the circuit a model uses to express bias,
running the same battery and comparing *mechanistic profiles* (not just effect sizes):
residual patching, head patching, head ablation, greedy knockout, steering, MLP
attribution — then layer/head/MLP stability, circuit concentration, linearity, and
cross-dataset transfer. The design lives in [`planning/`](planning/).

## Architecture

Object-oriented pipeline components carry their own config, model handles, and paths;
`scripts/` are thin orchestration. The shared foundations are centralized in `axiom/`:

| Module | Responsibility |
| --- | --- |
| `axiom.config` | Pydantic config models (the only source of run parameters) |
| `axiom.device` | Device/dtype resolution — `cuda` (pod) / `mps` (local) / `cpu` |
| `axiom.models` | The single HF + TransformerLens loader (GPT-2 and Llama) |
| `axiom.metrics` | Every metric formula, defined once (bias, restoration, Jaccard, Spearman, …) |
| `axiom.alignment` | Token alignment + continuation-span detection (pure) |
| `axiom.data` | Explicit cohort schemas + validation + `DatasetAdapter` interface |
| `axiom.cohorts` | `SegmentedCohortBuilder` — freeze balanced, stable cohorts |
| `axiom.interventions` | `InterventionRunner` ABC for the patching/ablation/steering steps |
| `axiom.paths` | Output layout `outputs/<model>/<dataset>/<step>/<run_id>/` |
| `axiom.provenance` | run_id, git sha, atomic writes, checksums, `*.meta.json` sidecars |
| `axiom.figures` | Publication-grade, **colorblind-safe** (Okabe-Ito) styling |

## Layout

```
axiom/        the importable package (foundations + pipeline classes)
scripts/      thin runners (run_scoring.py, build_cohorts.py, run_*_patching.py, …)
configs/      models.yaml + per-step run configs (validated as axiom.config)
data/
  raw/        upstream sources (gitignored; see manifests)
  processed/  parsed/scored intermediates (gitignored, except the combined candidate pool)
  cohorts/    frozen, balanced cohorts — the pod inputs (TRACKED)
  manifests/  per-dataset MANIFEST.json (TRACKED)
outputs/      all run outputs, by model -> dataset -> step -> run_id (gitignored)
models/ saes/ local model + SAE checkpoints (gitignored)
docs/adr/     architectural decision records
tests/        torch-free unit tests for the invariants that matter
```

### The pod inputs

Four frozen cohorts are what the pod analysis reads, tracked under `data/cohorts/`:

- `winoqueer/cohort.csv`
- `combined_bbq_crows/cohort.csv` — pooled BBQ+CrowS (head/MLP/resid)
- `residual_bbq_crows/bbq/cohort.csv`, `.../crows/cohort.csv` — per-source resid

## Running

Local development and smoke tests use **GPT-2** (MPS/CPU); pod runs use **Llama-3.1-8B**
(CUDA). Device is resolved automatically and recorded in provenance.

```bash
pip install -r requirements.txt
python -m pytest                                   # foundations (no model needed)

# Smoke-test scoring locally on GPT-2 (5 pairs):
python scripts/run_scoring.py --config configs/scoring/winoqueer_llama31_8b.yaml \
    --model gpt2 --max-pairs 5
```

Every runner supports a small smoke-test mode and writes valid artifacts with sidecars.
```
