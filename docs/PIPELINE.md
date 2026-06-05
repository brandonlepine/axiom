# Pipeline

End-to-end guide to the axiom bias-circuit pipeline: the stages, the scripts that run
them, the data that flows between them, and how to run locally (GPT-2) vs on the pod
(Llama-3.1-8B, CUDA). Design rationale lives in [`docs/adr/`](adr/); engineering
agreements in [`CLAUDE.md`](../CLAUDE.md).

## Philosophy

Localize and intervene on the circuit a model uses to express social bias, across
matched Wino-style datasets, and compare *mechanistic profiles* (not just effect sizes).
Two principles shape the flow:

- **Score with the canonical metric, then analyze only where the model is biased.**
  The published WinoQueer autoregressive metric decides which instances a *given model*
  is biased on; the expensive mechanistic steps run only on those (ADR 0003).
- **Everything is per model and reproducible.** Outputs are organized
  `outputs/<model>/<dataset>/<step>/<run_id>/`, and every artifact carries a
  `*.meta.json` sidecar (run id, git sha, input checksums, config, model runtime).

## Stages at a glance

```
 candidate pool ──score──▶ scored pool ──filter+balance──▶ analysis cohort ──┬─▶ residual patching
 (data/processed)         (wq_* columns)   (high-bias, per model)            ├─▶ head patching   (pending)
                                                                             ├─▶ head ablation   (pending)
                                                                             ├─▶ greedy knockout (pending)
                                                                             ├─▶ steering        (pending)
                                                                             └─▶ MLP attribution (pending)
                                                                                      │
                                                                          segmented / cross-dataset (pending)
```

| Stage | Script | Class | Output |
| --- | --- | --- | --- |
| 0 Cohort construction | `scripts/build_cohorts.py`, `scripts/write_manifests.py` | `SegmentedCohortBuilder` | `data/cohorts/*`, `data/manifests/*` |
| 1 Scoring | `scripts/run_scoring.py` | `BiasScorer` | `outputs/<m>/<d>/scoring/<run>/*_scoring_{raw,summary}.csv` |
| 2 Selection | `scripts/run_select_cohort.py` | `BiasCohortSelector` | `outputs/<m>/<d>/selection/<run>/*_analysis_cohort.csv` (+ scored_pool, selection_report) |
| 3 Residual patching | `scripts/run_residual_patching.py` | `ResidualPatcher` | `outputs/<m>/<d>/residual_patching/<run>/*` |
| 3+ remaining mechanistic steps | *(pending)* | head/MLP/steering/... | — |

> Stage 1 (`run_scoring.py`) scores a *fixed cohort* and reports the WinoQueer win-rate;
> Stage 2 (`run_select_cohort.py`) scores the *full pool* and freezes the high-bias
> analysis cohort the mechanistic steps consume. For circuit work you normally run Stage 2.

## Data layout

```
data/
  raw/<dataset>/          upstream sources (BBQ, CrowS, WinoQueer)        [gitignored]
  processed/
    combined/bbq_crows_candidates.csv     pooled BBQ+CrowS candidate pool [tracked]
    winoqueer/.../winoqueer_patching_candidates_all.csv   WinoQueer pool  [gitignored, large]
  cohorts/<dataset>/cohort.csv            frozen balanced cohorts (candidate universe) [tracked]
  manifests/<dataset>.MANIFEST.json       provenance + checksums          [tracked]
```

Datasets are registered once in `axiom/datasets.py` (`DATASETS`): each entry maps a slug
to its cohort path, candidate pool, manifest, schema, balancing cell, and (for WinoQueer)
the identity-taxonomy annotation. The four dataset slugs:

- `winoqueer` — LGBTQ+ identity × stereotype (pool 15,239; cohort 5,622)
- `combined_bbq_crows` — pooled BBQ+CrowS (pool 4,496; cohort 4,060)
- `bbq`, `crows` — per-source views of the combined pool/cohort

The combined `gender` axis is split into `gender_binary` / `gender_identity` (ADR 0002).

## The metric (Stage 1 / 2)

`BiasScorer` is the canonical WinoQueer autoregressive metric. For a minimal pair
(`sent_x` = stereotype variant, `sent_y` = reference), it difflib-aligns the tokens the
two sentences **share**, and with BOS prepended sums `log P(shared token | preceding)`:

- `sent_x_score`, `sent_y_score` — summed shared-span log-prob per side
- `wq_score_diff = sent_x_score - sent_y_score` — continuous bias signal
- `wq_stereo = 1` iff `wq_score_diff > 0` — the model prefers the stereotype (ties = neutral)
- headline **WinoQueer score** = % of pairs with `wq_stereo == 1`, overall + per group

Validated numerically identical (~1e-4) to the upstream per-token implementation.

## Selection (Stage 2)

`BiasCohortSelector` turns a scored pool into a model-specific high-bias analysis cohort
(ADR 0003):

1. **filter**: keep `wq_stereo == 1` AND `wq_score_diff >= tau` (default `tau = 0.5`)
2. **re-balance**: cap each `(identity/block, predicate)` cell, highest `wq_score_diff` kept
3. **freeze**: stable `cohort_pair_id` ordered by descending bias

`tau` and `cap` are the two knobs (`--tau`, `--cap`); the `selection_report.csv` shows per
-group scoreable / passed / kept and pass rates — check these on the first pod run and
adjust so each dataset lands at a sensible cohort size.

## Cohort resolution (which cohort a mechanistic step consumes)

Mechanistic runners default to the **latest selected analysis cohort** for the
(model, dataset) — you do not paste run-id paths. Precedence
(`axiom.datasets.resolve_analysis_cohort`):

1. `--cohort <path>` — explicit override.
2. `--raw-cohort` — the raw frozen pod cohort (the candidate universe, **not** bias-selected
   for this model); an escape hatch.
3. *(default)* — the most recent `outputs/<model>/<dataset>/selection/<run>/*_analysis_cohort.csv`.
   If none exists, the runner errors and tells you to run selection first.

This enforces ADR 0003: the circuit is localized on instances where *this* model is biased.

## Residual patching (Stage 3)

`ResidualPatcher` injects the stereotype variant's `resid_pre` activation into the
reference run, per `(layer, token)`, and re-scores the continuation:

- `bias_effect = patched - reference` (Δ logP of the continuation)
- `normalized_restoration = bias_effect / (stereotype - reference)`
- direction `stereotype_into_reference`; alignment via `align_minimal_pair`
  (prefix/suffix, identity span end-aligned)

Outputs: `*_raw.csv` (per pair/layer/token, resumable on `cohort_pair_id`),
`*_span_summary.csv`, `*_identity_by_layer.csv`, and a colorblind-safe (PuOr) layer×span
heatmap (png + pdf).

## Running

Local development uses **GPT-2** (MPS/CPU) for plumbing smoke tests; the pod uses
**Llama-3.1-8B** (CUDA) for results. Device resolves automatically; `--model gpt2`
overrides the config's model.

```bash
pip install -r requirements.txt
python -m pytest                          # torch-free foundation tests

# Pod: select high-bias analysis cohorts (all datasets, one model load), then patch.
python scripts/run_select_cohort.py     --config configs/scoring/winoqueer_llama31_8b.yaml --all
python scripts/run_residual_patching.py --config configs/scoring/winoqueer_llama31_8b.yaml --dataset winoqueer
#   (cohort auto-resolves to the latest selection; add --tau/--cap to selection as needed)

# Local smoke (small slices):
python scripts/run_select_cohort.py     --config ... --model gpt2 --all --max-pool-pairs 200 --cap 20
python scripts/run_residual_patching.py --config ... --model gpt2 --dataset winoqueer --max-pairs 2 --layers 0,1,2,3
```

> **MPS caveat**: TransformerLens may produce silently-incorrect numbers on MPS +
> PyTorch 2.11. Local GPT-2 runs validate *plumbing*; the **CUDA pod runs are
> authoritative**. Set `TRANSFORMERLENS_ALLOW_MPS=1` to acknowledge the warning.

## Provenance

Every artifact has a `<artifact>.meta.json` sidecar: `produced_by`, `run_id`, `git_sha`,
`git_dirty`, `dataset`, `dataset_manifest`, input `sha256`s, the full config, and the
model runtime (device/dtype/n_layers). Within a `run_select_cohort --all` or
`run_scoring --all` invocation, every dataset shares one `run_id`.

## Status (2026-06-05)

**Ported & validated**: cohort construction + gender split, WinoQueer scoring,
high-bias selection, residual patching, cohort auto-resolution.
**Pending**: head patching, head ablation, greedy knockout, steering, MLP attribution,
segmented analysis, cross-dataset comparison, transfer experiments. The
`InterventionRunner` ABC and the dataset/figure scaffolding are ready for them.
