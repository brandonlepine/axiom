# ADR 0001 — Architecture and on-disk layout

- Status: accepted
- Date: 2026-06-05

## Context

`axiom` is a clean restart of a mechanistic-interpretability bias project. The prior
repository accreted ad-hoc scripts, duplicated helpers, and outputs that could not be
traced to a dataset/run, making it impossible to tell which data belonged to which
analysis. `CLAUDE.md` codifies the working agreements; this ADR records the concrete
structural decisions taken at the start so they are not silently re-litigated.

## Decisions

1. **Object-oriented pipeline components; thin runners.** Each analysis step is a class
   (e.g. `BiasScorer`, `SegmentedCohortBuilder`, `ResidualPatcher`) carrying its config,
   model handle, and paths. `scripts/` only loads config, constructs the class, runs it,
   and writes outputs. Pure stateless helpers (alignment, metrics) stay free functions.

2. **Centralize the shared machinery.** Model loading (HF + TransformerLens), metric
   formulas, device/dtype resolution, token alignment, and provenance each live in one
   module and are imported everywhere. The prior repo duplicated `continuation_logp`,
   `align_pair`, and `resolve_device` across many scripts; that is now forbidden.

3. **Output layout: `outputs/<model>/<dataset>/<step>/<run_id>/`.** Human-navigable by
   model → dataset → step, with `run_id` as the leaf. `run_id`, git SHA, input checksums,
   and config are still recorded in a `*.meta.json` sidecar beside every artifact, so the
   model/dataset hierarchy is a convenience, not the system of record.

4. **Data layout per `CLAUDE.md`:** `data/raw/` (sources, gitignored), `data/processed/`
   (intermediates, gitignored except the small reproducible combined candidate pool),
   `data/cohorts/` (the frozen pod-input cohorts, **tracked**), `data/manifests/`
   (per-dataset provenance, **tracked**). The misleading `test_data/` name from the prior
   layout — which actually held the real pipeline inputs — is replaced by `data/cohorts/`.

5. **Local GPT-2, pod Llama, one code path.** Device resolves to `cuda` on the pod and
   `mps` locally; `mps + bfloat16` downgrades to `float16`. GPT-2 results slug to
   `outputs/gpt2/...` and never collide with pod Llama results.

6. **Tracked pod inputs.** The four frozen cohorts are small (≤ ~2 MB) and are committed
   so the pod can `git pull` them directly. Large raws, model checkpoints, and outputs
   are gitignored.

7. **Colorblind-safe figures by default.** All figures use the Okabe-Ito categorical
   palette and perceptually-uniform / colorblind-safe sequential & diverging maps via
   `axiom.figures`, saved as both PNG (raster) and PDF (vector) with stable filenames.

## Consequences

- Adding a dataset/model/metric/intervention means implementing an interface
  (`DatasetAdapter`, `InterventionRunner`, a metric function), not editing every runner.
- Every artifact is auditable; a result that exists only in a notebook is not part of the
  pipeline.
