# ADR 0003 — Model-specific high-bias analysis-cohort selection

- Status: accepted
- Date: 2026-06-05

## Context

The mechanistic analyses (residual/head patching, ablation, knockout, steering, MLP
attribution) localize the circuit a model uses to express bias. Running them on instances
where the model exhibits *no* bias dilutes the circuit estimate with noise. Bias is also
model-specific: each model has its own bias profile, so the analysis set must be derived
per model, not fixed once. The frozen pod cohorts were pre-filtered on the *prior* repo's
Llama continuation metric, which is neither the canonical WinoQueer metric nor model-general.

## Decision

Insert a **selection** step between scoring and the mechanistic analyses that freezes a
model-specific high-bias *analysis cohort* (`axiom.cohorts.BiasCohortSelector`,
`scripts/run_select_cohort.py`):

1. **Score the full candidate pool** (not the pre-filtered cohort) with the canonical
   WinoQueer autoregressive metric (ADR: `axiom.scoring.BiasScorer`) on the target model.
   WinoQueer pool = 15,239 pairs; combined = 4,496. WinoQueer pools are annotated with the
   identity taxonomy (axis/identity/is_umbrella) so the frozen cohort satisfies the schema.
2. **Filter to high-bias pairs**: keep where the model prefers the stereotype variant
   (`wq_stereo == 1`, i.e. `sent_x_score > sent_y_score`) by at least `tau`
   (`wq_score_diff >= tau`, default `tau = 0.5`). `wq_score_diff` is the sum of log-probs
   over the shared span; `tau` is in those units and is tunable per dataset.
3. **Re-balance** per `(identity/block, predicate)` cell (cap, highest `wq_score_diff`
   kept) so no identity or predicate dominates the aggregate circuit, then **freeze** a
   stable `cohort_pair_id` ordered by descending bias.

Outputs land under `outputs/<model>/<dataset>/selection/<run_id>/`: the scored pool, the
frozen analysis cohort, and a selection report (per-group scoreable / passed / kept +
pass rate). Downstream steps consume the analysis cohort via `--cohort`.

## Consequences

- The analysis set is regenerated per model and is fully provenance-tracked; it is an
  `outputs/` artifact (regenerable), not a tracked pod input.
- `tau` and `cap` are the two knobs that trade coverage against signal/balance; both are
  recorded in the selection report and every sidecar.
- The pre-existing frozen cohorts remain valid as the *candidate universe* selection draws
  from indirectly (via the pools); they are no longer the direct analysis input.
- Pairs the model is unbiased on are excluded by construction, so a near-zero mean effect
  in a downstream analysis reflects circuit structure, not a dilution by no-bias pairs.
