# ADR 0002 — Splitting the combined `gender` axis into `gender_binary` / `gender_identity`

- Status: accepted
- Date: 2026-06-05

## Context

The combined BBQ+CrowS cohorts pool a coarse `gender` axis that mixes two different
constructs:

- **`gender_binary`** — cis man/woman *role* bias (e.g. "a man" / "a woman"), the bulk
  of BBQ `Gender_identity` items and CrowS gender pairs;
- **`gender_identity`** — trans / nonbinary *identity* bias (e.g. "a transgender woman"),
  the construct that aligns with WinoQueer's gender_identity axis.

Conflating them makes the planned transfer experiments ill-posed: "WinoGender ↔ WinoQueer"
and "discover on WinoGender, test on WinoRace" need the trans/NB cells to be comparable to
WinoQueer specifically, while the binary cells are a separate comparison set. The frozen
pod cohorts from 2026-06-03 still carried the coarse `gender` label.

## Decision

Split `gender` by the identity string: an identity containing `trans`, `nonbinary`,
`non-binary`, or `enby` → `gender_identity`; otherwise → `gender_binary`
(`axiom.cohorts.builder.gender_subaxis`). The split is applied in the candidate pool and
re-applied idempotently by `SegmentedCohortBuilder` so output is correct regardless of
input state.

## Why this is balance-neutral (and therefore not a re-balancing)

Cohort balancing caps each `(block, predicate_label_provisional)` cell, and `block` is the
per-identity grouping. The binary blocks (`a man`, `a woman`, …) and the trans blocks
(`a transgender man`, `a transgender woman`, …) are **disjoint** — no block maps to both
sub-axes. So relabelling `axis` cannot move any row between cells: the kept set is
identical whether the split happens before or after capping.

This was verified empirically. Re-freezing the four combined cohorts from the candidate
pool at cap=80 reproduces the 2026-06-03 cohorts **row-for-row** (4060 / 3481 / 579),
with the *only* difference being `gender` → `{gender_binary, gender_identity}`. The
`build_cohorts.py --verify-against` check asserts exactly this invariant on every build.

## Consequences

- Combined `_all` axes: `gender_binary` (486) and `gender_identity` (117) replace
  `gender` (603); BBQ-only and CrowS-only cohorts split correspondingly.
- The split changes no effect estimate that does not condition on the gender axis label;
  it only enables the gender-specific transfer comparisons.
- WinoQueer is unaffected (already an identity-construct dataset).
