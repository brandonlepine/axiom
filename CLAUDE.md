# CLAUDE.md

> Engineering practices and working agreements for code written in this repository. This project contains mechanistic-interpretability pipelines for social-bias analysis across Wino-style datasets, including WinoQueer, WinoGender, WinoRace, BBQ-derived, and CrowS-derived variants. Read end-to-end before contributing.

---

## Core working agreements

### Ambiguity → ask before coding

If a requirement is ambiguous, stop and ask before writing code. This is especially important for:

- dataset schemas,
- run directories,
- model names,
- token-alignment assumptions,
- metric definitions,
- cohort-building rules,
- patching directionality,
- train/test split behavior,
- and artifact naming.

Do not invent missing paths, columns, default models, predicate labels, identity mappings, or magic constants. If something is unknown, say so and ask one narrowly scoped clarification question.

### No stubs, fake implementations, or placeholder logic

Do not write code that pretends to implement an analysis but silently skips the real work. No fake outputs, dummy metrics, placeholder datasets, or TODO: implement later branches.

If a step cannot be completed because an input artifact, schema, or design decision is missing, stop and document the blocker explicitly.

Structural scaffolding is acceptable only when it makes no behavioral claim, such as creating directories, empty __init__.py files, or config templates.

### Read before writing

Before adding or changing code, inspect the relevant nearby modules. Match existing conventions for:

- logging,
- schemas,
- output layout,
- run IDs,
- config loading,
- model loading,
- checkpointing,
- and CLI structure.

Do not introduce a new pattern when an existing one already works.

---

## Project architecture

### Use object-oriented pipeline components

Default to classes for analysis steps that carry state, configuration, model handles, tokenizer handles, paths, or lifecycle methods. Free functions are appropriate only for pure stateless utilities such as token alignment, metric computation, string normalization, or small mathematical helpers.

Each major analysis step should have a self-contained class, for example:

- BiasScorer
- ResidualPatcher
- HeadPatcher
- HeadAblator
- GreedyKnockoutRunner
- SteeringSweepRunner
- SegmentedCohortBuilder
- SegmentedHeadAnalyzer
- SegmentedMLPAnalyzer
- CrossDatasetComparator
- IdentityOnlyEvaluator

A class should be instantiable and testable in isolation. It should not silently depend on global state, environment variables, current working directory, or hard-coded paths.

### Separate orchestration from analysis logic

Analysis logic belongs in reusable classes or modules. Runner scripts should be thin orchestration layers that:

1. load config,
2. construct the relevant pipeline class,
3. run the step,
4. write outputs,
5. emit provenance.

Do not put core analysis logic directly inside CLI scripts.

### Centralized runners are required

The project should support both:

- running each step independently, and
- running the full pipeline in sequence.

Therefore, maintain centralized runner entry points such as:

text scripts/run_scoring.py scripts/run_residual_patching.py scripts/run_head_patching.py scripts/run_head_ablation.py scripts/run_greedy_knockout.py scripts/run_steering.py scripts/run_segmented_analysis.py scripts/run_cross_dataset_comparison.py scripts/run_full_pipeline.py 

Each step-specific runner must be callable on its own. The full-pipeline runner should call the same step classes rather than duplicating logic.

### Pipelines should be restartable and composable

Each pipeline step should define:

- required inputs,
- expected outputs,
- config schema,
- artifact names,
- checkpoint behavior,
- and failure modes.

Steps should not assume that previous steps were run in the same Python process. Every step must be able to load its inputs from disk.

---

## Dataset and schema discipline

### All dataset schemas are explicit

Every dataset consumed by the pipeline must have a schema model. This includes:

- WinoQueer,
- WinoGender,
- WinoRace,
- BBQ-derived Wino-style pairs,
- CrowS-derived Wino-style pairs,
- identity-only datasets,
- scored candidate files,
- balanced cohorts,
- patching raw outputs,
- ablation outputs,
- steering outputs,
- segmented-analysis summaries.

Use pydantic models for configs and artifact metadata. For row-level tabular data, maintain explicit schema documentation and validation utilities.

### Stable IDs are mandatory

Never rely on positional pair_id as a cross-script join key unless it is explicitly produced by a frozen cohort builder.

Preferred stable keys:

- row_id for source rows,
- cohort_pair_id for frozen cohorts,
- run_id for pipeline runs,
- dataset_id for dataset variants.

If a script internally re-sorts rows, this must be documented and outputs must preserve stable IDs.

### Dataset transformations must be auditable

Any adapted dataset must have a manifest documenting:

- original source,
- source version or acquisition date,
- transformation script,
- identity mappings,
- predicate mappings,
- filtering rules,
- dropped rows,
- schema version,
- and checksums.

This is especially important for BBQ-derived and CrowS-derived datasets, where the raw benchmark format differs from the Wino-style continuation-scoring format.

### No silent dataset swaps

Updating a dataset or changing a filtering rule requires updating the manifest and invalidating downstream stale outputs.

---

## Output naming and repository hygiene

### Output names must be explicit

Artifact names should encode:

- dataset,
- model,
- analysis step,
- split or cohort when relevant,
- metric when relevant,
- and whether the file is raw, summary, or figure.

Prefer names like:

text winoqueer_llama31_8b_resid_patching_raw.csv winoqueer_llama31_8b_head_patching_layer_head.csv winoqueer_llama31_8b_steering_sweep_raw.csv winoqueer_llama31_8b_segmented_head_jaccard_write.csv crossdataset_llama31_8b_attention_layer_spearman.csv 

Avoid vague names like:

text results.csv final.csv plot.png analysis_new.csv test_output.csv 

### Use consistent output directories

Use a predictable structure such as:

text data/   raw/   processed/   manifests/  outputs/   runs/     <run_id>/       config.json       provenance.json       logs/       scoring/       residual_patching/       head_patching/       head_ablation/       greedy_knockout/       steering/       segmented/       cross_dataset/  figures/   main/   appendices/   segmented-analysis/     attention/     mlp/     predicates/ 

Do not scatter outputs across ad hoc directories.

### Raw, intermediate, and final artifacts are distinct

Use clear subdirectories or filename suffixes:

- *_raw.csv for per-example or per-component raw outputs,
- *_summary.csv for aggregated summaries,
- *_ranking.csv for ordered component lists,
- *_profile.csv for layer/head/neuron profiles,
- *.png or *.pdf for figures.

Never overwrite raw outputs with aggregated summaries.

---

## Metrics and intervention conventions

### Directionality must be explicit

Every patching, ablation, or steering output must specify the direction of the intervention.

Examples:

- queer_to_control
- target_to_reference
- control_to_target
- ablate_on_target
- remove_from_target
- induce_on_reference

Do not use ambiguous terms like clean and corrupt in output columns unless the mapping is documented in the config and metadata.

### Metrics must be defined once

Core metrics should live in shared metric modules and be reused across scripts. This includes:

- continuation average log probability,
- bias score,
- bias effect,
- normalized restoration,
- bias fraction,
- fraction bias removed,
- attention readout-to-identity,
- KL readout divergence,
- Jaccard overlap,
- Spearman layer correlation,
- selectivity,
- rank-biased overlap.

Do not reimplement metric formulas independently in multiple scripts.

### Keep logits, log probabilities, and normalized metrics separate

Raw log-probability effects and normalized effects answer different questions. Store both when possible.

Recommended columns:

text control_cont_avg_logp target_cont_avg_logp patched_cont_avg_logp bias_score bias_effect normalized_restoration bias_fraction kl_readout 

Avoid columns whose meaning depends on context without documentation.

---

## Long-running model analyses

### Checkpointing is mandatory

Any model run expected to take more than five minutes must checkpoint progress. This includes:

- residual patching,
- head patching,
- head ablation,
- greedy knockout,
- steering sweeps,
- MLP attribution,
- SAE feature attribution.

Checkpoint files must be written atomically using temp files followed by replace.

### Resumability is required

A run interrupted midway should resume without corrupting outputs or duplicating completed work. Per-pair or per-chunk atomicity is preferred.

### Model loading is centralized

Model and tokenizer loading should happen through a single shared loader interface. Do not duplicate Hugging Face or TransformerLens loading logic across scripts.

The loader should make explicit:

- model path,
- tokenizer path,
- TransformerLens name if used,
- dtype,
- device,
- padding behavior,
- BOS behavior,
- and any model-specific quirks.

### Seeds must be explicit

Any stochastic process must accept a seed from config and write it to provenance metadata. This includes train/test splits, random steering directions, bootstrap confidence intervals, and randomized greedy knockout runs.

---

## Provenance and reproducibility

### Every artifact has metadata

Every written artifact must have a sidecar metadata file:

text <artifact>.meta.json 

Minimum metadata:

json {   "artifact": "filename.csv",   "produced_by": "module_or_runner_name",   "produced_at": "ISO-8601 timestamp",   "run_id": "uuid",   "git_sha": "commit hash if available",   "dataset": "winoqueer",   "dataset_manifest": "path/to/MANIFEST.json",   "model": "meta-llama/Llama-3.1-8B",   "input_artifacts": [     {"path": "input.csv", "sha256": "..."}   ],   "config": {},   "schema_version": "v1" } 

No artifact should be treated as reliable without provenance.

### Config is data, not code

All major runs should be configured through YAML, JSON, or TOML. Config files must be validated before use.

Do not hard-code model names, dataset paths, caps, layer lists, alpha grids, top-k values, or filtering rules inside analysis code.

### Run IDs thread through everything

A full pipeline run receives a run ID at the start. Every artifact produced during that run carries the same run ID.

---

## Testing expectations

### Test the invariants that matter

At minimum, tests should cover:

- token alignment,
- continuation-span detection,
- directionality of bias scoring,
- stable cohort IDs,
- metric formulas,
- output naming,
- config validation,
- artifact metadata creation,
- resume behavior,
- and dataset schema validation.

### Use small fixtures

Tests should use tiny synthetic datasets with 2–5 examples. Do not require loading large models in ordinary unit tests.

Model-dependent tests should be marked separately as integration tests.

### Golden tests for schema compatibility

For each major artifact type, maintain a tiny golden file and test that readers still parse it. Schema changes must be deliberate.

---

## Documentation standards

### Every public module has a top-level docstring

The docstring should state:

- what the module does,
- what it consumes,
- what it produces,
- and the typical caller.

### Every public class and method has a docstring

Docstrings should explain meaning, not restate type hints.

### Record architectural decisions

Significant design decisions belong in docs/adr/.

Examples:

- why Wino-style continuation scoring is used,
- why row_id is the stable join key,
- why attention and MLP analyses are separated,
- why identity-only datasets are used,
- why certain BBQ predicates are dropped,
- why CrowS-Pairs is treated as supplementary.

---

## Code quality

### Keep functions small and explicit

A function should generally fit on one screen. If it loads data, transforms it, runs a model, writes outputs, and plots figures, split it.

### Avoid global mutable state

No global lists of paths, implicit model handles, cached tokenizers, or mutable config objects.

Constants are acceptable only when they are genuine domain constants and documented.

### Composition over inheritance

Use small composable classes. Avoid deep inheritance hierarchies.

### Interfaces for extensibility

Adding a new dataset, model, metric, or intervention should not require editing every runner.

Prefer interfaces such as:

- DatasetAdapter
- BiasScoringTask
- InterventionRunner
- ComponentAttributor
- ResultAggregator
- FigureBuilder

---

## Failure handling

### Expected failures are recorded, not hidden

Examples:

- unalignable sentence pair,
- empty continuation,
- missing identity mapping,
- non-positive bias score,
- insufficient group size,
- failed tokenization,
- NaN metric.

These should be emitted as structured diagnostics where appropriate, not silently skipped.

### Catastrophic failures raise

Examples:

- missing required input artifact,
- invalid config,
- duplicated stable IDs,
- schema mismatch,
- model load failure,
- corrupt checkpoint.

Raise with clear error messages.

---

## Repository hygiene

### Do not commit large generated outputs unless intended

Large raw outputs, model caches, checkpoints, and temporary files should not be committed unless they are deliberately versioned artifacts.

Use .gitignore and DVC or another artifact system where appropriate.

### Delete dead code

Do not comment out old implementations. Delete them. Git preserves history.

### Keep notebooks out of the critical path

Exploratory notebooks are fine, but reproducible analysis must live in scripts and modules. A result that only exists in a notebook is not part of the pipeline.

### Figures should be reproducible

Every figure in the paper should be generated by a script from tracked inputs and saved with a stable filename.

---

## Process expectations

### Small commits

Each commit should make one coherent change.

Good examples:

text scoring: add Wino-style continuation scorer patching: centralize residual intervention runner datasets: add BBQ predicate manifest validation segmentation: add identity-level head overlap analysis 

### No merging on red

Tests, linting, and type checks must pass before merge.

### Run smoke tests before long jobs

Every major runner should support a small smoke-test mode, such as:

text --max_pairs 5 --layers 0,1 --heads 0,1 --alphas -1,0,1 

Smoke tests must complete quickly and write valid artifacts.

---

## When in doubt

Prefer code that is:

- explicit,
- testable,
- restartable,
- documented,
- schema-validated,
- and reproducible.

The goal is not just to produce figures. The goal is to build a repository where every figure, table, and claim can be traced back to a dataset, model, configuration, and exact analysis run.