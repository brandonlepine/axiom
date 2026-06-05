#!/usr/bin/env python3
"""Write per-dataset MANIFEST.json files documenting cohort provenance.

Every adapted dataset must have a manifest recording its source, transformation lineage,
identity/predicate mappings, schema version, and checksums (CLAUDE.md, "Dataset
transformations must be auditable"). This script encodes the fixed provenance facts and
fills in live row counts + sha256 from the frozen cohorts on disk, so the manifests never
drift from the actual files.

Manifests are written to data/manifests/<dataset>.MANIFEST.json (tracked).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from axiom.provenance import sha256_file, utc_now_iso

REPO = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "v1"


def _cohort_stat(path: Path) -> dict:
    df = pd.read_csv(path)
    axes = {str(k): int(v) for k, v in df["axis"].value_counts().items()} if "axis" in df.columns else {}
    return {
        "path": str(path.relative_to(REPO)),
        "rows": int(len(df)),
        "sha256": sha256_file(path),
        "axis_counts": axes,
    }


# Fixed provenance lineage per dataset. Paths are resolved + checksummed at write time.
MANIFESTS = {
    "combined_bbq_crows": {
        "description": (
            "Pooled BBQ + CrowS-Pairs biased pairs in Wino-style continuation-scoring format. "
            "Pod input for pooled head/MLP/residual analyses, plus per-source BBQ-only and "
            "CrowS-only cohorts for cross-source residual comparison."
        ),
        "sources": [
            {"name": "BBQ", "role": "template-based stereotype pairs",
             "url": "https://github.com/nyu-mll/BBQ", "acquired": "see data/raw/bbq"},
            {"name": "CrowS-Pairs", "role": "naturalistic stereotype pairs",
             "url": "https://github.com/nyu-mll/crows-pairs", "acquired": "see data/raw/crows-pairs"},
        ],
        "transformation": {
            "candidate_pool": "data/processed/combined/bbq_crows_candidates.csv",
            "pool_builder": "build_combined_candidates (ported from bias_llm); pools BBQ "
                            "final candidates + CrowS biased pairs (bias_score>0), fresh unique row_id.",
            "cohort_builder": "axiom.cohorts.SegmentedCohortBuilder via scripts/build_cohorts.py",
            "balancing": "cap=80 pairs per (block, predicate_label_provisional) cell, highest bias_score kept",
            "gender_split": (
                "axis 'gender' split into gender_binary (cis man/woman role bias) and "
                "gender_identity (trans/nonbinary identity bias) by identity string. Balance-neutral: "
                "the cap keys on block, and binary vs trans blocks are disjoint (ADR 0002). Verified "
                "row-identical to the pre-split pod cohorts; only the axis label changes."
            ),
        },
        "cohorts": {
            "_all": "data/cohorts/combined_bbq_crows/cohort.csv",
            "bbq_only": "data/cohorts/residual_bbq_crows/bbq/cohort.csv",
            "crows_only": "data/cohorts/residual_bbq_crows/crows/cohort.csv",
        },
        "schema": "axiom.data.schemas.COMBINED_SCHEMA",
    },
    "winoqueer": {
        "description": (
            "WinoQueer LGBTQ+ identity + stereotype-continuation pairs in Wino-style scoring "
            "format. Pod input for the identity-segmented circuit analyses."
        ),
        "sources": [
            {"name": "WinoQueer", "role": "LGBTQ+ bias benchmark",
             "url": "https://github.com/katyfelkner/winoqueer", "acquired": "see data/raw/winoqueer"},
        ],
        "transformation": {
            "candidate_pool": "data/processed/winoqueer/results/patching_candidates/"
                              "winoqueer_patching_candidates_all.csv",
            "cohort_builder": "build_winoqueer_segmented_cohort (bias_llm); cap=100 per "
                              "(Gender_ID_x, predicate_label_provisional) cell.",
            "gender_split": "n/a (WinoQueer is already an identity-construct dataset)",
            "note": "Carried over verbatim from the validated bias_llm build; not yet "
                    "rebuilt in-repo (requires porting the identity-taxonomy module). Tracked "
                    "as-is with this checksum; a future commit will reproduce it from the pool.",
        },
        "cohorts": {"_all": "data/cohorts/winoqueer/cohort.csv"},
        "schema": "axiom.data.schemas.WINOQUEER_SCHEMA",
    },
    "identity_only": {
        "description": (
            "Identity-only prompts (no stereotype continuation): minimal templates naming an "
            "identity, used to estimate the identity direction v_identity independent of stereotype "
            "expression (planning doc, 'Separate Identity Representation from Stereotype Expression')."
        ),
        "sources": [
            {"name": "axiom identity templates", "role": "authored templates x identity taxonomy",
             "url": "data/processed/identity_specific", "acquired": "authored"},
        ],
        "transformation": {
            "builder": "identity template x identity taxonomy expansion (bias_llm)",
            "note": "Carried over verbatim; tracked with checksum.",
        },
        "cohorts": {"prompts": "data/cohorts/identity_only/mi_identity_prompts.csv"},
        "schema": "prompt-level (prompt_id, identity_id, axis, canonical_label, prompt, ...)",
    },
}


def main() -> None:
    out_dir = REPO / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, meta in MANIFESTS.items():
        artifacts = {}
        for role, rel in meta["cohorts"].items():
            p = REPO / rel
            if not p.exists():
                raise FileNotFoundError(f"{dataset}: cohort {rel} not found; build it first.")
            artifacts[role] = _cohort_stat(p)
        manifest = {
            "dataset_id": dataset,
            "schema_version": SCHEMA_VERSION,
            "produced_at": utc_now_iso(),
            **meta,
            "artifacts": artifacts,
        }
        out = out_dir / f"{dataset}.MANIFEST.json"
        out.write_text(json.dumps(manifest, indent=2))
        total = sum(a["rows"] for a in artifacts.values())
        print(f"wrote {out.relative_to(REPO)}  ({len(artifacts)} cohort(s), {total} rows)")


if __name__ == "__main__":
    main()
