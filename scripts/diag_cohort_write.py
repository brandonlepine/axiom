#!/usr/bin/env python3
"""Model-free diagnostic for the empty analysis_cohort.csv bug (run: python scripts/diag_cohort_write.py).

Reproduces the selection write path for the crows pool with fabricated wq_* scores (no
model needed) and reports exactly where the cohort becomes a 0-byte file: in the frame
(rows/cols), in to_csv, or in the atomic writer.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from axiom.cohorts import BiasCohortSelector
from axiom.provenance import atomic_write_bytes

print("pandas:", pd.__version__)

pool = pd.read_csv("data/processed/combined/bbq_crows_candidates.csv")
c = pool[pool["source"] == "crows-pairs"].copy().reset_index(drop=True)
c["wq_score_diff"] = 1.0
c["wq_stereo"] = 1
c["scoreable"] = True
print("crows pool rows:", len(c))

cohort, report = BiasCohortSelector(
    tau=0.5, cap=100, cell_columns=["block", "predicate_label_provisional"]
).select(c)
print("cohort rows:", len(cohort), "| cohort cols:", cohort.shape[1])
print("cohort columns:", list(cohort.columns))

csv_text = cohort.to_csv(index=False)
print("to_csv chars:", len(csv_text), "| encoded bytes:", len(csv_text.encode("utf-8")))
print("to_csv first 100 chars:", repr(csv_text[:100]))

# Test the exact writer the selection runner uses.
tmp = Path(tempfile.gettempdir()) / "_diag_cohort.csv"
atomic_write_bytes(tmp, csv_text.encode("utf-8"))
print("atomic_write_bytes file size:", tmp.stat().st_size)

# And pandas' direct file writer, as the candidate fix.
tmp2 = Path(tempfile.gettempdir()) / "_diag_cohort_direct.csv"
cohort.to_csv(tmp2, index=False)
print("cohort.to_csv(path) file size:", tmp2.stat().st_size)

print("\nVERDICT:")
if cohort.shape[1] == 0:
    print("  -> cohort has 0 COLUMNS: keep_all_columns/freeze bug on this pandas.")
elif len(csv_text.encode()) == 0:
    print("  -> to_csv() returns EMPTY for a non-empty frame on this pandas.")
elif tmp.stat().st_size == 0:
    print("  -> atomic_write_bytes writes EMPTY on this filesystem.")
else:
    print("  -> write path is fine here; the empty files came from something else.")
