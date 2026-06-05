"""BiasScorer orchestration invariants without a model.

The batched model call (_score_side) is stubbed so we test the parts that carry the
scientific meaning: bias_score directionality, scoreable handling, and max_pairs --
exactly the invariants CLAUDE.md calls out ("directionality of bias scoring").
"""
import math

import pandas as pd

from axiom.config import ScoringConfig
from axiom.scoring import BiasScorer, ContinuationScore


def _df():
    return pd.DataFrame({
        "cohort_pair_id": [0, 1, 2],
        "row_id": [10, 11, 12],
        "sent_x": ["A is gay and sick", "B is straight and ok", "C empty"],
        "sent_y": ["A is straight and sick", "B is gay and ok", "C empty"],
        "prefix_x": ["A is gay and", "B is straight and", ""],   # row 2 unscoreable (empty prefix)
        "prefix_y": ["A is straight and", "B is gay and", ""],
        "continuation": ["sick", "ok", "x"],
    })


def _make_scorer(side_map, max_pairs=None):
    """A BiasScorer whose _score_side returns avg_logp from a {sentence: avg} map."""
    scorer = BiasScorer.__new__(BiasScorer)
    scorer.loaded = None
    scorer.config = ScoringConfig(max_pairs=max_pairs)

    def fake_side(fulls, prefixes):
        return [ContinuationScore(logp=side_map[s], avg_logp=side_map[s], token_count=1) for s in fulls]

    scorer._score_side = fake_side  # type: ignore[method-assign]
    return scorer


def test_bias_score_directionality_and_scoreable():
    # target (sent_x) more probable on row 0 -> positive; less on row 1 -> negative
    side = {
        "A is gay and sick": -1.0, "A is straight and sick": -3.0,   # bias +2
        "B is straight and ok": -3.0, "B is gay and ok": -1.0,       # bias -2
        "C empty": -2.0,
    }
    out = _make_scorer(side).score(_df())
    assert out.loc[0, "bias_score"] == 2.0
    assert out.loc[1, "bias_score"] == -2.0
    # row 2 has empty prefixes -> not scoreable -> NaN bias_score, flagged
    assert out.loc[2, "scoreable"] == False  # noqa: E712
    assert math.isnan(out.loc[2, "bias_score"])
    # original + new columns present
    for c in ["target_cont_avg_logp", "reference_cont_avg_logp", "bias_score", "row_id"]:
        assert c in out.columns


def test_max_pairs_truncates():
    side = {s: -1.0 for s in pd.concat([_df()["sent_x"], _df()["sent_y"]])}
    out = _make_scorer(side, max_pairs=2).score(_df())
    assert len(out) == 2
