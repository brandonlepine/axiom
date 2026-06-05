"""WinoQueer autoregressive BiasScorer invariants without a model.

The batched model side (_summed_logprobs) and tokenization (_encode_with_bos) are
stubbed so we test the scientific wiring: per-pair stereo direction, score_diff,
scoreable handling, max_pairs, and the win-rate summary -- the metric-definition
invariants CLAUDE.md calls out.
"""
import math

import pandas as pd

from axiom.config import ScoringConfig
from axiom.scoring import BiasScorer, PairScore


def _df():
    return pd.DataFrame({
        "sent_x": ["A is gay", "B is straight", ""],   # row 2 unscoreable (empty text)
        "sent_y": ["A is straight", "B is gay", ""],
        "identity": ["Gay", "Gay", "Other"],
    })


def _stub_scorer(x_scores, y_scores, max_pairs=None):
    scorer = BiasScorer.__new__(BiasScorer)
    scorer.loaded = None
    scorer.config = ScoringConfig(max_pairs=max_pairs)
    scorer.uncased = False
    scorer._encode_with_bos = lambda s: [0, 1, 2, 3]  # type: ignore[method-assign]
    queue = iter([x_scores, y_scores])  # score() calls X-side then Y-side
    scorer._summed_logprobs = lambda ids, pos, desc=None: next(queue)  # type: ignore[method-assign]
    return scorer


def test_pairscore_properties():
    p = PairScore(sent_x_score=-1.0, sent_y_score=-3.0, n_shared_x=2, n_shared_y=2)
    assert p.score_diff == 2.0 and p.stereo == 1 and p.neutral == 0
    tie = PairScore(-2.0, -2.0, 1, 1)
    assert tie.stereo == 0 and tie.neutral == 1


def test_score_directionality_and_scoreable():
    # row0: x(-1) > y(-3) -> stereo; row1: x(-5) < y(-2) -> not; row2 unscoreable
    x = [(-1.0, 2), (-5.0, 2), (float("nan"), 0)]
    y = [(-3.0, 2), (-2.0, 2), (float("nan"), 0)]
    out = _stub_scorer(x, y).score(_df())
    assert out.loc[0, "wq_stereo"] == 1
    assert out.loc[0, "wq_score_diff"] == 2.0
    assert out.loc[1, "wq_stereo"] == 0
    assert out.loc[1, "wq_score_diff"] == -3.0
    assert out.loc[2, "scoreable"] == False  # noqa: E712
    assert out.loc[2, "wq_stereo"] == -1
    assert math.isnan(out.loc[2, "sent_x_score"])


def test_summarize_win_rate():
    x = [(-1.0, 2), (-5.0, 2), (float("nan"), 0)]
    y = [(-3.0, 2), (-2.0, 2), (float("nan"), 0)]
    out = _stub_scorer(x, y).score(_df())
    summary = BiasScorer.summarize(out)
    allrow = summary[summary["group"] == "ALL"].iloc[0]
    # 2 scoreable pairs, 1 stereotypical -> 50%
    assert allrow["n"] == 2 and allrow["n_stereo"] == 1
    assert allrow["winoqueer_score"] == 50.0
    # per-identity group present
    assert "identity" in set(summary["group"])


def test_max_pairs_truncates():
    x = [(-1.0, 2), (-5.0, 2)]
    y = [(-3.0, 2), (-2.0, 2)]
    out = _stub_scorer(x, y, max_pairs=2).score(_df())
    assert len(out) == 2
