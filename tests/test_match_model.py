"""Tests for the MatchModel interface and its helpers (base.py)."""

from __future__ import annotations

import numpy as np

from worldcup.models.base import (
    MatchModel,
    MatchOutcome,
    outcome_probabilities,
    sample_scoreline,
)
from worldcup.rng import get_rng

# 2x2 matrix: rows = home goals (i), columns = away goals (j).
# (0,0)=0.4 draw, (0,1)=0.2 away win, (1,0)=0.3 home win, (1,1)=0.1 draw.
M = np.array([[0.4, 0.2], [0.3, 0.1]])


def test_outcome_probabilities_partitions_matrix() -> None:
    out = outcome_probabilities(M)
    assert isinstance(out, MatchOutcome)
    assert out.home_win == 0.3  # i > j (lower triangle)
    assert out.away_win == 0.2  # i < j (upper triangle)
    assert abs(out.draw - 0.5) < 1e-12  # diagonal: 0.4 + 0.1
    assert abs((out.home_win + out.draw + out.away_win) - 1.0) < 1e-12


def test_sample_scoreline_reproducible_and_valid() -> None:
    a = sample_scoreline(M, get_rng(42))
    b = sample_scoreline(M, get_rng(42))
    assert a == b  # same seed -> same scoreline
    assert a[0] in (0, 1) and a[1] in (0, 1)


def test_sample_scoreline_picks_the_only_nonzero_cell() -> None:
    m = np.zeros((3, 3))
    m[2, 1] = 1.0  # all mass on home=2, away=1
    assert sample_scoreline(m, get_rng(7)) == (2, 1)


def test_outcome_proba_default_derives_from_score_matrix() -> None:
    class _Const(MatchModel):
        def score_matrix(
            self, rating_home: float, rating_away: float, home_advantage: float = 0.0
        ) -> np.ndarray:
            return M

    out = _Const().outcome_proba(1500.0, 1500.0)
    assert out.home_win == 0.3
    assert abs((out.home_win + out.draw + out.away_win) - 1.0) < 1e-12
