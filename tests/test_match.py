"""Tests for simulate_match: regulation, extra time, and penalties (knockout)."""

from __future__ import annotations

import numpy as np

from worldcup.models.base import MatchModel
from worldcup.rng import get_rng
from worldcup.simulation.match import simulate_match


class _FixedDraw(MatchModel):
    """Test model that forces a ``g-g`` draw in regulation."""

    def __init__(self, goals: int) -> None:
        n = goals + 2
        self._m = np.zeros((n, n))
        self._m[goals, goals] = 1.0

    def score_matrix(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> np.ndarray:
        return self._m


class _HomeWins2_0(MatchModel):
    """Test model that forces a 2-0 home win."""

    def score_matrix(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> np.ndarray:
        m = np.zeros((3, 3))
        m[2, 0] = 1.0
        return m


def test_group_match_returns_score_and_no_winner() -> None:
    r = simulate_match(_HomeWins2_0(), "A", "B", 1600.0, 1400.0, get_rng(1))
    assert (r.home_goals, r.away_goals) == (2, 0)
    assert r.winner is None


def test_knockout_decisive_in_regulation() -> None:
    r = simulate_match(
        _HomeWins2_0(), "A", "B", 1600.0, 1400.0, get_rng(1), knockout=True
    )
    assert r.winner == "A"


def test_knockout_draw_always_produces_a_winner() -> None:
    # 1-1 forced in regulation -> extra time/penalties -> there's ALWAYS a winner.
    for seed in range(6):
        r = simulate_match(
            _FixedDraw(1), "A", "B", 1500.0, 1500.0, get_rng(seed), knockout=True
        )
        assert r.winner in ("A", "B")


def test_penalties_decide_when_extra_time_is_scoreless() -> None:
    # ET total 0 -> extra time always 0-0 -> penalties; huge home Elo edge, home wins.
    r = simulate_match(
        _FixedDraw(0),
        "A",
        "B",
        3000.0,
        1000.0,
        get_rng(3),
        knockout=True,
        extra_time_total_goals=0.0,
    )
    assert r.winner == "A"


def test_simulate_match_is_reproducible() -> None:
    a = simulate_match(
        _FixedDraw(1), "A", "B", 1500.0, 1500.0, get_rng(7), knockout=True
    )
    b = simulate_match(
        _FixedDraw(1), "A", "B", 1500.0, 1500.0, get_rng(7), knockout=True
    )
    assert a == b
