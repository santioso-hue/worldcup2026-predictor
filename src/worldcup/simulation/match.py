"""Match sampling: regulation + extra time + penalties (knockout stage).

Regulation: scoreline sampled from the Dixon-Coles matrix. If a knockout match ends
level, extra time is played (Poisson with ``extra_time_total_goals`` expected goals
total, split by each team's expected-goals share); if still level, penalties (an
Elo-weighted coin flip). All randomness goes through the project's seeded RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.elo import expected_score
from ..models.base import MatchModel, sample_scoreline


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Match result; ``home_goals``/``away_goals`` are the regulation scoreline.

    ``winner`` is ``None`` in the group stage (the scoreline decides W/D/L); in the
    knockout stage it's always the team that advances (after extra time/penalties
    if needed).
    """

    home: str
    away: str
    home_goals: int
    away_goals: int
    winner: str | None


def _expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    """Expected goals (home, away) derived from the matrix (model-agnostic)."""
    idx = np.arange(matrix.shape[0])
    e_home = float((idx[:, None] * matrix).sum())
    e_away = float((idx[None, :] * matrix).sum())
    return e_home, e_away


def simulate_match(
    model: MatchModel,
    home: str,
    away: str,
    rating_home: float,
    rating_away: float,
    rng: np.random.Generator,
    *,
    home_advantage: float = 0.0,
    knockout: bool = False,
    extra_time_total_goals: float = 0.8,
    elo_denominator: float = 400.0,
) -> MatchResult:
    """Sample a match and return its :class:`MatchResult`.

    In the group stage (``knockout=False``) this returns the regulation scoreline
    and ``winner=None``. In the knockout stage a winner is always resolved.
    """
    matrix = model.score_matrix(rating_home, rating_away, home_advantage)
    home_goals, away_goals = sample_scoreline(matrix, rng)

    if not knockout:
        return MatchResult(home, away, home_goals, away_goals, None)

    if home_goals != away_goals:
        winner = home if home_goals > away_goals else away
        return MatchResult(home, away, home_goals, away_goals, winner)

    # Level after regulation -> extra time (split expected total by offensive strength).
    e_home, e_away = _expected_goals(matrix)
    total = e_home + e_away
    home_share = 0.5 if total <= 0.0 else e_home / total
    et_home = int(rng.poisson(extra_time_total_goals * home_share))
    et_away = int(rng.poisson(extra_time_total_goals * (1.0 - home_share)))
    if et_home != et_away:
        winner = home if et_home > et_away else away
        return MatchResult(home, away, home_goals, away_goals, winner)

    # Still level -> penalties: P(home wins) from the Elo logistic.
    p_home = expected_score(rating_home, rating_away, home_advantage, elo_denominator)
    winner = home if rng.random() < p_home else away
    return MatchResult(home, away, home_goals, away_goals, winner)
