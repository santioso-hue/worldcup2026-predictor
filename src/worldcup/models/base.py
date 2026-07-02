"""Stable match-model interface plus scoreline helpers.

Any model (currently Dixon-Coles) implements :class:`MatchModel.score_matrix`
— the joint scoreline distribution. From that we derive 1X2 probabilities
(``outcome_proba``) and scoreline sampling for the simulation
(``sample_scoreline``). The rest of the project depends only on this
interface.

Matrix convention: ``score_matrix[i, j] = P(home scores i, away scores j)``;
rows = home goals, columns = away goals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """1X2 probabilities for a match (sum to 1)."""

    home_win: float
    draw: float
    away_win: float


def outcome_probabilities(score_matrix: np.ndarray) -> MatchOutcome:
    """Aggregate the scoreline matrix into 1X2 probabilities.

    Home win = lower triangle (``i > j``); draw = diagonal; away win = upper
    triangle (``i < j``).
    """
    home_win = float(np.tril(score_matrix, -1).sum())
    away_win = float(np.triu(score_matrix, 1).sum())
    draw = float(np.trace(score_matrix))
    return MatchOutcome(home_win=home_win, draw=draw, away_win=away_win)


def sample_scoreline(
    score_matrix: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """Sample a ``(home_goals, away_goals)`` scoreline from the matrix.

    Uses the project's seeded RNG (:func:`worldcup.rng.get_rng`) so results
    are reproducible. The matrix must be normalized (sum to 1).
    """
    n_cols = score_matrix.shape[1]
    flat = score_matrix.ravel()
    idx = int(rng.choice(flat.size, p=flat))
    return idx // n_cols, idx % n_cols


class MatchModel(ABC):
    """Match-model interface. Models implement ``score_matrix``."""

    @abstractmethod
    def score_matrix(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> np.ndarray:
        """Joint scoreline distribution; sums to 1, shape ``(n+1, n+1)``."""

    def outcome_proba(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> MatchOutcome:
        """1X2 probabilities derived from :meth:`score_matrix` (inheritable)."""
        return outcome_probabilities(
            self.score_matrix(rating_home, rating_away, home_advantage)
        )
