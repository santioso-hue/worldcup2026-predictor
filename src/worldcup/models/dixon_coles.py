"""Dixon-Coles model: Elo -> expected goals -> corrected scoreline matrix.

Maps the Elo difference (adjusted for home advantage) to ``λ_home``/``λ_away``,
builds the independent Poisson matrix, and applies the Dixon-Coles low-score
correction (``ρ`` parameter). The Poisson pmf is computed with numpy + stdlib
(scipy would be overkill for a handful of points).
"""

from __future__ import annotations

from math import exp, factorial

import numpy as np

from worldcup.config import DixonColesConfig, EloConfig

from .base import MatchModel


def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    """``[P(0), …, P(n-1)]`` vector for a Poisson(``lam``) (numpy + stdlib)."""
    return np.array([exp(-lam) * lam**k / factorial(k) for k in range(n)])


class DixonColesModel(MatchModel):
    """Dixon-Coles match model (primary).

    Parameters
    ----------
    elo_cfg:
        Elo config: ``base_lambda``, ``elo_per_goal_denominator``, ``lambda_min/max``.
    dc_cfg:
        Dixon-Coles config: ``rho`` and ``max_goals``.
    """

    def __init__(self, elo_cfg: EloConfig, dc_cfg: DixonColesConfig) -> None:
        self._base = elo_cfg.base_lambda
        self._denom = elo_cfg.elo_per_goal_denominator
        self._lambda_min = elo_cfg.lambda_min
        self._lambda_max = elo_cfg.lambda_max
        self._rho = dc_cfg.rho
        self._max_goals = dc_cfg.max_goals

    def expected_goals(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> tuple[float, float]:
        """Expected goals ``(λ_home, λ_away)`` from the Elo difference.

        Home-advantage-adjusted difference ``d = (R_home + HA) − R_away``;
        ``λ_home = clip(base + d/denom)``, ``λ_away = clip(base − d/denom)``.
        """
        diff = (rating_home + home_advantage) - rating_away
        lam_home = self._clip(self._base + diff / self._denom)
        lam_away = self._clip(self._base - diff / self._denom)
        return lam_home, lam_away

    def _clip(self, value: float) -> float:
        return min(self._lambda_max, max(self._lambda_min, value))

    def score_matrix(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> np.ndarray:
        """``(max_goals+1)²`` matrix of P(home=i, away=j), normalized to 1."""
        lam_home, lam_away = self.expected_goals(
            rating_home, rating_away, home_advantage
        )
        n = self._max_goals + 1
        p_home = _poisson_pmf(lam_home, n)  # P(home scores i)
        p_away = _poisson_pmf(lam_away, n)  # P(away scores j)
        matrix = np.outer(p_home, p_away)  # independent Poisson

        # Dixon-Coles correction on the 4 low-score cells.
        rho = self._rho
        matrix[0, 0] *= 1.0 - lam_home * lam_away * rho
        matrix[0, 1] *= 1.0 + lam_home * rho
        matrix[1, 0] *= 1.0 + lam_away * rho
        matrix[1, 1] *= 1.0 - rho

        return matrix / matrix.sum()  # renormalize (correction + max_goals truncation)
