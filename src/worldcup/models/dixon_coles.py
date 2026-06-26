"""Modelo Dixon-Coles: Elo -> goles esperados -> matriz de marcadores corregida.

Mapea la diferencia de Elo (ajustada por localía) a ``λ_home``/``λ_away``, construye la
matriz Poisson independiente y aplica la corrección de marcador bajo de Dixon-Coles
(parámetro ``ρ``). La pmf de Poisson se calcula con numpy + stdlib (no scipy para una
pmf de pocos puntos).
"""

from __future__ import annotations

from math import exp, factorial

import numpy as np

from worldcup.config import DixonColesConfig, EloConfig

from .base import MatchModel


def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    """Vector ``[P(0), …, P(n-1)]`` de una Poisson(``lam``) (numpy + stdlib)."""
    return np.array([exp(-lam) * lam**k / factorial(k) for k in range(n)])


class DixonColesModel(MatchModel):
    """Modelo de partido Dixon-Coles (primario).

    Parameters
    ----------
    elo_cfg:
        Config Elo: ``base_lambda``, ``elo_per_goal_denominator``, ``lambda_min/max``.
    dc_cfg:
        Config Dixon-Coles: ``rho`` y ``max_goals``.
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
        """Goles esperados ``(λ_home, λ_away)`` desde la diferencia de Elo.

        Diferencia ajustada por localía ``d = (R_home + HA) − R_away``;
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
        """Matriz ``(max_goals+1)²`` de P(local=i, visita=j), normalizada a 1."""
        lam_home, lam_away = self.expected_goals(
            rating_home, rating_away, home_advantage
        )
        n = self._max_goals + 1
        p_home = _poisson_pmf(lam_home, n)  # P(local marca i)
        p_away = _poisson_pmf(lam_away, n)  # P(visita marca j)
        matrix = np.outer(p_home, p_away)  # Poisson independiente

        # Corrección Dixon-Coles en las 4 celdas de marcador bajo.
        rho = self._rho
        matrix[0, 0] *= 1.0 - lam_home * lam_away * rho
        matrix[0, 1] *= 1.0 + lam_home * rho
        matrix[1, 0] *= 1.0 + lam_away * rho
        matrix[1, 1] *= 1.0 - rho

        return matrix / matrix.sum()  # renormaliza (corrección + truncado a max_goals)
