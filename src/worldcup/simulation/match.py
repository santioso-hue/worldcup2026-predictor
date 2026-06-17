"""Muestreo de un partido: reglamentario + prórroga + penales (eliminatoria).

Tiempo reglamentario: marcador muestreado de la matriz Dixon-Coles. Si un partido de
eliminatoria queda empatado, se juega prórroga (Poisson con ``extra_time_total_goals``
goles esperados totales, repartidos según los goles esperados de cada equipo); si sigue
empatado, penales (moneda ponderada por Elo). Toda la aleatoriedad pasa por el RNG
sembrado del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.elo import expected_score
from ..models.base import MatchModel, sample_scoreline


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Resultado de un partido; ``home_goals``/``away_goals`` son del reglamentario.

    ``winner`` es ``None`` en fase de grupos (el marcador decide W/D/L); en eliminatoria
    es siempre el equipo que avanza (tras prórroga/penales si hizo falta).
    """

    home: str
    away: str
    home_goals: int
    away_goals: int
    winner: str | None


def _expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    """Goles esperados (local, visitante) derivados de la matriz (model-agnostic)."""
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
    """Muestrea un partido y devuelve su :class:`MatchResult`.

    En fase de grupos (``knockout=False``) se devuelve el marcador reglamentario y
    ``winner=None``. En eliminatoria se resuelve siempre un ganador.
    """
    matrix = model.score_matrix(rating_home, rating_away, home_advantage)
    home_goals, away_goals = sample_scoreline(matrix, rng)

    if not knockout:
        return MatchResult(home, away, home_goals, away_goals, None)

    if home_goals != away_goals:
        winner = home if home_goals > away_goals else away
        return MatchResult(home, away, home_goals, away_goals, winner)

    # Empate reglamentario -> prórroga (reparto del total esperado por fuerza ofensiva).
    e_home, e_away = _expected_goals(matrix)
    total = e_home + e_away
    home_share = 0.5 if total <= 0.0 else e_home / total
    et_home = int(rng.poisson(extra_time_total_goals * home_share))
    et_away = int(rng.poisson(extra_time_total_goals * (1.0 - home_share)))
    if et_home != et_away:
        winner = home if et_home > et_away else away
        return MatchResult(home, away, home_goals, away_goals, winner)

    # Sigue empatado -> penales: P(gana local) por la logística de Elo.
    p_home = expected_score(rating_home, rating_away, home_advantage, elo_denominator)
    winner = home if rng.random() < p_home else away
    return MatchResult(home, away, home_goals, away_goals, winner)
