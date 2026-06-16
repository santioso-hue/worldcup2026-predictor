"""Interfaz estable de modelo de partido + helpers de marcador.

Cualquier modelo (Dixon-Coles primario, XGBoost alternativo) implementa
:class:`MatchModel.score_matrix` — la distribución conjunta de marcadores. De ahí se
derivan las probabilidades 1X2 (``outcome_proba``) y el muestreo de marcadores para la
simulación (``sample_scoreline``). El resto del proyecto depende solo de esta interfaz.

Convención de la matriz: ``score_matrix[i, j] = P(local marca i, visitante marca j)``;
filas = goles del local, columnas = goles del visitante.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Probabilidades 1X2 de un partido (suman 1)."""

    home_win: float
    draw: float
    away_win: float


def outcome_probabilities(score_matrix: np.ndarray) -> MatchOutcome:
    """Agrega la matriz de marcadores en probabilidades 1X2.

    Victoria local = triángulo inferior (``i > j``); empate = diagonal; victoria
    visitante = triángulo superior (``i < j``).
    """
    home_win = float(np.tril(score_matrix, -1).sum())
    away_win = float(np.triu(score_matrix, 1).sum())
    draw = float(np.trace(score_matrix))
    return MatchOutcome(home_win=home_win, draw=draw, away_win=away_win)


def sample_scoreline(
    score_matrix: np.ndarray, rng: np.random.Generator
) -> tuple[int, int]:
    """Muestrea un marcador ``(goles_local, goles_visitante)`` de la matriz.

    Usa el RNG sembrado del proyecto (:func:`worldcup.rng.get_rng`) para que sea
    reproducible. La matriz debe estar normalizada (sumar 1).
    """
    n_cols = score_matrix.shape[1]
    flat = score_matrix.ravel()
    idx = int(rng.choice(flat.size, p=flat))
    return idx // n_cols, idx % n_cols


class MatchModel(ABC):
    """Interfaz de modelo de partido. Los modelos implementan ``score_matrix``."""

    @abstractmethod
    def score_matrix(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> np.ndarray:
        """Distribución conjunta de marcadores; suma 1, forma ``(n+1, n+1)``."""

    def outcome_proba(
        self, rating_home: float, rating_away: float, home_advantage: float = 0.0
    ) -> MatchOutcome:
        """Probabilidades 1X2 derivadas de :meth:`score_matrix` (heredable)."""
        return outcome_probabilities(
            self.score_matrix(rating_home, rating_away, home_advantage)
        )
