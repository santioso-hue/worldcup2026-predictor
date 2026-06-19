"""Backtest walk-forward del modelo de partido (1X2) + métricas de puntuación.

Para cada partido histórico se predice con el MODELO QUE SE PUBLICA: ``fit_elo`` sobre
una ventana reciente anclada a la fecha del partido (recencia fiel), y se compara con el
resultado real. Sin peeking: solo partidos previos. Puras y deterministas (sin RNG).
Orden de resultados: 0=local, 1=empate, 2=visita.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from ..config import EloConfig
from ..data.historical import HistoricalMatch
from ..features.elo import fit_elo
from ..models.base import MatchModel

_EPS = 1e-15


@dataclass(frozen=True, slots=True)
class Prediction:
    """Una predicción 1X2 con su resultado real (0=local, 1=empate, 2=visita)."""

    probs: tuple[float, float, float]
    actual: int


@dataclass(frozen=True, slots=True)
class Metrics:
    """Métricas de puntuación del backtest (todas: menor es mejor salvo accuracy)."""

    log_loss: float
    brier: float
    rps: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Predicciones del walk-forward + sus métricas agregadas."""

    predictions: list[Prediction]
    metrics: Metrics


def _arrays(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("no hay predicciones que puntuar")
    probs = np.array([p.probs for p in predictions], dtype=float)
    actuals = np.array([p.actual for p in predictions], dtype=int)
    return probs, actuals


def _one_hot(probs: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actuals)), actuals] = 1.0
    return onehot


def log_loss(predictions: list[Prediction]) -> float:
    """Log-loss multiclase (con clip para evitar ``log(0)``)."""
    probs, actuals = _arrays(predictions)
    chosen = probs[np.arange(len(actuals)), actuals]
    return float(-np.log(np.clip(chosen, _EPS, 1.0)).mean())


def brier(predictions: list[Prediction]) -> float:
    """Brier multiclase: media de ``Σ_c (p_c − y_c)²``."""
    probs, actuals = _arrays(predictions)
    onehot = _one_hot(probs, actuals)
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def rps(predictions: list[Prediction]) -> float:
    """Ranked Probability Score (respeta el orden local>empate>visita)."""
    probs, actuals = _arrays(predictions)
    onehot = _one_hot(probs, actuals)
    cum_diff = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float((cum_diff**2).sum(axis=1).mean() / (probs.shape[1] - 1))


def accuracy(predictions: list[Prediction]) -> float:
    """Fracción de partidos cuyo resultado más probable (argmax) acertó."""
    probs, actuals = _arrays(predictions)
    return float((probs.argmax(axis=1) == actuals).mean())


def compute_metrics(predictions: list[Prediction]) -> Metrics:
    """Calcula las 4 métricas de un conjunto de predicciones."""
    return Metrics(
        log_loss=log_loss(predictions),
        brier=brier(predictions),
        rps=rps(predictions),
        accuracy=accuracy(predictions),
    )


def _actual_outcome(match: HistoricalMatch) -> int:
    if match.home_score > match.away_score:
        return 0
    if match.home_score == match.away_score:
        return 1
    return 2


def backtest(
    history: list[HistoricalMatch],
    model: MatchModel,
    elo_cfg: EloConfig,
    *,
    burn_in_matches: int,
    history_window_days: int,
) -> BacktestResult:
    """Walk-forward: predice cada partido (tras el burn-in) con el modelo recency-fiel.

    Para el partido en posición ``idx`` se reajusta ``fit_elo`` sobre la ventana
    ``[date − history_window_days, date)`` (vía ``bisect``); se predice el 1X2 y se
    registra junto al resultado real. Sin peeking.

    Raises
    ------
    ValueError
        Si no quedan partidos para evaluar (``burn_in_matches`` >= nº de partidos).
    """
    if burn_in_matches < 0:
        raise ValueError("burn_in_matches debe ser >= 0")
    if history_window_days <= 0:
        raise ValueError("history_window_days debe ser > 0")

    ordered = sorted(history, key=lambda m: m.date)
    dates = [m.date for m in ordered]
    window = timedelta(days=history_window_days)

    predictions: list[Prediction] = []
    for idx in range(burn_in_matches, len(ordered)):
        match = ordered[idx]
        lo = bisect_left(dates, match.date - window)
        hi = bisect_left(dates, match.date)  # ventana [., M.date): excluye el MISMO día
        prior = ordered[lo:hi]
        ratings = fit_elo(prior, elo_cfg, reference_date=match.date)
        rating_home = ratings.get(match.home_team, elo_cfg.initial_rating)
        rating_away = ratings.get(match.away_team, elo_cfg.initial_rating)
        home_adv = 0.0 if match.neutral else elo_cfg.home_advantage
        outcome = model.outcome_proba(rating_home, rating_away, home_adv)
        predictions.append(
            Prediction(
                probs=(outcome.home_win, outcome.draw, outcome.away_win),
                actual=_actual_outcome(match),
            )
        )

    if not predictions:
        raise ValueError(
            "no hay partidos para evaluar (burn_in_matches demasiado alto)"
        )
    return BacktestResult(predictions=predictions, metrics=compute_metrics(predictions))
