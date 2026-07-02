"""Walk-forward backtest for the match model (1X2) plus scoring metrics.

Each historical match is predicted with the MODEL THAT SHIPS: ``fit_elo`` refit over a
recent window anchored to the match date (true recency), compared against the actual
result. No peeking: only prior matches. Pure and deterministic (no RNG).
Outcome order: 0=home, 1=draw, 2=away.
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
    """A single 1X2 prediction with its actual result (0=home, 1=draw, 2=away)."""

    probs: tuple[float, float, float]
    actual: int


@dataclass(frozen=True, slots=True)
class Metrics:
    """Backtest scoring metrics (lower is better except accuracy)."""

    log_loss: float
    brier: float
    rps: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Walk-forward predictions plus their aggregate metrics."""

    predictions: list[Prediction]
    metrics: Metrics


def _arrays(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("no predictions to score")
    probs = np.array([p.probs for p in predictions], dtype=float)
    actuals = np.array([p.actual for p in predictions], dtype=int)
    return probs, actuals


def _one_hot(probs: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actuals)), actuals] = 1.0
    return onehot


def log_loss(predictions: list[Prediction]) -> float:
    """Multiclass log-loss (clipped to avoid ``log(0)``)."""
    probs, actuals = _arrays(predictions)
    chosen = probs[np.arange(len(actuals)), actuals]
    return float(-np.log(np.clip(chosen, _EPS, 1.0)).mean())


def brier(predictions: list[Prediction]) -> float:
    """Multiclass Brier score: mean of ``Σ_c (p_c − y_c)²``."""
    probs, actuals = _arrays(predictions)
    onehot = _one_hot(probs, actuals)
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def rps(predictions: list[Prediction]) -> float:
    """Ranked Probability Score (respects the home>draw>away ordering)."""
    probs, actuals = _arrays(predictions)
    onehot = _one_hot(probs, actuals)
    cum_diff = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float((cum_diff**2).sum(axis=1).mean() / (probs.shape[1] - 1))


def accuracy(predictions: list[Prediction]) -> float:
    """Fraction of matches where the most likely outcome (argmax) was correct."""
    probs, actuals = _arrays(predictions)
    return float((probs.argmax(axis=1) == actuals).mean())


def compute_metrics(predictions: list[Prediction]) -> Metrics:
    """Compute all 4 metrics for a set of predictions."""
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
    """Walk-forward: predict each match (after burn-in) with the recency-true model.

    For the match at position ``idx``, ``fit_elo`` is refit over the window
    ``[date − history_window_days, date)`` (via ``bisect``); the 1X2 is predicted and
    logged alongside the actual result. No peeking.

    Raises
    ------
    ValueError
        If no matches are left to evaluate (``burn_in_matches`` >= match count).
    """
    if burn_in_matches < 0:
        raise ValueError("burn_in_matches must be >= 0")
    if history_window_days <= 0:
        raise ValueError("history_window_days must be > 0")

    ordered = sorted(history, key=lambda m: m.date)
    dates = [m.date for m in ordered]
    window = timedelta(days=history_window_days)

    predictions: list[Prediction] = []
    for idx in range(burn_in_matches, len(ordered)):
        match = ordered[idx]
        lo = bisect_left(dates, match.date - window)
        hi = bisect_left(dates, match.date)  # [., M.date): excludes SAME-day matches
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
        raise ValueError("no matches to evaluate (burn_in_matches is too high)")
    return BacktestResult(predictions=predictions, metrics=compute_metrics(predictions))
