"""Match model calibration: reliability/ECE and per-class Platt recalibration.

``reliability_bins`` and ``ece`` are pure numpy and bucket, per class, the
predicted probability of each outcome against whether it happened. ``fit_platt``
fits a logistic regression (sklearn, lazy import) per class (home/draw/away) and
returns a calibrator that recalibrates a 1X2 prediction and renormalizes it to
sum to 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backtest import Prediction

_NUM_OUTCOMES = 3


def _pooled_points(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to ``(predicted_class_prob, occurred)`` points across the 3 classes."""
    probs: list[float] = []
    occurred: list[float] = []
    for pred in predictions:
        for outcome in range(_NUM_OUTCOMES):
            probs.append(pred.probs[outcome])
            occurred.append(1.0 if pred.actual == outcome else 0.0)
    return np.array(probs), np.array(occurred)


def reliability_bins(
    predictions: list[Prediction], n_bins: int
) -> list[tuple[float, float, int]]:
    """Reliability curve: ``[(prob, frequency, n)]`` for each non-empty bin."""
    if not predictions:
        raise ValueError("reliability_bins requires at least one prediction")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    probs, occurred = _pooled_points(predictions)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if mask.any():
            bins.append(
                (
                    float(probs[mask].mean()),
                    float(occurred[mask].mean()),
                    int(mask.sum()),
                )
            )
    return bins


def ece(predictions: list[Prediction], n_bins: int) -> float:
    """Expected Calibration Error: weighted mean of ``|mean_prob - observed|``."""
    bins = reliability_bins(predictions, n_bins)
    total = sum(count for _, _, count in bins)
    if total == 0:  # unreachable with valid input; never return 0.0 (ECE's optimum)
        raise ValueError("no samples to compute ECE")
    return sum((count / total) * abs(mean_p - obs) for mean_p, obs, count in bins)


class _ConstantModel:
    """Stand-in for when a class never/always occurs (logistic regression can't fit)."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.array([[1.0 - self._rate, self._rate] for _ in features])


class PlattCalibrator:
    """Recalibrates a 1X2 prediction with a per-outcome logistic and renormalizes."""

    def __init__(self, models: list[Any]) -> None:
        self._models = models

    def calibrate(
        self, probs: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        raw = [
            float(self._models[c].predict_proba([[probs[c]]])[0][1])
            for c in range(_NUM_OUTCOMES)
        ]
        total = sum(raw)
        if total <= 0.0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (raw[0] / total, raw[1] / total, raw[2] / total)


def fit_platt(predictions: list[Prediction]) -> PlattCalibrator:
    """Fit one logistic regression per class (Platt scaling) on backtest predictions."""
    if not predictions:
        raise ValueError("predictions is empty: cannot fit Platt scaling")
    from sklearn.linear_model import LogisticRegression

    models: list[Any] = []
    for outcome in range(_NUM_OUTCOMES):
        features = np.array([[p.probs[outcome]] for p in predictions])
        target = np.array([1 if p.actual == outcome else 0 for p in predictions])
        if len(set(target.tolist())) < 2:
            # Class never/always occurs: logistic regression can't fit.
            models.append(_ConstantModel(float(target.mean())))
        else:
            model = LogisticRegression()
            model.fit(features, target)
            models.append(model)
    return PlattCalibrator(models)
