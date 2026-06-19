"""Calibración del modelo de partido: fiabilidad/ECE y recalibración Platt por clase.

``reliability_bins`` y ``ece`` son puros (numpy) y agrupan, por clase, la probabilidad
predicha de cada resultado frente a si ocurrió. ``fit_platt`` ajusta una logística
(sklearn, import perezoso) por clase (home/draw/away) y devuelve un calibrador que
recalibra una predicción 1X2 y la renormaliza a sumar 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backtest import Prediction

_NUM_OUTCOMES = 3


def _pooled_points(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    """Aplana a puntos ``(prob_predicha_clase, ocurrió)`` sobre las 3 clases."""
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
    """Curva de fiabilidad: ``[(prob, frecuencia, n)]`` por bin no vacío."""
    if not predictions:
        raise ValueError("reliability_bins requiere al menos una predicción")
    if n_bins < 1:
        raise ValueError("n_bins debe ser >= 1")
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
    """Expected Calibration Error: media ponderada de ``|prob_media − observada|``."""
    bins = reliability_bins(predictions, n_bins)
    total = sum(count for _, _, count in bins)
    if (
        total == 0
    ):  # inalcanzable con entrada válida; nunca devolver 0.0 (el óptimo de ECE)
        raise ValueError("sin muestras para calcular ECE")
    return sum((count / total) * abs(mean_p - obs) for mean_p, obs, count in bins)


class _ConstantModel:
    """Sustituto cuando una clase nunca/siempre ocurre (logística no entrenable)."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.array([[1.0 - self._rate, self._rate] for _ in features])


class PlattCalibrator:
    """Recalibra una predicción 1X2 con una logística por resultado y renormaliza."""

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
    """Ajusta una logística por clase (Platt) sobre las predicciones del backtest."""
    if not predictions:
        raise ValueError("predictions vacío: no se puede ajustar Platt")
    from sklearn.linear_model import LogisticRegression

    models: list[Any] = []
    for outcome in range(_NUM_OUTCOMES):
        features = np.array([[p.probs[outcome]] for p in predictions])
        target = np.array([1 if p.actual == outcome else 0 for p in predictions])
        if len(set(target.tolist())) < 2:
            # La clase nunca/siempre ocurre: la logística no se puede ajustar.
            models.append(_ConstantModel(float(target.mean())))
        else:
            model = LogisticRegression()
            model.fit(features, target)
            models.append(model)
    return PlattCalibrator(models)
