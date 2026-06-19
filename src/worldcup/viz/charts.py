"""Gráficos: preparación de datos (pura, testeable) + render matplotlib (fino).

Cada figura separa ``prepare_*`` (datos → estructura lista para plotear; puro, sin
matplotlib) de ``render_*`` (dibuja la ``Figure``). Aquí viven los ``prepare_*``; los
``render_*`` se añaden con sus smoke-tests. Toda la marca vive en ``theme.py``.
Orden de resultados 1X2: local, empate, visita.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingRow:
    """Una fila del ranking de campeón."""

    rank: int
    team: str
    prob: float
    delta: str  # "up" | "down" | "flat"


def prepare_champion_ranking(
    probs: dict[str, float],
    previous: dict[str, float] | None = None,
    *,
    top_n: int = 10,
    min_delta: float = 0.005,
) -> list[RankingRow]:
    """Top-N por P(título) desc (desempate por nombre), con delta vs snapshot previo.

    Delta ``up``/``down`` solo si el cambio supera ``min_delta`` (evita ruido). Sin
    snapshot previo para el equipo, queda ``flat``.
    """
    if top_n < 1:
        raise ValueError("top_n debe ser >= 1")
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    rows: list[RankingRow] = []
    for rank, (team, prob) in enumerate(ordered, start=1):
        delta = "flat"
        if previous is not None and team in previous:
            change = prob - previous[team]
            if change >= min_delta:
                delta = "up"
            elif change <= -min_delta:
                delta = "down"
        rows.append(RankingRow(rank=rank, team=team, prob=prob, delta=delta))
    return rows


@dataclass(frozen=True)
class BarSegment:
    """Un segmento de la barra 1X2."""

    label: str
    value: float
    role: str  # "home" | "draw" | "away"


def prepare_match_bar(
    home: str,
    away: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    *,
    tol: float = 1e-6,
) -> list[BarSegment]:
    """Segmentos local/empate/visita; falla si las probabilidades no suman 1."""
    total = p_home + p_draw + p_away
    if abs(total - 1.0) > tol:
        raise ValueError(f"las probabilidades 1X2 deben sumar 1 (suma={total:.6f})")
    return [
        BarSegment(label=home, value=p_home, role="home"),
        BarSegment(label="Empate", value=p_draw, role="draw"),
        BarSegment(label=away, value=p_away, role="away"),
    ]


@dataclass(frozen=True)
class HeatmapData:
    """Región mostrada del heatmap y la celda modal (goles local, visita)."""

    grid: np.ndarray
    mode: tuple[int, int]


def prepare_score_heatmap(
    score_matrix: np.ndarray,
    *,
    max_goals: int = 5,
    tol: float = 1e-6,
) -> HeatmapData:
    """Valida la matriz y la recorta a la región 0..max_goals para mostrar."""
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("score_matrix debe ser 2D")
    if not np.isfinite(arr).all():
        raise ValueError("score_matrix no puede contener NaN/inf")
    if (arr < 0).any():
        raise ValueError("score_matrix no puede tener probabilidades negativas")
    total = float(arr.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"score_matrix debe sumar 1 (suma={total:.6f})")
    grid = arr[: max_goals + 1, : max_goals + 1]
    flat = int(grid.argmax())
    mode = (flat // grid.shape[1], flat % grid.shape[1])
    return HeatmapData(grid=grid, mode=mode)


@dataclass(frozen=True)
class ReliabilityData:
    """Puntos de la curva de fiabilidad (predicho vs observado) y conteo por bin."""

    pred: list[float]
    observed: list[float]
    counts: list[int]


def prepare_reliability(bins: list[tuple[float, float, int]]) -> ReliabilityData:
    """De ``calibration.reliability_bins``: separa columnas; falla si está vacío."""
    if not bins:
        raise ValueError("sin bins de fiabilidad que dibujar")
    return ReliabilityData(
        pred=[b[0] for b in bins],
        observed=[b[1] for b in bins],
        counts=[b[2] for b in bins],
    )
