"""Tests de preparación de datos de gráficos (puro, sin matplotlib)."""

from __future__ import annotations

import numpy as np
import pytest

from worldcup.viz.charts import (
    prepare_champion_ranking,
    prepare_match_bar,
    prepare_reliability,
    prepare_score_heatmap,
    render_champion_ranking,
    render_match_bar,
    render_reliability,
    render_score_heatmap,
)


def test_champion_ranking_sorts_and_caps() -> None:
    probs = {"A": 0.1, "B": 0.4, "C": 0.25, "D": 0.25}
    rows = prepare_champion_ranking(probs, top_n=3)
    assert [r.team for r in rows] == ["B", "C", "D"]  # desc; C/D empatan -> por nombre
    assert [r.rank for r in rows] == [1, 2, 3]
    assert all(r.delta == "flat" for r in rows)  # sin previous


def test_champion_ranking_deltas() -> None:
    probs = {"A": 0.20, "B": 0.10, "C": 0.30}
    previous = {"A": 0.10, "B": 0.15, "C": 0.30}
    delta = {r.team: r.delta for r in prepare_champion_ranking(probs, previous)}
    assert delta == {"A": "up", "B": "down", "C": "flat"}


def test_champion_ranking_delta_threshold() -> None:
    rows = prepare_champion_ranking({"A": 0.101}, {"A": 0.100}, min_delta=0.005)
    assert rows[0].delta == "flat"  # cambio < min_delta no marca flecha


def test_match_bar_segments_and_sum_guard() -> None:
    seg = prepare_match_bar("Brasil", "Francia", 0.46, 0.27, 0.27)
    assert [s.role for s in seg] == ["home", "draw", "away"]
    assert seg[0].label == "Brasil" and seg[2].label == "Francia"
    with pytest.raises(ValueError):
        prepare_match_bar("A", "B", 0.5, 0.3, 0.3)  # suma 1.1


def test_score_heatmap_validates_and_finds_mode() -> None:
    matrix = np.zeros((9, 9))
    matrix[1, 1] = 0.4
    matrix[2, 1] = 0.6
    data = prepare_score_heatmap(matrix, max_goals=5)
    assert data.grid.shape == (6, 6)
    assert data.mode == (2, 1)  # celda más probable
    with pytest.raises(ValueError):
        prepare_score_heatmap(np.full((3, 3), 0.5))  # no suma 1


def test_score_heatmap_rejects_nan() -> None:
    # NaN derrota a las guardas por comparación (<0, suma): exigir finitud explícita.
    matrix = np.zeros((6, 6))
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError):
        prepare_score_heatmap(matrix)


def test_reliability_passthrough_and_empty() -> None:
    data = prepare_reliability([(0.2, 0.18, 50), (0.8, 0.83, 40)])
    assert data.pred == [0.2, 0.8]
    assert data.observed == [0.18, 0.83]
    assert data.counts == [50, 40]
    with pytest.raises(ValueError):
        prepare_reliability([])


def test_render_champion_ranking_smoke() -> None:
    rows = prepare_champion_ranking({"A": 0.5, "B": 0.3, "C": 0.2})
    fig = render_champion_ranking(rows, stamp="actualizado hoy")
    ax = fig.axes[0]
    assert len(ax.patches) == 3  # una barra por equipo
    assert ax.get_title() == "Probabilidad de campeón"


def test_render_match_bar_smoke() -> None:
    segments = prepare_match_bar("Brasil", "Francia", 0.46, 0.27, 0.27)
    fig = render_match_bar(segments)
    assert len(fig.axes[0].patches) == 3  # tres segmentos 1X2


def test_render_score_heatmap_smoke() -> None:
    matrix = np.zeros((6, 6))
    matrix[1, 1] = 0.6
    matrix[2, 1] = 0.4
    fig = render_score_heatmap(prepare_score_heatmap(matrix))
    assert len(fig.axes[0].images) == 1


def test_render_reliability_smoke() -> None:
    fig = render_reliability(prepare_reliability([(0.2, 0.18, 50), (0.8, 0.83, 40)]))
    ax = fig.axes[0]
    assert len(ax.lines) >= 1  # diagonal ideal
    assert len(ax.collections) >= 1  # scatter de puntos
