"""Gráficos: preparación de datos (pura, testeable) + render matplotlib (fino).

Cada figura separa ``prepare_*`` (datos → estructura lista para plotear; puro, sin
matplotlib) de ``render_*`` (dibuja la ``Figure``; matplotlib en import perezoso, sin
pyplot). Toda la marca vive en ``theme.py``. Orden 1X2: local, empate, visita.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .theme import LANDSCAPE, PORTRAIT, THEME, ExportSpec, Theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
    if max_goals < 1:
        raise ValueError("max_goals debe ser >= 1")
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


# --- render: matplotlib (import perezoso, API orientada a objetos, sin pyplot) ---

_DELTA_GLYPH = {"up": "▲", "down": "▼", "flat": ""}


def _new_figure(spec: ExportSpec, theme: Theme) -> tuple[Figure, Axes]:
    """Figure headless del tamaño del spec, con fondo de marca (sin estado global)."""
    from matplotlib.figure import Figure

    fig = Figure(
        figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi
    )
    fig.patch.set_facecolor(theme.background)
    ax = fig.subplots()
    ax.set_facecolor(theme.background)
    return fig, ax


def _strip_chrome(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _apply_font(fig: Figure, theme: Theme) -> None:
    """Fija la tipografía de marca en todo el texto de la figura (determinista)."""
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        if isinstance(artist, Text):
            artist.set_fontfamily(theme.font_family)


def _draw_ranking_axes(
    ax: Axes, rows: list[RankingRow], theme: Theme, *, title: str
) -> None:
    """Dibuja el ranking sobre ``ax`` (compartido por render y animación)."""
    positions = list(range(len(rows)))
    ax.barh(positions, [r.prob * 100 for r in rows], color=theme.accent)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.team for r in rows], color=theme.text_primary)
    ax.invert_yaxis()  # rank 1 arriba
    color_by_delta = {"up": theme.up, "down": theme.down, "flat": theme.text_primary}
    for pos, row in zip(positions, rows, strict=True):
        ax.text(
            row.prob * 100,
            pos,
            f"  {row.prob * 100:.1f}% {_DELTA_GLYPH[row.delta]}",
            va="center",
            color=color_by_delta[row.delta],
            fontsize=theme.value_size,
        )
    ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size, pad=20)
    largest = max((r.prob * 100 for r in rows), default=1.0)
    ax.set_xlim(0, largest * 1.18)
    ax.set_xticks([])
    _strip_chrome(ax)


def render_champion_ranking(
    rows: list[RankingRow],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = PORTRAIT,
    title: str = "Probabilidad de campeón",
    stamp: str | None = None,
) -> Figure:
    """Barras horizontales de P(título), rank 1 arriba, con valor y flecha de delta."""
    fig, ax = _new_figure(spec, theme)
    _draw_ranking_axes(ax, rows, theme, title=title)
    if stamp is not None:
        fig.text(
            0.5,
            0.02,
            stamp,
            ha="center",
            color=theme.text_muted,
            fontsize=theme.stamp_size,
        )
    _apply_font(fig, theme)
    return fig


def animate_ranking(
    snapshots: list[dict[str, float]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = PORTRAIT,
    title: str = "Probabilidad de campeón",
    top_n: int = 10,
    name: str = "ranking",
    fps: int = 2,
    fmt: str = "mp4",
    outdir: Path | str = Path("outputs/videos"),
) -> Path:
    """Anima la evolución del ranking de campeón entre snapshots (el 'drumroll')."""
    from .export import save_animation

    if not snapshots:
        raise ValueError("se requiere al menos un snapshot")
    frames = [prepare_champion_ranking(s, top_n=top_n) for s in snapshots]
    fig, ax = _new_figure(spec, theme)

    def update(index: int) -> None:
        ax.clear()
        ax.set_facecolor(theme.background)
        _draw_ranking_axes(ax, frames[index], theme, title=title)
        _apply_font(fig, theme)

    return save_animation(
        fig, update, len(snapshots), name, fps=fps, fmt=fmt, outdir=outdir
    )


def render_match_bar(
    segments: list[BarSegment],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str | None = None,
) -> Figure:
    """Barra 1X2 apilada con el porcentaje rotulado en cada segmento."""
    fig, ax = _new_figure(spec, theme)
    color_by_role = {"home": theme.accent, "draw": theme.draw, "away": theme.away}
    left = 0.0
    for seg in segments:
        ax.barh([0], [seg.value], left=left, color=color_by_role[seg.role])
        ax.text(
            left + seg.value / 2,
            0,
            f"{seg.value * 100:.0f}%",
            ha="center",
            va="center",
            color=theme.background,
            fontsize=theme.value_size,
        )
        left += seg.value
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size, pad=16)
    _strip_chrome(ax)
    _apply_font(fig, theme)
    return fig


def render_score_heatmap(
    data: HeatmapData,
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str | None = "Marcadores más probables",
) -> Figure:
    """Heatmap de la matriz de marcadores con la celda modal resaltada."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    fig, ax = _new_figure(spec, theme)
    cmap = LinearSegmentedColormap.from_list("heat", [theme.background, theme.heat])
    ax.imshow(data.grid, cmap=cmap, origin="upper")
    mode_row, mode_col = data.mode
    ax.add_patch(
        Rectangle(
            (mode_col - 0.5, mode_row - 0.5),
            1,
            1,
            fill=False,
            edgecolor=theme.text_primary,
            linewidth=2,
        )
    )
    ax.set_xlabel("Goles visita", color=theme.text_primary)
    ax.set_ylabel("Goles local", color=theme.text_primary)
    ax.set_xticks(range(data.grid.shape[1]))
    ax.set_yticks(range(data.grid.shape[0]))
    ax.tick_params(colors=theme.text_primary)
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size, pad=16)
    _apply_font(fig, theme)
    return fig


def render_reliability(
    data: ReliabilityData,
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str | None = "Calibración (fiabilidad)",
) -> Figure:
    """Curva de fiabilidad: puntos predicho-vs-observado sobre la diagonal ideal."""
    fig, ax = _new_figure(spec, theme)
    ax.plot([0, 1], [0, 1], linestyle="--", color=theme.text_muted)
    ax.scatter(data.pred, data.observed, color=theme.accent)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidad predicha", color=theme.text_primary)
    ax.set_ylabel("Frecuencia observada", color=theme.text_primary)
    ax.tick_params(colors=theme.text_primary)
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size, pad=16)
    _apply_font(fig, theme)
    return fig


@dataclass(frozen=True)
class GroupRow:
    """Un equipo de un grupo con su probabilidad de avanzar."""

    team: str
    p_advance: float


def prepare_group_table(
    groups: dict[str, list[str]],
    probabilities: dict[str, dict[str, float]],
    *,
    advance_key: str = "advance",
) -> dict[str, list[GroupRow]]:
    """Por grupo, equipos ordenados por P(avance) desc; falla si falta un equipo."""
    if not groups:
        raise ValueError("groups vacío")
    table: dict[str, list[GroupRow]] = {}
    for letter, teams in groups.items():
        rows: list[GroupRow] = []
        for team in teams:
            probs = probabilities.get(team)
            if probs is None or advance_key not in probs:
                raise ValueError(f"falta P({advance_key}) para {team!r}")
            rows.append(GroupRow(team=team, p_advance=probs[advance_key]))
        rows.sort(key=lambda r: (-r.p_advance, r.team))
        table[letter] = rows
    return table


def render_group_table(
    table: dict[str, list[GroupRow]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str = "Probabilidad de avance",
) -> Figure:
    """Cuadrícula de paneles (uno por grupo) con barras de P(avance) por equipo."""
    from matplotlib.figure import Figure

    letters = sorted(table)
    cols = min(4, len(letters))
    n_rows = -(-len(letters) // cols)  # techo
    fig = Figure(
        figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi
    )
    fig.patch.set_facecolor(theme.background)
    axes = fig.subplots(n_rows, cols, squeeze=False)
    panels = list(axes.flat)
    for ax, letter in zip(panels, letters, strict=False):
        group_rows = table[letter]
        ax.set_facecolor(theme.background)
        ax.barh(
            range(len(group_rows)),
            [r.p_advance * 100 for r in group_rows],
            color=theme.accent,
        )
        ax.set_yticks(range(len(group_rows)))
        ax.set_yticklabels(
            [r.team for r in group_rows],
            color=theme.text_primary,
            fontsize=theme.stamp_size,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_title(
            f"Grupo {letter}", color=theme.text_primary, fontsize=theme.label_size
        )
        _strip_chrome(ax)
    for ax in panels[len(letters) :]:  # ocultar paneles sobrantes
        ax.set_visible(False)
    fig.suptitle(title, color=theme.text_primary, fontsize=theme.title_size)
    _apply_font(fig, theme)
    return fig
