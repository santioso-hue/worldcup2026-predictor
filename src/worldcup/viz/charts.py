"""Charts: pure data prep (testable) + thin matplotlib rendering.

Each figure splits ``prepare_*`` (data -> plot-ready structure; pure, no matplotlib)
from ``render_*`` (draws the ``Figure``; lazy-imports matplotlib, no pyplot). All
branding lives in ``theme.py``. 1X2 order: home, draw, away.
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
    """One row of the champion ranking."""

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
    """Top-N by P(title) desc (ties broken by name), with delta vs previous snapshot.

    Delta is ``up``/``down`` only if the change exceeds ``min_delta`` (avoids noise).
    Teams missing from the previous snapshot get ``flat``.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
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
    """One segment of the 1X2 bar."""

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
    """Home/draw/away segments; fails if the probabilities don't sum to 1."""
    total = p_home + p_draw + p_away
    if abs(total - 1.0) > tol:
        raise ValueError(f"1X2 probabilities must sum to 1 (sum={total:.6f})")
    return [
        BarSegment(label=home, value=p_home, role="home"),
        BarSegment(label="Draw", value=p_draw, role="draw"),
        BarSegment(label=away, value=p_away, role="away"),
    ]


@dataclass(frozen=True)
class HeatmapData:
    """Displayed region of the heatmap and its modal cell (home goals, away goals)."""

    grid: np.ndarray
    mode: tuple[int, int]


def prepare_score_heatmap(
    score_matrix: np.ndarray,
    *,
    max_goals: int = 5,
    tol: float = 1e-6,
) -> HeatmapData:
    """Validate the matrix and crop it to the 0..max_goals region for display."""
    if max_goals < 1:
        raise ValueError("max_goals must be >= 1")
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("score_matrix must be 2D")
    if not np.isfinite(arr).all():
        raise ValueError("score_matrix cannot contain NaN/inf")
    if (arr < 0).any():
        raise ValueError("score_matrix cannot have negative probabilities")
    total = float(arr.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"score_matrix must sum to 1 (sum={total:.6f})")
    grid = arr[: max_goals + 1, : max_goals + 1]
    flat = int(grid.argmax())
    mode = (flat // grid.shape[1], flat % grid.shape[1])
    return HeatmapData(grid=grid, mode=mode)


@dataclass(frozen=True)
class ReliabilityData:
    """Reliability curve points (predicted vs observed) and per-bin counts."""

    pred: list[float]
    observed: list[float]
    counts: list[int]


def prepare_reliability(bins: list[tuple[float, float, int]]) -> ReliabilityData:
    """From ``calibration.reliability_bins``: split into columns; fails if empty."""
    if not bins:
        raise ValueError("no reliability bins to plot")
    return ReliabilityData(
        pred=[b[0] for b in bins],
        observed=[b[1] for b in bins],
        counts=[b[2] for b in bins],
    )


# --- render: matplotlib (lazy import, object-oriented API, no pyplot) ---

_DELTA_GLYPH = {"up": "▲", "down": "▼", "flat": ""}


def _new_figure(spec: ExportSpec, theme: Theme) -> tuple[Figure, Axes]:
    """Headless Figure sized per spec, with the brand background (no global state)."""
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
    """Set the brand font on every text element in the figure (deterministic)."""
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        if isinstance(artist, Text):
            artist.set_fontfamily(theme.font_family)


def _draw_ranking_axes(
    ax: Axes, rows: list[RankingRow], theme: Theme, *, title: str
) -> None:
    """Draw the ranking onto ``ax`` (shared by render and animation)."""
    positions = list(range(len(rows)))
    ax.barh(positions, [r.prob * 100 for r in rows], color=theme.accent)
    ax.set_yticks(positions)
    ax.set_yticklabels([r.team for r in rows], color=theme.text_primary)
    ax.invert_yaxis()  # rank 1 on top
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
    title: str = "Championship probability",
    stamp: str | None = None,
) -> Figure:
    """Horizontal bars of P(title), rank 1 on top, with value and delta arrow."""
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
    title: str = "Championship probability",
    top_n: int = 10,
    name: str = "ranking",
    fps: int = 2,
    fmt: str = "mp4",
    outdir: Path | str = Path("outputs/videos"),
) -> Path:
    """Animate the champion ranking's evolution across snapshots."""
    from .export import save_animation

    if not snapshots:
        raise ValueError("at least one snapshot is required")
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
    """Stacked 1X2 bar with the percentage labeled on each segment."""
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
    title: str | None = "Most likely scorelines",
) -> Figure:
    """Heatmap of the score matrix with the modal cell highlighted."""
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
    ax.set_xlabel("Away goals", color=theme.text_primary)
    ax.set_ylabel("Home goals", color=theme.text_primary)
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
    title: str | None = "Calibration (reliability)",
) -> Figure:
    """Reliability curve: predicted-vs-observed points against the ideal diagonal."""
    fig, ax = _new_figure(spec, theme)
    ax.plot([0, 1], [0, 1], linestyle="--", color=theme.text_muted)
    ax.scatter(data.pred, data.observed, color=theme.accent)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", color=theme.text_primary)
    ax.set_ylabel("Observed frequency", color=theme.text_primary)
    ax.tick_params(colors=theme.text_primary)
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size, pad=16)
    _apply_font(fig, theme)
    return fig


@dataclass(frozen=True)
class GroupRow:
    """A team in a group with its probability of advancing."""

    team: str
    p_advance: float


def prepare_group_table(
    groups: dict[str, list[str]],
    probabilities: dict[str, dict[str, float]],
    *,
    advance_key: str = "advance",
) -> dict[str, list[GroupRow]]:
    """Per group, teams sorted by P(advance) desc; fails if a team is missing."""
    if not groups:
        raise ValueError("groups is empty")
    table: dict[str, list[GroupRow]] = {}
    for letter, teams in groups.items():
        rows: list[GroupRow] = []
        for team in teams:
            probs = probabilities.get(team)
            if probs is None or advance_key not in probs:
                raise ValueError(f"missing P({advance_key}) for {team!r}")
            rows.append(GroupRow(team=team, p_advance=probs[advance_key]))
        rows.sort(key=lambda r: (-r.p_advance, r.team))
        table[letter] = rows
    return table


def render_group_table(
    table: dict[str, list[GroupRow]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str = "Advance probability",
) -> Figure:
    """Grid of panels (one per group) with P(advance) bars per team."""
    from matplotlib.figure import Figure

    letters = sorted(table)
    cols = min(4, len(letters))
    n_rows = -(-len(letters) // cols)  # ceiling division
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
            f"Group {letter}", color=theme.text_primary, fontsize=theme.label_size
        )
        _strip_chrome(ax)
    for ax in panels[len(letters) :]:  # hide leftover panels
        ax.set_visible(False)
    fig.suptitle(title, color=theme.text_primary, fontsize=theme.title_size)
    _apply_font(fig, theme)
    return fig


_ROUND_LABELS = {
    "advance": "Advances",
    "round_of_16": "Round of 16",
    "quarter_finals": "Quarterfinals",
    "semi_finals": "Semifinals",
    "final": "Final",
    "champion": "Champion",
}


@dataclass(frozen=True)
class TeamRound:
    """A team's probability of reaching a round, with a readable label."""

    label: str
    prob: float


def prepare_team_detail(
    probabilities: dict[str, dict[str, float]], team: str
) -> list[TeamRound]:
    """Per-round probabilities for a team, in canonical order; fails if unknown."""
    probs = probabilities.get(team)
    if probs is None:
        raise ValueError(f"unknown team: {team!r}")
    return [
        TeamRound(label=label, prob=probs[key])
        for key, label in _ROUND_LABELS.items()
        if key in probs
    ]
