"""Knockout bracket: layout (pure) + matplotlib render.

``prepare_bracket`` positions each match into columns by round (pure, testable);
``render_bracket`` draws the boxes and connectors, bolding the winner in resolved
matchups. It takes the round structure as given: deriving matchups from results is
the pipeline's job, not ``viz``'s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .theme import LANDSCAPE, THEME, ExportSpec, Theme

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BOX_W = 0.7
_BOX_H = 0.7


@dataclass(frozen=True)
class BracketMatch:
    """A matchup: teams (``None`` = not yet decided) and the winner, if played."""

    home: str | None
    away: str | None
    winner: str | None = None


@dataclass(frozen=True)
class PositionedMatch:
    """A match with its layout position (column = round, ``y`` = centered row)."""

    match: BracketMatch
    column: int
    y: float


def prepare_bracket(rounds: list[list[BracketMatch]]) -> list[list[PositionedMatch]]:
    """Position each match: column = round; ``y`` = midpoint between its two children.

    Raises
    ------
    ValueError
        If the bracket is empty or a round doesn't have half the matches of the
        previous one.
    """
    if not rounds:
        raise ValueError("empty bracket")
    positioned: list[list[PositionedMatch]] = []
    for col, matches in enumerate(rounds):
        if col == 0:
            positioned.append(
                [PositionedMatch(m, 0, float(i)) for i, m in enumerate(matches)]
            )
            continue
        prev = positioned[col - 1]
        if len(matches) * 2 != len(prev):
            raise ValueError(
                f"round {col} must have half the matches of the previous round"
            )
        positioned.append(
            [
                PositionedMatch(m, col, (prev[2 * i].y + prev[2 * i + 1].y) / 2)
                for i, m in enumerate(matches)
            ]
        )
    return positioned


def _apply_font(fig: Figure, theme: Theme) -> None:
    """Apply the brand font to every text artist in the figure (deterministic)."""
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        if isinstance(artist, Text):
            artist.set_fontfamily(theme.font_family)


def render_bracket(
    positioned: list[list[PositionedMatch]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str | None = "Knockout bracket",
) -> Figure:
    """Draw one box per match and connectors between rounds; bold the winner."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig = Figure(
        figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi
    )
    fig.patch.set_facecolor(theme.background)
    ax = fig.subplots()
    ax.set_facecolor(theme.background)

    max_y = max((pm.y for col in positioned for pm in col), default=0.0)
    for col in positioned:
        for pm in col:
            ax.add_patch(
                Rectangle(
                    (pm.column, pm.y),
                    _BOX_W,
                    _BOX_H,
                    fill=False,
                    edgecolor=theme.text_muted,
                )
            )
            for slot, team in enumerate((pm.match.home, pm.match.away)):
                won = team is not None and team == pm.match.winner
                ax.text(
                    pm.column + 0.04,
                    pm.y + _BOX_H * (0.72 - 0.42 * slot),
                    team if team is not None else "—",
                    fontsize=theme.stamp_size,
                    color=theme.text_primary,
                    weight="bold" if won else "normal",
                    va="center",
                )

    for col_idx in range(1, len(positioned)):
        prev = positioned[col_idx - 1]
        for i, pm in enumerate(positioned[col_idx]):
            for child in (prev[2 * i], prev[2 * i + 1]):
                ax.plot(
                    [child.column + _BOX_W, pm.column],
                    [child.y + _BOX_H / 2, pm.y + _BOX_H / 2],
                    color=theme.text_muted,
                    linewidth=1,
                )

    ax.set_xlim(-0.2, len(positioned) + _BOX_W)
    ax.set_ylim(-0.5, max_y + 1.2)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size)
    _apply_font(fig, theme)
    return fig
