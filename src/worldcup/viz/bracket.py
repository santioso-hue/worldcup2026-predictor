"""Bracket eliminatorio: layout (puro) + render matplotlib.

``prepare_bracket`` posiciona cada partido en columnas por ronda (puro, testeable);
``render_bracket`` dibuja cajas y conectores, resaltando al ganador de las llaves
resueltas. Recibe la estructura de rondas ya dada: derivar los cruces a partir de los
resultados es trabajo del pipeline, no de ``viz``.
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
    """Una llave: equipos (``None`` = aún sin definir) y, si se jugó, el ganador."""

    home: str | None
    away: str | None
    winner: str | None = None


@dataclass(frozen=True)
class PositionedMatch:
    """Un partido con su posición de layout (columna = ronda, ``y`` = fila centrada)."""

    match: BracketMatch
    column: int
    y: float


def prepare_bracket(rounds: list[list[BracketMatch]]) -> list[list[PositionedMatch]]:
    """Posiciona cada partido: columna = ronda; ``y`` = centro entre sus dos hijos.

    Raises
    ------
    ValueError
        Si el bracket está vacío o una ronda no tiene la mitad que la anterior.
    """
    if not rounds:
        raise ValueError("bracket vacío")
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
                f"la ronda {col} debe tener la mitad de partidos que la anterior"
            )
        positioned.append(
            [
                PositionedMatch(m, col, (prev[2 * i].y + prev[2 * i + 1].y) / 2)
                for i, m in enumerate(matches)
            ]
        )
    return positioned


def _apply_font(fig: Figure, theme: Theme) -> None:
    """Fija la tipografía de marca en todo el texto de la figura (determinista)."""
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        if isinstance(artist, Text):
            artist.set_fontfamily(theme.font_family)


def render_bracket(
    positioned: list[list[PositionedMatch]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = LANDSCAPE,
    title: str | None = "Bracket eliminatorio",
) -> Figure:
    """Dibuja una caja por partido y conectores entre rondas; ganador en negrita."""
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
