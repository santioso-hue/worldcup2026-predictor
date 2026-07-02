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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_BOX_W = 0.7
_BOX_H = 0.7

# Mirrored layout gets its own (wider) box so long team names have room; the
# single-sided prepare_bracket/render_bracket keep using _BOX_W untouched.
# Sized so the worst-case real team name ("Bosnia and Herzegovina") still
# fits at the fontsize floor (see _fitted_fontsize / _MIN_STAMP_SIZE below).
_MIRROR_BOX_W = 1.8
_MIRROR_BOX_H = 0.7
# Column spacing factor: at width 1.8, boxes at consecutive integer columns
# would overlap (1.8 > 1.0), so mirrored x positions are scaled by this
# factor to keep columns clear of each other.
_MIRROR_COL_SPACING = 2.6

_MIN_STAMP_SIZE = 9
_CHAR_WIDTH_FACTOR = 0.62  # cheap deterministic point-width-per-char estimate
_TEXT_PADDING_PT = 4.0  # keep some breathing room from the box edge


def _estimate_text_width_pt(text: str, fontsize: float) -> float:
    """Deterministic text-width estimate in points (no bbox measurement)."""
    return len(text) * fontsize * _CHAR_WIDTH_FACTOR


def _fitted_fontsize(
    name: str,
    *,
    box_w_in: float = _MIRROR_BOX_W,
    max_fontsize: int = 16,
    min_fontsize: int = _MIN_STAMP_SIZE,
) -> int:
    """Largest fontsize (>= floor) whose estimated text width fits the box.

    Uses a cheap, deterministic heuristic (``len(name) * fontsize * 0.62``
    points) rather than measuring a real text bbox, so results are stable
    across environments/backends. Falls back to the floor if even that
    doesn't fit (better a slight overflow than unreadable tiny text).
    """
    box_w_pt = box_w_in * 72 - _TEXT_PADDING_PT
    for fontsize in range(max_fontsize, min_fontsize - 1, -1):
        if _estimate_text_width_pt(name, fontsize) <= box_w_pt:
            return fontsize
    return min_fontsize


@dataclass(frozen=True)
class BracketMatch:
    """A matchup: teams (``None`` = not yet decided) and the winner, if played."""

    home: str | None
    away: str | None
    winner: str | None = None
    annotation: str | None = None  # small text under the names (score, advance %)
    highlight: str | None = None  # team name to draw in the accent color


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


_MIRROR_COLUMNS = 4  # left half occupies columns 0..3, final at column 4


def prepare_bracket_mirrored(
    rounds: list[list[BracketMatch]],
) -> list[list[PositionedMatch]]:
    """Position a two-sided bracket: rounds ordered R32(16)...Final(1).

    Each round's list is split in half; the first half feeds the left side
    (columns ``0..3``, growing rightward toward the final) and the second half
    feeds the right side, mirrored (columns ``8..5``, growing leftward toward
    the final at column 4). Within each half, ``y`` follows the same
    invariant as :func:`prepare_bracket`: leaves at their row index, parents
    centered between their two children.

    Raises
    ------
    ValueError
        If the bracket is empty, a round doesn't have half the matches of
        the previous one, or a round can't be split into equal halves.
    """
    if not rounds:
        raise ValueError("empty bracket")
    if len(rounds[-1]) != 1:
        raise ValueError("final round must have exactly one match")
    if len(rounds) < 2:
        raise ValueError("mirrored bracket needs at least a semifinal and a final")
    if len(rounds[0]) % 2 != 0:
        raise ValueError("first round must split into equal left/right halves")

    # Every round but the final splits into a left half and a right half;
    # each half is itself a standard single-sided bracket feeding one
    # finalist (the last round of each half has exactly one match).
    semi_rounds = rounds[:-1]
    left_rounds = [matches[: len(matches) // 2] for matches in semi_rounds]
    right_rounds = [matches[len(matches) // 2 :] for matches in semi_rounds]

    left = prepare_bracket(left_rounds)
    right = prepare_bracket(right_rounds)

    positioned: list[list[PositionedMatch]] = []
    for col in range(len(semi_rounds)):
        left_col = [PositionedMatch(pm.match, col, pm.y) for pm in left[col]]
        right_col = [PositionedMatch(pm.match, 8 - col, pm.y) for pm in right[col]]
        positioned.append(left_col + right_col)

    # Final round: single match at the center column, centered between the
    # two finalists (the last entry of each half).
    final_y = (left[-1][0].y + right[-1][0].y) / 2
    positioned.append([PositionedMatch(rounds[-1][0], _MIRROR_COLUMNS, final_y)])
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


MIRRORED = ExportSpec(2400, 2000)  # near-square, for the two-sided bracket


def _mirror_x(column: int) -> float:
    """Scale a mirrored-bracket column index into an x position (inches).

    The mirrored box is wider than a single integer column (``_MIRROR_BOX_W``
    > 1), so columns are spread out by ``_MIRROR_COL_SPACING`` to keep
    neighboring boxes from touching.
    """
    return column * _MIRROR_COL_SPACING


def _draw_mirrored_box(
    ax: Axes,
    pm: PositionedMatch,
    *,
    theme: Theme,
    right_side: bool,
) -> None:
    """Draw one bracket box: names, bold winner, accent highlight, annotation."""
    from matplotlib.patches import Rectangle

    x0 = _mirror_x(pm.column)
    ax.add_patch(
        Rectangle(
            (x0, pm.y),
            _MIRROR_BOX_W,
            _MIRROR_BOX_H,
            fill=False,
            edgecolor=theme.text_muted,
        )
    )
    x_text = x0 + _MIRROR_BOX_W - 0.04 if right_side else x0 + 0.04
    ha = "right" if right_side else "left"
    for slot, team in enumerate((pm.match.home, pm.match.away)):
        won = team is not None and team == pm.match.winner
        highlighted = team is not None and team == pm.match.highlight
        color = theme.accent if highlighted else theme.text_primary
        label = team if team is not None else "—"
        ax.text(
            x_text,
            pm.y + _MIRROR_BOX_H * (0.78 - 0.34 * slot),
            label,
            fontsize=_fitted_fontsize(label),
            color=color,
            weight="bold" if won else "normal",
            va="center",
            ha=ha,
        )
    if pm.match.annotation is not None:
        annotation_size = min(
            _fitted_fontsize(pm.match.annotation), theme.stamp_size - 6
        )
        ax.text(
            x_text,
            pm.y + _MIRROR_BOX_H * 0.14,
            pm.match.annotation,
            fontsize=max(annotation_size, 6),
            color=theme.text_muted,
            va="center",
            ha=ha,
        )


def render_bracket_mirrored(
    positioned: list[list[PositionedMatch]],
    *,
    theme: Theme = THEME,
    spec: ExportSpec = MIRRORED,
    title: str | None = "Knockout bracket",
) -> Figure:
    """Draw a two-sided (mirrored) bracket: left half, final, right half.

    Right-half team names are right-aligned and their connectors run from
    the box's left edge toward the center, mirroring the left half's
    left-to-right flow.
    """
    from matplotlib.figure import Figure

    fig = Figure(
        figsize=(spec.width_px / spec.dpi, spec.height_px / spec.dpi), dpi=spec.dpi
    )
    fig.patch.set_facecolor(theme.background)
    ax = fig.subplots()
    ax.set_facecolor(theme.background)

    max_y = max((pm.y for col in positioned for pm in col), default=0.0)
    n_rounds = len(positioned)
    for col in positioned:
        for pm in col:
            right_side = pm.column > _MIRROR_COLUMNS
            _draw_mirrored_box(ax, pm, theme=theme, right_side=right_side)

    # Connectors: a parent on the left grows rightward (children's right edge
    # to its left edge); on the right it mirrors (children's left edge to its
    # right edge). The final (center column) receives one connector from each
    # side, each drawn from the appropriate child edge.
    for col_idx in range(1, n_rounds):
        prev = positioned[col_idx - 1]
        cur = positioned[col_idx]
        is_final = col_idx == n_rounds - 1
        half_prev = len(prev) // 2
        for i, pm in enumerate(cur):
            right_side = pm.column > _MIRROR_COLUMNS
            if is_final:
                children = (prev[0], prev[1])
            elif right_side:
                j = i - len(cur) // 2
                children = (prev[half_prev + 2 * j], prev[half_prev + 2 * j + 1])
            else:
                children = (prev[2 * i], prev[2 * i + 1])
            x_pm = _mirror_x(pm.column)
            x_parent = x_pm + _MIRROR_BOX_W if right_side else x_pm
            for child in children:
                child_right_side = child.column > _MIRROR_COLUMNS
                x_child_col = _mirror_x(child.column)
                x_child = (
                    x_child_col if child_right_side else x_child_col + _MIRROR_BOX_W
                )
                ax.plot(
                    [x_child, x_parent],
                    [child.y + _MIRROR_BOX_H / 2, pm.y + _MIRROR_BOX_H / 2],
                    color=theme.text_muted,
                    linewidth=1,
                )

    ax.set_xlim(-0.2, _mirror_x(8) + _MIRROR_BOX_W + 0.2)
    ax.set_ylim(-0.5, max_y + 1.2)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, color=theme.text_primary, fontsize=theme.title_size)
    _apply_font(fig, theme)
    return fig
