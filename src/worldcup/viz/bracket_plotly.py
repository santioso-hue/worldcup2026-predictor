"""Knockout bracket: interactive Plotly figure builder (pure).

``bracket_plotly_figure`` turns the same positioned-match layout used by the
matplotlib mirrored renderer (``viz.bracket.prepare_bracket_mirrored``) into a
``plotly.graph_objects.Figure``: one rectangle shape per box, a scatter-text
trace for team names (left/right anchored by half), a smaller gray annotation
trace, connector lines between rounds, and invisible hover points that carry
per-tie tooltip text supplied by the caller. The dashboard composes the hover
strings (kickoff, 1X2, advance %); this module only places them.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .bracket import (
    _MIRROR_BOX_H,
    _MIRROR_BOX_W,
    _MIRROR_COLUMNS,
    PositionedMatch,
    _mirror_x,
)
from .theme import THEME, Theme

_ANNOTATION_DY = _MIRROR_BOX_H * 0.14  # matches the matplotlib renderer's offset
_NAME_DY = (_MIRROR_BOX_H * 0.78, _MIRROR_BOX_H * 0.44)  # home, away row offsets
_TEXT_PAD = 0.06  # inset from the box edge, in the same x units as _mirror_x


def _team_label(team: str | None, *, winner: str | None, highlight: str | None) -> str:
    """Render one team's name: em dash for TBD, bold for the winner."""
    if team is None:
        return "—"
    return f"<b>{team}</b>" if team == winner else team


def _box_shape(pm: PositionedMatch, *, theme: Theme) -> dict[str, object]:
    x0 = _mirror_x(pm.column)
    return {
        "type": "rect",
        "x0": x0,
        "y0": pm.y,
        "x1": x0 + _MIRROR_BOX_W,
        "y1": pm.y + _MIRROR_BOX_H,
        "line": {"color": theme.text_muted, "width": 1},
        "fillcolor": "rgba(0,0,0,0)",
    }


def _connector_shape(
    child: PositionedMatch, parent: PositionedMatch, *, theme: Theme
) -> dict[str, object]:
    """One line from a child box's outer edge to its parent's inner edge.

    Mirrors ``render_bracket_mirrored``'s connector logic: a left-side box
    grows rightward (its right edge feeds the parent's left edge); a
    right-side box mirrors that (its left edge feeds the parent's right
    edge). The final's two children connect from whichever side they sit on.
    """
    child_right_side = child.column > _MIRROR_COLUMNS
    parent_right_side = parent.column > _MIRROR_COLUMNS
    x_child_col = _mirror_x(child.column)
    x_child = x_child_col if child_right_side else x_child_col + _MIRROR_BOX_W
    x_parent_col = _mirror_x(parent.column)
    x_parent = x_parent_col + _MIRROR_BOX_W if parent_right_side else x_parent_col
    return {
        "type": "line",
        "x0": x_child,
        "y0": child.y + _MIRROR_BOX_H / 2,
        "x1": x_parent,
        "y1": parent.y + _MIRROR_BOX_H / 2,
        "line": {"color": theme.text_muted, "width": 1},
    }


def _connector_shapes(
    positioned: list[list[PositionedMatch]], *, theme: Theme
) -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    n_rounds = len(positioned)
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
            for child in children:
                shapes.append(_connector_shape(child, pm, theme=theme))
    return shapes


def bracket_plotly_figure(
    positioned: list[list[PositionedMatch]],
    hover: dict[tuple[int, int], str],
    *,
    theme: Theme = THEME,
    title: str = "Knockout bracket",
) -> go.Figure:
    """Build an interactive bracket figure from positioned matches.

    Parameters
    ----------
    positioned : list[list[PositionedMatch]]
        Layout produced by ``prepare_bracket_mirrored``: one list per round
        column, each entry a match with its box position.
    hover : dict[tuple[int, int], str]
        Maps ``(column, row-index-in-column)`` to hover HTML for that tie.
        Ties with no entry get no hover point.
    theme : Theme, optional
        Brand palette/typography; defaults to ``THEME``.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with box shapes, connector lines, name/annotation text
        traces, and one invisible marker trace for hover.
    """
    shapes: list[dict[str, object]] = []
    left_x: list[float] = []
    left_y: list[float] = []
    left_text: list[str] = []
    left_color: list[str] = []
    right_x: list[float] = []
    right_y: list[float] = []
    right_text: list[str] = []
    right_color: list[str] = []
    annot_x: list[float] = []
    annot_y: list[float] = []
    annot_text: list[str] = []
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []

    for col_idx, col in enumerate(positioned):
        for row_idx, pm in enumerate(col):
            shapes.append(_box_shape(pm, theme=theme))
            x0 = _mirror_x(pm.column)
            right_side = pm.column > _MIRROR_COLUMNS
            x_text = x0 + _MIRROR_BOX_W - _TEXT_PAD if right_side else x0 + _TEXT_PAD

            match = pm.match
            for slot, team in enumerate((match.home, match.away)):
                label = _team_label(
                    team, winner=match.winner, highlight=match.highlight
                )
                highlighted = team is not None and team == match.highlight
                color = theme.accent if highlighted else theme.text_primary
                target_x = right_x if right_side else left_x
                target_y = right_y if right_side else left_y
                target_text = right_text if right_side else left_text
                target_color = right_color if right_side else left_color
                target_x.append(x_text)
                target_y.append(pm.y + _NAME_DY[slot])
                target_text.append(label)
                target_color.append(color)

            if match.annotation is not None:
                annot_x.append(x_text)
                annot_y.append(pm.y + _ANNOTATION_DY)
                annot_text.append(match.annotation)

            hover_key = (col_idx, row_idx)
            if hover_key in hover:
                hover_x.append(x0 + _MIRROR_BOX_W / 2)
                hover_y.append(pm.y + _MIRROR_BOX_H / 2)
                hover_text.append(hover[hover_key])

    shapes.extend(_connector_shapes(positioned, theme=theme))

    fig = go.Figure()
    if left_text:
        fig.add_trace(
            go.Scatter(
                x=left_x,
                y=left_y,
                mode="text",
                text=left_text,
                textposition="middle right",
                textfont={
                    "family": theme.font_family,
                    "size": theme.stamp_size,
                    "color": left_color,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if right_text:
        fig.add_trace(
            go.Scatter(
                x=right_x,
                y=right_y,
                mode="text",
                text=right_text,
                textposition="middle left",
                textfont={
                    "family": theme.font_family,
                    "size": theme.stamp_size,
                    "color": right_color,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if annot_text:
        fig.add_trace(
            go.Scatter(
                x=annot_x,
                y=annot_y,
                mode="text",
                text=annot_text,
                textposition="middle center",
                textfont={
                    "family": theme.font_family,
                    "size": max(theme.stamp_size - 6, 6),
                    "color": theme.text_muted,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if hover_text:
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={"size": 1, "opacity": 0},
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    max_y = max((pm.y for col in positioned for pm in col), default=0.0)
    max_x = _mirror_x(8) + _MIRROR_BOX_W

    fig.update_layout(
        title={"text": title, "font": {"color": theme.text_primary}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        shapes=shapes,
        xaxis={
            "visible": False,
            "range": [-0.2, max_x + 0.2],
            "constrain": "domain",
        },
        yaxis={
            "visible": False,
            "range": [-0.5, max_y + 1.2],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return fig
