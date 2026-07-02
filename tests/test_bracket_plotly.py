"""Tests for the interactive Plotly bracket builder (pure figure assembly)."""

from __future__ import annotations

import plotly.graph_objects as go

from worldcup.viz.bracket import BracketMatch, prepare_bracket_mirrored
from worldcup.viz.bracket_plotly import bracket_plotly_figure


def _mirrored_rounds() -> list[list[BracketMatch]]:
    """16 R32 leaves -> 8 R16 -> 4 QF -> 2 SF -> 1 Final (31 boxes total)."""
    r32 = [BracketMatch(f"T{i}", f"T{i}b") for i in range(16)]
    r16 = [BracketMatch(None, None) for _ in range(8)]
    qf = [BracketMatch(None, None) for _ in range(4)]
    sf = [BracketMatch(None, None) for _ in range(2)]
    final = [BracketMatch(None, None)]
    return [r32, r16, qf, sf, final]


def test_box_count_matches_full_bracket() -> None:
    positioned = prepare_bracket_mirrored(_mirrored_rounds())
    fig = bracket_plotly_figure(positioned, hover={})
    boxes = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(boxes) == 31  # 16 + 8 + 4 + 2 + 1


def test_winner_name_is_bold() -> None:
    rounds = [
        [
            BracketMatch("A", "B", winner="A", highlight="A", annotation="2-0"),
            BracketMatch("C", "D"),
        ],
        [BracketMatch("A", None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = bracket_plotly_figure(positioned, hover={})
    all_text = " ".join(
        text
        for trace in fig.data
        if isinstance(trace, go.Scatter) and trace.mode and "text" in trace.mode
        for text in (trace.text or [])
    )
    assert "<b>A</b>" in all_text


def test_right_half_trace_is_right_anchored() -> None:
    rounds = [
        [
            BracketMatch("A", "B"),
            BracketMatch("C", "D"),
        ],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = bracket_plotly_figure(positioned, hover={})
    right_traces = [
        trace
        for trace in fig.data
        if isinstance(trace, go.Scatter)
        and trace.mode
        and "text" in trace.mode
        and trace.textposition
        and "right" in trace.textposition
    ]
    assert right_traces
    left_traces = [
        trace
        for trace in fig.data
        if isinstance(trace, go.Scatter)
        and trace.mode
        and "text" in trace.mode
        and trace.textposition
        and "left" in trace.textposition
    ]
    assert left_traces


def test_hover_point_text_matches_dict() -> None:
    rounds = [
        [
            BracketMatch("A", "B"),
            BracketMatch("C", "D"),
        ],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    hover = {(0, 0): "Round of 16 — A vs B", (0, 1): "Round of 16 — C vs D"}
    fig = bracket_plotly_figure(positioned, hover=hover)
    hover_traces = [
        trace
        for trace in fig.data
        if isinstance(trace, go.Scatter) and trace.mode == "markers"
    ]
    assert len(hover_traces) == 1
    hover_texts = list(hover_traces[0].hovertext or hover_traces[0].text or [])
    assert set(hover_texts) == set(hover.values())


def test_hover_skips_ties_without_entry() -> None:
    rounds = [
        [
            BracketMatch("A", "B"),
            BracketMatch("C", "D"),
        ],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    hover = {(0, 0): "Round of 16 — A vs B"}
    fig = bracket_plotly_figure(positioned, hover=hover)
    hover_traces = [
        trace
        for trace in fig.data
        if isinstance(trace, go.Scatter) and trace.mode == "markers"
    ]
    assert len(hover_traces) == 1
    hover_texts = list(hover_traces[0].hovertext or hover_traces[0].text or [])
    assert hover_texts == ["Round of 16 — A vs B"]


def test_all_tbd_input_does_not_raise() -> None:
    rounds = [
        [BracketMatch(None, None), BracketMatch(None, None)],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = bracket_plotly_figure(positioned, hover={})
    boxes = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(boxes) == 3
    all_text = " ".join(
        text
        for trace in fig.data
        if isinstance(trace, go.Scatter) and trace.mode and "text" in trace.mode
        for text in (trace.text or [])
    )
    assert "—" in all_text  # em dash for TBD


def test_returns_plotly_figure_with_theme_defaults() -> None:
    rounds = [
        [BracketMatch("A", "B"), BracketMatch("C", "D")],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = bracket_plotly_figure(positioned, hover={})
    assert isinstance(fig, go.Figure)
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.dragmode == "pan"
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False
    assert fig.layout.showlegend is False
    assert fig.layout.title.text == "Knockout bracket"


def test_connectors_present_between_rounds() -> None:
    rounds = [
        [BracketMatch("A", "B"), BracketMatch("C", "D")],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = bracket_plotly_figure(positioned, hover={})
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    # Two boxes feed the final: two connector lines expected.
    assert len(lines) == 2


def test_deterministic_output() -> None:
    rounds = [
        [BracketMatch("A", "B"), BracketMatch("C", "D")],
        [BracketMatch(None, None)],
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig1 = bracket_plotly_figure(positioned, hover={})
    fig2 = bracket_plotly_figure(positioned, hover={})
    assert fig1.to_json() == fig2.to_json()
