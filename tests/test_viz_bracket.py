"""Tests for the viz bracket: layout (pure) + render smoke test."""

from __future__ import annotations

import pytest
from matplotlib.patches import Rectangle

from worldcup.viz.bracket import BracketMatch, prepare_bracket, render_bracket


def test_prepare_bracket_positions() -> None:
    rounds = [
        [
            BracketMatch("A", "B"),
            BracketMatch("C", "D"),
            BracketMatch("E", "F"),
            BracketMatch("G", "H"),
        ],
        [BracketMatch(None, None), BracketMatch(None, None)],
        [BracketMatch(None, None)],
    ]
    pos = prepare_bracket(rounds)
    assert [pm.y for pm in pos[0]] == [0.0, 1.0, 2.0, 3.0]
    assert [pm.y for pm in pos[1]] == [0.5, 2.5]  # midpoint between children
    assert [pm.y for pm in pos[2]] == [1.5]
    assert [pm.column for pm in pos[2]] == [2]


def test_prepare_bracket_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        prepare_bracket(
            [
                [BracketMatch("A", "B"), BracketMatch("C", "D")],
                [BracketMatch(None, None), BracketMatch(None, None)],  # should be 1
            ]
        )
    with pytest.raises(ValueError):
        prepare_bracket([])


def test_render_bracket_smoke() -> None:
    rounds = [
        [BracketMatch("A", "B", winner="A"), BracketMatch("C", "D")],
        [BracketMatch("A", None)],
    ]
    fig = render_bracket(prepare_bracket(rounds))
    rects = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(rects) == 3  # one box per match
