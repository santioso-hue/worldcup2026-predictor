"""Tests for the viz bracket: layout (pure) + render smoke test."""

from __future__ import annotations

import pytest
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from worldcup.viz.bracket import (
    BracketMatch,
    prepare_bracket,
    prepare_bracket_mirrored,
    render_bracket,
    render_bracket_mirrored,
)
from worldcup.viz.theme import THEME


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


def _mirrored_rounds() -> list[list[BracketMatch]]:
    """8 R16 leaves -> 4 QF -> 2 SF -> 1 Final; first half feeds the left."""
    r16 = [BracketMatch(t, t + "2") for t in "ABCDEFGH"]
    qf = [BracketMatch(None, None) for _ in range(4)]
    sf = [BracketMatch(None, None) for _ in range(2)]
    final = [BracketMatch(None, None)]
    return [r16, qf, sf, final]


def test_prepare_bracket_mirrored_positions() -> None:
    pos = prepare_bracket_mirrored(_mirrored_rounds())
    # Round 0 (leaves): 4 on the left at column 0, 4 on the right at column 8.
    leaves = pos[0]
    assert [pm.column for pm in leaves[:4]] == [0, 0, 0, 0]
    assert [pm.column for pm in leaves[4:]] == [8, 8, 8, 8]
    assert [pm.y for pm in leaves[:4]] == [0.0, 1.0, 2.0, 3.0]
    assert [pm.y for pm in leaves[4:]] == [0.0, 1.0, 2.0, 3.0]
    # A left leaf at row y has a mirrored counterpart at the same row on the right.
    for i in range(4):
        assert leaves[i].y == leaves[4 + i].y

    # Round 1 (QF): 2 left at column 1, 2 right at column 7.
    qf = pos[1]
    assert [pm.column for pm in qf[:2]] == [1, 1]
    assert [pm.column for pm in qf[2:]] == [7, 7]

    # Round 2 (SF): 1 left at column 2, 1 right at column 6.
    sf = pos[2]
    assert [pm.column for pm in sf] == [2, 6]

    # Final: single match at the center column 4.
    final = pos[3]
    assert len(final) == 1
    assert final[0].column == 4


def test_prepare_bracket_mirrored_centers_parents() -> None:
    pos = prepare_bracket_mirrored(_mirrored_rounds())
    leaves, qf, sf, final = pos
    # Left half: parent y centered between its two children.
    assert qf[0].y == (leaves[0].y + leaves[1].y) / 2
    assert qf[1].y == (leaves[2].y + leaves[3].y) / 2
    assert sf[0].y == (qf[0].y + qf[1].y) / 2
    # Right half mirrors the same centering invariant.
    assert qf[2].y == (leaves[4].y + leaves[5].y) / 2
    assert qf[3].y == (leaves[6].y + leaves[7].y) / 2
    assert sf[1].y == (qf[2].y + qf[3].y) / 2
    # Final centered between the two semifinal winners (left + right).
    assert final[0].y == (sf[0].y + sf[1].y) / 2


def test_prepare_bracket_mirrored_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        prepare_bracket_mirrored([])
    with pytest.raises(ValueError):
        prepare_bracket_mirrored(
            [
                [BracketMatch("A", "B"), BracketMatch("C", "D")],
                [BracketMatch(None, None), BracketMatch(None, None)],  # should be 1
            ]
        )
    with pytest.raises(ValueError):
        # Odd leaf count can't split into equal left/right halves.
        prepare_bracket_mirrored([[BracketMatch("A", "B"), BracketMatch("C", "D")]])


def test_render_bracket_mirrored_smoke() -> None:
    def leaves(n: int) -> list[BracketMatch]:
        return [
            BracketMatch(
                f"T{i}",
                f"T{i}b",
                annotation="2-1" if i == 0 else None,
                highlight=f"T{i}" if i == 0 else None,
            )
            for i in range(n)
        ]

    rounds = [
        leaves(16),
        [BracketMatch(None, None) for _ in range(8)],
        [BracketMatch(None, None) for _ in range(4)],
        [BracketMatch(None, None) for _ in range(2)],
        [BracketMatch(None, None)],
    ]
    fig = render_bracket_mirrored(prepare_bracket_mirrored(rounds))
    rects = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(rects) == 31  # 16 + 8 + 4 + 2 + 1

    texts = [t.get_text() for t in fig.axes[0].findobj(Text)]
    assert "2-1" in texts


def test_render_bracket_mirrored_highlight_and_bold() -> None:
    rounds = [
        [
            BracketMatch("A", "B", winner="A", highlight="A", annotation="2-0"),
            BracketMatch("C", "D"),
        ],
        [BracketMatch("A", None)],
    ]
    fig = render_bracket_mirrored(prepare_bracket_mirrored(rounds))
    ax = fig.axes[0]
    team_texts = [t for t in ax.findobj(Text) if t.get_text() == "A"]
    assert team_texts
    assert any(t.get_fontweight() in ("bold", 700) for t in team_texts)
    highlighted = [t for t in team_texts if t.get_color() == THEME.accent]
    assert highlighted
