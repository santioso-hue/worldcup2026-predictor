"""Tests for group standings (Art. 13: H2H before GD) and third-place ranking."""

from __future__ import annotations

import pytest

from worldcup.simulation.group_stage import (
    PlayedMatch,
    TeamStanding,
    rank_thirds,
    standings,
)

EQUAL_ELO = {t: 1500.0 for t in "ABCD"}


def _order(
    matches: list[PlayedMatch], teams: list[str], elo: dict[str, float]
) -> list[str]:
    return [s.team for s in standings(matches, teams, elo)]


def test_orders_by_points_when_no_ties() -> None:
    matches = [
        PlayedMatch("A", "B", 1, 0),
        PlayedMatch("A", "C", 1, 0),
        PlayedMatch("A", "D", 1, 0),
        PlayedMatch("B", "C", 1, 0),
        PlayedMatch("B", "D", 1, 0),
        PlayedMatch("C", "D", 1, 0),
    ]
    assert _order(matches, list("ABCD"), EQUAL_ELO) == ["A", "B", "C", "D"]


def test_head_to_head_beats_overall_goal_difference() -> None:
    # A and B tie on 6 pts; A has better overall GD (+5 vs +3) BUT B beat A H2H.
    # Art. 13 Step 1 (H2H) comes before GD -> B ranks above A.
    # C and D tie on 3; D beat C even though C has better GD -> D ranks above C.
    matches = [
        PlayedMatch("A", "C", 3, 0),
        PlayedMatch("A", "D", 3, 0),
        PlayedMatch("B", "A", 1, 0),
        PlayedMatch("C", "B", 1, 0),
        PlayedMatch("B", "D", 3, 0),
        PlayedMatch("D", "C", 1, 0),
    ]
    assert _order(matches, list("ABCD"), EQUAL_ELO) == ["B", "A", "D", "C"]


def test_falls_back_to_overall_gd_when_head_to_head_is_drawn() -> None:
    # A and B tie on 4 pts, drew H2H (1-1) -> H2H doesn't decide -> overall GD: A>B.
    matches = [
        PlayedMatch("A", "B", 1, 1),
        PlayedMatch("A", "C", 3, 0),
        PlayedMatch("A", "D", 0, 1),
        PlayedMatch("B", "C", 1, 0),
        PlayedMatch("B", "D", 0, 1),
        PlayedMatch("D", "C", 1, 0),
    ]
    order = _order(matches, list("ABCD"), EQUAL_ELO)
    assert order.index("A") < order.index("B")  # A before B on overall GD


def test_elo_breaks_total_ties() -> None:
    # A and B are identical on points, H2H (0-0), overall GD and GF -> Elo decides.
    matches = [
        PlayedMatch("A", "B", 0, 0),
        PlayedMatch("A", "C", 1, 0),
        PlayedMatch("A", "D", 0, 1),
        PlayedMatch("B", "C", 1, 0),
        PlayedMatch("B", "D", 0, 1),
        PlayedMatch("D", "C", 1, 0),
    ]
    elo = {"A": 1600.0, "B": 1500.0, "C": 1500.0, "D": 1500.0}
    order = _order(matches, list("ABCD"), elo)
    assert order.index("A") < order.index("B")  # A before B on higher Elo


def test_standings_returns_overall_records() -> None:
    matches = [
        PlayedMatch("A", "B", 2, 0),
        PlayedMatch("A", "C", 1, 0),
        PlayedMatch("A", "D", 1, 0),
        PlayedMatch("B", "C", 0, 0),
        PlayedMatch("B", "D", 0, 0),
        PlayedMatch("C", "D", 0, 0),
    ]
    top = standings(matches, list("ABCD"), EQUAL_ELO)[0]
    assert top == TeamStanding(team="A", points=9, goal_diff=4, goals_for=4)


def test_standings_fails_loud_when_elo_missing_a_team() -> None:
    # No ties (clear result): it still fails up front, regardless of the scoreline.
    matches = [
        PlayedMatch("A", "B", 3, 0),
        PlayedMatch("A", "C", 3, 0),
        PlayedMatch("A", "D", 3, 0),
        PlayedMatch("B", "C", 1, 0),
        PlayedMatch("B", "D", 1, 0),
        PlayedMatch("C", "D", 1, 0),
    ]
    with pytest.raises(ValueError, match="D"):
        standings(matches, list("ABCD"), {"A": 1500.0, "B": 1500.0, "C": 1500.0})


def test_rank_thirds_fails_loud_when_elo_missing_a_team() -> None:
    thirds = [TeamStanding("P", 3, 0, 2), TeamStanding("Q", 3, 0, 2)]
    with pytest.raises(ValueError, match="Q"):
        rank_thirds(thirds, {"P": 1500.0})


def test_rank_thirds_orders_by_points_gd_gf_then_elo() -> None:
    thirds = [
        TeamStanding("P", points=3, goal_diff=0, goals_for=2),
        TeamStanding("Q", points=4, goal_diff=-1, goals_for=1),
        TeamStanding("R", points=3, goal_diff=1, goals_for=2),
        TeamStanding("S", points=3, goal_diff=0, goals_for=2),
    ]
    elo = {"P": 1500.0, "Q": 1500.0, "R": 1500.0, "S": 1600.0}
    ranked = [s.team for s in rank_thirds(thirds, elo)]
    # Q first (4 pts); then R (GD +1); then S vs P tied on pts/GD/GF -> Elo: S > P.
    assert ranked == ["Q", "R", "S", "P"]
