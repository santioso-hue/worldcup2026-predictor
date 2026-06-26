"""Tests de standings de grupo (Art. 13: H2H ANTES de GD) y ranking de terceros."""

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
    # A y B empatan a 6 pts; A tiene mejor GD global (+5 vs +3) PERO B le ganó a A.
    # Art. 13 Step 1 (H2H) va ANTES que GD -> B por encima de A.
    # C y D empatan a 3; D le ganó a C aunque C tiene mejor GD -> D por encima de C.
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
    # A y B empatan a 4 pts y empataron su H2H (1-1) -> H2H no decide -> GD global: A>B.
    matches = [
        PlayedMatch("A", "B", 1, 1),
        PlayedMatch("A", "C", 3, 0),
        PlayedMatch("A", "D", 0, 1),
        PlayedMatch("B", "C", 1, 0),
        PlayedMatch("B", "D", 0, 1),
        PlayedMatch("D", "C", 1, 0),
    ]
    order = _order(matches, list("ABCD"), EQUAL_ELO)
    assert order.index("A") < order.index("B")  # A antes que B por GD global


def test_elo_breaks_total_ties() -> None:
    # A y B idénticos en puntos, H2H (0-0), GD y GF globales -> decide el Elo.
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
    assert order.index("A") < order.index("B")  # A antes que B por mayor Elo


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
    # Sin empates (resultado claro): aun así falla de entrada, no según el marcador.
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
