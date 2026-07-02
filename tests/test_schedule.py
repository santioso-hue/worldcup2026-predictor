"""Tests for the schedule backbone parser (openfootball -> normalized fixtures)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.data.schedule import (
    group_teams,
    is_knockout_stage,
    is_predictable,
    make_match_id,
    parse_match,
    parse_openfootball,
    remaining_fixtures,
    validate_schedule,
)

# Minimal sample with the VERIFIED openfootball schema (matchday + knockout).
SAMPLE: dict[str, Any] = {
    "rounds": [
        {
            "name": "Matchday 1",
            "matches": [
                {
                    "round": "Matchday 1",
                    "date": "2026-06-11",
                    "time": "13:00 UTC-6",
                    "team1": "Mexico",
                    "team2": "South Africa",
                    "group": "Group A",
                    "score": {"ft": [2, 0], "ht": [1, 0]},
                    "ground": "Mexico City",
                },
                {
                    "round": "Matchday 1",
                    "date": "2026-06-12",
                    "time": "18:00 UTC-4",
                    "team1": "Canada",
                    "team2": "Wales",
                    "group": "Group B",
                },
            ],
        },
        {
            "name": "Round of 32",
            "matches": [
                {
                    "round": "Round of 32",
                    "date": "2026-06-28",
                    "team1": "Winner Group A",
                    "team2": "Third Group C/D/E/F",
                },
            ],
        },
    ]
}


def test_make_match_id_is_stable_and_slugged() -> None:
    assert (
        make_match_id("2026-06-11", "Mexico", "South Africa")
        == "2026-06-11-mexico-vs-south-africa"
    )


def test_finished_match_parsed_with_scores() -> None:
    fixtures = parse_openfootball(SAMPLE)
    mex = fixtures[0]
    assert mex.status is MatchStatus.FINISHED
    assert (mex.ft_home, mex.ft_away) == (2, 0)
    assert (mex.ht_home, mex.ht_away) == (1, 0)
    assert mex.stage == "Group A"
    assert mex.venue == "Mexico City"


def test_scheduled_match_has_no_score() -> None:
    fixtures = parse_openfootball(SAMPLE)
    canada = fixtures[1]
    assert canada.status is MatchStatus.SCHEDULED
    assert canada.ft_home is None


def test_kickoff_converted_to_utc() -> None:
    fixtures = parse_openfootball(SAMPLE)
    # 13:00 UTC-6 -> 19:00 UTC.
    mex = fixtures[0]
    assert mex.kickoff_utc.tzinfo == timezone.utc
    assert (mex.kickoff_utc.hour, mex.kickoff_utc.minute) == (19, 0)


def test_knockout_stage_uses_round_name() -> None:
    fixtures = parse_openfootball(SAMPLE)
    ko = fixtures[2]
    assert ko.stage == "Round of 32"
    assert ko.status is MatchStatus.SCHEDULED


def test_group_teams_extraction() -> None:
    fixtures = parse_openfootball(SAMPLE)
    groups = group_teams(fixtures)
    assert groups["Group A"] == ["Mexico", "South Africa"]
    assert groups["Group B"] == ["Canada", "Wales"]
    # Knockout rounds don't show up as groups.
    assert "Round of 32" not in groups


def test_unknown_wrapper_raises() -> None:
    with pytest.raises(ValueError):
        parse_openfootball({"unexpected": []})


def test_flat_matches_wrapper_also_supported() -> None:
    flat = {"matches": SAMPLE["rounds"][0]["matches"]}
    fixtures = parse_openfootball(flat)
    assert len(fixtures) == 2
    assert fixtures[0].home_team == "Mexico"


# --- regressions: time parsing and structure validation -----


def test_missing_time_falls_back_to_midnight_utc() -> None:
    m = parse_match(
        {"team1": "A", "team2": "B", "date": "2026-06-20", "group": "Group A"}
    )
    assert (m.kickoff_utc.hour, m.kickoff_utc.minute) == (0, 0)


def test_parse_match_canonicalizes_team_names_to_martj42() -> None:
    # openfootball says "USA"; the Elo source (martj42) knows it as "United States".
    # Without canonicalizing, the pre_tournament baseline would treat it as a team
    # with no history.
    m = parse_match(
        {"team1": "USA", "team2": "B", "date": "2026-06-20", "group": "Group A"}
    )
    assert m.home_team == "United States"
    assert "united-states" in m.match_id


def test_present_but_unparseable_time_raises() -> None:
    # time present but offset unrecognized: fail loud (no silent midnight fallback).
    with pytest.raises(ValueError):
        parse_match(
            {"team1": "A", "team2": "B", "date": "2026-06-20", "time": "21:00 GMT-5"}
        )


def test_offset_with_minutes_converts_correctly() -> None:
    m = parse_match(
        {
            "team1": "A",
            "team2": "B",
            "date": "2026-06-20",
            "time": "12:30 UTC+5:30",
            "group": "Group A",
        }
    )
    assert (m.kickoff_utc.hour, m.kickoff_utc.minute) == (7, 0)  # 12:30 +5:30 -> 07:00Z


def test_validate_schedule_flags_incomplete_structure() -> None:
    # The sample has 3 matches, not 104: must report discrepancies.
    issues = validate_schedule(parse_openfootball(SAMPLE))
    assert issues


def _full_valid_schedule() -> list[NormalizedMatch]:
    """Structurally valid schedule: 12 groups of 4 + 32 knockout matches."""
    kickoff = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
    matches: list[NormalizedMatch] = []
    for g in "ABCDEFGHIJKL":  # 12 groups
        teams = [f"{g}{i}" for i in range(1, 5)]  # 4 teams
        for i in range(4):
            for j in range(i + 1, 4):  # 6 round-robin matches
                matches.append(
                    NormalizedMatch(
                        match_id=f"group-{g}-{i}{j}",
                        source="x",
                        source_match_id="x",
                        kickoff_utc=kickoff,
                        home_team=teams[i],
                        away_team=teams[j],
                        stage=f"Group {g}",
                        status=MatchStatus.SCHEDULED,
                    )
                )
    for k in range(32):  # 32 knockout matches
        matches.append(
            NormalizedMatch(
                match_id=f"ko-{k}",
                source="x",
                source_match_id="x",
                kickoff_utc=kickoff,
                home_team=f"K{k}a",
                away_team=f"K{k}b",
                stage="Round of 32",
                status=MatchStatus.SCHEDULED,
            )
        )
    return matches


def test_validate_schedule_accepts_complete_structure() -> None:
    matches = _full_valid_schedule()
    assert len(matches) == 104
    assert validate_schedule(matches) == []


# --- Fixture helpers -----


_UTC = timezone.utc


def _nm(
    home: str, away: str, stage: str, status: MatchStatus, **kw: Any
) -> NormalizedMatch:
    base: dict[str, Any] = dict(
        match_id=f"{stage}:{home}-{away}",
        source="openfootball",
        source_match_id="x",
        kickoff_utc=datetime(2026, 6, 11, tzinfo=_UTC),
        home_team=home,
        away_team=away,
        stage=stage,
        status=status,
    )
    base.update(kw)
    return NormalizedMatch(**base)


def test_is_knockout_stage() -> None:
    assert is_knockout_stage("Round of 32") is True
    assert is_knockout_stage("Quarter-finals") is True
    assert is_knockout_stage("Group A") is False


def test_remaining_fixtures_excludes_finished_and_sorts_by_kickoff() -> None:
    played = _nm(
        "A",
        "B",
        "Group A",
        MatchStatus.FINISHED,
        kickoff_utc=datetime(2026, 6, 11, tzinfo=_UTC),
    )
    later = _nm(
        "C",
        "D",
        "Group A",
        MatchStatus.SCHEDULED,
        kickoff_utc=datetime(2026, 6, 20, tzinfo=_UTC),
    )
    sooner = _nm(
        "E",
        "F",
        "Round of 32",
        MatchStatus.SCHEDULED,
        kickoff_utc=datetime(2026, 6, 15, tzinfo=_UTC),
    )
    out = remaining_fixtures([played, later, sooner])
    assert [m.home_team for m in out] == ["E", "C"]


def test_is_predictable_rejects_placeholder_team() -> None:
    known = {"Colombia", "Ghana"}
    assert is_predictable("Colombia", "Ghana", known) is True
    assert is_predictable("Colombia", "Winner 50", known) is False
