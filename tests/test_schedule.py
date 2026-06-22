"""Tests del parser del schedule backbone (openfootball -> fixtures normalizados)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.data.schedule import (
    group_teams,
    make_match_id,
    parse_match,
    parse_openfootball,
    validate_schedule,
)

# Muestra mínima con el esquema VERIFICADO de openfootball (matchday + knockout).
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
    # 13:00 en UTC-6 -> 19:00 UTC.
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
    # La eliminatoria no aparece como grupo.
    assert "Round of 32" not in groups


def test_unknown_wrapper_raises() -> None:
    with pytest.raises(ValueError):
        parse_openfootball({"unexpected": []})


def test_flat_matches_wrapper_also_supported() -> None:
    flat = {"matches": SAMPLE["rounds"][0]["matches"]}
    fixtures = parse_openfootball(flat)
    assert len(fixtures) == 2
    assert fixtures[0].home_team == "Mexico"


# --- regresiones del review: parsing de hora y validación de estructura -----


def test_missing_time_falls_back_to_midnight_utc() -> None:
    m = parse_match(
        {"team1": "A", "team2": "B", "date": "2026-06-20", "group": "Group A"}
    )
    assert (m.kickoff_utc.hour, m.kickoff_utc.minute) == (0, 0)


def test_present_but_unparseable_time_raises() -> None:
    # Con time presente pero offset no reconocido: fail loud (no medianoche silenciosa).
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
    # La muestra tiene 3 partidos, no 104: debe reportar discrepancias.
    issues = validate_schedule(parse_openfootball(SAMPLE))
    assert issues


def _full_valid_schedule() -> list[NormalizedMatch]:
    """Schedule estructuralmente válido: 12 grupos de 4 + 32 de eliminatoria."""
    kickoff = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
    matches: list[NormalizedMatch] = []
    for g in "ABCDEFGHIJKL":  # 12 grupos
        teams = [f"{g}{i}" for i in range(1, 5)]  # 4 equipos
        for i in range(4):
            for j in range(i + 1, 4):  # 6 partidos round-robin
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
    for k in range(32):  # 32 partidos de eliminatoria
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
