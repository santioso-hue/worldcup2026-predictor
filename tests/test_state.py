"""Tests for TournamentState: groups + locks (group and knockout) from the schedule."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pytest

from worldcup.config import load_config
from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.models.dixon_coles import DixonColesModel
from worldcup.simulation.bracket import load_annex_c
from worldcup.simulation.group_stage import PlayedMatch
from worldcup.simulation.state import _winner_of, build_state, run_from_state

UTC = timezone.utc
_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(_ROOT / "config" / "config.yaml")
ANNEX = load_annex_c(_ROOT / "data" / "raw" / "annex_c_2026.json")
MODEL = DixonColesModel(CFG.elo, CFG.dixon_coles)


def _nm(
    home: str, away: str, stage: str, status: MatchStatus, **kw: object
) -> NormalizedMatch:
    base: dict[str, object] = dict(
        match_id=f"{stage}:{home}-{away}",
        source="openfootball",
        source_match_id="x",
        kickoff_utc=datetime(2026, 6, 11, tzinfo=UTC),
        home_team=home,
        away_team=away,
        stage=stage,
        status=status,
    )
    base.update(kw)
    return NormalizedMatch(**base)  # type: ignore[arg-type]


def test_groups_use_single_letter_keys() -> None:
    fx = [
        _nm("A1", "A2", "Group A", MatchStatus.FINISHED, ft_home=2, ft_away=0),
        _nm("A3", "A4", "Group A", MatchStatus.SCHEDULED),
    ]
    state = build_state(fx, {f"A{i}": 1500.0 for i in range(1, 5)})
    assert state.groups["A"] == ["A1", "A2", "A3", "A4"]


def test_locks_finished_group_matches_only() -> None:
    fx = [
        _nm("A1", "A2", "Group A", MatchStatus.FINISHED, ft_home=2, ft_away=0),
        _nm("A3", "A4", "Group A", MatchStatus.SCHEDULED),
    ]
    state = build_state(fx, {f"A{i}": 1500.0 for i in range(1, 5)})
    assert state.locked_group[frozenset(("A1", "A2"))] == PlayedMatch("A1", "A2", 2, 0)
    assert frozenset(("A3", "A4")) not in state.locked_group


def test_locks_finished_knockout_with_winner() -> None:
    fx = [
        _nm(
            "X",
            "Y",
            "Round of 16",
            MatchStatus.FINISHED,
            ft_home=1,
            ft_away=1,
            et_home=1,
            et_away=1,
            pen_home=4,
            pen_away=2,
        ),
    ]
    state = build_state(fx, {"X": 1500.0, "Y": 1500.0})
    assert state.locked_knockout[frozenset(("X", "Y"))] == "X"  # won on penalties


def test_winner_of_uses_phase_priority() -> None:
    reg = _nm("H", "A", "Final", MatchStatus.FINISHED, ft_home=2, ft_away=1)
    et = _nm(
        "H",
        "A",
        "Final",
        MatchStatus.FINISHED,
        ft_home=0,
        ft_away=0,
        et_home=2,
        et_away=1,
    )
    pen = _nm(
        "H",
        "A",
        "Final",
        MatchStatus.FINISHED,
        ft_home=1,
        ft_away=1,
        et_home=1,
        et_away=1,
        pen_home=5,
        pen_away=4,
    )
    assert _winner_of(reg) == "H"
    assert _winner_of(et) == "H"
    assert _winner_of(pen) == "H"


def test_winner_of_raises_on_unresolved_knockout() -> None:
    # KO "finished" drawn at 90' with no ET/penalties (corrupt data) -> fail loud.
    drawn = _nm("X", "Y", "Round of 16", MatchStatus.FINISHED, ft_home=1, ft_away=1)
    with pytest.raises(ValueError):
        _winner_of(drawn)


def test_winner_of_raises_on_incomplete_phase() -> None:
    # Penalties with only one side recorded (partial data) -> don't guess, fail.
    partial = _nm(
        "X",
        "Y",
        "Final",
        MatchStatus.FINISHED,
        ft_home=1,
        ft_away=1,
        et_home=1,
        et_away=1,
        pen_home=5,
    )
    with pytest.raises(ValueError):
        _winner_of(partial)


def test_run_from_state_end_to_end() -> None:
    fx = []
    for g in "ABCDEFGHIJKL":
        teams = [f"{g}{i}" for i in range(1, 5)]
        for home, away in combinations(teams, 2):
            fx.append(_nm(home, away, f"Group {g}", MatchStatus.SCHEDULED))
    ratings = {f"{g}{i}": 1500.0 for g in "ABCDEFGHIJKL" for i in range(1, 5)}
    state = build_state(fx, ratings)
    probs = run_from_state(state, MODEL, ANNEX, runs=100, seed=42)
    assert abs(sum(probs[t]["champion"] for t in ratings) - 1.0) < 1e-9
