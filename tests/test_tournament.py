"""Tests for the conditional Monte Carlo (bracket invariants + reproducibility)."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from worldcup.config import load_config
from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.models.dixon_coles import DixonColesModel
from worldcup.simulation.bracket import load_annex_c
from worldcup.simulation.group_stage import PlayedMatch
from worldcup.simulation.state import TournamentState
from worldcup.simulation.tournament import (
    _net_host_advantage,
    resolve_bracket,
    run_tournament,
)

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
ANNEX = load_annex_c(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "annex_c_2026.json"
)
MODEL = DixonColesModel(CFG.elo, CFG.dixon_coles)

GROUPS = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCDEFGHIJKL"}
ALL_TEAMS = [t for teams in GROUPS.values() for t in teams]
BASE_RATINGS = {t: 1500.0 for t in ALL_TEAMS}

UTC = timezone.utc


def _all_groups_locked() -> dict[frozenset[str], PlayedMatch]:
    """Lock every group match with a fixed intra-group pecking order (Xn beats Xm, n<m).

    Gives fully deterministic standings (X1 > X2 > X3 > X4 in every group) so the
    resolved bracket is pinned down exactly, with no Elo tiebreak ambiguity.
    """
    locked: dict[frozenset[str], PlayedMatch] = {}
    for teams in GROUPS.values():
        for home, away in combinations(teams, 2):
            if teams.index(home) < teams.index(away):
                locked[frozenset((home, away))] = PlayedMatch(home, away, 3, 0)
            else:
                locked[frozenset((home, away))] = PlayedMatch(home, away, 0, 3)
    return locked


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


def _full_state(
    locked_knockout: dict[frozenset[str], str] | None = None,
    *,
    incomplete_group: bool = False,
) -> TournamentState:
    """Full 12-group state with every group match locked (unless incomplete)."""
    locked_group = _all_groups_locked()
    if incomplete_group:
        # Drop one A-group result so group A is no longer complete.
        del locked_group[frozenset(("A1", "A2"))]
    return TournamentState(
        groups=GROUPS,
        ratings=BASE_RATINGS,
        locked_group=locked_group,
        locked_knockout=locked_knockout or {},
    )


def _run(
    ratings: dict[str, float], runs: int = 200, seed: int = 42, **kw: Any
) -> dict[str, dict[str, float]]:
    return run_tournament(GROUPS, ratings, MODEL, ANNEX, runs=runs, seed=seed, **kw)


def test_exactly_one_champion_per_run() -> None:
    probs = _run(BASE_RATINGS)
    total = sum(probs[t]["champion"] for t in ALL_TEAMS)
    assert abs(total - 1.0) < 1e-9  # P(champion) across all teams sums to 1


def test_advance_probabilities_average_to_two_thirds() -> None:
    # 32 of 48 advance to R32 -> mean P(advance) is 32/48.
    probs = _run(BASE_RATINGS)
    mean_advance = sum(probs[t]["advance"] for t in ALL_TEAMS) / len(ALL_TEAMS)
    assert abs(mean_advance - 32 / 48) < 1e-9


def test_reproducible_for_same_seed() -> None:
    assert _run(BASE_RATINGS, seed=7) == _run(BASE_RATINGS, seed=7)


def test_strongest_team_is_most_likely_champion() -> None:
    ratings = dict(BASE_RATINGS)
    ratings["A1"] = 2400.0  # far above the rest
    probs = _run(ratings, runs=300)
    champ = {t: probs[t]["champion"] for t in ALL_TEAMS}
    assert max(champ, key=lambda t: champ[t]) == "A1"
    assert champ["A1"] > 0.25


def test_net_host_advantage_is_signed_and_neutral_between_hosts() -> None:
    hosts = {"USA": 37.5}
    assert _net_host_advantage("USA", "Brazil", hosts) == 37.5  # home is host
    assert _net_host_advantage("Brazil", "USA", hosts) == -37.5  # away is host
    assert _net_host_advantage("Brazil", "Argentina", hosts) == 0.0  # neither is host


def test_host_advantage_raises_host_champion_probability() -> None:
    # With equal ratings, the host bonus should raise the host's P(champion).
    base = _run(BASE_RATINGS, runs=400, seed=11)
    boosted = _run(BASE_RATINGS, runs=400, seed=11, host_advantage={"A1": 300.0})
    assert boosted["A1"]["champion"] > base["A1"]["champion"]


def test_rejects_malformed_groups() -> None:
    # 11 groups instead of 12 -> clear input error (not an opaque KeyError later).
    bad = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCDEFGHIJK"}
    ratings = {t: 1500.0 for teams in bad.values() for t in teams}
    with pytest.raises(ValueError, match="12 groups"):
        run_tournament(bad, ratings, MODEL, ANNEX, runs=10, seed=1)


def test_rejects_group_with_wrong_team_count() -> None:
    bad = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCDEFGHIJKL"}
    bad["A"] = ["A1", "A2", "A3"]  # only 3 teams
    ratings = {t: 1500.0 for teams in bad.values() for t in teams}
    with pytest.raises(ValueError, match="!= 4"):
        run_tournament(bad, ratings, MODEL, ANNEX, runs=10, seed=1)


def test_locked_group_result_eliminates_a_team() -> None:
    # Lock group A so A4 loses all 3 -> finishes 4th -> never advances.
    locked = {
        frozenset(("A1", "A2")): PlayedMatch("A1", "A2", 1, 0),
        frozenset(("A1", "A3")): PlayedMatch("A1", "A3", 1, 0),
        frozenset(("A1", "A4")): PlayedMatch("A1", "A4", 5, 0),
        frozenset(("A2", "A3")): PlayedMatch("A2", "A3", 1, 0),
        frozenset(("A2", "A4")): PlayedMatch("A2", "A4", 5, 0),
        frozenset(("A3", "A4")): PlayedMatch("A3", "A4", 5, 0),
    }
    probs = _run(BASE_RATINGS, locked_group=locked)
    assert probs["A4"]["advance"] == 0.0  # eliminated in groups, conditioned
    assert probs["A1"]["advance"] == 1.0  # group winner, always advances


# --- resolve_bracket ---------------------------------------------------------


def test_resolve_bracket_all_groups_locked_no_ko_results() -> None:
    # (a) Every group complete, no knockout locks: every R32 tie has both teams
    # known ("scheduled" once joined, never "tbd"); every R16+ tie is "tbd".
    state = _full_state()
    resolved = resolve_bracket(state, ANNEX, fixtures=[])
    for match_id in range(73, 89):
        tie = resolved[match_id]
        assert tie.home is not None and tie.away is not None
        assert tie.status != "tbd"
    for match_id in range(89, 105):
        assert resolved[match_id].status == "tbd"


def test_resolve_bracket_propagates_locked_r32_result_to_r16() -> None:
    # (b) Locking M74 (E1 vs C3) and M77 (I1 vs F3), per Annex C with our
    # deterministic standings, resolves M89 (MW74 vs MW77) to those winners,
    # and M74's score carries through unchanged.
    fixtures = [
        _nm(
            "E1",
            "C3",
            "Round of 32",
            MatchStatus.FINISHED,
            ft_home=2,
            ft_away=1,
            kickoff_utc=datetime(2026, 6, 30, tzinfo=UTC),
        ),
        _nm("I1", "F3", "Round of 32", MatchStatus.FINISHED, ft_home=1, ft_away=0),
    ]
    state = _full_state(
        locked_knockout={
            frozenset(("E1", "C3")): "E1",
            frozenset(("I1", "F3")): "I1",
        }
    )
    resolved = resolve_bracket(state, ANNEX, fixtures)

    m74 = resolved[74]
    assert {m74.home, m74.away} == {"E1", "C3"}
    assert m74.status == "finished"
    assert m74.winner == "E1"
    assert m74.ft_home == 2 and m74.ft_away == 1
    assert m74.kickoff == datetime(2026, 6, 30, tzinfo=UTC).isoformat()
    assert m74.stage == "Round of 32"

    m89 = resolved[89]
    assert {m89.home, m89.away} == {"E1", "I1"}  # winners of M74/M77 propagated
    assert m89.winner is None  # M89 itself isn't locked
    assert m89.status == "scheduled"


def test_resolve_bracket_incomplete_group_leaves_r32_tbd() -> None:
    # (c) One group incomplete -> group-derived slots are all unresolved.
    state = _full_state(incomplete_group=True)
    resolved = resolve_bracket(state, ANNEX, fixtures=[])
    for match_id in range(73, 89):
        assert resolved[match_id].status == "tbd"
        assert resolved[match_id].home is None
        assert resolved[match_id].away is None


def test_resolve_bracket_is_deterministic() -> None:
    # (d) Same inputs -> equal outputs, twice; no RNG argument exists.
    state = _full_state(locked_knockout={frozenset(("E1", "C3")): "E1"})
    fixtures = [
        _nm("E1", "C3", "Round of 32", MatchStatus.FINISHED, ft_home=2, ft_away=1),
    ]
    first = resolve_bracket(state, ANNEX, fixtures)
    second = resolve_bracket(state, ANNEX, fixtures)
    assert first == second


def test_resolve_bracket_unknown_team_is_tbd() -> None:
    # A slot whose team isn't determined yet (all R16+ before any KO lock) TBDs
    # with the round label attached, both teams None, and no winner/score.
    state = _full_state()
    resolved = resolve_bracket(state, ANNEX, fixtures=[])
    tie = resolved[89]
    assert tie.home is None
    assert tie.away is None
    assert tie.winner is None
    assert tie.ft_home is None and tie.ft_away is None
    assert tie.stage == "Round of 16"


def test_resolve_bracket_no_simulate_match_import() -> None:
    # The resolver must be RNG-free: simulate_match must not appear in the module.
    import inspect

    from worldcup.simulation import tournament as tournament_module

    source = inspect.getsource(tournament_module)
    resolver_source = source[source.index("def resolve_bracket") :]
    assert "simulate_match" not in resolver_source
