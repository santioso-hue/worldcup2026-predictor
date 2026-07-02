"""Tests for the conditional Monte Carlo (bracket invariants + reproducibility)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from worldcup.config import load_config
from worldcup.models.dixon_coles import DixonColesModel
from worldcup.simulation.bracket import load_annex_c
from worldcup.simulation.group_stage import PlayedMatch
from worldcup.simulation.tournament import _net_host_advantage, run_tournament

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
ANNEX = load_annex_c(
    Path(__file__).resolve().parents[1] / "data" / "raw" / "annex_c_2026.json"
)
MODEL = DixonColesModel(CFG.elo, CFG.dixon_coles)

GROUPS = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCDEFGHIJKL"}
ALL_TEAMS = [t for teams in GROUPS.values() for t in teams]
BASE_RATINGS = {t: 1500.0 for t in ALL_TEAMS}


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
