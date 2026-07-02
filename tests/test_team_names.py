"""Tests for team-name canonicalization (football-data -> martj42)."""

from __future__ import annotations

from worldcup.data.team_names import (
    canonical_footballdata_team,
    canonical_openfootball_team,
)


def test_known_aliases_map_to_martj42_names() -> None:
    assert canonical_footballdata_team("Congo DR") == "DR Congo"
    assert canonical_footballdata_team("Czechia") == "Czech Republic"
    assert canonical_footballdata_team("Cape Verde Islands") == "Cape Verde"
    assert canonical_footballdata_team("Bosnia-Herzegovina") == "Bosnia and Herzegovina"


def test_unaliased_name_is_identity() -> None:
    # Most teams already match: no alias means the name passes through unchanged.
    assert canonical_footballdata_team("Colombia") == "Colombia"
    assert canonical_footballdata_team("Brazil") == "Brazil"


def test_congo_is_not_collapsed_into_dr_congo() -> None:
    # "Congo" (Republic of Congo) is a different team from "DR Congo": don't merge them.
    assert canonical_footballdata_team("Congo") == "Congo"


def test_openfootball_aliases_map_to_martj42_names() -> None:
    # openfootball spells 2 of the 48 teams differently from martj42; without this
    # they'd fall back to Elo 1500 and 'USA' (the host) would lose its host bonus.
    assert canonical_openfootball_team("USA") == "United States"
    bih = canonical_openfootball_team("Bosnia & Herzegovina")
    assert bih == "Bosnia and Herzegovina"
    assert canonical_openfootball_team("Colombia") == "Colombia"  # identity
