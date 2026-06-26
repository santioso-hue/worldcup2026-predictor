"""Tests de canonicalización de nombres de selección (football-data -> martj42)."""

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
    # La mayoría de selecciones coinciden: sin alias, se devuelve igual.
    assert canonical_footballdata_team("Colombia") == "Colombia"
    assert canonical_footballdata_team("Brazil") == "Brazil"


def test_congo_is_not_collapsed_into_dr_congo() -> None:
    # "Congo" (Rep. del Congo) es una selección distinta de "DR Congo": no se fusiona.
    assert canonical_footballdata_team("Congo") == "Congo"


def test_openfootball_aliases_map_to_martj42_names() -> None:
    # openfootball escribe 2 de las 48 distinto a martj42; sin esto caerían a Elo 1500
    # y 'USA' (anfitrión) perdería el bono de sede.
    assert canonical_openfootball_team("USA") == "United States"
    bih = canonical_openfootball_team("Bosnia & Herzegovina")
    assert bih == "Bosnia and Herzegovina"
    assert canonical_openfootball_team("Colombia") == "Colombia"  # identidad
