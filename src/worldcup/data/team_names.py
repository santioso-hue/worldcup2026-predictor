"""Canonicalize team names across data sources.

The Elo model trains on martj42 (~49k matches); that's the project's **canonical**
naming. Live providers use different spellings ("Congo DR" vs "DR Congo", "Czechia"
vs "Czech Republic"): unmapped, a team can't find its history and falls back to the
default rating (``initial_rating`` = 1500), skewing the prediction.

Each alias here is **verified** against both sources. A wrong mapping would merge
two different teams — e.g. ``"Congo"`` (Republic of Congo) is not ``"DR Congo"`` —
which is worse than falling back to 1500.
"""

from __future__ import annotations

# football-data.org v4 -> martj42 (the Elo base). Verified 22 Jun 2026 against both
# feeds (live name on the left, martj42 name with real history on the right).
_FOOTBALLDATA_TO_CANONICAL: dict[str, str] = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def canonical_footballdata_team(name: str) -> str:
    """Translate a football-data.org name to its martj42 canonical form.

    Identity if there's no alias (most teams already match).
    """
    return _FOOTBALLDATA_TO_CANONICAL.get(name, name)


# openfootball/worldcup.json -> martj42. Verified 24 Jun 2026 against the real
# WC2026 schedule: 2 of the 48 teams differ; the other 46 already match martj42.
_OPENFOOTBALL_TO_CANONICAL: dict[str, str] = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def canonical_openfootball_team(name: str) -> str:
    """Translate an openfootball name to its martj42 canonical form.

    Identity if there's no alias. Without this, the pre_tournament baseline would
    fall back to the default rating (1500) for these teams, and 'USA' (the host)
    would miss its host bonus.
    """
    return _OPENFOOTBALL_TO_CANONICAL.get(name, name)
