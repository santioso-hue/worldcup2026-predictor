"""Standings de grupo (FWC2026 Art. 13) y ranking de los mejores terceros.

Cadena de desempate VERIFICADA (Art. 13): puntos → **head-to-head** (pts/GD/GF entre los
empatados, recursivo "matches between the remaining teams only") → GD global → goles
globales → [conduct score: omitido, no modelamos tarjetas] → **Elo** (proxy determinista
del ranking FIFA, el paso final). El head-to-head va ANTES que el GD global.

Funciones puras (sin I/O, sin RNG).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayedMatch:
    """Un partido ya jugado/simulado (marcador final reglamentario)."""

    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True, slots=True)
class TeamStanding:
    """Registro global de un equipo en su grupo."""

    team: str
    points: int
    goal_diff: int
    goals_for: int


def _stats(
    matches: list[PlayedMatch], subset: list[str]
) -> dict[str, tuple[int, int, int]]:
    """``{team: (pts, gd, gf)}`` contando solo partidos entre equipos de ``subset``."""
    members = set(subset)
    pts = {t: 0 for t in subset}
    gd = {t: 0 for t in subset}
    gf = {t: 0 for t in subset}
    for m in matches:
        if m.home not in members or m.away not in members:
            continue
        if m.home_goals > m.away_goals:
            pts[m.home] += 3
        elif m.home_goals < m.away_goals:
            pts[m.away] += 3
        else:
            pts[m.home] += 1
            pts[m.away] += 1
        gd[m.home] += m.home_goals - m.away_goals
        gd[m.away] += m.away_goals - m.home_goals
        gf[m.home] += m.home_goals
        gf[m.away] += m.away_goals
    return {t: (pts[t], gd[t], gf[t]) for t in subset}


def _bucket(items: list[str], key: Callable[[str], object]) -> list[list[str]]:
    """Ordena descendente por ``key`` y agrupa elementos con la misma clave."""
    ordered = sorted(items, key=key, reverse=True)  # type: ignore[arg-type]
    buckets: list[list[str]] = []
    for it in ordered:
        if buckets and key(buckets[-1][0]) == key(it):
            buckets[-1].append(it)
        else:
            buckets.append([it])
    return buckets


def standings(
    matches: list[PlayedMatch], teams: list[str], elo_ratings: dict[str, float]
) -> list[TeamStanding]:
    """Ordena los equipos de un grupo (1º→4º) según el Art. 13.

    Parameters
    ----------
    matches:
        Los 6 partidos del grupo (marcadores reglamentarios).
    teams:
        Los 4 equipos del grupo.
    elo_ratings:
        Ratings Elo (proxy del ranking FIFA para el desempate final).

    Returns
    -------
    list[TeamStanding]
        Equipos en orden de clasificación, con su registro global.
    """
    overall = _stats(matches, teams)

    def rank_tied(subset: list[str]) -> list[str]:
        """Ordena un subconjunto empatado en puntos (head-to-head recursivo)."""
        if len(subset) == 1:
            return list(subset)
        h2h = _stats(matches, subset)  # "remaining teams only" al recursar
        buckets = _bucket(subset, key=lambda t: h2h[t])
        if len(buckets) > 1:
            ordered: list[str] = []
            for bucket in buckets:
                ordered.extend(rank_tied(bucket))
            return ordered
        # H2H no separó: GD global → goles globales → Elo (conduct omitido).
        return sorted(
            subset,
            key=lambda t: (overall[t][1], overall[t][2], elo_ratings[t]),
            reverse=True,
        )

    ordered_teams: list[str] = []
    for bucket in _bucket(teams, key=lambda t: overall[t][0]):  # por puntos
        ordered_teams.extend(rank_tied(bucket) if len(bucket) > 1 else bucket)
    return [TeamStanding(t, *overall[t]) for t in ordered_teams]


def rank_thirds(
    thirds: list[TeamStanding], elo_ratings: dict[str, float]
) -> list[TeamStanding]:
    """Ordena los terceros (sin H2H — distintos grupos): pts → GD → goles → Elo.

    El proxy Elo cubre el paso final (ranking FIFA); el conduct score se omite.
    """
    return sorted(
        thirds,
        key=lambda s: (s.points, s.goal_diff, s.goals_for, elo_ratings[s.team]),
        reverse=True,
    )
