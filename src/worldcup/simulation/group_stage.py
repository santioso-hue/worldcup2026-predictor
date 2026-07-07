"""Group standings (FWC2026 Art. 13) and best-thirds ranking.

Verified tiebreak chain (Art. 13): points -> **head-to-head** (pts/GD/GF among tied
teams, recursive "matches between the remaining teams only") -> overall GD -> overall
goals -> [conduct score: skipped, we don't model cards] -> **Elo** (deterministic proxy
for the FIFA ranking, the final step). Head-to-head comes BEFORE overall GD.

Pure functions (no I/O, no RNG).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True, slots=True)
class PlayedMatch:
    """A played/simulated match (regulation final score)."""

    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True, slots=True)
class TeamStanding:
    """A team's overall record within its group."""

    team: str
    points: int
    goal_diff: int
    goals_for: int


def _stats(
    matches: list[PlayedMatch], subset: list[str]
) -> dict[str, tuple[int, int, int]]:
    """``{team: (pts, gd, gf)}`` counting only matches between teams in ``subset``."""
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


def _require_elo(teams: list[str], elo_ratings: dict[str, float]) -> None:
    """Fail loudly if the Elo (final-tiebreak proxy) doesn't cover every team.

    Elo is the deterministic final tiebreak (Art. 13 Step 3); a missing team means the
    caller wired something wrong. We validate up front so it fails regardless of the
    scoreline, instead of only when the tiebreak branch is actually reached.
    """
    missing = sorted({t for t in teams if t not in elo_ratings})
    if missing:
        raise ValueError(f"elo_ratings is missing teams: {missing}")


def _bucket(items: list[str], key: Callable[[str], object]) -> list[list[str]]:
    """Sort descending by ``key`` and group items sharing the same key."""
    ordered = sorted(items, key=key, reverse=True)  # type: ignore[arg-type]
    return [list(group) for _, group in groupby(ordered, key=key)]


def standings(
    matches: list[PlayedMatch], teams: list[str], elo_ratings: dict[str, float]
) -> list[TeamStanding]:
    """Order a group's teams (1st-4th) per Art. 13.

    Parameters
    ----------
    matches:
        The group's 6 matches (regulation scores).
    teams:
        The group's 4 teams.
    elo_ratings:
        Elo ratings (FIFA-ranking proxy for the final tiebreak).

    Returns
    -------
    list[TeamStanding]
        Teams in ranked order, with their overall record.
    """
    _require_elo(teams, elo_ratings)
    overall = _stats(matches, teams)

    def rank_tied(subset: list[str]) -> list[str]:
        """Order a subset tied on points (recursive head-to-head)."""
        if len(subset) == 1:
            return list(subset)
        h2h = _stats(matches, subset)  # "remaining teams only" when recursing
        buckets = _bucket(subset, key=lambda t: h2h[t])
        if len(buckets) > 1:
            ordered: list[str] = []
            for bucket in buckets:
                ordered.extend(rank_tied(bucket))
            return ordered
        # H2H didn't separate them: overall GD -> overall goals -> Elo (skip conduct).
        return sorted(
            subset,
            key=lambda t: (overall[t][1], overall[t][2], elo_ratings[t]),
            reverse=True,
        )

    ordered_teams: list[str] = []
    for bucket in _bucket(teams, key=lambda t: overall[t][0]):  # by points
        ordered_teams.extend(rank_tied(bucket) if len(bucket) > 1 else bucket)
    return [TeamStanding(t, *overall[t]) for t in ordered_teams]


def rank_thirds(
    thirds: list[TeamStanding], elo_ratings: dict[str, float]
) -> list[TeamStanding]:
    """Order third-place teams (no H2H, different groups): pts -> GD -> goals -> Elo.

    The Elo proxy covers the final step (FIFA ranking); conduct score is skipped.
    """
    _require_elo([s.team for s in thirds], elo_ratings)
    return sorted(
        thirds,
        key=lambda s: (s.points, s.goal_diff, s.goals_for, elo_ratings[s.team]),
        reverse=True,
    )
