"""Conditional Monte Carlo for the tournament: simulate what's pending, tally results.

Each run: simulate the pending group matches (using locked results where they
exist), compute standings (Art. 13), determine winners/runners-up and the 8
best third-placed teams (Annex C), build the bracket, and simulate the
knockout stage (conditioned on locks). Aggregating over ``runs`` gives
P(advance / R16 / QF / SF / final / champion) per team. Deterministic given
the snapshot (locks) + seed.

*Host bonus:* hosts (``config.simulation.hosts``) get ``elo.host_advantage``
in their simulated matches, via ``host_advantage`` (``team -> bonus`` map); in
a given matchup the bonus is net (home host minus away host), so two hosts
playing each other is neutral.
*Performance:* no matrix caching (correctness first); for 50k runs it'd be
worth caching ``score_matrix`` per matchup (future work).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..models.base import MatchModel
from ..rng import spawn_rngs
from .bracket import KNOCKOUT_BRACKET, assign_best_thirds
from .group_stage import PlayedMatch, TeamStanding, rank_thirds, standings
from .match import simulate_match

# Rounds in order of depth. "advance" = qualify for the Round of 32.
ROUNDS = (
    "advance",
    "round_of_16",
    "quarter_finals",
    "semi_finals",
    "final",
    "champion",
)
_ROUND_INDEX = {r: i for i, r in enumerate(ROUNDS)}

# The bracket and Annex C are WC2026-specific: 12 groups A-L of 4 teams each.
_GROUP_KEYS = frozenset("ABCDEFGHIJKL")


def _net_host_advantage(
    home: str, away: str, host_advantage: dict[str, float]
) -> float:
    """Net host bonus: +bonus if home is a host, -bonus if away is a host.

    Two hosts facing each other (or neither) -> 0. The bracket's home/away
    label doesn't reflect the real venue (everything is played in US/MX/CA),
    so the bonus goes to whichever side is a host.
    """
    return host_advantage.get(home, 0.0) - host_advantage.get(away, 0.0)


def _validate_groups(groups: dict[str, list[str]]) -> None:
    """Fail loudly if ``groups`` isn't the WC2026 structure (12 groups A-L of 4).

    Without this, a malformed input (wrong group count, unexpected group
    labels) would blow up with an opaque ``KeyError`` mid-bracket instead of
    a clear error up front.
    """
    if set(groups) != _GROUP_KEYS:
        raise ValueError(f"expected 12 groups A-L; got {sorted(groups)}")
    for group, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"group {group} has {len(teams)} teams != 4")


def _simulate_group(
    teams: list[str],
    ratings: dict[str, float],
    model: MatchModel,
    rng: np.random.Generator,
    locked_group: dict[frozenset[str], PlayedMatch],
    et_total: float,
    denom: float,
    host_advantage: dict[str, float],
) -> list[TeamStanding]:
    """Simulate (or use the lock for) the group's 6 matches; return standings."""
    matches: list[PlayedMatch] = []
    for home, away in combinations(teams, 2):
        locked = locked_group.get(frozenset((home, away)))
        if locked is not None:
            matches.append(locked)
        else:
            result = simulate_match(
                model,
                home,
                away,
                ratings[home],
                ratings[away],
                rng,
                home_advantage=_net_host_advantage(home, away, host_advantage),
                extra_time_total_goals=et_total,
                elo_denominator=denom,
            )
            matches.append(
                PlayedMatch(
                    result.home, result.away, result.home_goals, result.away_goals
                )
            )
    return standings(matches, teams, ratings)


def _simulate_once(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    model: MatchModel,
    annex_c: dict[frozenset[str], dict[str, str]],
    rng: np.random.Generator,
    locked_group: dict[frozenset[str], PlayedMatch],
    locked_knockout: dict[frozenset[str], str],
    et_total: float,
    denom: float,
    host_advantage: dict[str, float],
) -> dict[str, str]:
    """One tournament realization -> ``{team: deepest round reached}``."""
    group_standings = {
        g: _simulate_group(
            teams, ratings, model, rng, locked_group, et_total, denom, host_advantage
        )
        for g, teams in groups.items()
    }
    winners = {g: gs[0].team for g, gs in group_standings.items()}
    runners = {g: gs[1].team for g, gs in group_standings.items()}
    third_of_group = {g: gs[2] for g, gs in group_standings.items()}
    group_of_third = {ts.team: g for g, ts in third_of_group.items()}

    ranked_thirds = rank_thirds(list(third_of_group.values()), ratings)
    top_thirds = ranked_thirds[:8]
    qualifying_groups = {group_of_third[ts.team] for ts in top_thirds}
    assignment = assign_best_thirds(qualifying_groups, annex_c)  # {"1A": group}

    def resolve(slot: tuple[str, object]) -> str:
        kind, ref = slot
        if kind == "W":
            return winners[str(ref)]
        if kind == "R":
            return runners[str(ref)]
        if kind == "T":
            return third_of_group[assignment[str(ref)]].team
        assert isinstance(ref, int)  # "MW"/"ML" refs are match ids
        return match_winner[ref] if kind == "MW" else match_loser[ref]

    reached: dict[str, str] = {}
    for team in (*winners.values(), *runners.values(), *(t.team for t in top_thirds)):
        reached[team] = "advance"

    match_winner: dict[int, str] = {}
    match_loser: dict[int, str] = {}
    for match_id in sorted(KNOCKOUT_BRACKET):
        slot_a, slot_b = KNOCKOUT_BRACKET[match_id]
        home, away = resolve(slot_a), resolve(slot_b)
        locked_winner = locked_knockout.get(frozenset((home, away)))
        if locked_winner is not None:
            winner = locked_winner
        else:
            winner = (
                simulate_match(
                    model,
                    home,
                    away,
                    ratings[home],
                    ratings[away],
                    rng,
                    home_advantage=_net_host_advantage(home, away, host_advantage),
                    knockout=True,
                    extra_time_total_goals=et_total,
                    elo_denominator=denom,
                ).winner
                or home
            )
        match_winner[match_id] = winner
        match_loser[match_id] = away if winner == home else home
        if 73 <= match_id <= 88:
            reached[winner] = "round_of_16"
        elif 89 <= match_id <= 96:
            reached[winner] = "quarter_finals"
        elif 97 <= match_id <= 100:
            reached[winner] = "semi_finals"
        elif match_id in (101, 102):
            reached[winner] = "final"
    reached[match_winner[104]] = "champion"
    return reached


def run_tournament(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    model: MatchModel,
    annex_c: dict[frozenset[str], dict[str, str]],
    *,
    runs: int,
    seed: int,
    extra_time_total_goals: float = 0.8,
    elo_denominator: float = 400.0,
    locked_group: dict[frozenset[str], PlayedMatch] | None = None,
    locked_knockout: dict[frozenset[str], str] | None = None,
    host_advantage: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Run the conditional Monte Carlo and return ``{team: {round: probability}}``.

    ``locked_group`` maps ``frozenset({home, away}) -> PlayedMatch`` (results
    already played); ``locked_knockout`` maps ``frozenset({home, away}) ->
    winner``. Locked matches are never re-sampled. ``host_advantage`` maps
    ``host -> Elo bonus`` (applied net per matchup; empty = neutral venue).
    Deterministic given ``(locks, seed)``.
    """
    _validate_groups(groups)
    locked_group = locked_group or {}
    locked_knockout = locked_knockout or {}
    host_advantage = host_advantage or {}
    counts = {t: dict.fromkeys(ROUNDS, 0) for t in ratings}

    for run_rng in spawn_rngs(seed, runs):
        reached = _simulate_once(
            groups,
            ratings,
            model,
            annex_c,
            run_rng,
            locked_group,
            locked_knockout,
            extra_time_total_goals,
            elo_denominator,
            host_advantage,
        )
        for team, deepest in reached.items():
            for i in range(
                _ROUND_INDEX[deepest] + 1
            ):  # reached every round up to the deepest one
                counts[team][ROUNDS[i]] += 1

    return {team: {rnd: counts[team][rnd] / runs for rnd in ROUNDS} for team in ratings}
