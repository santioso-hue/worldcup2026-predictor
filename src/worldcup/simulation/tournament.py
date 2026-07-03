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

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ..data.live_results import NormalizedMatch
from ..data.schedule import is_knockout_stage
from ..models.base import MatchModel
from ..rng import spawn_rngs
from .bracket import KNOCKOUT_BRACKET, assign_best_thirds
from .group_stage import PlayedMatch, TeamStanding, rank_thirds, standings
from .match import simulate_match

if TYPE_CHECKING:
    from .state import TournamentState

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


# --- Deterministic bracket resolution (no simulation, real results only) ----

# Round label by match-number range, for slots with no joined fixture yet.
_ROUND_LABELS: tuple[tuple[range, str], ...] = (
    (range(73, 89), "Round of 32"),
    (range(89, 97), "Round of 16"),
    (range(97, 101), "Quarter-finals"),
    (range(101, 103), "Semi-finals"),
    (range(103, 104), "Third place"),
    (range(104, 105), "Final"),
)


def _round_label(match_id: int) -> str | None:
    """Round label for ``match_id``, by the ranges in ``_ROUND_LABELS``."""
    for span, label in _ROUND_LABELS:
        if match_id in span:
            return label
    return None


@dataclass(frozen=True, slots=True)
class ResolvedTie:
    """One knockout slot's real-world state: decided, scheduled, or unresolved.

    ``home``/``away`` are ``None`` when the slot's team isn't determined yet.
    ``status`` is ``"finished"`` (locked result), ``"scheduled"`` (both teams
    known, not yet decided) or ``"tbd"`` (at least one team unknown).
    """

    home: str | None
    away: str | None
    status: str
    ft_home: int | None
    ft_away: int | None
    winner: str | None
    kickoff: str | None
    stage: str | None
    pen_home: int | None = None
    pen_away: int | None = None
    et_home: int | None = None
    et_away: int | None = None


def resolve_bracket(
    state: TournamentState,
    annex_c: dict[frozenset[str], dict[str, str]],
    fixtures: list[NormalizedMatch],
) -> dict[int, ResolvedTie]:
    """Resolve the real knockout bracket from locked results only (RNG-free).

    Mirrors the ``_simulate_once`` recipe (group standings -> winners/runners
    -> best thirds -> knockout slots) but never simulates: a slot with an
    undetermined input resolves to ``None`` instead of being sampled. Real
    knockout results come only from ``state.locked_knockout`` (team-pair
    keyed); a tie with an unknown team, or whose result isn't locked, has no
    winner.

    Parameters
    ----------
    state:
        Live tournament state (groups, ratings, locked group/knockout results).
    annex_c:
        Best-thirds assignment table (:func:`load_annex_c`).
    fixtures:
        All fixtures (group + knockout); used to join real scores/kickoffs
        onto resolved knockout ties via ``frozenset((home, away))``.

    Returns
    -------
    dict[int, ResolvedTie]
        One entry per match in :data:`KNOCKOUT_BRACKET` (73-104, incl. 103).
    """
    winners: dict[str, str] = {}
    runners: dict[str, str] = {}
    third_of_group: dict[str, TeamStanding] = {}
    assignment: dict[str, str] = {}

    all_groups_complete = all(
        frozenset((home, away)) in state.locked_group
        for teams in state.groups.values()
        for home, away in combinations(teams, 2)
    )
    if all_groups_complete:
        group_standings = {
            g: standings(
                [state.locked_group[frozenset((h, a))] for h, a in combinations(t, 2)],
                t,
                state.ratings,
            )
            for g, t in state.groups.items()
        }
        winners = {g: gs[0].team for g, gs in group_standings.items()}
        runners = {g: gs[1].team for g, gs in group_standings.items()}
        third_of_group = {g: gs[2] for g, gs in group_standings.items()}
        group_of_third = {ts.team: g for g, ts in third_of_group.items()}
        ranked_thirds = rank_thirds(list(third_of_group.values()), state.ratings)
        top_thirds = ranked_thirds[:8]
        qualifying_groups = {group_of_third[ts.team] for ts in top_thirds}
        assignment = assign_best_thirds(qualifying_groups, annex_c)

    def resolve(slot: tuple[str, object]) -> str | None:
        kind, ref = slot
        if kind == "W":
            return winners.get(str(ref))
        if kind == "R":
            return runners.get(str(ref))
        if kind == "T":
            group = assignment.get(str(ref))
            return third_of_group[group].team if group is not None else None
        assert isinstance(ref, int)  # "MW"/"ML" refs are match ids
        return match_winner.get(ref) if kind == "MW" else match_loser.get(ref)

    fixture_by_pair = {
        frozenset((m.home_team, m.away_team)): m
        for m in fixtures
        if is_knockout_stage(m.stage)
    }

    match_winner: dict[int, str | None] = {}
    match_loser: dict[int, str | None] = {}
    resolved: dict[int, ResolvedTie] = {}
    for match_id in sorted(KNOCKOUT_BRACKET):
        slot_a, slot_b = KNOCKOUT_BRACKET[match_id]
        home, away = resolve(slot_a), resolve(slot_b)

        locked_winner = (
            state.locked_knockout.get(frozenset((home, away)))
            if home is not None and away is not None
            else None
        )
        match_winner[match_id] = locked_winner
        if locked_winner is not None:
            match_loser[match_id] = away if locked_winner == home else home
        else:
            match_loser[match_id] = None

        resolved[match_id] = _build_tie(
            home, away, locked_winner, fixture_by_pair, match_id
        )
    return resolved


def _build_tie(
    home: str | None,
    away: str | None,
    locked_winner: str | None,
    fixture_by_pair: dict[frozenset[str], NormalizedMatch],
    match_id: int,
) -> ResolvedTie:
    """Join a resolved slot pair against real fixtures/results for one tie."""
    label = _round_label(match_id)
    if home is None or away is None:
        return ResolvedTie(None, None, "tbd", None, None, None, None, label)

    fixture = fixture_by_pair.get(frozenset((home, away)))
    if fixture is None:
        return ResolvedTie(
            home, away, "scheduled", None, None, locked_winner, None, label
        )

    status = "finished" if fixture.is_finished else "scheduled"
    return ResolvedTie(
        home,
        away,
        status,
        fixture.ft_home,
        fixture.ft_away,
        locked_winner,
        fixture.kickoff_utc.isoformat(),
        fixture.stage,
        fixture.pen_home,
        fixture.pen_away,
        fixture.et_home,
        fixture.et_away,
    )


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
