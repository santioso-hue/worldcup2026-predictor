"""Tournament state: schedule + live snapshot to Monte Carlo inputs.

``build_state`` takes the **backbone** fixtures (``stage`` ``"Group A"`` and KO rounds)
already overlaid with live results, plus Elo ratings, and produces a
:class:`TournamentState`: the 12 groups (single-letter keys ``A``...``L``, matching the
bracket/Annex C), and the results already locked in for group and knockout play. The
Monte Carlo only simulates what's still pending, conditioned on those locks (live
reconditioning).

Knockout matches lock by **team pair** (``frozenset``), not match-id: in single
elimination two teams meet at most once, and the KO stage starts only once the group
stage is closed (deterministic bracket), so the pair identifies the tie unambiguously.
A match with no ``time``/no score stays unlocked.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.live_results import NormalizedMatch, is_lockable
from ..data.schedule import group_teams
from ..models.base import MatchModel
from .group_stage import PlayedMatch
from .tournament import run_tournament


@dataclass(frozen=True, slots=True)
class TournamentState:
    """Tournament state ready for the conditional Monte Carlo."""

    groups: dict[str, list[str]]  # "A".."L" -> teams
    ratings: dict[str, float]
    locked_group: dict[frozenset[str], PlayedMatch]
    locked_knockout: dict[frozenset[str], str]  # pair -> winner


def _winner_of(match: NormalizedMatch) -> str:
    """Winner of a FINISHED knockout match, by phase (penalties > extra time > 90').

    Fails loudly instead of guessing when there's no clear winner (incomplete phase,
    or a "finished" KO match that's still drawn). ``build_state`` assumes fixtures
    have already been reconciled (``clean.reconcile`` drops anything suspicious
    before it gets here).
    """
    phases = (
        (match.pen_home, match.pen_away),
        (match.et_home, match.et_away),
        (match.ft_home, match.ft_away),
    )
    for home_score, away_score in phases:
        if home_score is None and away_score is None:
            continue  # phase not played
        if home_score is None or away_score is None:
            raise ValueError(f"incomplete phase in match {match.match_id!r}")
        if home_score != away_score:
            return match.home_team if home_score > away_score else match.away_team
    raise ValueError(f"knockout match finished with no winner: {match.match_id!r}")


def build_state(
    fixtures: list[NormalizedMatch], ratings: dict[str, float]
) -> TournamentState:
    """Build the :class:`TournamentState` from backbone fixtures + live data."""
    # group_teams groups by the stage label ("Group A"); we re-key to the letter.
    groups = {
        stage.split()[-1]: teams for stage, teams in group_teams(fixtures).items()
    }
    locked_group: dict[frozenset[str], PlayedMatch] = {}
    locked_knockout: dict[frozenset[str], str] = {}
    for match in fixtures:
        if not is_lockable(match):
            continue
        pair = frozenset((match.home_team, match.away_team))
        if match.stage.startswith("Group"):
            assert match.ft_home is not None and match.ft_away is not None
            locked_group[pair] = PlayedMatch(
                match.home_team, match.away_team, match.ft_home, match.ft_away
            )
        else:
            locked_knockout[pair] = _winner_of(match)
    return TournamentState(groups, ratings, locked_group, locked_knockout)


def run_from_state(
    state: TournamentState,
    model: MatchModel,
    annex_c: dict[frozenset[str], dict[str, str]],
    *,
    runs: int,
    seed: int,
    extra_time_total_goals: float = 0.8,
    elo_denominator: float = 400.0,
    host_advantage: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Run the conditional Monte Carlo from a :class:`TournamentState`."""
    return run_tournament(
        state.groups,
        state.ratings,
        model,
        annex_c,
        runs=runs,
        seed=seed,
        extra_time_total_goals=extra_time_total_goals,
        elo_denominator=elo_denominator,
        locked_group=state.locked_group,
        locked_knockout=state.locked_knockout,
        host_advantage=host_advantage,
    )
