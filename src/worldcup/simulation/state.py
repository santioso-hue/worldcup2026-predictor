"""Estado real del torneo: del schedule + snapshot live a las entradas del Monte Carlo.

``build_state`` toma los fixtures del **backbone** (``stage`` ``"Group A"`` y rondas KO)
ya superpuestos con los resultados live, más los ratings Elo, y produce
un :class:`TournamentState`: los 12 grupos (claves de una letra ``A``…``L``, como el
bracket/Annex C), y los resultados YA bloqueados de grupo y de eliminatoria. El Monte
Carlo solo simula lo pendiente, condicionado a esos locks (reconditioning live).

Los partidos de eliminatoria se bloquean por **par de equipos** (``frozenset``), no por
match-id: en eliminatoria directa dos selecciones se cruzan a lo sumo una vez, y la KO
empieza con la fase de grupos cerrada (bracket determinista), así que el par identifica
el cruce sin ambigüedad. Un partido sin ``time``/sin marcador no se bloquea.
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
    """Estado del torneo listo para el Monte Carlo condicional."""

    groups: dict[str, list[str]]  # "A".."L" -> equipos
    ratings: dict[str, float]
    locked_group: dict[frozenset[str], PlayedMatch]
    locked_knockout: dict[frozenset[str], str]  # par -> ganador


def _winner_of(match: NormalizedMatch) -> str:
    """Ganador de un KO FINISHED por fase (penales > prórroga > 90').

    Falla ruidosamente si no hay un ganador claro (fase incompleta, o un KO "finalizado"
    empatado sin resolución) en vez de fabricar uno. ``build_state`` asume fixtures ya
    reconciliados (``clean.reconcile`` descarta lo sospechoso antes de llegar aquí).
    """
    phases = (
        (match.pen_home, match.pen_away),
        (match.et_home, match.et_away),
        (match.ft_home, match.ft_away),
    )
    for home_score, away_score in phases:
        if home_score is None and away_score is None:
            continue  # fase no disputada
        if home_score is None or away_score is None:
            raise ValueError(f"fase incompleta en el partido {match.match_id!r}")
        if home_score != away_score:
            return match.home_team if home_score > away_score else match.away_team
    raise ValueError(f"eliminatoria finalizada sin ganador en {match.match_id!r}")


def build_state(
    fixtures: list[NormalizedMatch], ratings: dict[str, float]
) -> TournamentState:
    """Construye el :class:`TournamentState` desde los fixtures del backbone + live."""
    # group_teams agrupa por la etiqueta de stage ("Group A"); re-llaveamos a la letra.
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
    """Corre el Monte Carlo condicional a partir de un :class:`TournamentState`."""
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
