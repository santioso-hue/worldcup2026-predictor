"""Monte Carlo condicional del torneo: simula solo lo pendiente y agrega frecuencias.

Cada corrida: simula los partidos de grupo pendientes (usando los resultados
bloqueados donde existan), calcula las posiciones (Art. 13), determina
ganadores/subcampeones y los 8 mejores terceros (Annex C), arma el bracket y simula
la eliminatoria (condicionada a lo bloqueado). Agrega sobre ``runs`` corridas ->
P(avanzar / R16 / QF / SF / final / campeón) por selección. Determinista dado el
snapshot (locks) + la semilla.

*Bono de sede:* las anfitrionas (``config.simulation.hosts``) reciben
``elo.host_advantage`` en sus partidos simulados, vía ``host_advantage`` (mapa
``team -> bono``); en un cruce el bono es neto (local anfitrión menos visitante
anfitrión), así dos anfitrionas entre sí juegan neutral.
*Rendimiento:* sin caché de matrices (correctness-first); para 50k corridas conviene
cachear ``score_matrix`` por emparejamiento (futuro).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..models.base import MatchModel
from ..rng import spawn_rngs
from .bracket import KNOCKOUT_BRACKET, assign_best_thirds
from .group_stage import PlayedMatch, TeamStanding, rank_thirds, standings
from .match import simulate_match

# Rondas en orden de profundidad. "advance" = clasificar al Round of 32.
ROUNDS = (
    "advance",
    "round_of_16",
    "quarter_finals",
    "semi_finals",
    "final",
    "champion",
)
_ROUND_INDEX = {r: i for i, r in enumerate(ROUNDS)}

# El bracket y Annex C son específicos del WC2026: 12 grupos A-L de 4 equipos.
_GROUP_KEYS = frozenset("ABCDEFGHIJKL")


def _net_host_advantage(
    home: str, away: str, host_advantage: dict[str, float]
) -> float:
    """Bono de sede neto: +bono si el local es anfitrión, -bono si lo es el visitante.

    Dos anfitrionas entre sí (o ninguna) -> 0. La etiqueta home/away del bracket no
    refleja la sede real (todo se juega en US/MX/CA), así que el bono va al anfitrión.
    """
    return host_advantage.get(home, 0.0) - host_advantage.get(away, 0.0)


def _validate_groups(groups: dict[str, list[str]]) -> None:
    """Falla ruidosamente si ``groups`` no es la estructura WC2026 (12 grupos A-L de 4).

    Sin esto, una entrada malformada (un nº de grupos != 12, o etiquetas de grupo
    inesperadas) reventaría con un ``KeyError`` opaco en mitad de la resolución del
    bracket en vez de dar un error claro.
    """
    if set(groups) != _GROUP_KEYS:
        raise ValueError(f"se esperan 12 grupos A-L; se recibieron {sorted(groups)}")
    for group, teams in groups.items():
        if len(teams) != 4:
            raise ValueError(f"el grupo {group} tiene {len(teams)} equipos != 4")


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
    """Simula (o usa el lock de) los 6 partidos del grupo; devuelve las posiciones."""
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
    """Una realización del torneo -> ``{team: ronda más profunda alcanzada}``."""
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
        assert isinstance(ref, int)  # los refs "MW"/"ML" son match ids
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
    """Corre el Monte Carlo condicional y devuelve ``{team: {ronda: probabilidad}}``.

    ``locked_group`` mapea ``frozenset({home, away}) -> PlayedMatch`` (resultados ya
    jugados); ``locked_knockout`` mapea ``frozenset({home, away}) -> ganador``. Lo
    bloqueado no se re-muestrea. ``host_advantage`` mapea ``anfitrión -> bono Elo`` (se
    aplica neto por cruce; vacío = cancha neutral). Determinista dado ``(locks, seed)``.
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
            ):  # alcanzó todas hasta la más profunda
                counts[team][ROUNDS[i]] += 1

    return {team: {rnd: counts[team][rnd] / runs for rnd in ROUNDS} for team in ratings}
