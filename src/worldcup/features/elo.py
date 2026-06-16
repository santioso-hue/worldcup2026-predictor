"""Elo dinámico (funciones puras, sin I/O, sin RNG — el Elo es determinista).

Ajuste secuencial cronológico sobre el histórico martj42:

    Δ = K_base(importancia) · G(margen) · recency · (resultado − E)

donde ``E`` es la expectativa logística con bono de localía. Ver
``docs/specs/2026-06-16-elo-design.md``.
"""

from __future__ import annotations

from datetime import date

from worldcup.config import EloConfig, GoalMarginConfig
from worldcup.data.historical import HistoricalMatch

# Constante de calendario (no es un hiperparámetro): días promedio por mes = 365.25/12.
DAYS_PER_MONTH = 30.4375


def expected_score(
    rating_home: float,
    rating_away: float,
    home_advantage: float = 0.0,
    denominator: float = 400.0,
) -> float:
    """Probabilidad logística de que el local puntúe (victoria=1, empate=0.5).

    ``E = 1 / (1 + 10^(-(R_home + home_advantage - R_away) / denominator))``.

    Parameters
    ----------
    rating_home, rating_away:
        Ratings Elo de local y visitante.
    home_advantage:
        Bono de localía en puntos Elo (``0`` en cancha neutral).
    denominator:
        Escala Elo (``400`` estándar; ``elo.elo_per_goal_denominator``).
    """
    diff = (rating_home + home_advantage) - rating_away
    return 1.0 / (1.0 + 10.0 ** (-diff / denominator))


def recency_weight(
    match_date: date, reference_date: date, half_life_months: float
) -> float:
    """Peso temporal del partido: ``0.5^(edad_meses / half_life_months)``.

    La edad es ``reference_date − match_date``. En la referencia el peso es ``1``; a
    ``half_life_months`` de antigüedad, ``0.5``. ``reference_date`` debe ser ≥ todas las
    fechas (típicamente la fecha del último partido del snapshot), lo que hace el peso
    determinista por snapshot.
    """
    age_months = (reference_date - match_date).days / DAYS_PER_MONTH
    return 0.5 ** (age_months / half_life_months)


_CONTINENTAL_KEYWORDS = (
    "euro",
    "copa am",  # Copa América
    "gold cup",
    "asian cup",
    "africa",  # Africa/African Cup of Nations
    "confederations",
)


def classify_importance(tournament: str) -> str:
    """Mapea el nombre de un torneo a una clave de ``elo.k_factors``.

    Reglas por palabra clave (documentadas y ajustables, ver el spec). Lo no reconocido
    cae en ``"default"``.
    """
    t = tournament.lower()
    if "world cup" in t and "qualif" in t:
        return "world_cup_qualifier"
    if "world cup" in t:
        return "world_cup"
    if "nations league" in t:
        return "nations_league"
    if "friendl" in t:
        return "friendly"
    if any(keyword in t for keyword in _CONTINENTAL_KEYWORDS):
        return "continental"
    return "default"


def goal_margin_multiplier(margin: int, cfg: GoalMarginConfig) -> float:
    """Multiplicador del factor K por margen de gol (eloratings.net).

    ``G = 1`` si ``|margin| ≤ 1``; ``cfg.two_goal`` si ``|margin| == 2``; si no
    ``(cfg.offset + |margin|) / cfg.divisor`` (rendimientos decrecientes).

    Parameters
    ----------
    margin:
        Diferencia de goles (puede ser negativa; se usa el valor absoluto).
    cfg:
        Parámetros del multiplicador.
    """
    d = abs(margin)
    if d <= 1:
        return 1.0
    if d == 2:
        return cfg.two_goal
    return (cfg.offset + d) / cfg.divisor


def _result(home_score: int, away_score: int) -> float:
    """Resultado desde la óptica del local: victoria=1, empate=0.5, derrota=0."""
    if home_score > away_score:
        return 1.0
    if home_score == away_score:
        return 0.5
    return 0.0


def fit_elo(
    matches: list[HistoricalMatch],
    cfg: EloConfig,
    reference_date: date | None = None,
) -> dict[str, float]:
    """Ajusta ratings Elo con un paso secuencial cronológico sobre el histórico.

    Para cada partido:
    ``Δ = K_base · G(margen) · recency · (resultado − E)``. El local suma ``Δ`` y el
    visitante resta ``Δ`` (simétrico). El orden es determinista — ``(fecha, local,
    visitante)`` — así que el resultado no depende del orden de entrada.

    Parameters
    ----------
    matches:
        Partidos históricos. Vacío -> ``{}``.
    cfg:
        Hiperparámetros Elo (``config.elo``).
    reference_date:
        Ancla del peso de recencia. Por defecto ``max(match.date)`` (el último partido
        del snapshot), lo que mantiene la salida determinista por snapshot.

    Returns
    -------
    dict[str, float]
        ``team -> rating`` final.
    """
    if not matches:
        return {}
    if reference_date is None:
        reference_date = max(m.date for m in matches)

    ratings: dict[str, float] = {}
    init = cfg.initial_rating
    default_k = cfg.k_factors["default"]
    ordered = sorted(matches, key=lambda m: (m.date, m.home_team, m.away_team))
    for m in ordered:
        r_home = ratings.get(m.home_team, init)
        r_away = ratings.get(m.away_team, init)
        home_adv = 0.0 if m.neutral else cfg.home_advantage
        expected = expected_score(
            r_home, r_away, home_adv, cfg.elo_per_goal_denominator
        )
        k = cfg.k_factors.get(classify_importance(m.tournament), default_k)
        g = goal_margin_multiplier(m.home_score - m.away_score, cfg.goal_margin)
        w = recency_weight(m.date, reference_date, cfg.recency_half_life_months)
        delta = k * g * w * (_result(m.home_score, m.away_score) - expected)
        ratings[m.home_team] = r_home + delta
        ratings[m.away_team] = r_away - delta
    return ratings
