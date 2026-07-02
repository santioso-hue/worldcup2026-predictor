"""Dynamic Elo (pure functions, no I/O, no RNG — Elo is deterministic).

Sequential chronological update over the martj42 history:

    Δ = K_base(importance) · G(margin) · recency · (result − E)

where ``E`` is the logistic expectation with home-advantage bonus.
"""

from __future__ import annotations

from datetime import date

from worldcup.config import EloConfig, GoalMarginConfig
from worldcup.data.historical import HistoricalMatch

# Calendar constant (not a hyperparameter): average days per month = 365.25/12.
DAYS_PER_MONTH = 30.4375


def expected_score(
    rating_home: float,
    rating_away: float,
    home_advantage: float = 0.0,
    denominator: float = 400.0,
) -> float:
    """Logistic probability that the home team scores (win=1, draw=0.5).

    ``E = 1 / (1 + 10^(-(R_home + home_advantage - R_away) / denominator))``.

    Parameters
    ----------
    rating_home, rating_away:
        Elo ratings for home and away teams.
    home_advantage:
        Home-field bonus in Elo points (``0`` on neutral ground).
    denominator:
        Elo scale (``400`` standard; ``elo.elo_per_goal_denominator``).
    """
    diff = (rating_home + home_advantage) - rating_away
    return 1.0 / (1.0 + 10.0 ** (-diff / denominator))


def recency_weight(
    match_date: date, reference_date: date, half_life_months: float
) -> float:
    """Time weight of a match: ``0.5^(age_months / half_life_months)``.

    Age is ``reference_date − match_date``. At the reference the weight is ``1``; at
    ``half_life_months`` old, ``0.5``. ``reference_date`` should be ≥ every match date
    (typically the last match date in the snapshot), which keeps the weight
    deterministic per snapshot.
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
    """Map a tournament name to an ``elo.k_factors`` key.

    Keyword-based rules (documented and tunable, see the spec). Anything unrecognized
    falls back to ``"default"``.
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
    """K-factor multiplier by goal margin (eloratings.net).

    ``G = 1`` if ``|margin| ≤ 1``; ``cfg.two_goal`` if ``|margin| == 2``; otherwise
    ``(cfg.offset + |margin|) / cfg.divisor`` (diminishing returns).

    Parameters
    ----------
    margin:
        Goal difference (can be negative; the absolute value is used).
    cfg:
        Multiplier parameters.
    """
    d = abs(margin)
    if d <= 1:
        return 1.0
    if d == 2:
        return cfg.two_goal
    return (cfg.offset + d) / cfg.divisor


def _result(home_score: int, away_score: int) -> float:
    """Result from the home team's perspective: win=1, draw=0.5, loss=0."""
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
    """Fit Elo ratings with a sequential chronological pass over the history.

    For each match:
    ``Δ = K_base · G(margin) · recency · (result − E)``. The home team gains
    ``Δ`` and the away team loses ``Δ`` (symmetric). The processing order is
    deterministic — ``(date, home, away)`` — so the result doesn't depend on
    input order.

    Parameters
    ----------
    matches:
        Historical matches. Empty -> ``{}``.
    cfg:
        Elo hyperparameters (``config.elo``).
    reference_date:
        Anchor for the recency weight. Defaults to ``max(match.date)`` (the last
        match in the snapshot), which keeps output deterministic per snapshot.

    Returns
    -------
    dict[str, float]
        Final ``team -> rating`` mapping.
    """
    if not matches:
        return {}
    if reference_date is None:
        reference_date = max(m.date for m in matches)

    ratings: dict[str, float] = {}
    init = cfg.initial_rating
    default_k = cfg.k_factors["default"]
    # Include the score in the sort key: two rows with the same date+teams but a
    # different result (happens in martj42) sort reproducibly, so the
    # order-independence invariant holds even with duplicate keys.
    ordered = sorted(
        matches,
        key=lambda m: (m.date, m.home_team, m.away_team, m.home_score, m.away_score),
    )
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
