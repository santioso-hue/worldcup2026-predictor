"""Tests del Elo dinámico (TDD)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from worldcup.config import GoalMarginConfig, load_config
from worldcup.data.historical import HistoricalMatch
from worldcup.features.elo import (
    classify_importance,
    expected_score,
    fit_elo,
    goal_margin_multiplier,
    recency_weight,
)

GM = GoalMarginConfig(two_goal=1.5, offset=11.0, divisor=8.0)
ELO = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml").elo


def _hm(
    d: date, home: str, away: str, hs: int, as_: int, *, neutral: bool = True
) -> HistoricalMatch:
    return HistoricalMatch(
        date=d,
        home_team=home,
        away_team=away,
        home_score=hs,
        away_score=as_,
        tournament="Friendly",
        neutral=neutral,
    )


def test_goal_margin_draw_or_one_goal_is_one() -> None:
    assert goal_margin_multiplier(0, GM) == 1.0
    assert goal_margin_multiplier(1, GM) == 1.0
    assert goal_margin_multiplier(-1, GM) == 1.0  # usa |margen|


def test_goal_margin_two_goals() -> None:
    assert goal_margin_multiplier(2, GM) == 1.5


def test_goal_margin_three_plus_diminishing() -> None:
    assert goal_margin_multiplier(3, GM) == (11.0 + 3) / 8.0  # 1.75
    assert goal_margin_multiplier(5, GM) == (11.0 + 5) / 8.0  # 2.0


def test_expected_score_equal_neutral_is_half() -> None:
    assert expected_score(1500.0, 1500.0, home_advantage=0.0) == 0.5


def test_expected_score_home_advantage_raises_it() -> None:
    assert expected_score(1500.0, 1500.0, home_advantage=75.0) > 0.5


def test_expected_score_is_symmetric() -> None:
    e_home = expected_score(1600.0, 1400.0, home_advantage=0.0)
    e_away = expected_score(1400.0, 1600.0, home_advantage=0.0)
    assert abs((e_home + e_away) - 1.0) < 1e-12


def test_expected_score_strong_favorite() -> None:
    # +400 Elo -> ~0.91 esperado.
    assert expected_score(1800.0, 1400.0, home_advantage=0.0) > 0.9


def test_recency_weight_at_reference_is_one() -> None:
    d = date(2026, 6, 16)
    assert recency_weight(d, d, half_life_months=18.0) == 1.0


def test_recency_weight_at_half_life_is_about_half() -> None:
    # ~18 meses antes de la referencia -> ~0.5.
    w = recency_weight(date(2024, 12, 16), date(2026, 6, 16), half_life_months=18.0)
    assert abs(w - 0.5) < 0.02


def test_recency_weight_older_matches_count_less() -> None:
    ref = date(2026, 6, 16)
    recent = recency_weight(date(2025, 6, 16), ref, half_life_months=18.0)
    old = recency_weight(date(2020, 6, 16), ref, half_life_months=18.0)
    assert old < recent < 1.0


def test_classify_importance_maps_known_tournaments() -> None:
    assert classify_importance("FIFA World Cup") == "world_cup"
    assert classify_importance("FIFA World Cup qualification") == "world_cup_qualifier"
    assert classify_importance("UEFA Nations League") == "nations_league"
    assert classify_importance("Friendly") == "friendly"
    assert classify_importance("UEFA Euro") == "continental"
    assert classify_importance("Copa América") == "continental"
    assert classify_importance("African Cup of Nations") == "continental"
    assert classify_importance("AFC Asian Cup") == "continental"
    assert classify_importance("CONCACAF Gold Cup") == "continental"


def test_classify_importance_unknown_is_default() -> None:
    assert classify_importance("Kirin Cup") == "default"


def test_fit_elo_winner_gains_loser_loses_zero_sum() -> None:
    r = fit_elo([_hm(date(2026, 6, 16), "A", "B", 1, 0)], ELO)
    assert r["A"] > ELO.initial_rating > r["B"]
    # actualización simétrica: suma constante.
    assert abs((r["A"] + r["B"]) - 2 * ELO.initial_rating) < 1e-9


def test_fit_elo_is_order_independent_and_deterministic() -> None:
    matches = [
        _hm(date(2026, 6, 1), "A", "B", 1, 0),
        _hm(date(2026, 6, 2), "B", "C", 2, 0),
        _hm(date(2026, 6, 3), "A", "C", 0, 0),
    ]
    assert fit_elo(matches, ELO) == fit_elo(list(reversed(matches)), ELO)


def test_fit_elo_bigger_margin_moves_more() -> None:
    narrow = fit_elo([_hm(date(2026, 6, 16), "A", "B", 1, 0)], ELO)
    blowout = fit_elo([_hm(date(2026, 6, 16), "A", "B", 5, 0)], ELO)
    assert (blowout["A"] - ELO.initial_rating) > (narrow["A"] - ELO.initial_rating) > 0


def test_fit_elo_old_match_moves_less_than_recent() -> None:
    ref = date(2026, 6, 16)
    recent = fit_elo([_hm(date(2026, 6, 16), "A", "B", 1, 0)], ELO, reference_date=ref)
    old = fit_elo([_hm(date(2020, 6, 16), "A", "B", 1, 0)], ELO, reference_date=ref)
    assert (recent["A"] - ELO.initial_rating) > (old["A"] - ELO.initial_rating) > 0


def test_fit_elo_home_advantage_reduces_gain_on_a_win() -> None:
    # Mismo 1-0: en casa el esperado era mayor, así que la subida es menor que
    # en cancha neutral.
    at_home = fit_elo([_hm(date(2026, 6, 16), "A", "B", 1, 0, neutral=False)], ELO)
    neutral = fit_elo([_hm(date(2026, 6, 16), "A", "B", 1, 0, neutral=True)], ELO)
    assert (at_home["A"] - ELO.initial_rating) < (neutral["A"] - ELO.initial_rating)
