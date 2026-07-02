"""Tests for the walk-forward backtest and the 1X2 metrics."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest

from worldcup.config import load_config
from worldcup.data.historical import HistoricalMatch
from worldcup.evaluation.backtest import (
    Prediction,
    accuracy,
    backtest,
    brier,
    log_loss,
    rps,
)
from worldcup.models.dixon_coles import DixonColesModel

_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(_ROOT / "config" / "config.yaml")
MODEL = DixonColesModel(CFG.elo, CFG.dixon_coles)


def test_perfect_predictions_score_zero() -> None:
    preds = [Prediction((1.0, 0.0, 0.0), 0), Prediction((0.0, 0.0, 1.0), 2)]
    assert log_loss(preds) == 0.0
    assert brier(preds) == 0.0
    assert rps(preds) == 0.0
    assert accuracy(preds) == 1.0


def test_rps_respects_outcome_ordering() -> None:
    # actual=home: predicting all-away is penalized more than predicting all-draw.
    predicted_away = rps([Prediction((0.0, 0.0, 1.0), 0)])
    predicted_draw = rps([Prediction((0.0, 1.0, 0.0), 0)])
    assert predicted_away > predicted_draw
    assert predicted_away == 1.0 and predicted_draw == 0.5


def test_brier_matches_hand_computation() -> None:
    # (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert abs(brier([Prediction((0.5, 0.3, 0.2), 0)]) - 0.38) < 1e-12


def test_log_loss_matches_hand_computation() -> None:
    assert abs(log_loss([Prediction((0.5, 0.3, 0.2), 0)]) - (-math.log(0.5))) < 1e-12


def test_accuracy_uses_argmax() -> None:
    preds = [Prediction((0.6, 0.3, 0.1), 0), Prediction((0.2, 0.3, 0.5), 1)]
    assert accuracy(preds) == 0.5  # 1st correct (argmax 0); 2nd wrong (argmax 2 != 1)


def _hm(
    d: date, hs: int, as_: int, home: str = "A", away: str = "B"
) -> HistoricalMatch:
    return HistoricalMatch(
        date=d,
        home_team=home,
        away_team=away,
        home_score=hs,
        away_score=as_,
        tournament="Friendly",
        neutral=True,
    )


def test_backtest_is_deterministic() -> None:
    history = [_hm(date(2020, 1, 1 + i), 1, 0) for i in range(10)]
    r1 = backtest(history, MODEL, CFG.elo, burn_in_matches=0, history_window_days=3650)
    r2 = backtest(history, MODEL, CFG.elo, burn_in_matches=0, history_window_days=3650)
    assert r1.predictions == r2.predictions


def test_backtest_does_not_peek_at_the_future() -> None:
    history = [_hm(date(2020, m, 1), 1, 0) for m in (1, 2, 3)]
    base = backtest(
        history, MODEL, CFG.elo, burn_in_matches=1, history_window_days=3650
    )
    extended = backtest(
        history + [_hm(date(2020, 4, 1), 5, 0)],
        MODEL,
        CFG.elo,
        burn_in_matches=1,
        history_window_days=3650,
    )
    # adding a LATER match doesn't change predictions for the earlier ones.
    assert extended.predictions[: len(base.predictions)] == base.predictions


def test_backtest_rejects_negative_burn_in() -> None:
    history = [_hm(date(2020, 1, 1 + i), 1, 0) for i in range(5)]
    with pytest.raises(ValueError):
        backtest(history, MODEL, CFG.elo, burn_in_matches=-1, history_window_days=3650)


def test_backtest_rejects_nonpositive_window() -> None:
    history = [_hm(date(2020, 1, 1 + i), 1, 0) for i in range(5)]
    with pytest.raises(ValueError):
        backtest(history, MODEL, CFG.elo, burn_in_matches=0, history_window_days=0)


def test_metrics_reject_empty_predictions() -> None:
    for metric in (log_loss, brier, rps, accuracy):
        with pytest.raises(ValueError):
            metric([])


def test_backtest_excludes_same_day_matches_from_prior() -> None:
    # Two matches on the SAME day must not see each other: the window is [., M.date)
    # by date, so reordering them in the input doesn't change any prediction.
    earlier = _hm(date(2020, 1, 1), 1, 0, "A", "B")
    m_ab = _hm(date(2020, 6, 1), 1, 0, "A", "B")
    m_cd = _hm(date(2020, 6, 1), 2, 0, "C", "D")
    kw = dict(burn_in_matches=0, history_window_days=3650)
    ab_first = backtest([earlier, m_ab, m_cd], MODEL, CFG.elo, **kw)
    cd_first = backtest([earlier, m_cd, m_ab], MODEL, CFG.elo, **kw)
    assert ab_first.predictions[1] == cd_first.predictions[2]  # A-B: same prior in both
    assert ab_first.predictions[2] == cd_first.predictions[1]  # C-D: same prior in both
