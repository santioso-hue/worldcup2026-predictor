"""Calibration tests: reliability/ECE (pure numpy) and Platt (sklearn)."""

from __future__ import annotations

import pytest

from worldcup.evaluation.backtest import Prediction, log_loss
from worldcup.evaluation.calibration import ece, fit_platt, reliability_bins


def _alternating(
    p_home: float, p_draw: float, p_away: float, n: int = 100
) -> list[Prediction]:
    # home and away alternate (each occurs half the time); never a draw.
    return [
        Prediction((p_home, p_draw, p_away), 0 if i % 2 == 0 else 2) for i in range(n)
    ]


def test_reliability_bin_observed_matches_predicted_when_calibrated() -> None:
    bins = reliability_bins(_alternating(0.5, 0.0, 0.5), n_bins=10)
    half = [b for b in bins if abs(b[0] - 0.5) < 0.06]
    assert half and abs(half[0][1] - 0.5) < 0.05  # observed ~ predicted (0.5)


def test_ece_low_when_calibrated() -> None:
    assert ece(_alternating(0.5, 0.0, 0.5), n_bins=10) < 0.1


def test_ece_high_when_overconfident() -> None:
    assert ece(_alternating(0.9, 0.05, 0.05), n_bins=10) > 0.2


def test_platt_recalibration_valid_and_improves_logloss() -> None:
    pytest.importorskip("sklearn")
    preds = _alternating(0.9, 0.05, 0.05, n=200)
    calibrator = fit_platt(preds)

    out = calibrator.calibrate((0.9, 0.05, 0.05))
    assert abs(sum(out) - 1.0) < 1e-9
    assert all(0.0 <= x <= 1.0 for x in out)

    recalibrated = [Prediction(calibrator.calibrate(p.probs), p.actual) for p in preds]
    assert log_loss(recalibrated) <= log_loss(preds) + 1e-9


def test_reliability_bins_rejects_empty_or_bad_bins() -> None:
    sample = _alternating(0.5, 0.0, 0.5, n=10)
    with pytest.raises(ValueError):
        reliability_bins([], n_bins=10)
    with pytest.raises(ValueError):
        reliability_bins(sample, n_bins=0)
    with pytest.raises(ValueError):
        ece(sample, n_bins=0)


def test_fit_platt_rejects_empty_predictions() -> None:
    # Guard before importing sklearn: doesn't need the dependency to fail.
    with pytest.raises(ValueError):
        fit_platt([])
