"""Tests del modelo Dixon-Coles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldcup.config import DixonColesConfig, load_config
from worldcup.models.dixon_coles import DixonColesModel

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
MODEL = DixonColesModel(CFG.elo, CFG.dixon_coles)


def test_expected_goals_equal_neutral_is_base_lambda() -> None:
    lh, la = MODEL.expected_goals(1500.0, 1500.0, home_advantage=0.0)
    assert lh == CFG.elo.base_lambda
    assert la == CFG.elo.base_lambda


def test_expected_goals_stronger_home_scores_more() -> None:
    lh, la = MODEL.expected_goals(1800.0, 1400.0, home_advantage=0.0)
    assert lh > la


def test_expected_goals_respects_clip_bounds() -> None:
    lh, la = MODEL.expected_goals(3000.0, 1000.0, home_advantage=0.0)
    assert lh == CFG.elo.lambda_max
    assert la == CFG.elo.lambda_min


def test_score_matrix_shape_and_normalized() -> None:
    m = MODEL.score_matrix(1500.0, 1500.0)
    n = CFG.dixon_coles.max_goals + 1
    assert m.shape == (n, n)
    assert abs(m.sum() - 1.0) < 1e-12


def test_stronger_home_has_higher_home_win() -> None:
    strong = MODEL.outcome_proba(1800.0, 1400.0)
    weak = MODEL.outcome_proba(1400.0, 1800.0)
    assert strong.home_win > weak.home_win


def test_score_matrix_symmetric_for_equal_teams_on_neutral() -> None:
    m = MODEL.score_matrix(1500.0, 1500.0, home_advantage=0.0)
    assert np.allclose(m, m.T)


def test_dixon_coles_increases_draw_vs_plain_poisson() -> None:
    # ρ<0 sube los empates de marcador bajo (0-0, 1-1) y baja 1-0/0-1.
    plain = DixonColesModel(CFG.elo, DixonColesConfig(rho=0.0, max_goals=8))
    dc = DixonColesModel(CFG.elo, DixonColesConfig(rho=-0.13, max_goals=8))
    assert (
        dc.outcome_proba(1500.0, 1500.0).draw > plain.outcome_proba(1500.0, 1500.0).draw
    )
