"""Tests de carga/validación de config.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from worldcup.config import Config, TournamentMode, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_loads_real_config() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.project.seed == 42
    assert cfg.project.mode is TournamentMode.live
    assert cfg.data.live.provider == "football_data"
    assert cfg.data.live.competition_code == "WC"
    assert cfg.data.live.token_env == "FOOTBALL_DATA_TOKEN"
    assert cfg.elo.k_factors["world_cup"] == 66.0  # afinado vía backtest (×1.2)
    assert cfg.elo.goal_margin.two_goal == 1.5
    assert cfg.elo.goal_margin.offset == 11.0
    assert cfg.elo.goal_margin.divisor == 8.0
    assert cfg.dixon_coles.max_goals == 8
    assert cfg.evaluation.history_window_days == 36500


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("config/does_not_exist.yaml"))


def test_invalid_mode_rejected() -> None:
    cfg = load_config(CONFIG_PATH)
    data = cfg.model_dump()
    data["project"]["mode"] = "banana"  # modo inexistente
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_unknown_key_rejected() -> None:
    # extra="forbid": un typo en el YAML debe fallar, no pasar desapercibido.
    cfg = load_config(CONFIG_PATH)
    data = cfg.model_dump()
    data["project"]["speed"] = 99  # typo de "seed"
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_k_factors_requires_default() -> None:
    cfg = load_config(CONFIG_PATH)
    data = cfg.model_dump()
    del data["elo"]["k_factors"]["default"]
    with pytest.raises(ValidationError):
        Config.model_validate(data)
