"""Carga y valida ``config.yaml`` con pydantic.

Es la ÚNICA fuente de hiperparámetros, rutas, semilla y modo (PROJECT.md §9). Los
modelos usan ``extra="forbid"`` para que un typo en el YAML falle ruidosamente en vez
de pasar desapercibido, y ``frozen=True`` para que la config sea inmutable en runtime.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TournamentMode(str, Enum):
    """Modo de torneo (PROJECT.md §2)."""

    live = "live"
    pre_tournament = "pre_tournament"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Base):
    name: str
    seed: int
    mode: TournamentMode


class PathsConfig(_Base):
    data_raw: Path
    data_interim: Path
    data_processed: Path
    figures: Path
    videos: Path


class HistoricalConfig(_Base):
    source: str
    results_url: str
    shootouts_url: str


class ScheduleConfig(_Base):
    source: str
    url: str


class FallbackConfig(_Base):
    provider: str
    base_url: str
    competition_code: str
    token_env: str


class LiveConfig(_Base):
    provider: str
    base_url: str
    league_id: int
    season: int
    api_key_env: str
    poll_interval_seconds: int = Field(gt=0)
    daily_request_budget: int = Field(gt=0)
    fallback: FallbackConfig


class SnapshotsConfig(_Base):
    dir: Path
    filename_pattern: str
    latest_pointer: str


class DataConfig(_Base):
    historical: HistoricalConfig
    schedule: ScheduleConfig
    live: LiveConfig
    snapshots: SnapshotsConfig


class GoalMarginConfig(_Base):
    """Parámetros del multiplicador de margen de gol (eloratings.net)."""

    two_goal: float = Field(gt=0)
    offset: float = Field(gt=0)
    divisor: float = Field(gt=0)


class EloConfig(_Base):
    initial_rating: float
    home_advantage: float
    host_advantage: float
    recency_half_life_months: float = Field(gt=0)
    base_lambda: float = Field(gt=0)
    elo_per_goal_denominator: float = Field(gt=0)
    lambda_min: float = Field(gt=0)
    lambda_max: float = Field(gt=0)
    k_factors: dict[str, float]
    goal_margin: GoalMarginConfig

    @field_validator("k_factors")
    @classmethod
    def _must_have_default(cls, v: dict[str, float]) -> dict[str, float]:
        if "default" not in v:
            raise ValueError("elo.k_factors debe incluir la clave 'default'")
        return v


class DixonColesConfig(_Base):
    rho: float
    max_goals: int = Field(gt=0)


class SimulationConfig(_Base):
    runs: int = Field(gt=0)
    extra_time_total_goals: float = Field(ge=0)
    hosts: list[str] = Field(default_factory=list)  # sedes -> bono elo.host_advantage


class EvaluationConfig(_Base):
    burn_in_matches: int = Field(ge=0)
    history_window_days: int = Field(gt=0)


class Config(_Base):
    """Configuración raíz validada del proyecto."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    elo: EloConfig
    dixon_coles: DixonColesConfig
    simulation: SimulationConfig
    evaluation: EvaluationConfig


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lee y valida ``config.yaml``.

    Parameters
    ----------
    path:
        Ruta al YAML. Por defecto ``config/config.yaml`` relativo al cwd.

    Returns
    -------
    Config
        Configuración validada e inmutable.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    pydantic.ValidationError
        Si falta un campo, sobra una clave o un valor viola sus restricciones.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config.model_validate(raw)
