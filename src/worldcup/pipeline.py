"""Orquestación del pipeline: núcleo puro (sin red/fs) + helpers de I/O finos.

``run_pipeline`` encadena ``reconcile`` → ``fit_elo`` → ``build_state`` → Monte Carlo y
devuelve las probabilidades + los matches reconciliados (para snapshotear). El núcleo no
toca red ni fs, así que se testea con fixtures; el I/O (fetch, snapshot, figuras) va en
``scripts/``. ``reconcile`` alimenta ``build_state`` (antes no estaba cableado).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .config import Config
from .data.clean import reconcile
from .data.historical import HistoricalMatch
from .data.live_results import NormalizedMatch
from .data.schedule import group_teams
from .features.elo import fit_elo
from .models.base import MatchOutcome
from .models.dixon_coles import DixonColesModel
from .simulation.state import build_state, run_from_state


@dataclass(frozen=True)
class PipelineResult:
    """Salida del pipeline: probabilidades, ratings, grupos y anomalías de reconcile."""

    probabilities: dict[str, dict[str, float]]
    ratings: dict[str, float]
    anomalies: list[tuple[str, str]]
    groups: dict[str, list[str]]


def run_pipeline(
    incoming: list[NormalizedMatch],
    previous: list[NormalizedMatch],
    history: list[HistoricalMatch],
    config: Config,
    annex_c: dict[frozenset[str], dict[str, str]],
    *,
    runs: int,
    seed: int,
    reference_date: date | None = None,
    history_cutoff: date | None = None,
) -> tuple[PipelineResult, list[NormalizedMatch]]:
    """``reconcile`` → ``fit_elo`` → ``build_state`` → Monte Carlo (sin I/O).

    Cada equipo del backbone recibe un rating: el de ``fit_elo`` o, si no tiene
    histórico, ``initial_rating`` (``build_state`` exige rating para todos, fail-loud).
    ``history_cutoff`` (baseline pre-torneo) descarta los partidos en/posteriores a esa
    fecha, para no contaminar el Elo con resultados del torneo en curso. Devuelve el
    resultado y los matches reconciliados (para snapshotear).
    """
    rec = reconcile(previous, incoming)
    if history_cutoff is not None:
        history = [m for m in history if m.date < history_cutoff]
    fitted = fit_elo(history, config.elo, reference_date=reference_date)
    # El universo de equipos son los participantes de la fase de grupos (los 48 reales);
    # las llaves KO del backbone traen etiquetas de slot ("2A", "1E"), no selecciones.
    # `sorted` -> orden determinista de equipos (independiente de PYTHONHASHSEED), para
    # que el artefacto JSON sea byte-reproducible entre corridas/procesos.
    teams = sorted(
        {team for members in group_teams(rec.matches).values() for team in members}
    )
    ratings = {t: fitted.get(t, config.elo.initial_rating) for t in teams}
    # Bono de sede: las anfitrionas reciben elo.host_advantage en sus partidos.
    hosts = set(config.simulation.hosts)
    host_advantage = {t: config.elo.host_advantage for t in teams if t in hosts}
    state = build_state(rec.matches, ratings)
    model = DixonColesModel(config.elo, config.dixon_coles)
    probabilities = run_from_state(
        state,
        model,
        annex_c,
        runs=runs,
        seed=seed,
        extra_time_total_goals=config.simulation.extra_time_total_goals,
        elo_denominator=config.elo.elo_per_goal_denominator,
        host_advantage=host_advantage,
    )
    result = PipelineResult(
        probabilities=probabilities,
        ratings=ratings,
        anomalies=rec.anomalies,
        groups=state.groups,
    )
    return result, rec.matches


def predict_match(
    home: str,
    away: str,
    history: list[HistoricalMatch],
    config: Config,
    *,
    host: str | None = None,
    reference_date: date | None = None,
) -> MatchOutcome:
    """1X2 de un partido puntual desde los ratings Elo del histórico.

    ``host`` aplica la ventaja de **sede del Mundial** (``elo.host_advantage``, no la
    localía plena de un amistoso) al equipo anfitrión (``home`` o ``away``); en sede
    neutral (``host=None``) no hay ventaja. Sin histórico se usa ``initial_rating``.
    """
    fitted = fit_elo(history, config.elo, reference_date=reference_date)
    rating_home = fitted.get(home, config.elo.initial_rating)
    rating_away = fitted.get(away, config.elo.initial_rating)
    if host == home:
        advantage = config.elo.host_advantage
    elif host == away:
        advantage = -config.elo.host_advantage
    else:
        advantage = 0.0
    model = DixonColesModel(config.elo, config.dixon_coles)
    return model.outcome_proba(rating_home, rating_away, advantage)


# --- helpers de I/O finos (testeables con tmp_path, sin red) ---


def write_probabilities(
    probabilities: dict[str, dict[str, float]],
    outdir: Path | str,
    ts: str,
    *,
    groups: dict[str, list[str]] | None = None,
    latest_pointer: str = "latest.json",
    update_pointer: bool = True,
) -> Path:
    """Escribe ``probabilities_<ts>.json`` y, si ``update_pointer``, el ``latest``.

    El payload incluye ``groups`` (para las tablas del dashboard). Los replays/baselines
    escriben su JSON pero NO mueven el puntero live.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probabilities_{ts}.json"
    payload = {"timestamp": ts, "groups": groups or {}, "probabilities": probabilities}
    # sort_keys -> bytes estables del artefacto (el dashboard/replay lo consumen).
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    if update_pointer:
        (out / latest_pointer).write_text(json.dumps({"timestamp": ts}))
    return path


def load_latest_probabilities(
    outdir: Path | str, *, latest_pointer: str = "latest.json"
) -> dict[str, dict[str, float]] | None:
    """Carga las probabilidades del último run (para deltas ↑/↓); ``None`` si no hay."""
    out = Path(outdir)
    pointer = out / latest_pointer
    if not pointer.exists():
        return None
    ts = json.loads(pointer.read_text())["timestamp"]
    path = out / f"probabilities_{ts}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    probabilities: dict[str, dict[str, float]] = payload["probabilities"]
    return probabilities


@dataclass(frozen=True)
class RunArtifact:
    """Una corrida persistida: timestamp, grupos y probabilidades."""

    timestamp: str
    groups: dict[str, list[str]]
    probabilities: dict[str, dict[str, float]]


def load_latest_run(
    outdir: Path | str, *, latest_pointer: str = "latest.json"
) -> RunArtifact | None:
    """Carga la última corrida (timestamp + grupos + probabilidades); ``None``."""
    out = Path(outdir)
    pointer = out / latest_pointer
    if not pointer.exists():
        return None
    ts = json.loads(pointer.read_text())["timestamp"]
    path = out / f"probabilities_{ts}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return RunArtifact(
        timestamp=payload["timestamp"],
        groups=payload.get("groups", {}),
        probabilities=payload["probabilities"],
    )


def render_outputs(
    result: PipelineResult,
    previous: dict[str, dict[str, float]] | None,
    outdir: Path | str,
    *,
    ts: str,
) -> list[Path]:
    """Renderiza el ranking de campeón (con deltas vs el run previo) y lo guarda."""
    from .viz.charts import prepare_champion_ranking, render_champion_ranking
    from .viz.export import save_figure
    from .viz.theme import PORTRAIT

    champion = {team: probs["champion"] for team, probs in result.probabilities.items()}
    previous_champion = (
        {team: probs["champion"] for team, probs in previous.items()}
        if previous is not None
        else None
    )
    rows = prepare_champion_ranking(champion, previous_champion)
    figure = render_champion_ranking(rows, stamp=f"Actualizado {ts}")
    return [save_figure(figure, f"champion_{ts}", PORTRAIT, outdir=outdir)]
