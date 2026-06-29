"""CLI del pipeline completo: carga/descarga → reconcile → simula → snapshot + figuras.

Modos: ``live`` (fetch del proveedor), ``--snapshot <ts>`` (replay) y
``--mode pre_tournament`` (solo el schedule). ``--watch`` hace polling con
``WatchTrigger``; sin ``--watch`` es una sola corrida (modelo cron). Todo el I/O vive
aquí; la lógica pura está en ``worldcup.pipeline``.

El replay congela los *fixtures* (snapshot parquet) pero NO el histórico martj42
(``results.csv``, caché compartida que las corridas live refrescan): reproduce
exactamente dada la MISMA caché de histórico + semilla. Para reproducibilidad a prueba
de refrescos habría que pinear el histórico junto al snapshot (pendiente).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import typer

from worldcup.config import Config, load_config
from worldcup.data.download import (
    fetch_openfootball,
    load_latest_snapshot,
    load_snapshot,
    make_timestamp,
    save_snapshot,
)
from worldcup.data.football_data import FootballDataProvider
from worldcup.data.historical import HistoricalMatch, fetch_martj42, parse_results_csv
from worldcup.data.live_results import LiveResultsProvider, NormalizedMatch
from worldcup.data.schedule import parse_openfootball, validate_schedule
from worldcup.data.triggers import CronTrigger, RefreshTrigger, WatchTrigger
from worldcup.pipeline import (
    load_latest_probabilities,
    render_outputs,
    run_pipeline,
    write_probabilities,
)
from worldcup.simulation.bracket import load_annex_c

_AnnexC = dict[frozenset[str], dict[str, str]]
_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Carga ``_ROOT/.env`` (KEY=VALUE) sin pisar variables ya definidas."""
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_history(
    config: Config, *, force_refresh: bool = False
) -> list[HistoricalMatch]:
    dest = config.paths.data_raw / "results.csv"
    # Live re-descarga para Elo fresco; el replay usa la caché (determinista).
    fetch_martj42(config.data.historical.results_url, dest, force=force_refresh)
    return parse_results_csv(dest.read_text())


def _make_live_provider(config: Config) -> LiveResultsProvider:
    """Construye el proveedor live (football-data.org) desde config (fail-loud)."""
    live = config.data.live
    if live.provider != "football_data":
        raise typer.BadParameter(f"proveedor live desconocido: {live.provider}")
    token = os.environ.get(live.token_env)
    if not token:
        raise typer.BadParameter(f"falta la variable de entorno {live.token_env}")
    return FootballDataProvider(token, live.base_url, live.competition_code)


def _load_incoming(
    config: Config, mode: str, snapshot: str | None
) -> list[NormalizedMatch]:
    snaps = config.data.snapshots
    if snapshot is not None:
        return load_snapshot(
            snapshot, snaps.dir, filename_pattern=snaps.filename_pattern
        )
    if mode == "pre_tournament":
        # El backbone openfootball trae los 104 partidos (72 grupo + 32 KO con etiquetas
        # de slot): validamos la estructura y abortamos ruidosamente si está incompleta,
        # en vez de simular un torneo roto. (El feed live tiene otra forma: no se valida
        # aquí porque omite los 32 slots KO sin resolver.)
        schedule = parse_openfootball(fetch_openfootball(config.data.schedule.url))
        issues = validate_schedule(schedule)
        if issues:
            raise typer.BadParameter(
                "schedule openfootball inválido: " + "; ".join(issues)
            )
        return schedule
    if mode != "live":
        raise typer.BadParameter(f"modo desconocido: {mode}")
    return _make_live_provider(config).get_schedule()


def _run_once(
    config: Config,
    annex_c: _AnnexC,
    *,
    mode: str,
    snapshot: str | None,
    runs: int,
    seed: int,
) -> str:
    snaps = config.data.snapshots
    # El snapshot de reconcile (parquet + latest.txt) es el REGISTRO DE RESULTADOS LIVE:
    # solo `live` lo escribe y solo `live` lo lee como `previous`. Si pre_tournament lo
    # escribiera, una corrida live posterior lo tomaría como `previous` y reconciliaría
    # fixtures de openfootball contra el feed live -> grupos corruptos. pre_tournament y
    # replay igual escriben probabilities/figuras (dashboard), pero NO tocan esa cadena.
    is_live = snapshot is None and mode == "live"
    is_replay = snapshot is not None
    incoming = _load_incoming(config, mode, snapshot)
    previous = (
        (
            load_latest_snapshot(
                snaps.dir,
                filename_pattern=snaps.filename_pattern,
                latest_pointer=snaps.latest_pointer,
            )
            or []
        )
        if is_live
        else []
    )
    history = _load_history(config, force_refresh=not is_replay)
    # Baseline pre-torneo: el Elo solo usa partidos ANTES del primer fixture (excluye
    # los resultados del torneo en curso). Live usa el Elo actual.
    history_cutoff: date | None = None
    if mode == "pre_tournament" and incoming:
        history_cutoff = min(m.kickoff_utc for m in incoming).date()
    result, reconciled = run_pipeline(
        incoming,
        previous,
        history,
        config,
        annex_c,
        runs=runs,
        seed=seed,
        history_cutoff=history_cutoff,
    )
    ts = snapshot or make_timestamp()
    # El replay escribe sus salidas con sufijo _replay para NO pisar el artefacto curado
    # del mismo ts (p. ej. una corrida live de 50k); el puntero live tampoco se mueve.
    out_ts = f"{ts}_replay" if is_replay else ts
    if is_live:  # solo live escribe la cadena de snapshots (parquet + latest.txt)
        save_snapshot(
            reconciled,
            ts,
            snaps.dir,
            filename_pattern=snaps.filename_pattern,
            latest_pointer=snaps.latest_pointer,
        )
    previous_probs = (
        load_latest_probabilities(config.paths.data_processed) if is_live else None
    )
    write_probabilities(
        result.probabilities,
        config.paths.data_processed,
        out_ts,
        groups=result.groups,
        fixtures=reconciled,
        update_pointer=not is_replay,
    )
    render_outputs(result, previous_probs, config.paths.figures, ts=out_ts)
    for match_id, anomaly in result.anomalies:
        typer.echo(f"anomalía: {match_id}: {anomaly}")
    typer.echo(f"listo: snapshot {out_ts}, {len(result.probabilities)} equipos")
    return out_ts


def main(
    config: Path = Path("config/config.yaml"),
    snapshot: str | None = None,
    runs: int | None = None,
    seed: int | None = None,
    mode: str | None = None,
    watch: bool = False,
    interval: int | None = None,
) -> None:
    _load_dotenv()  # carga la API key desde .env si existe (modo live)
    cfg = load_config(config)
    annex_c = load_annex_c(cfg.paths.data_raw / "annex_c_2026.json")
    resolved_runs = runs if runs is not None else cfg.simulation.runs
    resolved_seed = seed if seed is not None else cfg.project.seed
    # mode cae a config.yaml (única fuente) si no se pasa por CLI.
    resolved_mode = mode if mode is not None else cfg.project.mode.value
    resolved_interval = (
        interval if interval is not None else cfg.data.live.poll_interval_seconds
    )

    def on_refresh(_tick: int) -> None:
        _run_once(
            cfg,
            annex_c,
            mode=resolved_mode,
            snapshot=snapshot,
            runs=resolved_runs,
            seed=resolved_seed,
        )

    trigger: RefreshTrigger = (
        WatchTrigger(resolved_interval) if watch else CronTrigger()
    )
    trigger.run(on_refresh)


if __name__ == "__main__":
    typer.run(main)
