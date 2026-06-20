"""CLI del pipeline completo: carga/descarga → reconcile → simula → snapshot + figuras.

Modos: ``live`` (fetch del proveedor), ``--snapshot <ts>`` (replay reproducible) y
``--mode pre_tournament`` (solo el schedule). ``--watch`` hace polling con
``WatchTrigger``; sin ``--watch`` es una sola corrida (modelo cron). Todo el I/O vive
aquí; la lógica pura está en ``worldcup.pipeline``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from worldcup.config import Config, load_config
from worldcup.data.api_football import APIFootballProvider
from worldcup.data.download import (
    fetch_openfootball,
    load_latest_snapshot,
    load_snapshot,
    save_snapshot,
)
from worldcup.data.historical import HistoricalMatch, fetch_martj42, parse_results_csv
from worldcup.data.live_results import NormalizedMatch
from worldcup.data.schedule import parse_openfootball
from worldcup.data.triggers import CronTrigger, RefreshTrigger, WatchTrigger
from worldcup.pipeline import (
    load_latest_probabilities,
    render_outputs,
    run_pipeline,
    write_probabilities,
)
from worldcup.simulation.bracket import load_annex_c

_AnnexC = dict[frozenset[str], dict[str, str]]


def _load_history(config: Config) -> list[HistoricalMatch]:
    dest = config.paths.data_raw / "results.csv"
    fetch_martj42(config.data.historical.results_url, dest)  # idempotente
    return parse_results_csv(dest.read_text())


def _load_incoming(
    config: Config, mode: str, snapshot: str | None
) -> list[NormalizedMatch]:
    snaps = config.data.snapshots
    if snapshot is not None:
        return load_snapshot(
            snapshot, snaps.dir, filename_pattern=snaps.filename_pattern
        )
    if mode == "pre_tournament":
        return parse_openfootball(fetch_openfootball(config.data.schedule.url))
    if mode != "live":
        raise typer.BadParameter(f"modo desconocido: {mode}")
    live = config.data.live
    api_key = os.environ.get(live.api_key_env)
    if not api_key:
        raise typer.BadParameter(f"falta la variable de entorno {live.api_key_env}")
    provider = APIFootballProvider(api_key, live.base_url, live.league_id, live.season)
    return provider.get_schedule()


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
    # Solo el modo live toca los punteros (lee previous, snapshotea, mueve latest).
    # El replay (--snapshot) y el baseline (pre_tournament) son herméticos: previous=[],
    # sin re-snapshotear ni mover los punteros (preserva determinismo y deltas live).
    is_live = snapshot is None and mode == "live"
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
    history = _load_history(config)
    result, reconciled = run_pipeline(
        incoming, previous, history, config, annex_c, runs=runs, seed=seed
    )
    ts = snapshot or datetime.now(timezone.utc).strftime("%Y%m%dt%H%M")
    if is_live:
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
        ts,
        groups=result.groups,
        update_pointer=is_live,
    )
    render_outputs(result, previous_probs, config.paths.figures, ts=ts)
    for match_id, anomaly in result.anomalies:
        typer.echo(f"anomalía: {match_id}: {anomaly}")
    typer.echo(f"listo: snapshot {ts}, {len(result.probabilities)} equipos")
    return ts


def main(
    config: Path = Path("config/config.yaml"),
    snapshot: str | None = None,
    runs: int | None = None,
    seed: int | None = None,
    mode: str = "live",
    watch: bool = False,
    interval: int | None = None,
) -> None:
    cfg = load_config(config)
    annex_c = load_annex_c(cfg.paths.data_raw / "annex_c_2026.json")
    resolved_runs = runs if runs is not None else cfg.simulation.runs
    resolved_seed = seed if seed is not None else cfg.project.seed
    resolved_interval = (
        interval if interval is not None else cfg.data.live.poll_interval_seconds
    )

    def on_refresh(_tick: int) -> None:
        _run_once(
            cfg,
            annex_c,
            mode=mode,
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
