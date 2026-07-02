"""Full-pipeline CLI: load/fetch -> reconcile -> simulate -> snapshot + figures.

Modes: ``live`` (fetch from the provider), ``--snapshot <ts>`` (replay), and
``--mode pre_tournament`` (schedule only). ``--watch`` polls via ``WatchTrigger``;
without ``--watch`` it's a single run (cron model). All I/O lives here; the pure
logic is in ``worldcup.pipeline``.

Replay freezes the *fixtures* (parquet snapshot) but NOT the martj42 history
(``results.csv``, a shared cache that live runs refresh): it reproduces exactly
given the SAME history cache + seed. Refresh-proof reproducibility would need
pinning the history alongside the snapshot (not done yet).
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
    """Load ``_ROOT/.env`` (KEY=VALUE) without overriding already-set variables."""
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
    # Live re-fetches for a fresh Elo; replay uses the cache (deterministic).
    fetch_martj42(config.data.historical.results_url, dest, force=force_refresh)
    return parse_results_csv(dest.read_text())


def _make_live_provider(config: Config) -> LiveResultsProvider:
    """Build the live provider (football-data.org) from config (fail-loud)."""
    live = config.data.live
    if live.provider != "football_data":
        raise typer.BadParameter(f"unknown live provider: {live.provider}")
    token = os.environ.get(live.token_env)
    if not token:
        raise typer.BadParameter(f"missing environment variable {live.token_env}")
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
        # The openfootball backbone has all 104 matches (72 group + 32 KO with slot
        # labels): validate the structure and abort loudly if incomplete, rather than
        # simulate a broken tournament. (The live feed has a different shape and isn't
        # validated here — it omits the 32 unresolved KO slots.)
        schedule = parse_openfootball(fetch_openfootball(config.data.schedule.url))
        issues = validate_schedule(schedule)
        if issues:
            raise typer.BadParameter(
                "invalid openfootball schedule: " + "; ".join(issues)
            )
        return schedule
    if mode != "live":
        raise typer.BadParameter(f"unknown mode: {mode}")
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
    # The reconcile snapshot (parquet + latest.txt) is the LIVE RESULTS RECORD: only
    # `live` writes it and only `live` reads it as `previous`. If pre_tournament wrote
    # it, a later live run would pick it up as `previous` and reconcile openfootball
    # fixtures against the live feed -> corrupted groups. pre_tournament and replay
    # still write probabilities/figures (dashboard), but they don't touch this chain.
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
    # Pre-tournament baseline: Elo only uses matches BEFORE the first fixture (excludes
    # results from the ongoing tournament). Live uses the current Elo.
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
    # Replay writes its outputs with a _replay suffix so it doesn't overwrite the
    # curated artifact for the same ts (e.g. a 50k live run); the live pointer doesn't
    # move either.
    out_ts = f"{ts}_replay" if is_replay else ts
    if is_live:  # only live writes the snapshot chain (parquet + latest.txt)
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
        bracket=result.bracket,
        update_pointer=not is_replay,
    )
    render_outputs(result, previous_probs, config.paths.figures, ts=out_ts)
    for match_id, anomaly in result.anomalies:
        typer.echo(f"anomaly: {match_id}: {anomaly}")
    typer.echo(f"done: snapshot {out_ts}, {len(result.probabilities)} teams")
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
    _load_dotenv()  # load the API key from .env if present (live mode)
    cfg = load_config(config)
    annex_c = load_annex_c(cfg.paths.data_raw / "annex_c_2026.json")
    resolved_runs = runs if runs is not None else cfg.simulation.runs
    resolved_seed = seed if seed is not None else cfg.project.seed
    # mode falls back to config.yaml (single source of truth) if not passed via CLI.
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
