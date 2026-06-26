"""CLI de predicción puntual: 1X2 de un partido desde los ratings Elo del histórico.

Los nombres deben coincidir con el histórico martj42 (p. ej. "United States", no "USA";
"DR Congo", no "Congo DR"). Un nombre desconocido cae al rating por defecto y avisa.

Ejemplos:
    python scripts/predict_match.py "Brazil" "France"
    python scripts/predict_match.py "United States" "Mexico" --host "United States"
"""

from __future__ import annotations

from pathlib import Path

import typer

from worldcup.config import load_config
from worldcup.data.historical import fetch_martj42, parse_results_csv
from worldcup.pipeline import predict_match


def main(
    home: str,
    away: str,
    host: str | None = None,
    config: Path = Path("config/config.yaml"),
) -> None:
    cfg = load_config(config)
    dest = cfg.paths.data_raw / "results.csv"
    fetch_martj42(cfg.data.historical.results_url, dest)  # idempotente
    history = parse_results_csv(dest.read_text())
    known = {m.home_team for m in history} | {m.away_team for m in history}
    for label, team in (("local", home), ("visitante", away)):
        if team not in known:
            typer.secho(
                f"aviso: '{team}' ({label}) no está en el histórico; se usa el rating "
                f"por defecto ({cfg.elo.initial_rating:.0f}). Revisa el nombre "
                "(p. ej. 'United States', no 'USA').",
                err=True,
                fg=typer.colors.YELLOW,
            )
    if host is not None and host not in (home, away):
        typer.secho(
            f"aviso: host '{host}' no coincide con ningún equipo; sin ventaja de sede.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    outcome = predict_match(home, away, history, cfg, host=host)
    typer.echo(
        f"{home} {outcome.home_win:.1%}  ·  "
        f"empate {outcome.draw:.1%}  ·  "
        f"{away} {outcome.away_win:.1%}"
    )


if __name__ == "__main__":
    typer.run(main)
