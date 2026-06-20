"""CLI de predicción puntual: 1X2 de un partido desde los ratings Elo del histórico.

Ejemplos:
    python scripts/predict_match.py "Brazil" "France"
    python scripts/predict_match.py "USA" "Mexico" --host "USA"
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
    outcome = predict_match(home, away, history, cfg, host=host)
    typer.echo(
        f"{home} {outcome.home_win:.1%}  ·  "
        f"empate {outcome.draw:.1%}  ·  "
        f"{away} {outcome.away_win:.1%}"
    )


if __name__ == "__main__":
    typer.run(main)
