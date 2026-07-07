"""One-off CLI: 1X2 prediction for a single match from historical Elo ratings.

Team names must match the martj42 historical dataset (e.g. "United States", not
"USA"; "DR Congo", not "Congo DR"). An unknown name falls back to the default
rating and prints a warning.

Examples:
    python scripts/predict_match.py "Brazil" "France"
    python scripts/predict_match.py "United States" "Mexico" --host "United States"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same shim as the dashboard: make `worldcup` importable without an editable
# install, so the script works from any caller.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer  # noqa: E402

from worldcup.config import load_config  # noqa: E402
from worldcup.data.historical import (  # noqa: E402
    fetch_martj42,
    parse_results_csv,
    teams_from_history,
)
from worldcup.pipeline import predict_match  # noqa: E402


def main(
    home: str,
    away: str,
    host: str | None = None,
    config: Path = Path("config/config.yaml"),
) -> None:
    cfg = load_config(config)
    dest = cfg.paths.data_raw / "results.csv"
    fetch_martj42(cfg.data.historical.results_url, dest)  # idempotent
    history = parse_results_csv(dest.read_text())
    known = teams_from_history(history)
    for label, team in (("home", home), ("away", away)):
        if team not in known:
            typer.secho(
                f"warning: '{team}' ({label}) not in the historical data; using "
                f"the default rating ({cfg.elo.initial_rating:.0f}). Check the "
                "name (e.g. 'United States', not 'USA').",
                err=True,
                fg=typer.colors.YELLOW,
            )
    if host is not None and host not in (home, away):
        typer.secho(
            f"warning: host '{host}' doesn't match either team; no host advantage.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    outcome = predict_match(home, away, history, cfg, host=host)
    typer.echo(
        f"{home} {outcome.home_win:.1%}  ·  "
        f"draw {outcome.draw:.1%}  ·  "
        f"{away} {outcome.away_win:.1%}"
    )


if __name__ == "__main__":
    typer.run(main)
