"""Champion CLI: prints the P(champion) ranking from the pipeline's latest run.

Reads the pointer ``data/processed/latest.json`` -> ``probabilities_<ts>.json``
(written by ``run_pipeline.py``). Doesn't simulate anything, just formats the
artifact that's already there, which is why it's instant. To refresh the
numbers, run ``make run`` (live) or
``python scripts/run_pipeline.py --mode pre_tournament`` first.

Examples:
    python scripts/predict_champion.py                 # top-10 + last-updated date
    python scripts/predict_champion.py --top 15
    python scripts/predict_champion.py --team Colombia  # highlight one team
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same shim as the dashboard: make `worldcup` importable without an editable
# install, so the script works from any caller (subprocess, cron, bare run).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer  # noqa: E402

from worldcup.pipeline import load_latest_run  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"


def main(top: int = 10, team: str | None = None) -> None:
    run = load_latest_run(_PROCESSED)
    if run is None:
        raise typer.BadParameter(
            "no readable run under data/processed. Run `make run` or "
            "`python scripts/run_pipeline.py --mode pre_tournament` first."
        )
    ts, probs = run.timestamp, run.probabilities
    ranking = sorted(
        ((name, p["champion"]) for name, p in probs.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    width = max((len(n) for n, _ in ranking[:top]), default=0)
    typer.secho(f"P(champion) — updated {ts}", bold=True)
    for i, (name, p) in enumerate(ranking[:top], 1):
        hit = team is not None and name.lower() == team.lower()
        line = f"{i:2d}. {name:<{width}}  {p:6.2%}"
        typer.secho(line, fg=typer.colors.GREEN if hit else None, bold=hit)

    if team is not None:
        found = next(
            (
                (rank, name, p)
                for rank, (name, p) in enumerate(ranking, 1)
                if name.lower() == team.lower()
            ),
            None,
        )
        if found is None:
            typer.secho(
                f"\n'{team}' isn't in the latest run (check the name, e.g. "
                "'United States', not 'USA').",
                err=True,
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        rank, name, p = found
        typer.secho(
            f"\n{name}: P(champion) = {p:.2%}  ·  rank #{rank} of {len(ranking)}",
            fg=typer.colors.GREEN,
            bold=True,
        )


if __name__ == "__main__":
    typer.run(main)
