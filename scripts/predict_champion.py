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

import json
from pathlib import Path

import typer

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"


def _load_latest() -> tuple[str, dict[str, dict[str, float]]]:
    """Return ``(timestamp, probabilities)`` from the latest run; fails loudly."""
    pointer = _PROCESSED / "latest.json"
    if not pointer.exists():
        raise typer.BadParameter(
            "no runs yet: data/processed/latest.json is missing. "
            "Run `make run` or `python scripts/run_pipeline.py "
            "--mode pre_tournament` first."
        )
    ts = json.loads(pointer.read_text())["timestamp"]
    probs_path = _PROCESSED / f"probabilities_{ts}.json"
    if not probs_path.exists():
        raise typer.BadParameter(f"pointer targets {ts} but {probs_path} is missing.")
    return ts, json.loads(probs_path.read_text())["probabilities"]


def main(top: int = 10, team: str | None = None) -> None:
    ts, probs = _load_latest()
    ranking = sorted(
        ((name, p["champion"]) for name, p in probs.items()), key=lambda x: -x[1]
    )
    width = max((len(n) for n, _ in ranking[:top]), default=0)
    typer.secho(f"P(champion) — updated {ts}", bold=True)
    for i, (name, p) in enumerate(ranking[:top], 1):
        hit = team is not None and name.lower() == team.lower()
        line = f"{i:2d}. {name:<{width}}  {p:6.2%}"
        typer.secho(line, fg=typer.colors.GREEN if hit else None, bold=hit)

    if team is not None:
        match = next((r for r in ranking if r[0].lower() == team.lower()), None)
        if match is None:
            typer.secho(
                f"\n'{team}' isn't in the latest run (check the name, e.g. "
                "'United States', not 'USA').",
                err=True,
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        rank = [n for n, _ in ranking].index(match[0]) + 1
        typer.secho(
            f"\n{match[0]}: P(champion) = {match[1]:.2%}  ·  rank #{rank} "
            f"of {len(ranking)}",
            fg=typer.colors.GREEN,
            bold=True,
        )


if __name__ == "__main__":
    typer.run(main)
