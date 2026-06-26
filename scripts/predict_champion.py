"""CLI de campeón: imprime el ranking de P(campeón) de la última corrida del pipeline.

Lee el puntero ``data/processed/latest.json`` -> ``probabilities_<ts>.json`` (lo que
dejó ``run_pipeline.py``). No simula nada: solo formatea el artefacto ya generado, por
eso es instantáneo. Para refrescar los números, corre antes ``make run`` (live) o
``python scripts/run_pipeline.py --mode pre_tournament``.

Ejemplos:
    python scripts/predict_champion.py                 # top-10 + fecha de actualización
    python scripts/predict_champion.py --top 15
    python scripts/predict_champion.py --team Colombia  # resalta una selección
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"


def _load_latest() -> tuple[str, dict[str, dict[str, float]]]:
    """Devuelve ``(timestamp, probabilities)`` de la última corrida; falla ruidoso."""
    pointer = _PROCESSED / "latest.json"
    if not pointer.exists():
        raise typer.BadParameter(
            "no hay corridas: falta data/processed/latest.json. "
            "Corre primero `make run` o `python scripts/run_pipeline.py "
            "--mode pre_tournament`."
        )
    ts = json.loads(pointer.read_text())["timestamp"]
    probs_path = _PROCESSED / f"probabilities_{ts}.json"
    if not probs_path.exists():
        raise typer.BadParameter(f"el puntero apunta a {ts} pero falta {probs_path}.")
    return ts, json.loads(probs_path.read_text())["probabilities"]


def main(top: int = 10, team: str | None = None) -> None:
    ts, probs = _load_latest()
    ranking = sorted(
        ((name, p["champion"]) for name, p in probs.items()), key=lambda x: -x[1]
    )
    width = max((len(n) for n, _ in ranking[:top]), default=0)
    typer.secho(f"P(campeón) — actualizado {ts}", bold=True)
    for i, (name, p) in enumerate(ranking[:top], 1):
        hit = team is not None and name.lower() == team.lower()
        line = f"{i:2d}. {name:<{width}}  {p:6.2%}"
        typer.secho(line, fg=typer.colors.GREEN if hit else None, bold=hit)

    if team is not None:
        match = next((r for r in ranking if r[0].lower() == team.lower()), None)
        if match is None:
            typer.secho(
                f"\n'{team}' no está en la última corrida (revisa el nombre, p. ej. "
                "'United States', no 'USA').",
                err=True,
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        rank = [n for n, _ in ranking].index(match[0]) + 1
        typer.secho(
            f"\n{match[0]}: P(campeón) = {match[1]:.2%}  ·  puesto #{rank} "
            f"de {len(ranking)}",
            fg=typer.colors.GREEN,
            bold=True,
        )


if __name__ == "__main__":
    typer.run(main)
