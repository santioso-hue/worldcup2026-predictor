"""Histórico internacional (martj42/international_results): descarga + parseo.

``parse_results_csv`` es **puro** (stdlib ``csv`` sobre un string, sin pandas) y produce
:class:`HistoricalMatch` — la base del ajuste Elo (features/elo.py). ``fetch_martj42``
hace el I/O (descarga y cachea el CSV en ``data/raw/``).

Columnas VERIFICADAS de ``results.csv`` (ver data/raw/SOURCES.md):
``date, home_team, away_team, home_score, away_score, tournament, city, country,
neutral``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HistoricalMatch:
    """Un partido internacional histórico (solo los campos que usa el Elo)."""

    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool


def parse_results_csv(text: str) -> list[HistoricalMatch]:
    """Parsea el CSV de martj42 a :class:`HistoricalMatch` (función pura).

    Las filas sin marcador entero (no disputadas / datos incompletos) se omiten: no son
    útiles para el Elo. ``neutral`` se interpreta de ``"TRUE"``/``"FALSE"``
    (case-insensitive).
    """
    matches: list[HistoricalMatch] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
        except (ValueError, TypeError, KeyError):
            continue  # fila sin marcador numérico -> no usable para Elo
        matches.append(
            HistoricalMatch(
                date=date.fromisoformat(row["date"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=home_score,
                away_score=away_score,
                tournament=row["tournament"],
                neutral=row["neutral"].strip().lower() == "true",
            )
        )
    return matches


def fetch_martj42(
    url: str, dest: Path | str, *, timeout: float = 60.0, force: bool = False
) -> Path:
    """Descarga el CSV de martj42 y lo cachea en ``dest`` (I/O; ``requests`` perezoso).

    Idempotente salvo ``force``: si ``dest`` ya existe y no se fuerza, no re-descarga.
    Las corridas live hacia adelante fuerzan el refresco (el Elo debe absorber los
    últimos resultados, no quedar congelado en una caché vieja); el replay
    (``--snapshot``) usa la copia cacheada (reproduce dada esa misma caché).
    """
    import requests

    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    return dest
