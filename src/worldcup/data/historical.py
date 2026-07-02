"""International match history (martj42/international_results): download + parsing.

``parse_results_csv`` is **pure** (stdlib ``csv`` over a string, no pandas) and
produces :class:`HistoricalMatch` — the basis for the Elo fit (features/elo.py).
``fetch_martj42`` does the I/O (downloads and caches the CSV in ``data/raw/``).

VERIFIED columns of ``results.csv``:
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
    """A historical international match (only the fields the Elo model uses)."""

    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool


def parse_results_csv(text: str) -> list[HistoricalMatch]:
    """Parse martj42's CSV into :class:`HistoricalMatch` (pure function).

    Rows without an integer score (not played / incomplete data) are dropped: they're
    no use to the Elo fit. ``neutral`` is parsed from ``"TRUE"``/``"FALSE"``
    (case-insensitive).
    """
    matches: list[HistoricalMatch] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
        except (ValueError, TypeError, KeyError):
            continue  # no numeric score -> unusable for Elo
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
    """Download martj42's CSV and cache it at ``dest`` (I/O; lazy ``requests`` import).

    Idempotent unless ``force``: if ``dest`` already exists and force isn't set, skips
    the download. Live forward-looking runs force a refresh (the Elo fit needs the
    latest results, not a stale cache); replay runs (``--snapshot``) use the cached
    copy so they reproduce against that same cache.
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
