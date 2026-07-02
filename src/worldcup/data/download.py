"""Download + timestamped **snapshotting** (reproducible record of model state).

Each run writes an immutable ``results_YYYYMMDDtHHMM.parquet`` snapshot and updates a
``latest.txt`` pointer. A previous snapshot is never overwritten: they're the record of
"what the model knew and when." Pinning a snapshot with ``--snapshot <ts>`` reproduces a
given set of figures exactly.

The ``NormalizedMatch`` <-> record (``list[dict]``) conversion is **pure** (stdlib,
ISO-8601 dates, ``status`` as a string) and is tested without pandas. Parquet
serialization imports ``pandas``/``pyarrow`` lazily, isolating the dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .live_results import MatchStatus, NormalizedMatch

_SCORE_FIELDS = (
    "ht_home",
    "ht_away",
    "ft_home",
    "ft_away",
    "et_home",
    "et_away",
    "pen_home",
    "pen_away",
)

TIMESTAMP_FORMAT = "%Y%m%dt%H%M"


def make_timestamp(moment: datetime | None = None) -> str:
    """Format a ``datetime`` as ``YYYYMMDDtHHMM`` (UTC).

    Parameters
    ----------
    moment:
        Instant to format. Defaults to ``datetime.now(UTC)`` when ``None``. Pass an
        explicit ``moment`` for deterministic output (tests, recordings).
    """
    if moment is None:
        moment = datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


def match_to_record(match: NormalizedMatch) -> dict[str, Any]:
    """Serialize a match to a flat, JSON-compatible record (pure, no pandas)."""
    return {
        "match_id": match.match_id,
        "source": match.source,
        "source_match_id": match.source_match_id,
        "kickoff_utc": match.kickoff_utc.isoformat(),
        "home_team": match.home_team,
        "away_team": match.away_team,
        "stage": match.stage,
        "status": match.status.value,
        "ht_home": match.ht_home,
        "ht_away": match.ht_away,
        "ft_home": match.ft_home,
        "ft_away": match.ft_away,
        "et_home": match.et_home,
        "et_away": match.et_away,
        "pen_home": match.pen_home,
        "pen_away": match.pen_away,
        "venue": match.venue,
        "fetched_at": match.fetched_at.isoformat() if match.fetched_at else None,
    }


def _opt_int(value: Any) -> int | None:
    """Coerce to ``int`` or ``None``, tolerating parquet ``NaN``/``<NA>``/``float``."""
    if value is None:
        return None
    # NaN != NaN; also catches pandas.NA, whose bool is ambiguous, via the try.
    try:
        if value != value:  # noqa: PLR0124 - NaN check
            return None
    except (TypeError, ValueError):
        return None
    return int(value)


def _opt_str(value: Any) -> str | None:
    """Coerce to ``str`` or ``None``, tolerating parquet's ``NaN``/``<NA>``.

    pandas fills missing cells in a text column with ``NaN`` (float), not ``None``.
    Without this guard, a missing ``venue`` would round-trip back as the string
    ``"nan"`` (``str(nan)``), breaking snapshot immutability and determinism.
    """
    if value is None:
        return None
    try:
        if value != value:  # noqa: PLR0124 - NaN check
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


def record_to_match(record: dict[str, Any]) -> NormalizedMatch:
    """Rebuild a :class:`NormalizedMatch` from a flat record (pure)."""
    fetched = _opt_str(record.get("fetched_at"))
    return NormalizedMatch(
        match_id=str(record["match_id"]),
        source=str(record["source"]),
        source_match_id=str(record["source_match_id"]),
        kickoff_utc=datetime.fromisoformat(record["kickoff_utc"]),
        home_team=str(record["home_team"]),
        away_team=str(record["away_team"]),
        stage=str(record["stage"]),
        status=MatchStatus(record["status"]),
        ht_home=_opt_int(record.get("ht_home")),
        ht_away=_opt_int(record.get("ht_away")),
        ft_home=_opt_int(record.get("ft_home")),
        ft_away=_opt_int(record.get("ft_away")),
        et_home=_opt_int(record.get("et_home")),
        et_away=_opt_int(record.get("et_away")),
        pen_home=_opt_int(record.get("pen_home")),
        pen_away=_opt_int(record.get("pen_away")),
        venue=_opt_str(record.get("venue")),
        fetched_at=datetime.fromisoformat(fetched) if fetched else None,
    )


def snapshot_path(raw_dir: Path | str, filename_pattern: str, ts: str) -> Path:
    """Snapshot path for a given timestamp ``ts``."""
    return Path(raw_dir) / filename_pattern.format(ts=ts)


def save_snapshot(
    matches: list[NormalizedMatch],
    ts: str,
    raw_dir: Path | str,
    *,
    filename_pattern: str = "results_{ts}.parquet",
    latest_pointer: str = "latest.txt",
    overwrite: bool = False,
) -> Path:
    """Write an immutable parquet snapshot and update the ``latest`` pointer.

    Parameters
    ----------
    matches:
        Normalized matches to save.
    ts:
        Timestamp ``YYYYMMDDtHHMM`` (see :func:`make_timestamp`).
    raw_dir:
        ``data/raw`` directory (created if missing).
    overwrite:
        Defaults to ``False``: raises ``FileExistsError`` if the snapshot already
        exists (snapshots are immutable). Use a new ``ts`` instead.

    Returns
    -------
    pathlib.Path
        Path of the parquet file written.
    """
    import pandas as pd  # lazy import

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(raw_dir, filename_pattern, ts)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Snapshot {path} already exists; snapshots are immutable. "
            "Use a new timestamp or overwrite=True."
        )
    records = [match_to_record(m) for m in matches]
    pd.DataFrame(records).to_parquet(path, index=False)
    (raw_dir / latest_pointer).write_text(ts, encoding="utf-8")
    return path


def latest_timestamp(
    raw_dir: Path | str, *, latest_pointer: str = "latest.txt"
) -> str | None:
    """Read the timestamp of the latest snapshot, or ``None`` if there's no pointer."""
    pointer = Path(raw_dir) / latest_pointer
    if not pointer.exists():
        return None
    return pointer.read_text(encoding="utf-8").strip() or None


def load_snapshot(
    ts: str,
    raw_dir: Path | str,
    *,
    filename_pattern: str = "results_{ts}.parquet",
) -> list[NormalizedMatch]:
    """Load a parquet snapshot and rebuild it into ``NormalizedMatch`` objects.

    Raises
    ------
    FileNotFoundError
        If no snapshot exists for that ``ts``.
    """
    import pandas as pd  # lazy import

    path = snapshot_path(raw_dir, filename_pattern, ts)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    df = pd.read_parquet(path)
    # to_dict types keys as Hashable; here they're column names (str).
    records = cast("list[dict[str, Any]]", df.to_dict(orient="records"))
    return [record_to_match(r) for r in records]


def load_latest_snapshot(
    raw_dir: Path | str,
    *,
    filename_pattern: str = "results_{ts}.parquet",
    latest_pointer: str = "latest.txt",
) -> list[NormalizedMatch] | None:
    """Load the snapshot pointed to by ``latest``; ``None`` if there isn't one."""
    ts = latest_timestamp(raw_dir, latest_pointer=latest_pointer)
    if ts is None:
        return None
    return load_snapshot(ts, raw_dir, filename_pattern=filename_pattern)


def fetch_openfootball(url: str, *, timeout: float = 30.0) -> dict:
    """Download the openfootball fixtures JSON (I/O; ``requests`` imported lazily)."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
