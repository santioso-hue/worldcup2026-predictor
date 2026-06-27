"""Descarga + **snapshotting** con timestamp (registro reproducible del estado).

Cada corrida guarda un snapshot inmutable ``results_YYYYMMDDtHHMM.parquet`` y actualiza
un puntero ``latest.txt``. Nunca se machaca un snapshot previo: son el registro de "qué
sabía el modelo y cuándo". Fijar un snapshot ``--snapshot <ts>``
reproduce exactamente unas figuras.

La conversión ``NormalizedMatch`` <-> registros (``list[dict]``) es **pura** (stdlib,
ISO-8601 para fechas y ``status`` como string) y se testea sin pandas. La serialización
a parquet importa ``pandas``/``pyarrow`` de forma perezosa, aislando la dependencia.
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
    """Formatea un ``datetime`` como ``YYYYMMDDtHHMM`` (UTC).

    Parameters
    ----------
    moment:
        Instante a formatear. Si es ``None`` se usa ``datetime.now(UTC)``. Para salidas
        deterministas (tests, grabación), pasa un ``moment`` explícito.
    """
    if moment is None:
        moment = datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


def match_to_record(match: NormalizedMatch) -> dict[str, Any]:
    """Serializa un partido a un registro plano JSON-compatible (puro, sin pandas)."""
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
    """Coerce a ``int`` o ``None``, tolerando ``NaN``/``<NA>``/``float`` de parquet."""
    if value is None:
        return None
    # NaN != NaN; también cubre pandas.NA cuyo bool es ambiguo, evitado por el try.
    try:
        if value != value:  # noqa: PLR0124 - chequeo de NaN
            return None
    except (TypeError, ValueError):
        return None
    return int(value)


def _opt_str(value: Any) -> str | None:
    """Coerce a ``str`` o ``None``, tolerando ``NaN``/``<NA>`` de parquet.

    pandas rellena las celdas ausentes de una columna de texto con ``NaN`` (float), no
    con ``None``. Sin la guarda, un ``venue`` ausente volvería como la cadena ``"nan"``
    tras un round-trip (``str(nan)``), rompiendo la inmutabilidad del snapshot y el
    determinismo.
    """
    if value is None:
        return None
    try:
        if value != value:  # noqa: PLR0124 - chequeo de NaN
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


def record_to_match(record: dict[str, Any]) -> NormalizedMatch:
    """Reconstruye un :class:`NormalizedMatch` desde un registro plano (puro)."""
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
    """Ruta del snapshot para un timestamp ``ts``."""
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
    """Escribe un snapshot parquet inmutable y actualiza el puntero ``latest``.

    Parameters
    ----------
    matches:
        Partidos normalizados a guardar.
    ts:
        Timestamp ``YYYYMMDDtHHMM`` (ver :func:`make_timestamp`).
    raw_dir:
        Directorio ``data/raw`` (se crea si no existe).
    overwrite:
        Por defecto ``False``: si el snapshot ya existe se lanza ``FileExistsError``
        (los snapshots son inmutables). Usa un ``ts`` nuevo en su lugar.

    Returns
    -------
    pathlib.Path
        Ruta del parquet escrito.
    """
    import pandas as pd  # import perezoso

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(raw_dir, filename_pattern, ts)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"El snapshot {path} ya existe; los snapshots son inmutables. "
            "Usa un timestamp nuevo o overwrite=True."
        )
    records = [match_to_record(m) for m in matches]
    pd.DataFrame(records).to_parquet(path, index=False)
    (raw_dir / latest_pointer).write_text(ts, encoding="utf-8")
    return path


def latest_timestamp(
    raw_dir: Path | str, *, latest_pointer: str = "latest.txt"
) -> str | None:
    """Lee el timestamp del último snapshot, o ``None`` si no hay puntero."""
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
    """Carga un snapshot parquet y lo reconstruye a ``NormalizedMatch``.

    Raises
    ------
    FileNotFoundError
        Si no existe el snapshot para ese ``ts``.
    """
    import pandas as pd  # import perezoso

    path = snapshot_path(raw_dir, filename_pattern, ts)
    if not path.exists():
        raise FileNotFoundError(f"No existe el snapshot: {path}")
    df = pd.read_parquet(path)
    # to_dict tipa las claves como Hashable; aquí son nombres de columna (str).
    records = cast("list[dict[str, Any]]", df.to_dict(orient="records"))
    return [record_to_match(r) for r in records]


def load_latest_snapshot(
    raw_dir: Path | str,
    *,
    filename_pattern: str = "results_{ts}.parquet",
    latest_pointer: str = "latest.txt",
) -> list[NormalizedMatch] | None:
    """Carga el snapshot apuntado por ``latest``; ``None`` si no hay ninguno."""
    ts = latest_timestamp(raw_dir, latest_pointer=latest_pointer)
    if ts is None:
        return None
    return load_snapshot(ts, raw_dir, filename_pattern=filename_pattern)


def fetch_openfootball(url: str, *, timeout: float = 30.0) -> dict:
    """Descarga el JSON de fixtures de openfootball (I/O; ``requests`` perezoso)."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
