"""Schedule backbone del WC2026: parser de openfootball -> fixtures normalizados.

Es la **espina dorsal** del torneo: los 104 partidos (72 de grupo + 32 de
eliminatoria), los 12 grupos de 4 y la estructura del bracket. Los resultados live
(football-data.org) se adjuntan luego a estos fixtures por ``match_id``, de modo que un
cambio de proveedor no rompe la identidad de los partidos.

Funciones **puras** (sin I/O): reciben el JSON ya cargado. La descarga vive en
``download.py``. El esquema por partido está VERIFICADO contra openfootball:

    { "round": "Matchday 1", "date": "YYYY-MM-DD", "time": "13:00 UTC-6",
      "team1": "...", "team2": "...", "group": "Group A",
      "score": {"ft": [h, a], "ht": [h, a]}, "ground": "..." }

*Supuesto (envoltorio):* el archivo agrupa los partidos bajo ``"matches"`` o bajo
``"rounds"[].matches``; soportamos ambos y fallamos ruidosamente si no reconocemos el
formato (no inventamos estructura). openfootball no trae estado: inferimos ``FINISHED``
si hay ``score.ft`` y ``SCHEDULED`` en caso contrario (un archivo estático no tiene
``in_play``). Prórroga/penales de eliminatorias se toman del proveedor live, no de aquí.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .live_results import MatchStatus, NormalizedMatch
from .team_names import canonical_openfootball_team

# Estructura esperada del torneo (para validación, verificado).
EXPECTED_GROUPS = 12
EXPECTED_TEAMS_PER_GROUP = 4
EXPECTED_GROUP_MATCHES = 72
EXPECTED_KNOCKOUT_MATCHES = 32
EXPECTED_TOTAL_MATCHES = 104

# Rondas de eliminatoria 2026 y su n.º de partidos (para validar/etiquetar).
KNOCKOUT_MATCH_COUNTS: dict[str, int] = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarter-finals": 4,
    "Semi-finals": 2,
    "Third-place play-off": 1,
    "Final": 1,
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TZ_RE = re.compile(r"UTC([+-]\d{1,2})(?::?(\d{2}))?")


def _slug(text: str) -> str:
    """Normaliza un nombre a un slug estable (minúsculas, sin signos)."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def make_match_id(date: str, home: str, away: str) -> str:
    """Construye un ``match_id`` determinista y estable entre proveedores.

    Forma: ``"{date}-{home}-vs-{away}"`` con nombres en slug. La clave de unión real
    con el feed live es (fecha, local, visita); por eso el id la codifica.

    Parameters
    ----------
    date:
        Fecha ``"YYYY-MM-DD"``. Los llamadores usan la **fecha UTC del kickoff** para
        que el id coincida entre proveedores (ver ``parse_match`` y ``football_data``).
    home, away:
        Nombres de las selecciones local y visitante.

    Returns
    -------
    str
        Identificador estable, p. ej. ``"2026-06-11-mexico-vs-south-africa"``.
    """
    return f"{date}-{_slug(home)}-vs-{_slug(away)}"


def _parse_kickoff(date_str: str, time_str: str | None) -> datetime:
    """Convierte ``date`` + ``time`` de openfootball a un ``datetime`` en UTC.

    ``time`` viene como ``"HH:MM UTC±H"`` (offset embebido).

    - Si ``time`` **falta**, se usa medianoche UTC de la fecha *local* de openfootball.
      En ese caso el ``match_id`` queda anclado a esa fecha local, no a la UTC real del
      kickoff: la unión cross-proveedor solo está garantizada para partidos con hora
      conocida (todos los jugados/finalizados la tienen). Los partidos sin hora son
      eliminatorias TBD aún sin resolver, que todavía no se unen a resultados live; por
      eso aquí NO se hace fail-loud (rompería el parseo de esas entradas legítimas).
    - Si ``time`` **está presente pero no se puede parsear**, se lanza ``ValueError``
      (fail loud): como el ``match_id`` se deriva de la fecha UTC del kickoff, un tiempo
      mal interpretado corrompería silenciosamente la clave de unión. Mejor enterarse.
    """
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if not time_str:
        return base
    try:
        hhmm, _, tz_part = time_str.strip().partition(" ")
        hour, minute = (int(x) for x in hhmm.split(":"))
        match = _TZ_RE.search(tz_part)
        if match is None:
            raise ValueError(f"offset de zona no reconocido en {time_str!r}")
        off_hours = int(match.group(1))
        off_minutes = int(match.group(2) or 0)
        sign = -1 if off_hours < 0 else 1
        offset = sign * timedelta(hours=abs(off_hours), minutes=off_minutes)
        local = base.replace(hour=hour, minute=minute, tzinfo=timezone(offset))
        return local.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(
            f"time inválido {time_str!r} para fecha {date_str!r}: {exc}"
        ) from exc


def _pair(value: object) -> tuple[int, int] | None:
    """Devuelve ``(int, int)`` si ``value`` es un par de enteros, si no ``None``."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        return int(value[0]), int(value[1])
    return None


def parse_match(raw: dict, source: str = "openfootball") -> NormalizedMatch:
    """Convierte una entrada de partido de openfootball a :class:`NormalizedMatch`.

    Parameters
    ----------
    raw:
        Dict con el esquema verificado de openfootball.
    source:
        Etiqueta de origen (default ``"openfootball"``).

    Returns
    -------
    NormalizedMatch
        Fixture normalizado. ``FINISHED`` si trae ``score.ft`` válido, si no
        ``SCHEDULED``.
    """
    # Canonicalizamos al espacio de nombres de martj42 (mismo criterio que el cliente
    # live) ANTES de derivar el match_id, para que el Elo y el bono de sede acierten.
    home = canonical_openfootball_team(raw["team1"])
    away = canonical_openfootball_team(raw["team2"])
    date = raw["date"]
    kickoff = _parse_kickoff(date, raw.get("time"))

    score = raw.get("score") or {}
    ft = _pair(score.get("ft"))
    ht = _pair(score.get("ht"))
    status = MatchStatus.FINISHED if ft is not None else MatchStatus.SCHEDULED

    group = raw.get("group")
    stage = group if group else raw.get("round", "")
    # match_id desde la fecha UTC del kickoff (mismo ancla que el cliente live).
    mid = make_match_id(kickoff.date().isoformat(), home, away)

    return NormalizedMatch(
        match_id=mid,
        source=source,
        source_match_id=mid,
        kickoff_utc=kickoff,
        home_team=home,
        away_team=away,
        stage=stage,
        status=status,
        ht_home=ht[0] if ht is not None else None,
        ht_away=ht[1] if ht is not None else None,
        ft_home=ft[0] if ft is not None else None,
        ft_away=ft[1] if ft is not None else None,
        venue=raw.get("ground"),
    )


def parse_fixtures(matches: list[dict]) -> list[NormalizedMatch]:
    """Parsea una lista de entradas de partido (ya aplanada)."""
    return [parse_match(m) for m in matches]


def parse_openfootball(doc: dict) -> list[NormalizedMatch]:
    """Aplana el envoltorio de openfootball y parsea todos los fixtures.

    Soporta ``doc["matches"]`` o ``doc["rounds"][].matches``.

    Raises
    ------
    ValueError
        Si el documento no tiene ninguna de las dos claves (formato no reconocido).
    """
    if "matches" in doc:
        matches = doc["matches"]
    elif "rounds" in doc:
        matches = [m for r in doc["rounds"] for m in r.get("matches", [])]
    else:
        raise ValueError(
            "Formato openfootball no reconocido: falta 'matches' o 'rounds'."
        )
    return parse_fixtures(matches)


def group_teams(matches: list[NormalizedMatch]) -> dict[str, list[str]]:
    """Extrae los grupos y sus selecciones de los fixtures de fase de grupos.

    Returns
    -------
    dict[str, list[str]]
        Mapa ``"Group A" -> [equipos ordenados]``, ordenado por grupo.
    """
    groups: dict[str, set[str]] = {}
    for m in matches:
        if m.stage.startswith("Group"):
            members = groups.setdefault(m.stage, set())
            members.add(m.home_team)
            members.add(m.away_team)
    return {g: sorted(teams) for g, teams in sorted(groups.items())}


def validate_schedule(matches: list[NormalizedMatch]) -> list[str]:
    """Comprueba que el schedule parseado tenga la estructura del WC2026.

    Usa las constantes ``EXPECTED_*`` para fallar ruidosamente ante un backbone parcial
    o malformado (en vez de simular sobre un torneo incompleto). El pipeline debería
    llamar a esta función tras ``parse_openfootball`` y abortar si devuelve problemas.

    Returns
    -------
    list[str]
        Lista de discrepancias (vacía si la estructura es correcta).
    """
    issues: list[str] = []
    total = len(matches)
    if total != EXPECTED_TOTAL_MATCHES:
        issues.append(f"total {total} != {EXPECTED_TOTAL_MATCHES} partidos")

    groups = group_teams(matches)
    if len(groups) != EXPECTED_GROUPS:
        issues.append(f"{len(groups)} grupos != {EXPECTED_GROUPS}")
    for name, teams in groups.items():
        if len(teams) != EXPECTED_TEAMS_PER_GROUP:
            issues.append(f"{name}: {len(teams)} equipos != {EXPECTED_TEAMS_PER_GROUP}")

    group_matches = sum(1 for m in matches if m.stage.startswith("Group"))
    if group_matches != EXPECTED_GROUP_MATCHES:
        issues.append(f"{group_matches} partidos de grupo != {EXPECTED_GROUP_MATCHES}")
    knockout_matches = total - group_matches
    if knockout_matches != EXPECTED_KNOCKOUT_MATCHES:
        issues.append(f"eliminatoria {knockout_matches} != {EXPECTED_KNOCKOUT_MATCHES}")
    return issues
