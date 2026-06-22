"""Cliente football-data.org (v4) — proveedor LIVE gratuito (cubre el Mundial).

Implementa :class:`~worldcup.data.live_results.LiveResultsProvider`.
:func:`parse_footballdata_match` es **puro**; :class:`FootballDataProvider` hace el I/O
HTTP tras una sesión inyectable (tests sin red). El plan free cubre ``WC`` y no está
limitado por temporada (a diferencia de API-Football). Auth: header ``X-Auth-Token``.

Esquema v4: ``GET /competitions/{code}/matches`` -> ``{matches: [...]}``; cada match
trae ``id``, ``utcDate``, ``status``, ``stage``, ``group``, los equipos y
``score.{winner, duration, fullTime, halfTime}``.

Penales: v4 NO desglosa la tanda y pliega la prórroga en ``fullTime``. Si hay ganador
con ``fullTime`` empatado (penales), codificamos ``pen_*`` (1,0)/(0,1) con el ganador y
reflejamos ``et_* = ft_*`` (empate tras prórroga). El ``et`` no es decorativo: sin él,
``validate_match`` marcaría "penales sin prórroga" y ``reconcile`` descartaría la llave.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from .live_results import (
    LiveResultsProvider,
    MatchStatus,
    NormalizedMatch,
    normalize_status,
)
from .schedule import make_match_id
from .team_names import canonical_footballdata_team

PROVIDER_NAME = "football_data"

_KO_STAGE = {
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-finals",
    "SEMI_FINALS": "Semi-finals",
    "THIRD_PLACE": "Third place",
    "FINAL": "Final",
}


def _iso_to_utc(value: str) -> datetime:
    """Parsea un timestamp ISO-8601 (acepta sufijo ``Z``) y lo normaliza a UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _stage_label(stage: str, group: str | None) -> str:
    """Stage de football-data al rótulo del proyecto ('Group A' / 'Round of 16')."""
    if stage == "GROUP_STAGE" and group:
        return "Group " + group.split("_")[-1]
    return _KO_STAGE.get(stage, stage.replace("_", " ").title())


def _is_determined(raw: dict) -> bool:
    """¿El partido tiene ambos equipos definidos? (puro, sin I/O).

    v4 devuelve las 32 llaves de eliminatoria con ``homeTeam.name=None`` hasta que se
    resuelven (son slots del bracket pendientes). El pipeline reconstruye el bracket
    desde los grupos + Annex C, así que esos placeholders no aportan nada y romperían
    ``make_match_id`` (slug de ``None``). Los descartamos en la frontera del proveedor.
    """
    return (
        raw.get("homeTeam", {}).get("name") is not None
        and raw.get("awayTeam", {}).get("name") is not None
    )


def parse_footballdata_match(
    raw: dict, fetched_at: datetime | None = None
) -> NormalizedMatch:
    """Convierte un match de football-data.org v4 a :class:`NormalizedMatch` (puro).

    El ``match_id`` se deriva de la fecha UTC del kickoff + equipos (mismo ancla que el
    backbone y API-Football, para unir feeds). En un KO por penales se codifican
    ``pen_*`` desde ``score.winner`` y se refleja ``et_* = ft_*`` (ver módulo).
    """
    score = raw.get("score") or {}
    full_time = score.get("fullTime") or {}
    half_time = score.get("halfTime") or {}
    # Canonicalizamos a los nombres de martj42 ANTES de derivar el match_id, para que
    # el id, reconcile, los grupos y el lookup de Elo compartan una sola identidad.
    home = canonical_footballdata_team(raw["homeTeam"]["name"])
    away = canonical_footballdata_team(raw["awayTeam"]["name"])
    kickoff = _iso_to_utc(raw["utcDate"])

    ft_home = _opt_int(full_time.get("home"))
    ft_away = _opt_int(full_time.get("away"))
    pen_home: int | None = None
    pen_away: int | None = None
    et_home: int | None = None
    et_away: int | None = None
    winner = score.get("winner")
    if (
        winner in ("HOME_TEAM", "AWAY_TEAM")
        and ft_home is not None
        and ft_home == ft_away
    ):
        # v4 pliega la prórroga en fullTime: empate (ft) con ganador => penales tras
        # una prórroga también empatada. Reflejamos et = ft además de pen_*, para que
        # validate_match lo lea como "empate resuelto por penales"; sin el et, la regla
        # "penales sin prórroga" lo haría sospechoso y reconcile lo descartaría.
        et_home, et_away = ft_home, ft_away
        pen_home, pen_away = (1, 0) if winner == "HOME_TEAM" else (0, 1)

    return NormalizedMatch(
        match_id=make_match_id(kickoff.date().isoformat(), home, away),
        source=PROVIDER_NAME,
        source_match_id=str(raw["id"]),
        kickoff_utc=kickoff,
        home_team=home,
        away_team=away,
        stage=_stage_label(raw.get("stage", ""), raw.get("group")),
        status=normalize_status(PROVIDER_NAME, raw["status"]),
        ht_home=_opt_int(half_time.get("home")),
        ht_away=_opt_int(half_time.get("away")),
        ft_home=ft_home,
        ft_away=ft_away,
        et_home=et_home,
        et_away=et_away,
        pen_home=pen_home,
        pen_away=pen_away,
        venue=None,
        fetched_at=fetched_at,
    )


class _HttpResponse(Protocol):
    def json(self) -> dict: ...

    def raise_for_status(self) -> None: ...


class _HttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: float,
    ) -> _HttpResponse: ...


class FootballDataProvider(LiveResultsProvider):
    """Cliente HTTP de football-data.org v4 tras la interfaz estable.

    Parameters
    ----------
    token:
        Clave (de ``FOOTBALL_DATA_TOKEN``); header ``X-Auth-Token``.
    base_url:
        Base de la API (``https://api.football-data.org/v4``).
    competition_code:
        Código de competición (``WC`` para el Mundial).
    session:
        Sesión HTTP inyectable (tests). ``None`` -> ``requests.Session`` perezoso.
    timeout:
        Timeout por request en segundos.
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        competition_code: str,
        *,
        session: _HttpSession | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._competition = competition_code
        self._session = session
        self._timeout = timeout
        self.request_count = 0

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def _session_or_default(self) -> _HttpSession:
        if self._session is None:
            import requests  # import perezoso: el módulo no depende de requests

            self._session = requests.Session()
        return self._session

    def _fetch_matches(self, params: dict[str, Any]) -> list[dict]:
        """GET /competitions/{code}/matches; los errores HTTP (4xx/5xx) abortan."""
        session = self._session_or_default()
        url = f"{self._base_url}/competitions/{self._competition}/matches"
        headers = {"X-Auth-Token": self._token}
        resp = session.get(url, headers=headers, params=params, timeout=self._timeout)
        resp.raise_for_status()
        self.request_count += 1
        matches: list[dict] = resp.json().get("matches", [])
        return matches

    def _parse_determined(self, params: dict[str, Any]) -> list[NormalizedMatch]:
        """GET + parse, omitiendo fixtures sin equipos definidos (slots pendientes)."""
        return [
            parse_footballdata_match(m)
            for m in self._fetch_matches(params)
            if _is_determined(m)
        ]

    def get_schedule(self) -> list[NormalizedMatch]:
        return self._parse_determined({})

    def get_live_fixtures(self) -> list[NormalizedMatch]:
        return self._parse_determined({"status": "IN_PLAY,PAUSED"})

    def get_finished_results(
        self, since: datetime | None = None
    ) -> list[NormalizedMatch]:
        # `since` filtra por kickoff_utc (hora de inicio), no por la de finalización.
        out: list[NormalizedMatch] = []
        for match in self._parse_determined({"status": "FINISHED"}):
            if match.status is not MatchStatus.FINISHED:
                continue
            if since is not None and match.kickoff_utc < since:
                continue
            out.append(match)
        return out
