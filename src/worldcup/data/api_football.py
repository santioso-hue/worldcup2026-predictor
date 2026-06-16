"""Cliente API-Football (v3) — proveedor LIVE primario.

Implementa :class:`~worldcup.data.live_results.LiveResultsProvider`. El parser
:func:`parse_apifootball_fixture` es **puro** (mapea un dict de fixture a
``NormalizedMatch``) y se testea sin red; la clase
:class:`APIFootballProvider` hace el I/O HTTP detrás de una sesión inyectable, de modo
que los tests usan una sesión falsa y nunca tocan la red.

Esquema VERIFICADO contra la doc oficial (v3, 16 jun 2026; ver data/raw/SOURCES.md):

    fixture.id · fixture.date (ISO con offset) · fixture.status.short
    teams.home.name / teams.away.name
    score.halftime/fulltime/extratime/penalty -> {home, away}
    league.id / league.round · fixture.venue.name

Auth: header ``x-apisports-key``. Endpoint ``GET /fixtures``; ``live=all`` devuelve
todos los partidos en juego en una llamada. La disciplina de polling por ventanas vive
en el pipeline/triggers, no aquí: el proveedor solo hace llamadas puntuales.
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

PROVIDER_NAME = "api_football"


def _iso_to_utc(value: str) -> datetime:
    """Parsea un timestamp ISO-8601 (con o sin offset) y lo normaliza a UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _opt_int(value: Any) -> int | None:
    """Devuelve ``int`` o ``None`` (los marcadores ausentes llegan como ``null``)."""
    return None if value is None else int(value)


def parse_apifootball_fixture(
    raw: dict, fetched_at: datetime | None = None
) -> NormalizedMatch:
    """Convierte un item de ``/fixtures`` de API-Football a :class:`NormalizedMatch`.

    El ``match_id`` se deriva de la **fecha UTC del kickoff** + los nombres de equipo,
    el mismo ancla que usa el schedule backbone, para que el id sea idéntico aunque el
    partido venga de openfootball o de API-Football (la fecha *local* de cada feed puede
    diferir cerca de medianoche; la UTC es neutral). La canonicalización de nombres es
    responsabilidad de ``clean.py``.

    ``stage`` lleva ``league.round`` (p.ej. ``"Group Stage - 1"`` en fase de grupos, o
    ``"Round of 16"``); **no** es la letra del grupo — esa la fija el backbone vía el
    ``match_id`` compartido.

    Parameters
    ----------
    raw:
        Un elemento de la lista ``response`` de API-Football.
    fetched_at:
        Momento de la descarga (para el sello de "última actualización").

    Returns
    -------
    NormalizedMatch
        Partido normalizado. Los marcadores se toman por fase de ``score.*`` (no de
        ``goals``, cuya semántica de prórroga es ambigua).
    """
    fixture = raw["fixture"]
    teams = raw["teams"]
    league = raw.get("league") or {}
    score = raw.get("score") or {}

    home = teams["home"]["name"]
    away = teams["away"]["name"]
    kickoff = _iso_to_utc(fixture["date"])
    utc_date = kickoff.date().isoformat()  # ancla de join neutral entre proveedores

    ht = score.get("halftime") or {}
    ft = score.get("fulltime") or {}
    et = score.get("extratime") or {}
    pen = score.get("penalty") or {}
    venue = fixture.get("venue") or {}

    return NormalizedMatch(
        match_id=make_match_id(utc_date, home, away),
        source=PROVIDER_NAME,
        source_match_id=str(fixture["id"]),
        kickoff_utc=kickoff,
        home_team=home,
        away_team=away,
        stage=str(league.get("round", "")),
        status=normalize_status(PROVIDER_NAME, fixture["status"]["short"]),
        ht_home=_opt_int(ht.get("home")),
        ht_away=_opt_int(ht.get("away")),
        ft_home=_opt_int(ft.get("home")),
        ft_away=_opt_int(ft.get("away")),
        et_home=_opt_int(et.get("home")),
        et_away=_opt_int(et.get("away")),
        pen_home=_opt_int(pen.get("home")),
        pen_away=_opt_int(pen.get("away")),
        venue=venue.get("name"),
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


class APIFootballProvider(LiveResultsProvider):
    """Cliente HTTP de API-Football detrás de la interfaz estable.

    Parameters
    ----------
    api_key:
        Clave (de la variable de entorno ``API_FOOTBALL_KEY``); nunca hardcodeada.
    base_url:
        Base de la API (``https://v3.football.api-sports.io``).
    league_id, season:
        Identificadores del WC2026 (``1`` y ``2026``, verificados).
    session:
        Sesión HTTP inyectable (para tests). Si es ``None``, se crea
        ``requests.Session`` de forma perezosa (el módulo importa sin ``requests``).
    timeout:
        Timeout por request en segundos.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        league_id: int,
        season: int,
        *,
        session: _HttpSession | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._league_id = league_id
        self._season = season
        self._session = session
        self._timeout = timeout
        self.request_count = 0  # observabilidad del presupuesto de requests

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def _session_or_default(self) -> _HttpSession:
        if self._session is None:
            import requests  # import perezoso: el módulo no depende de requests

            self._session = requests.Session()
        return self._session

    def _fetch_fixtures(self, params: dict[str, Any]) -> list[dict]:
        """GET /fixtures con manejo de paginación y del envoltorio de errores.

        Un ``status.short`` no mapeado hace que ``parse_*`` lance ``KeyError`` y aborte
        el refresh entero (fail loud). Es deliberado: un código nuevo es un cambio de
        esquema que queremos detectar, y el pipeline conserva el último snapshot válido
        ante un refresh fallido (no se degrada la predicción).
        """
        session = self._session_or_default()
        url = f"{self._base_url}/fixtures"
        headers = {"x-apisports-key": self._api_key}
        out: list[dict] = []
        page = 1
        while True:
            page_params = {**params, "page": page}
            resp = session.get(
                url, headers=headers, params=page_params, timeout=self._timeout
            )
            resp.raise_for_status()
            self.request_count += 1
            data = resp.json()
            if data.get("errors"):
                raise RuntimeError(f"API-Football devolvió errores: {data['errors']}")
            out.extend(data.get("response", []))
            paging = data.get("paging") or {}
            current = int(paging.get("current", 1))
            total = int(paging.get("total", 1))
            if current >= total:
                break
            page += 1
        return out

    def get_schedule(self) -> list[NormalizedMatch]:
        raws = self._fetch_fixtures({"league": self._league_id, "season": self._season})
        return [parse_apifootball_fixture(r) for r in raws]

    def get_live_fixtures(self) -> list[NormalizedMatch]:
        raws = self._fetch_fixtures({"live": "all"})
        # live=all trae todas las ligas; filtramos a la nuestra.
        return [
            parse_apifootball_fixture(r)
            for r in raws
            if (r.get("league") or {}).get("id") == self._league_id
        ]

    def get_finished_results(
        self, since: datetime | None = None
    ) -> list[NormalizedMatch]:
        # `since` filtra por kickoff_utc (hora de inicio), no por la de finalización.
        raws = self._fetch_fixtures({"league": self._league_id, "season": self._season})
        out: list[NormalizedMatch] = []
        for r in raws:
            match = parse_apifootball_fixture(r)
            if match.status is not MatchStatus.FINISHED:
                continue
            if since is not None and match.kickoff_utc < since:
                continue
            out.append(match)
        return out
