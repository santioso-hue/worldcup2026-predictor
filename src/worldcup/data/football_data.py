"""football-data.org (v4) client -- free LIVE provider (covers the World Cup).

Implements :class:`~worldcup.data.live_results.LiveResultsProvider`.
:func:`parse_footballdata_match` is **pure**; :class:`FootballDataProvider` does the
HTTP I/O behind an injectable session (tests run without network). The free plan
covers ``WC`` and isn't season-limited (unlike API-Football). Auth: ``X-Auth-Token``
header.

v4 schema: ``GET /competitions/{code}/matches`` -> ``{matches: [...]}``; each match
carries ``id``, ``utcDate``, ``status``, ``stage``, ``group``, the teams, and
``score.{winner, duration, fullTime, halfTime}``. Decided knockouts additionally
carry ``regularTime``, ``extraTime``, and ``penalties``.

Penalties: v4 folds everything into ``fullTime``, shootout goals included (a 1-1
match decided 3-4 on penalties arrives as ``fullTime`` 4-5). When the phase
breakdown is present we rebuild the real phases from ``regularTime``/``extraTime``/
``penalties``. When it's absent we fall back to inference: a level ``fullTime``
with a winner means penalties, encoded as a synthetic (1,0)/(0,1) shootout with
``et_* = ft_*`` mirrored — without the mirror, ``validate_match`` flags "penalties
without extra time" and ``reconcile`` drops the tie.
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
    """Parse an ISO-8601 timestamp (accepts a ``Z`` suffix) and normalize to UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _stage_label(stage: str, group: str | None) -> str:
    """Map a football-data stage to the project's label ('Group A' / 'Round of 16')."""
    if stage == "GROUP_STAGE" and group:
        return "Group " + group.split("_")[-1]
    return _KO_STAGE.get(stage, stage.replace("_", " ").title())


def _is_determined(raw: dict) -> bool:
    """Does the match have both teams set? (pure, no I/O)

    v4 returns all 32 knockout ties with ``homeTeam.name=None`` until they're
    resolved -- they're pending bracket slots. The pipeline rebuilds the bracket from
    the groups + Annex C, so these placeholders add nothing and would break
    ``make_match_id`` (slugging ``None``). We drop them at the provider boundary.
    """
    return (
        raw.get("homeTeam", {}).get("name") is not None
        and raw.get("awayTeam", {}).get("name") is not None
    )


def parse_footballdata_match(
    raw: dict, fetched_at: datetime | None = None
) -> NormalizedMatch:
    """Convert a football-data.org v4 match to :class:`NormalizedMatch` (pure).

    ``match_id`` is derived from the UTC kickoff date + teams (same anchor as the
    openfootball backbone, so feeds can be joined).
    """
    score = raw.get("score") or {}
    full_time = score.get("fullTime") or {}
    half_time = score.get("halfTime") or {}
    regular_time = score.get("regularTime") or {}
    extra_time = score.get("extraTime") or {}
    penalties = score.get("penalties") or {}
    # Canonicalize to martj42 names BEFORE deriving match_id, so the id, reconcile,
    # groups, and Elo lookup all share one identity.
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
    duration = score.get("duration")
    rt_home = _opt_int(regular_time.get("home"))
    rt_away = _opt_int(regular_time.get("away"))
    if (
        duration in ("EXTRA_TIME", "PENALTY_SHOOTOUT")
        and rt_home is not None
        and rt_away is not None
    ):
        # fullTime bundles even shootout goals (module docstring); rebuild the
        # real phases: ft = 90' score, et = cumulative, pen = the shootout.
        # A partial regularTime falls through to the inference below instead.
        ft_home, ft_away = rt_home, rt_away
        et_home = rt_home + (_opt_int(extra_time.get("home")) or 0)
        et_away = rt_away + (_opt_int(extra_time.get("away")) or 0)
        if duration == "PENALTY_SHOOTOUT":
            pen_home = _opt_int(penalties.get("home"))
            pen_away = _opt_int(penalties.get("away"))
            if (pen_home is None or pen_away is None) and winner in (
                "HOME_TEAM",
                "AWAY_TEAM",
            ):
                # Shootout numbers can lag the result on the feed; encode the
                # winner as a synthetic 1-0/0-1 so the decided tie isn't
                # dropped as "finished knockout without a resolution".
                pen_home, pen_away = (1, 0) if winner == "HOME_TEAM" else (0, 1)
    elif (
        winner in ("HOME_TEAM", "AWAY_TEAM")
        and ft_home is not None
        and ft_home == ft_away
    ):
        # No phase breakdown: infer penalties from the level fullTime + winner
        # (synthetic shootout, mirrored et — see module docstring).
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
    """HTTP client for football-data.org v4 behind the stable interface.

    Parameters
    ----------
    token:
        API key (from ``FOOTBALL_DATA_TOKEN``); sent as the ``X-Auth-Token`` header.
    base_url:
        API base (``https://api.football-data.org/v4``).
    competition_code:
        Competition code (``WC`` for the World Cup).
    session:
        Injectable HTTP session (tests). ``None`` -> lazily built ``requests.Session``.
    timeout:
        Per-request timeout in seconds.
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

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def _session_or_default(self) -> _HttpSession:
        if self._session is None:
            import requests  # lazy import: the module has no hard requests dependency

            self._session = requests.Session()
        return self._session

    def _fetch_matches(self, params: dict[str, Any]) -> list[dict]:
        """GET /competitions/{code}/matches; HTTP errors (4xx/5xx) abort."""
        session = self._session_or_default()
        url = f"{self._base_url}/competitions/{self._competition}/matches"
        headers = {"X-Auth-Token": self._token}
        resp = session.get(url, headers=headers, params=params, timeout=self._timeout)
        resp.raise_for_status()
        matches: list[dict] = resp.json().get("matches", [])
        return matches

    def _parse_determined(self, params: dict[str, Any]) -> list[NormalizedMatch]:
        """GET + parse, skipping fixtures with undetermined teams (pending slots)."""
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
        # `since` filters on kickoff_utc (start time), not finish time.
        out: list[NormalizedMatch] = []
        for match in self._parse_determined({"status": "FINISHED"}):
            if match.status is not MatchStatus.FINISHED:
                continue
            if since is not None and match.kickoff_utc < since:
                continue
            out.append(match)
        return out
