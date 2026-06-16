"""Tests del cliente API-Football: parser puro + provider con sesión falsa (sin red)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from worldcup.data.api_football import APIFootballProvider, parse_apifootball_fixture
from worldcup.data.live_results import MatchStatus
from worldcup.data.schedule import parse_match

FINAL_PEN = {
    "fixture": {
        "id": 999,
        "date": "2026-07-19T15:00:00-04:00",
        "status": {"short": "PEN"},
        "venue": {"name": "MetLife Stadium"},
    },
    "league": {"id": 1, "season": 2026, "round": "Final"},
    "teams": {
        "home": {"id": 10, "name": "Argentina"},
        "away": {"id": 20, "name": "France"},
    },
    "goals": {"home": 3, "away": 3},
    "score": {
        "halftime": {"home": 1, "away": 1},
        "fulltime": {"home": 2, "away": 2},
        "extratime": {"home": 3, "away": 3},
        "penalty": {"home": 4, "away": 2},
    },
}

GROUP_INPLAY = {
    "fixture": {
        "id": 5,
        "date": "2026-06-11T19:00:00+00:00",
        "status": {"short": "1H"},
        "venue": {"name": "Estadio Azteca"},
    },
    "league": {"id": 1, "season": 2026, "round": "Group Stage - 1"},
    "teams": {
        "home": {"id": 1, "name": "Mexico"},
        "away": {"id": 2, "name": "South Africa"},
    },
    "goals": {"home": 1, "away": 0},
    "score": {
        "halftime": {"home": None, "away": None},
        "fulltime": {"home": None, "away": None},
        "extratime": {"home": None, "away": None},
        "penalty": {"home": None, "away": None},
    },
}


def test_parse_penalty_final() -> None:
    m = parse_apifootball_fixture(FINAL_PEN)
    assert m.status is MatchStatus.FINISHED
    assert (m.ft_home, m.ft_away) == (2, 2)
    assert (m.et_home, m.et_away) == (3, 3)
    assert (m.pen_home, m.pen_away) == (4, 2)
    assert m.decided_by == "PENALTIES"
    assert m.match_id == "2026-07-19-argentina-vs-france"
    assert m.source_match_id == "999"
    assert m.venue == "MetLife Stadium"
    assert m.kickoff_utc.tzinfo == timezone.utc
    assert m.kickoff_utc.hour == 19  # 15:00 -04:00 -> 19:00Z


def test_parse_inplay_has_no_fulltime_score() -> None:
    m = parse_apifootball_fixture(GROUP_INPLAY)
    assert m.status is MatchStatus.IN_PLAY
    assert m.ft_home is None and m.ft_away is None
    assert m.match_id == "2026-06-11-mexico-vs-south-africa"


# --- Provider con sesión falsa ---------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Devuelve payloads en orden; registra las llamadas."""

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers, "params": params})
        if not self._payloads:
            raise AssertionError("FakeSession: sin payloads restantes")
        return _FakeResponse(self._payloads.pop(0))


def _provider(session: _FakeSession) -> APIFootballProvider:
    return APIFootballProvider(
        api_key="TEST",
        base_url="https://v3.football.api-sports.io",
        league_id=1,
        season=2026,
        session=session,
    )


def test_get_schedule_maps_and_sends_auth_header() -> None:
    session = _FakeSession(
        [{"response": [FINAL_PEN, GROUP_INPLAY], "paging": {"current": 1, "total": 1}}]
    )
    provider = _provider(session)
    matches = provider.get_schedule()
    assert len(matches) == 2
    assert session.calls[0]["headers"]["x-apisports-key"] == "TEST"
    assert provider.request_count == 1


def test_pagination_follows_total_pages() -> None:
    session = _FakeSession(
        [
            {"response": [GROUP_INPLAY], "paging": {"current": 1, "total": 2}},
            {"response": [FINAL_PEN], "paging": {"current": 2, "total": 2}},
        ]
    )
    provider = _provider(session)
    matches = provider.get_schedule()
    assert len(matches) == 2
    assert provider.request_count == 2  # dos páginas


def test_live_fixtures_filtered_to_our_league() -> None:
    other_league = {**GROUP_INPLAY, "league": {"id": 39, "round": "X"}}
    session = _FakeSession(
        [
            {
                "response": [GROUP_INPLAY, other_league],
                "paging": {"current": 1, "total": 1},
            }
        ]
    )
    matches = _provider(session).get_live_fixtures()
    assert len(matches) == 1  # se descarta la liga ajena
    assert matches[0].home_team == "Mexico"


def test_api_errors_raise() -> None:
    session = _FakeSession([{"errors": ["Invalid API key"], "response": []}])
    with pytest.raises(RuntimeError, match="API-Football"):
        _provider(session).get_schedule()


def test_finished_results_filtered_by_status_and_since() -> None:
    session = _FakeSession(
        [{"response": [FINAL_PEN, GROUP_INPLAY], "paging": {"current": 1, "total": 1}}]
    )
    finished = _provider(session).get_finished_results()
    assert [m.source_match_id for m in finished] == ["999"]  # solo el FINISHED


def test_finished_results_since_filters_on_kickoff() -> None:
    # FINAL_PEN arranca 2026-07-19 19:00Z. `since` filtra por kickoff, no por fin.
    payload = {"response": [FINAL_PEN], "paging": {"current": 1, "total": 1}}
    after = _provider(_FakeSession([payload])).get_finished_results(
        since=datetime(2026, 7, 20, tzinfo=timezone.utc)
    )
    assert after == []  # kickoff anterior a `since` -> excluido
    before = _provider(_FakeSession([payload])).get_finished_results(
        since=datetime(2026, 7, 1, tzinfo=timezone.utc)
    )
    assert [m.source_match_id for m in before] == ["999"]


# --- regresiones del review adversarial ------------------------------------


def test_match_id_stable_across_providers_near_midnight() -> None:
    # Mismo partido visto por openfootball (21:00 UTC-5 del 15) y API-Football
    # (02:00Z del 16): el id debe coincidir porque ambos usan la FECHA UTC del kickoff.
    of_match = parse_match(
        {
            "team1": "USA",
            "team2": "Mexico",
            "date": "2026-06-15",
            "time": "21:00 UTC-5",
            "group": "Group D",
        }
    )
    af_match = parse_apifootball_fixture(
        {
            "fixture": {
                "id": 1,
                "date": "2026-06-16T02:00:00+00:00",
                "status": {"short": "NS"},
                "venue": {},
            },
            "league": {"id": 1, "round": "Group Stage - 1"},
            "teams": {
                "home": {"id": 1, "name": "USA"},
                "away": {"id": 2, "name": "Mexico"},
            },
            "goals": {"home": None, "away": None},
            "score": {
                "halftime": {"home": None, "away": None},
                "fulltime": {"home": None, "away": None},
                "extratime": {"home": None, "away": None},
                "penalty": {"home": None, "away": None},
            },
        }
    )
    assert of_match.match_id == af_match.match_id == "2026-06-16-usa-vs-mexico"
