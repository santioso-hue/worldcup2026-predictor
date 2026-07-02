"""football-data.org tests: pure parser + provider with a fake session (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from worldcup.data.clean import reconcile, validate_match
from worldcup.data.football_data import FootballDataProvider, parse_footballdata_match
from worldcup.data.live_results import MatchStatus

GROUP_FINISHED = {
    "id": 100,
    "utcDate": "2026-06-11T16:00:00Z",
    "status": "FINISHED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_A",
    "homeTeam": {"id": 1, "name": "Mexico"},
    "awayTeam": {"id": 2, "name": "South Africa"},
    "score": {
        "winner": "HOME_TEAM",
        "duration": "REGULAR",
        "fullTime": {"home": 2, "away": 1},
        "halfTime": {"home": 1, "away": 0},
    },
}

KO_PENALTIES = {
    "id": 200,
    "utcDate": "2026-07-19T19:00:00Z",
    "status": "FINISHED",
    "stage": "FINAL",
    "group": None,
    "homeTeam": {"id": 10, "name": "Argentina"},
    "awayTeam": {"id": 20, "name": "France"},
    "score": {
        "winner": "AWAY_TEAM",
        "duration": "PENALTY_SHOOTOUT",
        "fullTime": {"home": 1, "away": 1},
        "halfTime": {"home": 0, "away": 1},
    },
}

TIMED = {
    "id": 300,
    "utcDate": "2026-06-12T19:00:00Z",
    "status": "TIMED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_B",
    "homeTeam": {"id": 3, "name": "Spain"},
    "awayTeam": {"id": 4, "name": "Japan"},
    "score": {
        "winner": None,
        "duration": "REGULAR",
        "fullTime": {"home": None, "away": None},
        "halfTime": {"home": None, "away": None},
    },
}

# Unresolved knockout slot: v4 returns all 32 ties with `None` teams.
UNDETERMINED_KO = {
    "id": 400,
    "utcDate": "2026-06-28T19:00:00Z",
    "status": "TIMED",
    "stage": "LAST_32",
    "group": None,
    "homeTeam": {"id": None, "name": None},
    "awayTeam": {"id": None, "name": None},
    "score": {
        "winner": None,
        "duration": "REGULAR",
        "fullTime": {"home": None, "away": None},
        "halfTime": {"home": None, "away": None},
    },
}


def test_parse_group_finished() -> None:
    m = parse_footballdata_match(GROUP_FINISHED)
    assert m.status is MatchStatus.FINISHED
    assert (m.ft_home, m.ft_away) == (2, 1)
    assert m.stage == "Group A"
    assert m.match_id == "2026-06-11-mexico-vs-south-africa"
    assert m.source == "football_data" and m.source_match_id == "100"
    assert (m.pen_home, m.pen_away) == (None, None)  # no penalties


def test_parse_penalty_ko_encodes_winner() -> None:
    # v4 doesn't break out the shootout: fullTime stays 1-1, the winner is in `winner`.
    m = parse_footballdata_match(KO_PENALTIES)
    assert m.status is MatchStatus.FINISHED
    assert (m.ft_home, m.ft_away) == (1, 1)
    assert (m.et_home, m.et_away) == (1, 1)  # et mirrored = ft (draw after extra time)
    assert (m.pen_home, m.pen_away) == (0, 1)  # AWAY won -> _winner_of picks away
    assert m.stage == "Final"


def test_penalty_ko_survives_validation_and_reconcile() -> None:
    # Regression: a penalty-decided KO from the live feed must pass validate_match and
    # NOT get dropped by reconcile (previously "penalties without extra time" flagged
    # it as suspicious and the already-played tie got re-simulated).
    m = parse_footballdata_match(KO_PENALTIES)
    assert validate_match(m) == []  # clean
    rec = reconcile([], [m])
    assert rec.anomalies == []
    assert [x.source_match_id for x in rec.matches] == ["200"]  # kept, not dropped


def test_parse_timed_is_scheduled() -> None:
    m = parse_footballdata_match(TIMED)
    assert m.status is MatchStatus.SCHEDULED
    assert m.ft_home is None and m.stage == "Group B"


def test_parse_canonicalizes_team_names_to_martj42() -> None:
    # football-data says "Congo DR"; the Elo source (martj42) knows it as "DR Congo".
    # Without canonicalizing, the team would fall back to the default rating (1500)
    # -> broken prediction.
    raw = {
        "id": 500,
        "utcDate": "2026-06-24T02:00:00Z",
        "status": "TIMED",
        "stage": "GROUP_STAGE",
        "group": "GROUP_K",
        "homeTeam": {"id": 5, "name": "Colombia"},
        "awayTeam": {"id": 6, "name": "Congo DR"},
        "score": {
            "winner": None,
            "duration": "REGULAR",
            "fullTime": {"home": None, "away": None},
            "halfTime": {"home": None, "away": None},
        },
    }
    m = parse_footballdata_match(raw)
    assert m.away_team == "DR Congo"  # canonicalized
    assert m.match_id == "2026-06-24-colombia-vs-dr-congo"  # id uses canonical name


# --- Provider with a fake session -------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
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
        return _FakeResponse(self._payload)


def _provider(session: _FakeSession) -> FootballDataProvider:
    return FootballDataProvider(
        "TOKEN", "https://api.football-data.org/v4", "WC", session=session
    )


def test_get_schedule_sends_token_and_hits_competition() -> None:
    session = _FakeSession({"matches": [GROUP_FINISHED, KO_PENALTIES, TIMED]})
    matches = _provider(session).get_schedule()
    assert len(matches) == 3
    assert session.calls[0]["headers"]["X-Auth-Token"] == "TOKEN"
    assert "/competitions/WC/matches" in session.calls[0]["url"]


def test_get_schedule_skips_undetermined_knockout_slots() -> None:
    # All 32 knockout ties arrive with `None` teams until resolved: they get dropped
    # (the pipeline rebuilds the bracket from the groups + Annex C).
    session = _FakeSession({"matches": [GROUP_FINISHED, UNDETERMINED_KO, TIMED]})
    matches = _provider(session).get_schedule()
    assert {m.source_match_id for m in matches} == {"100", "300"}  # 400 excluded


def test_get_finished_filters_status_and_since() -> None:
    session = _FakeSession({"matches": [GROUP_FINISHED, KO_PENALTIES, TIMED]})
    finished = _provider(session).get_finished_results()
    assert sorted(m.source_match_id for m in finished) == [
        "100",
        "200",
    ]  # FINISHED only
    after = _provider(
        _FakeSession({"matches": [GROUP_FINISHED, KO_PENALTIES]})
    ).get_finished_results(since=datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert [m.source_match_id for m in after] == ["200"]  # since filters on kickoff
