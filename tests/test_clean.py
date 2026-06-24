"""Tests de validación y reconciliación (conservar el último snapshot válido)."""

from __future__ import annotations

from datetime import datetime, timezone

from worldcup.data.clean import (
    apply_team_aliases,
    reconcile,
    validate_match,
)
from worldcup.data.live_results import MatchStatus, NormalizedMatch

UTC = timezone.utc


def _m(match_id: str, **kw: object) -> NormalizedMatch:
    base: dict[str, object] = dict(
        match_id=match_id,
        source="football_data",
        source_match_id=match_id,
        kickoff_utc=datetime(2026, 6, 11, 19, 0, tzinfo=UTC),
        home_team="A",
        away_team="B",
        stage="Group A",
        status=MatchStatus.SCHEDULED,
    )
    base.update(kw)
    return NormalizedMatch(**base)  # type: ignore[arg-type]


# --- validación -------------------------------------------------------------


def test_healthy_finished_has_no_issues() -> None:
    assert (
        validate_match(_m("x", status=MatchStatus.FINISHED, ft_home=2, ft_away=0)) == []
    )


def test_finished_without_score_flagged() -> None:
    assert validate_match(_m("x", status=MatchStatus.FINISHED)) != []


def test_scheduled_with_score_flagged() -> None:
    assert (
        validate_match(_m("x", status=MatchStatus.SCHEDULED, ft_home=1, ft_away=0))
        != []
    )


def test_negative_score_flagged() -> None:
    assert (
        validate_match(_m("x", status=MatchStatus.FINISHED, ft_home=-1, ft_away=0))
        != []
    )


def test_penalties_without_extratime_flagged() -> None:
    m = _m(
        "x", status=MatchStatus.FINISHED, ft_home=1, ft_away=1, pen_home=4, pen_away=3
    )
    assert validate_match(m) != []


def test_extratime_one_sided_flagged() -> None:
    m = _m(
        "x", status=MatchStatus.FINISHED, ft_home=1, ft_away=1, et_home=2
    )  # et_away None
    assert any("prórroga incompleta" in i for i in validate_match(m))


def test_knockout_draw_without_resolution_flagged() -> None:
    ko = _m("k", stage="Round of 16", status=MatchStatus.FINISHED, ft_home=1, ft_away=1)
    assert any("empate sin resolución" in i for i in validate_match(ko))


def test_knockout_draw_resolved_by_penalties_is_clean() -> None:
    ko = _m(
        "k",
        stage="Round of 16",
        status=MatchStatus.FINISHED,
        ft_home=1,
        ft_away=1,
        et_home=1,
        et_away=1,
        pen_home=4,
        pen_away=3,
    )
    assert validate_match(ko) == []


def test_group_draw_is_legal() -> None:
    g = _m("g", stage="Group A", status=MatchStatus.FINISHED, ft_home=1, ft_away=1)
    assert validate_match(g) == []


# --- reconciliación ---------------------------------------------------------


def test_normal_progression_accepts_incoming() -> None:
    prev = [_m("a", status=MatchStatus.SCHEDULED)]
    inc = [_m("a", status=MatchStatus.FINISHED, ft_home=1, ft_away=0)]
    res = reconcile(prev, inc)
    assert res.matches[0].status is MatchStatus.FINISHED
    assert res.anomalies == []


def test_suspicious_incoming_keeps_previous() -> None:
    prev = [_m("a", status=MatchStatus.SCHEDULED)]
    inc = [_m("a", status=MatchStatus.FINISHED)]  # finished sin marcador
    res = reconcile(prev, inc)
    assert res.matches[0].status is MatchStatus.SCHEDULED  # se conservó el previo
    assert res.anomalies and res.anomalies[0][0] == "a"


def test_suspicious_incoming_without_previous_is_dropped() -> None:
    res = reconcile([], [_m("a", status=MatchStatus.FINISHED)])
    assert res.matches == []
    assert res.anomalies and "descartado" in res.anomalies[0][1]


def test_locked_result_is_immutable() -> None:
    prev = [_m("a", status=MatchStatus.FINISHED, ft_home=2, ft_away=0)]
    inc = [_m("a", status=MatchStatus.FINISHED, ft_home=3, ft_away=0)]  # contradice
    res = reconcile(prev, inc)
    assert (res.matches[0].ft_home, res.matches[0].ft_away) == (2, 0)
    assert res.anomalies and "bloqueado" in res.anomalies[0][1]


def test_locked_resolution_is_immutable() -> None:
    # Mismo 90' (1-1) pero resolución distinta: ET-decidido no puede pasar a PEN.
    prev = [
        _m(
            "k",
            stage="Round of 16",
            status=MatchStatus.FINISHED,
            ft_home=1,
            ft_away=1,
            et_home=2,
            et_away=1,
        )
    ]
    inc = [
        _m(
            "k",
            stage="Round of 16",
            status=MatchStatus.FINISHED,
            ft_home=1,
            ft_away=1,
            et_home=1,
            et_away=1,
            pen_home=4,
            pen_away=3,
        )
    ]
    res = reconcile(prev, inc)
    assert (res.matches[0].et_home, res.matches[0].et_away) == (2, 1)  # conserva previo
    assert res.anomalies and "bloqueado" in res.anomalies[0][1]


def test_partial_feed_preserves_missing_matches() -> None:
    prev = [
        _m("a", status=MatchStatus.FINISHED, ft_home=1, ft_away=0),
        _m("b", status=MatchStatus.SCHEDULED),
    ]
    inc = [_m("a", status=MatchStatus.FINISHED, ft_home=1, ft_away=0)]  # falta "b"
    res = reconcile(prev, inc)
    ids = {m.match_id for m in res.matches}
    assert ids == {"a", "b"}  # "b" se conserva aunque no vino en el feed


# --- canonicalización -------------------------------------------------------


def test_apply_team_aliases_recomputes_match_id() -> None:
    m = _m(
        "2026-06-15-usa-vs-mexico",
        home_team="USA",
        away_team="Mexico",
    )
    out = apply_team_aliases([m], {"USA": "United States"})
    assert out[0].home_team == "United States"
    assert out[0].match_id == "2026-06-15-united-states-vs-mexico"


def test_apply_team_aliases_noop_preserves_match() -> None:
    # Sin alias coincidente: pass-through (mismo objeto, mismo match_id).
    m = _m("2026-06-15-usa-vs-mexico", home_team="USA", away_team="Mexico")
    out = apply_team_aliases([m], {"Brazil": "Brasil"})
    assert out[0] is m
    assert out[0].match_id == "2026-06-15-usa-vs-mexico"
