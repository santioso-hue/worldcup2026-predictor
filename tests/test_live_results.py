"""Tests de la interfaz live: mapeo de estado, esquema normalizado e ``is_lockable``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from worldcup.data.live_results import (
    MatchStatus,
    NormalizedMatch,
    is_lockable,
    normalize_status,
)


def _match(**kw: object) -> NormalizedMatch:
    base: dict[str, object] = dict(
        match_id="WC2026-1",
        source="api_football",
        source_match_id="123",
        kickoff_utc=datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc),
        home_team="Mexico",
        away_team="South Africa",
        stage="Group A",
        status=MatchStatus.FINISHED,
    )
    base.update(kw)
    return NormalizedMatch(**base)  # type: ignore[arg-type]


def test_status_mapping_both_providers() -> None:
    assert normalize_status("api_football", "FT") is MatchStatus.FINISHED
    assert normalize_status("api_football", "1H") is MatchStatus.IN_PLAY
    assert normalize_status("api_football", "NS") is MatchStatus.SCHEDULED
    assert normalize_status("football_data", "FINISHED") is MatchStatus.FINISHED
    assert normalize_status("football_data", "IN_PLAY") is MatchStatus.IN_PLAY


def test_unknown_status_raises_not_guesses() -> None:
    # Edge case: un código desconocido debe fallar, no bloquear un partido por error.
    with pytest.raises(KeyError):
        normalize_status("api_football", "ZZ")


def test_decided_by_and_is_finished() -> None:
    regular = _match(ft_home=2, ft_away=0)
    assert regular.decided_by == "REGULAR_TIME"
    assert regular.is_finished is True

    extra = _match(ft_home=1, ft_away=1, et_home=2, et_away=1)
    assert extra.decided_by == "EXTRA_TIME"

    pens = _match(ft_home=1, ft_away=1, et_home=1, et_away=1, pen_home=4, pen_away=3)
    assert pens.decided_by == "PENALTIES"


def test_scheduled_match_has_no_scores() -> None:
    scheduled = _match(status=MatchStatus.SCHEDULED)
    assert scheduled.ft_home is None
    assert scheduled.is_finished is False


# --- is_lockable: bloquear FINISHED con marcador de 90' ---------------------


def test_lockable_finished_with_score() -> None:
    assert is_lockable(_match(status=MatchStatus.FINISHED, ft_home=2, ft_away=0))


def test_not_lockable_when_in_play_or_scheduled() -> None:
    assert not is_lockable(_match(status=MatchStatus.IN_PLAY, ft_home=1, ft_away=0))
    assert not is_lockable(_match(status=MatchStatus.SCHEDULED))


def test_not_lockable_finished_without_score() -> None:
    # Edge case: FINISHED sin marcador = dato parcial/sospechoso -> NO bloquear.
    suspicious = _match(status=MatchStatus.FINISHED, ft_home=None, ft_away=None)
    assert not is_lockable(suspicious)
