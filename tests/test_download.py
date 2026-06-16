"""Tests de snapshotting: round-trip de registros (puro) y de parquet (con pandas)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from worldcup.data.download import (
    _opt_int,
    latest_timestamp,
    load_latest_snapshot,
    load_snapshot,
    make_timestamp,
    match_to_record,
    record_to_match,
    save_snapshot,
)
from worldcup.data.live_results import MatchStatus, NormalizedMatch

UTC = timezone.utc


def _finished() -> NormalizedMatch:
    return NormalizedMatch(
        match_id="2026-07-19-argentina-vs-france",
        source="api_football",
        source_match_id="999",
        kickoff_utc=datetime(2026, 7, 19, 19, 0, tzinfo=UTC),
        home_team="Argentina",
        away_team="France",
        stage="Final",
        status=MatchStatus.FINISHED,
        ht_home=1,
        ht_away=1,
        ft_home=2,
        ft_away=2,
        et_home=3,
        et_away=3,
        pen_home=4,
        pen_away=2,
        venue="MetLife Stadium",
        fetched_at=datetime(2026, 7, 19, 21, 30, tzinfo=UTC),
    )


def _scheduled() -> NormalizedMatch:
    return NormalizedMatch(
        match_id="2026-06-12-canada-vs-wales",
        source="openfootball",
        source_match_id="2026-06-12-canada-vs-wales",
        kickoff_utc=datetime(2026, 6, 12, 22, 0, tzinfo=UTC),
        home_team="Canada",
        away_team="Wales",
        stage="Group B",
        status=MatchStatus.SCHEDULED,
    )


def test_make_timestamp_is_utc_and_formatted() -> None:
    assert make_timestamp(datetime(2026, 6, 16, 18, 30, tzinfo=UTC)) == "20260616t1830"
    # convierte a UTC antes de formatear
    tz_minus4 = timezone.utc.utcoffset(None)  # noqa: F841 - claridad
    from datetime import timedelta

    other = datetime(2026, 6, 16, 14, 30, tzinfo=timezone(timedelta(hours=-4)))
    assert make_timestamp(other) == "20260616t1830"


def test_record_round_trip_is_lossless() -> None:
    for original in (_finished(), _scheduled()):
        assert record_to_match(match_to_record(original)) == original


def test_parquet_round_trip(tmp_path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    matches = [_finished(), _scheduled()]
    ts = "20260719t2130"
    path = save_snapshot(matches, ts, tmp_path)
    assert path.exists()
    assert latest_timestamp(tmp_path) == ts
    loaded = load_snapshot(ts, tmp_path)
    assert loaded == matches
    assert load_latest_snapshot(tmp_path) == matches


def test_snapshots_are_immutable(tmp_path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    ts = "20260719t2130"
    save_snapshot([_finished()], ts, tmp_path)
    with pytest.raises(FileExistsError):
        save_snapshot([_finished()], ts, tmp_path)  # no se machaca


def test_load_latest_none_when_empty(tmp_path) -> None:
    assert latest_timestamp(tmp_path) is None
    assert load_latest_snapshot(tmp_path) is None


def test_opt_int_coerces_nan_and_floats() -> None:
    # Guarda del round-trip parquet: NaN/float deben volver a None/int, no corromper.
    assert _opt_int(None) is None
    assert _opt_int(float("nan")) is None
    assert _opt_int(2.0) == 2
    assert _opt_int(3) == 3


def test_record_to_match_coerces_nan_strings_to_none() -> None:
    # pandas devuelve NaN (float) para celdas de texto ausentes (venue/fetched_at):
    # deben volver a None, NO a la cadena "nan" (regresión del round-trip de snapshot).
    rec = match_to_record(_scheduled())  # venue y fetched_at son None
    rec["venue"] = float("nan")
    rec["fetched_at"] = float("nan")
    m = record_to_match(rec)
    assert m.venue is None
    assert m.fetched_at is None
