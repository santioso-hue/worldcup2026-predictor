"""RefreshTrigger tests: watch cadence and single cron fire."""

from __future__ import annotations

import pytest

from worldcup.data.triggers import CronTrigger, WatchTrigger


def test_watch_fires_max_ticks_and_sleeps_between() -> None:
    ticks: list[int] = []
    sleeps: list[float] = []
    trigger = WatchTrigger(600, max_ticks=3, sleep=sleeps.append)
    trigger.run(ticks.append)
    assert ticks == [0, 1, 2]
    assert sleeps == [600, 600]  # sleeps between refreshes, not after the last one


def test_watch_max_ticks_zero_never_fires() -> None:
    ticks: list[int] = []
    sleeps: list[float] = []
    WatchTrigger(600, max_ticks=0, sleep=sleeps.append).run(ticks.append)
    assert ticks == []
    assert sleeps == []


def test_watch_survives_error_in_refresh() -> None:
    # A transient failure (timeout/5xx/429) in a refresh must not kill the loop:
    # it gets logged and the loop continues to the next tick.
    ticks: list[int] = []

    def on_refresh(tick: int) -> None:
        ticks.append(tick)
        if tick == 0:
            raise RuntimeError("network blip")

    WatchTrigger(600, max_ticks=3, sleep=lambda _s: None).run(on_refresh)
    assert ticks == [0, 1, 2]  # kept going despite the exception on tick 0


def test_cron_fires_once() -> None:
    ticks: list[int] = []
    CronTrigger().run(ticks.append)
    assert ticks == [0]


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError):
        WatchTrigger(0)
