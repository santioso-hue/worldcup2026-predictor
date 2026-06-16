"""Tests de RefreshTrigger: cadencia del watch y disparo único del cron."""

from __future__ import annotations

import pytest

from worldcup.data.triggers import CronTrigger, WatchTrigger


def test_watch_fires_max_ticks_and_sleeps_between() -> None:
    ticks: list[int] = []
    sleeps: list[float] = []
    trigger = WatchTrigger(600, max_ticks=3, sleep=sleeps.append)
    trigger.run(ticks.append)
    assert ticks == [0, 1, 2]
    assert sleeps == [600, 600]  # duerme entre refreshes, no tras el último


def test_watch_max_ticks_zero_never_fires() -> None:
    ticks: list[int] = []
    sleeps: list[float] = []
    WatchTrigger(600, max_ticks=0, sleep=sleeps.append).run(ticks.append)
    assert ticks == []
    assert sleeps == []


def test_cron_fires_once() -> None:
    ticks: list[int] = []
    CronTrigger().run(ticks.append)
    assert ticks == [0]


def test_invalid_interval_raises() -> None:
    with pytest.raises(ValueError):
        WatchTrigger(0)
