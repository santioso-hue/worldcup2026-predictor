"""``RefreshTrigger``: what fires a re-run of the live pipeline.

The trigger sits behind an interface so it can evolve without touching the pipeline:

- :class:`WatchTrigger` — a loop that polls every N seconds while
  matches are in progress. Simple, no infrastructure.
- :class:`CronTrigger` — models a single invocation per tick of an
  external scheduler (cron / systemd timer / GitHub Actions schedule): more resilient
  to crashes than a long-running process. Runs the pipeline once per tick.
- **Webhook / push** (not implemented here) — the provider notifies the
  final score instantly; a ``WebhookTrigger`` would block waiting for incoming
  events and call ``on_refresh`` for each one (needs a server).

The ``on_refresh(tick)`` callback is provided by the pipeline: it pulls a fresh
snapshot and re-simulates what's pending. The trigger knows nothing about the model.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

OnRefresh = Callable[[int], None]


class RefreshTrigger(ABC):
    """Trigger interface. ``run`` calls ``on_refresh`` per its own timing policy."""

    @abstractmethod
    def run(self, on_refresh: OnRefresh) -> None:
        """Run the trigger policy, calling ``on_refresh(tick)`` on each refresh."""


class WatchTrigger(RefreshTrigger):
    """Polls in a loop every ``interval_seconds``.

    Parameters
    ----------
    interval_seconds:
        Seconds between refreshes (typically 600-900: polling within a match window).
    max_ticks:
        If set, stop after that many refreshes (useful for tests and bounded runs).
        ``None`` means loop indefinitely.
    sleep:
        Injectable sleep function (defaults to ``time.sleep``); tests pass a stub so
        nothing actually sleeps.
    """

    def __init__(
        self,
        interval_seconds: float,
        *,
        max_ticks: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if max_ticks is not None and max_ticks < 0:
            raise ValueError("max_ticks must be >= 0 or None")
        self._interval_seconds = interval_seconds
        self._max_ticks = max_ticks
        self._sleep = sleep

    def run(self, on_refresh: OnRefresh) -> None:
        tick = 0
        while True:
            if self._max_ticks is not None and tick >= self._max_ticks:
                return
            # Polling daemon: a network blip (timeout, 5xx, rate-limit 429) must not
            # kill the loop. Log it and move to the next tick; a persistent failure
            # will just show up repeatedly. (CronTrigger fails hard instead, since a
            # scheduler is watching it.)
            try:
                on_refresh(tick)
            except Exception as exc:  # noqa: BLE001 — survive a transient failure
                print(
                    f"refresh tick {tick} failed: {exc!r}; retrying next tick",
                    file=sys.stderr,
                )
            tick += 1
            if self._max_ticks is not None and tick >= self._max_ticks:
                return
            self._sleep(self._interval_seconds)


class CronTrigger(RefreshTrigger):
    """Fires **exactly once** per invocation (cron / systemd / GH Actions model).

    Under an external scheduler, the pipeline runs fresh each tick; this trigger
    represents that single invocation by calling ``on_refresh(0)`` and returning.
    More resilient than :class:`WatchTrigger` since it doesn't depend on a
    long-running process.
    """

    def run(self, on_refresh: OnRefresh) -> None:
        on_refresh(0)
