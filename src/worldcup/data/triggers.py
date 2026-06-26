"""``RefreshTrigger``: qué dispara una re-ejecución del pipeline live.

El disparador vive tras una interfaz para poder evolucionar sin tocar el pipeline
(SOURCES.md, metodología §5.1):

- :class:`WatchTrigger` — un loop que sondea cada N segundos
  mientras hay partidos. Simple, sin infraestructura.
- :class:`CronTrigger` — modela una invocación única por tick de un
  scheduler externo (cron / systemd timer / GitHub Actions schedule): más robusto ante
  caídas que un proceso vivo. Corre el pipeline una vez por tick.
- **Webhook / push** (no implementado aquí) — el proveedor notifica
  el full-time al instante; un ``WebhookTrigger`` bloquearía esperando eventos entrantes
  y llamaría ``on_refresh`` por cada uno (requiere un servidor).

El callback ``on_refresh(tick)`` lo provee el pipeline: baja un snapshot nuevo y
re-simula lo pendiente. El trigger no sabe nada del modelo.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

OnRefresh = Callable[[int], None]


class RefreshTrigger(ABC):
    """Interfaz de disparo. ``run`` invoca ``on_refresh`` según su política temporal."""

    @abstractmethod
    def run(self, on_refresh: OnRefresh) -> None:
        """Ejecuta la política de disparo, llamando ``on_refresh(tick)`` por refresh."""


class WatchTrigger(RefreshTrigger):
    """Sondea en un loop cada ``interval_seconds``.

    Parameters
    ----------
    interval_seconds:
        Segundos entre refreshes (típico 600–900: polling por ventanas, SOURCES.md).
    max_ticks:
        Si se indica, para tras ese n.º de refreshes (útil en tests y corridas
        acotadas). ``None`` = loop indefinido.
    sleep:
        Función de espera inyectable (por defecto ``time.sleep``); en tests se pasa un
        doble para no dormir de verdad.
    """

    def __init__(
        self,
        interval_seconds: float,
        *,
        max_ticks: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser > 0")
        if max_ticks is not None and max_ticks < 0:
            raise ValueError("max_ticks debe ser >= 0 o None")
        self._interval_seconds = interval_seconds
        self._max_ticks = max_ticks
        self._sleep = sleep

    def run(self, on_refresh: OnRefresh) -> None:
        tick = 0
        while True:
            if self._max_ticks is not None and tick >= self._max_ticks:
                return
            # Daemon de sondeo: un blip de red (timeout, 5xx, 429 del rate-limit) NO
            # debe matar el loop. Lo registramos y seguimos al próximo tick; un error
            # persistente se ve repetido. (CronTrigger sí falla fuerte: hay scheduler.)
            try:
                on_refresh(tick)
            except Exception as exc:  # noqa: BLE001 — sobrevivir a un fallo transitorio
                print(
                    f"refresh tick {tick} falló: {exc!r}; reintento al próximo tick",
                    file=sys.stderr,
                )
            tick += 1
            if self._max_ticks is not None and tick >= self._max_ticks:
                return
            self._sleep(self._interval_seconds)


class CronTrigger(RefreshTrigger):
    """Dispara **una sola vez** por invocación (modelo cron / systemd / GH Actions).

    Bajo un scheduler externo, el pipeline se ejecuta de cero cada tick; este trigger
    representa esa única invocación llamando ``on_refresh(0)`` y retornando. Más robusto
    que :class:`WatchTrigger` porque no depende de un proceso vivo.
    """

    def run(self, on_refresh: OnRefresh) -> None:
        on_refresh(0)
