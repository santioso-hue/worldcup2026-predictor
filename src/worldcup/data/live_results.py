"""Interfaz provider-agnostic para resultados live + esquema normalizado.

Toda fuente live (football-data.org, openfootball) se accede tras
:class:`LiveResultsProvider`, de modo que el modelo nunca sabe de qué API vienen los
datos: se puede cambiar de proveedor sin tocar simulación ni viz (PROJECT.md §6,
SOURCES.md).

Los mapeos de estado y los campos de marcador están **verificados** contra la doc
oficial de cada proveedor (16 jun 2026). Cada campo de
:class:`NormalizedMatch` corresponde a un campo documentado. Ver tabla en SOURCES.md.

Este módulo define SOLO la interfaz + el esquema + los mapeos (funciones puras). Los
clientes HTTP concretos (con I/O) viven aparte y heredan de
:class:`LiveResultsProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchStatus(str, Enum):
    """Estado normalizado de un partido, colapsado a 4 categorías para el modelo.

    El modelo solo distingue: pendiente (``SCHEDULED``), en juego (``IN_PLAY``),
    hecho consumado (``FINISHED``) o no disputado (``NOT_PLAYED``: aplazado, suspendido,
    cancelado). El reconditioning live bloquea únicamente lo que :func:`is_lockable`
    considere un hecho.
    """

    SCHEDULED = "scheduled"
    IN_PLAY = "in_play"
    FINISHED = "finished"
    NOT_PLAYED = "not_played"


# --- Mapeos raw -> normalizado (VERIFICADOS, ver SOURCES.md) -----------------
# football-data.org v4: campo `match.status`.
FOOTBALLDATA_STATUS: dict[str, MatchStatus] = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "LIVE": MatchStatus.IN_PLAY,
    "IN_PLAY": MatchStatus.IN_PLAY,
    "PAUSED": MatchStatus.IN_PLAY,
    "FINISHED": MatchStatus.FINISHED,
    "POSTPONED": MatchStatus.NOT_PLAYED,
    "SUSPENDED": MatchStatus.NOT_PLAYED,
    "CANCELLED": MatchStatus.NOT_PLAYED,
}

_STATUS_MAPS: dict[str, dict[str, MatchStatus]] = {
    "football_data": FOOTBALLDATA_STATUS,
}


def normalize_status(provider: str, raw_code: str) -> MatchStatus:
    """Traduce el código de estado crudo de un proveedor a :class:`MatchStatus`.

    Parameters
    ----------
    provider:
        Clave del proveedor (``"football_data"``).
    raw_code:
        Código tal cual lo devuelve la API (p. ej. ``"FINISHED"``, ``"IN_PLAY"``).

    Returns
    -------
    MatchStatus
        Estado normalizado.

    Raises
    ------
    KeyError
        Si el proveedor o el código no están mapeados. Preferimos fallar ruidosamente
        a adivinar: un código desconocido podría hacernos bloquear un partido por error.
    """
    table = _STATUS_MAPS[provider]
    return table[raw_code]


@dataclass(frozen=True, slots=True)
class NormalizedMatch:
    """Un partido normalizado, agnóstico al proveedor.

    Los marcadores son ``None`` mientras esa fase no se haya jugado. Se guardan por
    fase (no agregados) para no perder información ni adivinar semánticas: ``ft_*`` es
    el resultado a los 90', ``et_*`` solo si hubo prórroga, ``pen_*`` solo si hubo
    tanda. Esto permite calcular puntos de grupo (90') y resolver eliminatorias sin
    ambigüedad.

    Attributes
    ----------
    match_id:
        Identificador estable nuestro (idealmente derivado del schedule, no del
        proveedor, para que sobreviva un cambio de fuente).
    source / source_match_id:
        Proveedor de origen y su id nativo (trazabilidad).
    kickoff_utc:
        Hora de inicio en UTC.
    home_team / away_team:
        Nombres de selección (se canonicalizan en ``clean.py``).
    stage:
        Fase/grupo, p. ej. ``"Group A"`` o ``"Round of 16"``.
    status:
        Estado normalizado.
    ht_home/ht_away, ft_home/ft_away, et_home/et_away, pen_home/pen_away:
        Marcadores por fase (``None`` si no aplica/no disputada).
    venue:
        Sede (opcional).
    fetched_at:
        Cuándo se obtuvo este dato (para el sello de "última actualización").
    """

    match_id: str
    source: str
    source_match_id: str
    kickoff_utc: datetime
    home_team: str
    away_team: str
    stage: str
    status: MatchStatus
    ht_home: int | None = None
    ht_away: int | None = None
    ft_home: int | None = None
    ft_away: int | None = None
    et_home: int | None = None
    et_away: int | None = None
    pen_home: int | None = None
    pen_away: int | None = None
    venue: str | None = None
    fetched_at: datetime | None = None

    @property
    def is_finished(self) -> bool:
        """``True`` si el estado normalizado es ``FINISHED``."""
        return self.status is MatchStatus.FINISHED

    @property
    def decided_by(self) -> str:
        """Cómo se decidió: ``"PENALTIES"``, ``"EXTRA_TIME"`` o ``"REGULAR_TIME"``."""
        if self.pen_home is not None:
            return "PENALTIES"
        if self.et_home is not None:
            return "EXTRA_TIME"
        return "REGULAR_TIME"


def is_lockable(match: NormalizedMatch) -> bool:
    """Decide si un partido es un HECHO que el reconditioning live debe bloquear.

    Bloquear (lock) significa: su marcador deja de muestrearse en Monte Carlo y se
    trata como dato cierto (PROJECT.md §5.1).

    Política: se bloquea un partido ``FINISHED`` que traiga marcador de los 90'
    (``ft_home`` y ``ft_away`` no ``None``).

    - ``SCHEDULED``, ``IN_PLAY`` y ``NOT_PLAYED`` nunca se bloquean.
    - Un ``FINISHED`` sin marcador (dato parcial/sospechoso) NO se bloquea: se conserva
      el último snapshot válido (PROJECT.md §5.1 / §6).
    - Los resultados técnicos (``AWD``/``WO``) no reciben trato especial: en un Mundial
      moderno no ha ocurrido nunca un walkover/resultado adjudicado, así que no vale la
      complejidad. Si llegara uno sin marcador de 90', simplemente no se bloquearía.

    Parameters
    ----------
    match:
        Partido normalizado candidato a bloqueo.

    Returns
    -------
    bool
        ``True`` si debe bloquearse como hecho consumado; ``False`` si debe seguir
        simulándose (o ignorarse) por estar pendiente, en juego o ser dudoso.
    """
    if match.status is not MatchStatus.FINISHED:
        return False
    return match.ft_home is not None and match.ft_away is not None


class LiveResultsProvider(ABC):
    """Interfaz estable para cualquier fuente de resultados (live o fallback).

    Cualquier cliente (football-data.org, openfootball) la implementa devolviendo
    siempre :class:`NormalizedMatch`; el resto del proyecto depende solo de esta
    interfaz.

    Política de uso (SOURCES.md): polling **solo por ventanas** con partidos en juego,
    cada 10–15 min; cachear lo que cambia lento. Nunca polling continuo.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Clave del proveedor (p. ej. ``"football_data"``); usada en ``source``."""

    @abstractmethod
    def get_schedule(self) -> list[NormalizedMatch]:
        """Devuelve los 104 fixtures del WC2026 (jugados y por jugar)."""

    @abstractmethod
    def get_live_fixtures(self) -> list[NormalizedMatch]:
        """Devuelve los partidos actualmente en juego (status IN_PLAY/PAUSED)."""

    @abstractmethod
    def get_finished_results(
        self, since: datetime | None = None
    ) -> list[NormalizedMatch]:
        """Devuelve resultados FINALIZADOS.

        Parameters
        ----------
        since:
            Si se indica, excluye partidos cuyo **kickoff** (``kickoff_utc``) sea
            anterior a ``since`` (UTC). El filtro es sobre la hora de inicio, no la de
            finalización: un partido que arrancó antes de ``since`` pero terminó después
            queda excluido.
        """
