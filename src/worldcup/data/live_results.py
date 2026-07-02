"""Provider-agnostic interface for live results + normalized schema.

Every live source (football-data.org, openfootball) is accessed through
:class:`LiveResultsProvider`, so the model never knows which API the data came
from: we can swap providers without touching simulation or viz.

Status mappings and score fields are **verified** against each provider's
official docs (2026-06-16). Every field on :class:`NormalizedMatch` maps to a
documented field.

This module defines ONLY the interface + schema + mappings (pure functions).
The concrete HTTP clients (with I/O) live elsewhere and inherit from
:class:`LiveResultsProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchStatus(str, Enum):
    """Normalized match status, collapsed to 4 categories for the model.

    The model only distinguishes: not yet started (``SCHEDULED``), in progress
    (``IN_PLAY``), done (``FINISHED``), or not played (``NOT_PLAYED``:
    postponed, suspended, cancelled). Live reconditioning only locks what
    :func:`is_lockable` considers a settled fact.
    """

    SCHEDULED = "scheduled"
    IN_PLAY = "in_play"
    FINISHED = "finished"
    NOT_PLAYED = "not_played"


# --- Raw -> normalized mappings (VERIFIED) -----------------
# football-data.org v4: `match.status` field.
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
    """Translate a provider's raw status code into :class:`MatchStatus`.

    Parameters
    ----------
    provider:
        Provider key (``"football_data"``).
    raw_code:
        Code as returned by the API (e.g. ``"FINISHED"``, ``"IN_PLAY"``).

    Returns
    -------
    MatchStatus
        Normalized status.

    Raises
    ------
    KeyError
        If the provider or code isn't mapped. We'd rather fail loudly than
        guess: an unknown code could make us lock a match by mistake.
    """
    table = _STATUS_MAPS[provider]
    return table[raw_code]


@dataclass(frozen=True, slots=True)
class NormalizedMatch:
    """A normalized match, agnostic to the provider.

    Scores are ``None`` until that phase has been played. They're stored per
    phase (not aggregated) to avoid losing information or guessing semantics:
    ``ft_*`` is the 90' result, ``et_*`` only if there was extra time,
    ``pen_*`` only if it went to a shootout. This lets us compute group points
    (90') and resolve knockouts without ambiguity.

    Attributes
    ----------
    match_id:
        Our own stable identifier (ideally derived from the schedule, not the
        provider, so it survives a source change).
    source / source_match_id:
        Origin provider and its native id (traceability).
    kickoff_utc:
        Kickoff time in UTC.
    home_team / away_team:
        Team names (canonicalized in ``clean.py``).
    stage:
        Stage/group, e.g. ``"Group A"`` or ``"Round of 16"``.
    status:
        Normalized status.
    ht_home/ht_away, ft_home/ft_away, et_home/et_away, pen_home/pen_away:
        Scores per phase (``None`` if not applicable/not played).
    venue:
        Venue (optional).
    fetched_at:
        When this data was fetched (for the "last updated" stamp).
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
        """``True`` if the normalized status is ``FINISHED``."""
        return self.status is MatchStatus.FINISHED

    @property
    def decided_by(self) -> str:
        """How it was decided: ``"PENALTIES"``, ``"EXTRA_TIME"``, ``"REGULAR_TIME"``."""
        if self.pen_home is not None:
            return "PENALTIES"
        if self.et_home is not None:
            return "EXTRA_TIME"
        return "REGULAR_TIME"


def is_lockable(match: NormalizedMatch) -> bool:
    """Decide if a match is a settled FACT that live reconditioning should lock.

    Locking means: its score stops being sampled in Monte Carlo and is
    treated as certain.

    Policy: lock a ``FINISHED`` match that has a 90' score (``ft_home`` and
    ``ft_away`` both not ``None``).

    - ``SCHEDULED``, ``IN_PLAY`` and ``NOT_PLAYED`` are never locked.
    - A ``FINISHED`` match with no score (partial/suspect data) is NOT
      locked: we keep the last valid snapshot instead.
    - Walkovers/awarded results (``AWD``/``WO``) get no special handling: a
      modern World Cup has never had one, so it's not worth the complexity.
      If one showed up without a 90' score, it just wouldn't get locked.

    Parameters
    ----------
    match:
        Normalized match candidate for locking.

    Returns
    -------
    bool
        ``True`` if it should be locked as a settled fact; ``False`` if it
        should keep being simulated (or ignored) because it's pending, in
        play, or doubtful.
    """
    if match.status is not MatchStatus.FINISHED:
        return False
    return match.ft_home is not None and match.ft_away is not None


class LiveResultsProvider(ABC):
    """Stable interface for any results source (live or fallback).

    Any client (football-data.org, openfootball) implements it and always
    returns :class:`NormalizedMatch`; the rest of the project depends only on
    this interface.

    Usage policy: poll **only in windows** with matches in play, every
    10-15 min; cache anything that changes slowly. Never poll continuously.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider key (e.g. ``"football_data"``); used in ``source``."""

    @abstractmethod
    def get_schedule(self) -> list[NormalizedMatch]:
        """Return all 104 WC2026 fixtures (played and upcoming)."""

    @abstractmethod
    def get_live_fixtures(self) -> list[NormalizedMatch]:
        """Return matches currently in play (status IN_PLAY/PAUSED)."""

    @abstractmethod
    def get_finished_results(
        self, since: datetime | None = None
    ) -> list[NormalizedMatch]:
        """Return FINISHED results.

        Parameters
        ----------
        since:
            If given, excludes matches whose **kickoff** (``kickoff_utc``) is
            before ``since`` (UTC). The filter is on kickoff time, not finish
            time: a match that started before ``since`` but finished after it
            is excluded.
        """
