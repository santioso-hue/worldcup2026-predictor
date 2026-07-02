"""Validate and reconcile match results (pure functions, no I/O).

Two responsibilities:

1. **Validate** each match's status and score (``scheduled``/``in_play``/``finished``)
   and flag suspicious data.
2. **Reconcile** a new snapshot against the previous one: on suspicious data or an
   attempt to change an already-locked result, **keep the last valid snapshot**
   instead of degrading the prediction.

Name canonicalization across providers lives in ``data/team_names.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .live_results import MatchStatus, NormalizedMatch, is_lockable


def validate_match(match: NormalizedMatch) -> list[str]:
    """Return the list of issues found in a match (empty if it's clean).

    Rules:

    - No score can be negative.
    - A ``FINISHED`` match must carry a 90' score (``ft_home``/``ft_away``).
    - A ``SCHEDULED`` match shouldn't carry a score (contradictory data).
    - Extra time must be complete (both sides or neither).
    - No penalties without extra time (penalties only follow extra time).
    """
    issues: list[str] = []
    scores = (
        match.ht_home,
        match.ht_away,
        match.ft_home,
        match.ft_away,
        match.et_home,
        match.et_away,
        match.pen_home,
        match.pen_away,
    )
    if any(s is not None and s < 0 for s in scores):
        issues.append("negative score")
    if match.status is MatchStatus.FINISHED and (
        match.ft_home is None or match.ft_away is None
    ):
        issues.append("finished without a 90' score")
    if match.status is MatchStatus.SCHEDULED and (
        match.ft_home is not None or match.ft_away is not None
    ):
        issues.append("scheduled with a score")
    if (match.et_home is None) != (match.et_away is None):
        issues.append("incomplete extra time (only one side)")
    if match.pen_home is not None and match.et_home is None:
        issues.append("penalties without extra time")
    # A FINISHED knockout match can't end in a draw with no resolution: if it does,
    # treat it as suspicious (locking it would leave a bracket with no winner).
    is_group = match.stage.startswith("Group")
    if (
        match.status is MatchStatus.FINISHED
        and not is_group
        and match.ft_home is not None
        and match.ft_away is not None
        and match.ft_home == match.ft_away
    ):
        et_breaks = (
            match.et_home is not None
            and match.et_away is not None
            and match.et_home != match.et_away
        )
        pen_breaks = (
            match.pen_home is not None
            and match.pen_away is not None
            and match.pen_home != match.pen_away
        )
        if not (et_breaks or pen_breaks):
            issues.append("finished knockout match ended in a draw with no resolution")
    return issues


def is_suspicious(match: NormalizedMatch) -> bool:
    """``True`` if the match has any validation issue."""
    return bool(validate_match(match))


@dataclass(frozen=True)
class ReconcileResult:
    """Result of :func:`reconcile`: chosen matches + logged anomalies."""

    matches: list[NormalizedMatch]
    anomalies: list[tuple[str, str]]  # (match_id, reason)


def _choose(
    previous: NormalizedMatch | None, incoming: NormalizedMatch
) -> tuple[NormalizedMatch | None, str | None]:
    """Pick between the previous and incoming match; returns (chosen, anomaly)."""
    if is_suspicious(incoming):
        reason = "; ".join(validate_match(incoming))
        if previous is not None:
            return previous, f"suspicious incoming data ({reason}); keeping previous"
        return None, f"suspicious incoming data with no previous ({reason}); dropped"

    if previous is not None and is_lockable(previous):
        # Locked result: immutable, including its resolution (extra time/penalties).
        # The incoming match is only accepted if it keeps the same 90' score AND
        # doesn't CHANGE any phase already known from the previous match (it can fill
        # in phases that were None before, e.g. adding penalty detail to the same 1-1).
        same_score = (
            incoming.status is MatchStatus.FINISHED
            and incoming.ft_home == previous.ft_home
            and incoming.ft_away == previous.ft_away
        )
        changes_resolution = (
            (previous.et_home is not None and incoming.et_home != previous.et_home)
            or (previous.et_away is not None and incoming.et_away != previous.et_away)
            or (
                previous.pen_home is not None and incoming.pen_home != previous.pen_home
            )
            or (
                previous.pen_away is not None and incoming.pen_away != previous.pen_away
            )
        )
        if same_score and not changes_resolution:
            return incoming, None
        return (
            previous,
            "attempted to modify an already-locked result; keeping previous",
        )

    return incoming, None


def reconcile(
    previous: list[NormalizedMatch], incoming: list[NormalizedMatch]
) -> ReconcileResult:
    """Merge an incoming snapshot with the previous one, keeping what's valid.

    - A suspicious match in ``incoming`` is dropped in favor of the previous one (or
      skipped if there was no previous).
    - An already-locked result (finished with a score) can't mutate: if the incoming
      match contradicts it, the previous one is kept and the anomaly is logged.
    - A match that was in ``previous`` but is missing from ``incoming`` is kept (a
      partial feed shouldn't erase what's already known).

    Parameters
    ----------
    previous:
        Last valid snapshot (can be empty on first run).
    incoming:
        Freshly downloaded snapshot.

    Returns
    -------
    ReconcileResult
        Reconciled ``matches`` (order: ``incoming`` first, then anything exclusive to
        ``previous``) and the list of ``anomalies``.
    """
    prev_by = {m.match_id: m for m in previous}
    anomalies: list[tuple[str, str]] = []
    result: list[NormalizedMatch] = []
    seen: set[str] = set()

    for inc in incoming:
        seen.add(inc.match_id)
        chosen, anomaly = _choose(prev_by.get(inc.match_id), inc)
        if anomaly is not None:
            anomalies.append((inc.match_id, anomaly))
        if chosen is not None:
            result.append(chosen)

    for prev in previous:
        if prev.match_id not in seen:
            result.append(prev)

    return ReconcileResult(matches=result, anomalies=anomalies)
