"""WC2026 schedule backbone: openfootball parser -> normalized fixtures.

This is the tournament's **backbone**: the 104 matches (72 group + 32 knockout),
the 12 groups of 4, and the bracket structure. Live results (football-data.org) get
attached to these fixtures later via ``match_id``, so a provider swap never breaks
match identity.

**Pure** functions (no I/O): they take already-loaded JSON. Downloading lives in
``download.py``. The per-match schema is VERIFIED against openfootball:

    { "round": "Matchday 1", "date": "YYYY-MM-DD", "time": "13:00 UTC-6",
      "team1": "...", "team2": "...", "group": "Group A",
      "score": {"ft": [h, a], "ht": [h, a]}, "ground": "..." }

*Assumption (wrapper):* the file groups matches under ``"matches"`` or under
``"rounds"[].matches``; we support both and fail loudly if neither shape shows up
(no guessing at structure). openfootball carries no status field: we infer
``FINISHED`` when ``score.ft`` is present and ``SCHEDULED`` otherwise (a static file
has no ``in_play``). Knockout extra-time/penalties come from the live provider, not
from here.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .live_results import MatchStatus, NormalizedMatch
from .team_names import canonical_openfootball_team

# Expected tournament structure (for validation, verified).
EXPECTED_GROUPS = 12
EXPECTED_TEAMS_PER_GROUP = 4
EXPECTED_GROUP_MATCHES = 72
EXPECTED_KNOCKOUT_MATCHES = 32
EXPECTED_TOTAL_MATCHES = 104

# 2026 knockout rounds and their match counts (for validation/labeling).
KNOCKOUT_MATCH_COUNTS: dict[str, int] = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarter-finals": 4,
    "Semi-finals": 2,
    "Third-place play-off": 1,
    "Final": 1,
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TZ_RE = re.compile(r"UTC([+-]\d{1,2})(?::?(\d{2}))?")


def _slug(text: str) -> str:
    """Normalize a name into a stable slug (lowercase, no punctuation)."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def make_match_id(date: str, home: str, away: str) -> str:
    """Build a deterministic ``match_id`` that's stable across providers.

    Shape: ``"{date}-{home}-vs-{away}"`` with slugged names. The real join key
    against the live feed is (date, home, away), which is why the id encodes it.

    Parameters
    ----------
    date:
        Date as ``"YYYY-MM-DD"``. Callers pass the **UTC kickoff date** so the id
        lines up across providers (see ``parse_match`` and ``football_data``).
    home, away:
        Home and away team names.

    Returns
    -------
    str
        Stable identifier, e.g. ``"2026-06-11-mexico-vs-south-africa"``.
    """
    return f"{date}-{_slug(home)}-vs-{_slug(away)}"


def _parse_kickoff(date_str: str, time_str: str | None) -> datetime:
    """Convert openfootball's ``date`` + ``time`` into a UTC ``datetime``.

    ``time`` comes as ``"HH:MM UTC±H"`` (offset embedded).

    - If ``time`` is **missing**, we fall back to UTC midnight on openfootball's
      *local* date. The ``match_id`` then anchors to that local date rather than
      the real UTC kickoff date, so cross-provider joins are only guaranteed for
      matches with a known time (every played/finished match has one). Matches
      without a time are unresolved knockout TBDs that don't join to live results
      yet, so we deliberately do NOT fail loud here — that would break parsing of
      those legitimate entries.
    - If ``time`` **is present but unparseable**, we raise ``ValueError`` (fail
      loud): since ``match_id`` derives from the UTC kickoff date, a misread time
      would silently corrupt the join key. Better to find out immediately.
    """
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if not time_str:
        return base
    try:
        hhmm, _, tz_part = time_str.strip().partition(" ")
        hour, minute = (int(x) for x in hhmm.split(":"))
        match = _TZ_RE.search(tz_part)
        if match is None:
            raise ValueError(f"unrecognized timezone offset in {time_str!r}")
        off_hours = int(match.group(1))
        off_minutes = int(match.group(2) or 0)
        sign = -1 if off_hours < 0 else 1
        offset = sign * timedelta(hours=abs(off_hours), minutes=off_minutes)
        local = base.replace(hour=hour, minute=minute, tzinfo=timezone(offset))
        return local.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(
            f"invalid time {time_str!r} for date {date_str!r}: {exc}"
        ) from exc


def _pair(value: object) -> tuple[int, int] | None:
    """Return ``(int, int)`` if ``value`` is a pair of ints, else ``None``."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        return int(value[0]), int(value[1])
    return None


def parse_match(raw: dict, source: str = "openfootball") -> NormalizedMatch:
    """Convert an openfootball match entry into a :class:`NormalizedMatch`.

    Parameters
    ----------
    raw:
        Dict following the verified openfootball schema.
    source:
        Source label (default ``"openfootball"``).

    Returns
    -------
    NormalizedMatch
        Normalized fixture. ``FINISHED`` if it carries a valid ``score.ft``,
        ``SCHEDULED`` otherwise.
    """
    # Canonicalize to the martj42 namespace (same convention as the live client)
    # BEFORE deriving match_id, so Elo lookups and the host bonus line up.
    home = canonical_openfootball_team(raw["team1"])
    away = canonical_openfootball_team(raw["team2"])
    date = raw["date"]
    kickoff = _parse_kickoff(date, raw.get("time"))

    score = raw.get("score") or {}
    ft = _pair(score.get("ft"))
    ht = _pair(score.get("ht"))
    status = MatchStatus.FINISHED if ft is not None else MatchStatus.SCHEDULED

    group = raw.get("group")
    stage = group if group else raw.get("round", "")
    # match_id from the UTC kickoff date (same anchor as the live client).
    mid = make_match_id(kickoff.date().isoformat(), home, away)

    return NormalizedMatch(
        match_id=mid,
        source=source,
        source_match_id=mid,
        kickoff_utc=kickoff,
        home_team=home,
        away_team=away,
        stage=stage,
        status=status,
        ht_home=ht[0] if ht is not None else None,
        ht_away=ht[1] if ht is not None else None,
        ft_home=ft[0] if ft is not None else None,
        ft_away=ft[1] if ft is not None else None,
        venue=raw.get("ground"),
    )


def parse_fixtures(matches: list[dict]) -> list[NormalizedMatch]:
    """Parse a (already flattened) list of match entries."""
    return [parse_match(m) for m in matches]


def parse_openfootball(doc: dict) -> list[NormalizedMatch]:
    """Flatten the openfootball wrapper and parse all fixtures.

    Supports ``doc["matches"]`` or ``doc["rounds"][].matches``.

    Raises
    ------
    ValueError
        If the document has neither key (unrecognized format).
    """
    if "matches" in doc:
        matches = doc["matches"]
    elif "rounds" in doc:
        matches = [m for r in doc["rounds"] for m in r.get("matches", [])]
    else:
        raise ValueError(
            "Unrecognized openfootball format: missing 'matches' or 'rounds'."
        )
    return parse_fixtures(matches)


def group_teams(matches: list[NormalizedMatch]) -> dict[str, list[str]]:
    """Extract groups and their teams from group-stage fixtures.

    Returns
    -------
    dict[str, list[str]]
        Map of ``"Group A" -> [sorted teams]``, ordered by group.
    """
    groups: dict[str, set[str]] = {}
    for m in matches:
        if m.stage.startswith("Group"):
            members = groups.setdefault(m.stage, set())
            members.add(m.home_team)
            members.add(m.away_team)
    return {g: sorted(teams) for g, teams in sorted(groups.items())}


def validate_schedule(matches: list[NormalizedMatch]) -> list[str]:
    """Check that the parsed schedule matches the WC2026 structure.

    Uses the ``EXPECTED_*`` constants to fail loudly on a partial or malformed
    backbone instead of simulating an incomplete tournament. The pipeline should
    call this right after ``parse_openfootball`` and abort if it returns issues.

    Returns
    -------
    list[str]
        List of discrepancies (empty if the structure checks out).
    """
    issues: list[str] = []
    total = len(matches)
    if total != EXPECTED_TOTAL_MATCHES:
        issues.append(f"total {total} != {EXPECTED_TOTAL_MATCHES} matches")

    groups = group_teams(matches)
    if len(groups) != EXPECTED_GROUPS:
        issues.append(f"{len(groups)} groups != {EXPECTED_GROUPS}")
    for name, teams in groups.items():
        if len(teams) != EXPECTED_TEAMS_PER_GROUP:
            issues.append(f"{name}: {len(teams)} teams != {EXPECTED_TEAMS_PER_GROUP}")

    group_matches = sum(1 for m in matches if m.stage.startswith("Group"))
    if group_matches != EXPECTED_GROUP_MATCHES:
        issues.append(f"{group_matches} group matches != {EXPECTED_GROUP_MATCHES}")
    knockout_matches = total - group_matches
    if knockout_matches != EXPECTED_KNOCKOUT_MATCHES:
        issues.append(f"knockout {knockout_matches} != {EXPECTED_KNOCKOUT_MATCHES}")
    return issues


def is_knockout_stage(stage: str) -> bool:
    """``True`` if the stage is a knockout round (not the group stage)."""
    return not stage.startswith("Group")


def remaining_fixtures(matches: list[NormalizedMatch]) -> list[NormalizedMatch]:
    """Fixtures still to be played (status != FINISHED), sorted by kickoff."""
    pending = [m for m in matches if m.status is not MatchStatus.FINISHED]
    return sorted(pending, key=lambda m: m.kickoff_utc)


def is_predictable(home: str, away: str, known_teams: set[str]) -> bool:
    """``True`` if both teams are actual national teams (not unresolved KO slots)."""
    return home in known_teams and away in known_teams
