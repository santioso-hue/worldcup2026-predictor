"""Pipeline orchestration: pure core (no network/fs) + thin I/O helpers.

``run_pipeline`` chains ``reconcile`` -> ``fit_elo`` -> ``build_state`` -> Monte Carlo
and returns the probabilities + the reconciled matches (for snapshotting). The core
never touches the network or filesystem, so it's tested with fixtures; I/O (fetch,
snapshot, figures) lives in ``scripts/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .config import Config
from .data.clean import reconcile
from .data.historical import HistoricalMatch
from .data.live_results import NormalizedMatch
from .data.schedule import group_teams, remaining_fixtures
from .features.elo import fit_elo
from .models.base import CachedMatchModel, MatchOutcome, outcome_probabilities
from .models.dixon_coles import DixonColesModel
from .rng import DEFAULT_SEED, get_rng
from .simulation.match import simulate_match
from .simulation.state import build_state, run_from_state
from .simulation.tournament import ResolvedTie, resolve_bracket


@dataclass(frozen=True)
class PipelineResult:
    """Pipeline output: probabilities, ratings, groups, and reconcile anomalies."""

    probabilities: dict[str, dict[str, float]]
    ratings: dict[str, float]
    anomalies: list[tuple[str, str]]
    groups: dict[str, list[str]]
    bracket: dict[int, ResolvedTie]


def run_pipeline(
    incoming: list[NormalizedMatch],
    previous: list[NormalizedMatch],
    history: list[HistoricalMatch],
    config: Config,
    annex_c: dict[frozenset[str], dict[str, str]],
    *,
    runs: int,
    seed: int,
    reference_date: date | None = None,
    history_cutoff: date | None = None,
) -> tuple[PipelineResult, list[NormalizedMatch]]:
    """``reconcile`` -> ``fit_elo`` -> ``build_state`` -> Monte Carlo (no I/O).

    Every team in the backbone gets a rating: the one from ``fit_elo``, or
    ``initial_rating`` if it has no history (``build_state`` requires a rating for
    everyone, fail-loud). ``history_cutoff`` (pre-tournament baseline) drops matches on
    or after that date, so the Elo doesn't get contaminated by in-tournament results.
    Returns the result plus the reconciled matches (for snapshotting).
    """
    rec = reconcile(previous, incoming)
    if history_cutoff is not None:
        history = [m for m in history if m.date < history_cutoff]
    fitted = fit_elo(history, config.elo, reference_date=reference_date)
    # Team universe = group-stage participants (the real 48); the backbone's KO slots
    # carry placeholder labels ("2A", "1E"), not actual teams.
    # `sorted` -> deterministic team order (independent of PYTHONHASHSEED), so the JSON
    # artifact is byte-reproducible across runs/processes.
    teams = sorted(
        {team for members in group_teams(rec.matches).values() for team in members}
    )
    ratings = {t: fitted.get(t, config.elo.initial_rating) for t in teams}
    # Host bonus: host nations get elo.host_advantage in their matches.
    hosts = set(config.simulation.hosts)
    host_advantage = {t: config.elo.host_advantage for t in teams if t in hosts}
    state = build_state(rec.matches, ratings)
    bracket = resolve_bracket(state, annex_c, rec.matches)
    model = DixonColesModel(config.elo, config.dixon_coles)
    probabilities = run_from_state(
        state,
        model,
        annex_c,
        runs=runs,
        seed=seed,
        extra_time_total_goals=config.simulation.extra_time_total_goals,
        elo_denominator=config.elo.elo_per_goal_denominator,
        host_advantage=host_advantage,
    )
    result = PipelineResult(
        probabilities=probabilities,
        ratings=ratings,
        anomalies=rec.anomalies,
        groups=state.groups,
        bracket=bracket,
    )
    return result, rec.matches


def outcome_from_ratings(
    home: str,
    away: str,
    ratings: dict[str, float],
    config: Config,
    *,
    host: str | None = None,
) -> tuple[MatchOutcome, np.ndarray]:
    """1X2 + Dixon-Coles matrix from ratings that are ALREADY fit (no re-fitting Elo).

    Lets you predict many matches while reusing a single ``fit_elo`` call: the
    dashboard fits Elo once and calls this per fixture. ``host`` applies the World Cup
    host advantage to whichever side (``home`` or ``away``) is hosting; on neutral
    ground there's no advantage. Teams without a rating fall back to
    ``initial_rating``.
    """
    rating_home = ratings.get(home, config.elo.initial_rating)
    rating_away = ratings.get(away, config.elo.initial_rating)
    if host == home:
        advantage = config.elo.host_advantage
    elif host == away:
        advantage = -config.elo.host_advantage
    else:
        advantage = 0.0
    model = DixonColesModel(config.elo, config.dixon_coles)
    matrix = model.score_matrix(rating_home, rating_away, advantage)
    return outcome_probabilities(matrix), matrix


def predict_fixture(
    home: str,
    away: str,
    history: list[HistoricalMatch],
    config: Config,
    *,
    host: str | None = None,
    reference_date: date | None = None,
) -> tuple[MatchOutcome, np.ndarray]:
    """1X2 + Dixon-Coles scoreline matrix for one match (pure, no I/O).

    Fits Elo from history and delegates to :func:`outcome_from_ratings`. ``host``
    applies the World Cup host advantage; without history it falls back to
    ``initial_rating``.
    """
    fitted = fit_elo(history, config.elo, reference_date=reference_date)
    return outcome_from_ratings(home, away, fitted, config, host=host)


def predict_match(
    home: str,
    away: str,
    history: list[HistoricalMatch],
    config: Config,
    *,
    host: str | None = None,
    reference_date: date | None = None,
) -> MatchOutcome:
    """1X2 for a single match from the historical Elo ratings.

    Thin wrapper over :func:`predict_fixture` (returns only the 1X2). ``host`` applies
    the World Cup host advantage; without history it falls back to ``initial_rating``.
    """
    outcome, _ = predict_fixture(
        home, away, history, config, host=host, reference_date=reference_date
    )
    return outcome


def knockout_advance_probability(
    home: str,
    away: str,
    ratings: dict[str, float],
    config: Config,
    *,
    host: str | None = None,
    runs: int = 2000,
    seed: int = DEFAULT_SEED,
) -> float:
    """P(``home`` advances) in a knockout tie, with real extra time and penalties.

    Samples the match ``runs`` times with :func:`simulate_match` (``knockout=True``),
    which resolves draws via extra time (Poisson) and penalties (Elo-weighted coin
    flip), and returns the fraction of times ``home`` advances. Deterministic given
    ``seed``. By construction ``advance >= P(win in 90)``: draws can also break
    ``home``'s way.
    """
    if runs <= 0:
        raise ValueError("runs must be > 0")
    rating_home = ratings.get(home, config.elo.initial_rating)
    rating_away = ratings.get(away, config.elo.initial_rating)
    if host == home:
        advantage = config.elo.host_advantage
    elif host == away:
        advantage = -config.elo.host_advantage
    else:
        advantage = 0.0
    # All `runs` samples share one matchup; cache the matrix across them.
    model = CachedMatchModel(DixonColesModel(config.elo, config.dixon_coles))
    rng = get_rng(seed)
    wins = sum(
        simulate_match(
            model,
            home,
            away,
            rating_home,
            rating_away,
            rng,
            home_advantage=advantage,
            knockout=True,
            extra_time_total_goals=config.simulation.extra_time_total_goals,
            elo_denominator=config.elo.elo_per_goal_denominator,
        ).winner
        == home
        for _ in range(runs)
    )
    return wins / runs


# --- thin I/O helpers (testable with tmp_path, no network) ---


def _fixture_row(match: NormalizedMatch) -> dict[str, str]:
    """Serializable fixture row for the dashboard artifact."""
    return {
        "home": match.home_team,
        "away": match.away_team,
        "stage": match.stage,
        "kickoff": match.kickoff_utc.isoformat(),
    }


def write_probabilities(
    probabilities: dict[str, dict[str, float]],
    outdir: Path | str,
    ts: str,
    *,
    groups: dict[str, list[str]] | None = None,
    fixtures: list[NormalizedMatch] | None = None,
    bracket: dict[int, ResolvedTie] | None = None,
    latest_pointer: str = "latest.json",
    update_pointer: bool = True,
) -> Path:
    """Write ``probabilities_<ts>.json`` plus ``latest`` if ``update_pointer`` is set.

    The payload includes ``groups`` (dashboard tables), ``fixtures`` (matches still to
    be played, for the predictor panel), and ``bracket`` (resolved knockout ties keyed
    by FIFA match number, for the bracket panel). Replays/baselines write their JSON
    but do NOT move the live pointer.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probabilities_{ts}.json"
    pending = remaining_fixtures(fixtures or [])
    payload = {
        "timestamp": ts,
        "groups": groups or {},
        "fixtures": [_fixture_row(m) for m in pending],
        "bracket": {str(n): asdict(tie) for n, tie in (bracket or {}).items()},
        "probabilities": probabilities,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    if update_pointer:
        (out / latest_pointer).write_text(json.dumps({"timestamp": ts}))
    return path


def load_latest_probabilities(
    outdir: Path | str, *, latest_pointer: str = "latest.json"
) -> dict[str, dict[str, float]] | None:
    """Load probabilities from the latest run (for up/down deltas); ``None`` if none."""
    out = Path(outdir)
    pointer = out / latest_pointer
    if not pointer.exists():
        return None
    ts = json.loads(pointer.read_text())["timestamp"]
    path = out / f"probabilities_{ts}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    probabilities: dict[str, dict[str, float]] = payload["probabilities"]
    return probabilities


@dataclass(frozen=True)
class RunArtifact:
    """A persisted run: timestamp, groups, remaining fixtures, and probabilities."""

    timestamp: str
    groups: dict[str, list[str]]
    probabilities: dict[str, dict[str, float]]
    fixtures: list[dict[str, str]] = field(default_factory=list)
    bracket: dict[str, dict] = field(default_factory=dict)


def load_latest_run(
    outdir: Path | str, *, latest_pointer: str = "latest.json"
) -> RunArtifact | None:
    """Load the latest run (timestamp + groups + probabilities); ``None`` if none."""
    out = Path(outdir)
    pointer = out / latest_pointer
    if not pointer.exists():
        return None
    ts = json.loads(pointer.read_text())["timestamp"]
    path = out / f"probabilities_{ts}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return RunArtifact(
        timestamp=payload["timestamp"],
        groups=payload.get("groups", {}),
        probabilities=payload["probabilities"],
        fixtures=payload.get("fixtures", []),
        bracket=payload.get("bracket", {}),
    )


def render_outputs(
    result: PipelineResult,
    previous: dict[str, dict[str, float]] | None,
    outdir: Path | str,
    *,
    ts: str,
) -> list[Path]:
    """Render the champion ranking (with deltas vs. the previous run) and save it."""
    from .viz.charts import prepare_champion_ranking, render_champion_ranking
    from .viz.export import save_figure
    from .viz.theme import PORTRAIT

    champion = {team: probs["champion"] for team, probs in result.probabilities.items()}
    previous_champion = (
        {team: probs["champion"] for team, probs in previous.items()}
        if previous is not None
        else None
    )
    rows = prepare_champion_ranking(champion, previous_champion)
    figure = render_champion_ranking(rows, stamp=f"Updated {ts}")
    return [save_figure(figure, f"champion_{ts}", PORTRAIT, outdir=outdir)]
