"""Tests del pipeline: núcleo puro (run_pipeline, predict_match) + helpers de I/O."""

from __future__ import annotations

from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path

import pytest

from worldcup.config import load_config
from worldcup.data.historical import HistoricalMatch
from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.features.elo import fit_elo
from worldcup.pipeline import (
    knockout_advance_probability,
    load_latest_probabilities,
    load_latest_run,
    outcome_from_ratings,
    predict_fixture,
    predict_match,
    render_outputs,
    run_pipeline,
    write_probabilities,
)
from worldcup.simulation.bracket import load_annex_c

UTC = timezone.utc
_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(_ROOT / "config" / "config.yaml")
ANNEX = load_annex_c(_ROOT / "data" / "raw" / "annex_c_2026.json")

_HISTORY = [
    HistoricalMatch(
        date=date(2025, 6, 1),
        home_team="A1",
        away_team="A2",
        home_score=2,
        away_score=0,
        tournament="Friendly",
        neutral=True,
    ),
]


def _nm(
    home: str, away: str, stage: str, status: MatchStatus, **kw: object
) -> NormalizedMatch:
    base: dict[str, object] = dict(
        match_id=f"{stage}:{home}-{away}",
        source="openfootball",
        source_match_id="x",
        kickoff_utc=datetime(2026, 6, 11, tzinfo=UTC),
        home_team=home,
        away_team=away,
        stage=stage,
        status=status,
    )
    base.update(kw)
    return NormalizedMatch(**base)  # type: ignore[arg-type]


def _backbone() -> list[NormalizedMatch]:
    fx: list[NormalizedMatch] = []
    for group in "ABCDEFGHIJKL":
        teams = [f"{group}{i}" for i in range(1, 5)]
        for home, away in combinations(teams, 2):
            fx.append(_nm(home, away, f"Group {group}", MatchStatus.SCHEDULED))
    return fx


def test_run_pipeline_sums_to_one_and_is_deterministic() -> None:
    fx = _backbone()
    r1, _ = run_pipeline(fx, [], _HISTORY, CFG, ANNEX, runs=100, seed=7)
    r2, _ = run_pipeline(fx, [], _HISTORY, CFG, ANNEX, runs=100, seed=7)
    champion = sum(p["champion"] for p in r1.probabilities.values())
    assert abs(champion - 1.0) < 1e-9
    assert r1.probabilities == r2.probabilities  # determinista por semilla


def test_run_pipeline_reconcile_drops_suspicious_before_build_state() -> None:
    # KO finalizado en empate sin resolución es sospechoso: reconcile lo descarta, así
    # build_state nunca intenta bloquearlo (lo que dispararía _winner_of). Si NO se
    # descartara, run_pipeline lanzaría. Fija el hilo reconcile -> build_state.
    suspicious = _nm(
        "A1", "B1", "Round of 16", MatchStatus.FINISHED, ft_home=1, ft_away=1
    )
    result, reconciled = run_pipeline(
        _backbone() + [suspicious], [], _HISTORY, CFG, ANNEX, runs=50, seed=1
    )
    assert result.anomalies  # registró la anomalía
    assert suspicious not in reconciled  # descartado antes de build_state


def test_run_pipeline_excludes_knockout_slot_labels() -> None:
    # Las llaves KO del backbone traen etiquetas de slot ("2A", "1B"), no selecciones;
    # no deben colarse como equipos en la salida (solo los 48 de fase de grupos).
    ko_slots = [
        _nm("2A", "1B", "Round of 16", MatchStatus.SCHEDULED),
        _nm("1C", "2D", "Round of 16", MatchStatus.SCHEDULED),
    ]
    result, _ = run_pipeline(
        _backbone() + ko_slots, [], _HISTORY, CFG, ANNEX, runs=20, seed=1
    )
    assert len(result.probabilities) == 48
    assert not any(slot in result.probabilities for slot in ("2A", "1B", "1C", "2D"))


def test_predict_match_outcome_sums_to_one() -> None:
    out = predict_match("A1", "A2", _HISTORY, CFG, host="A1")
    assert abs(out.home_win + out.draw + out.away_win - 1.0) < 1e-9


def test_predict_fixture_returns_normalized_outcome_and_matrix() -> None:
    outcome, matrix = predict_fixture("A1", "A2", _HISTORY, CFG)
    assert abs(outcome.home_win + outcome.draw + outcome.away_win - 1.0) < 1e-9
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert abs(float(matrix.sum()) - 1.0) < 1e-6


def test_predict_fixture_outcome_matches_predict_match() -> None:
    outcome, _ = predict_fixture("A1", "A2", _HISTORY, CFG, host="A1")
    assert outcome == predict_match("A1", "A2", _HISTORY, CFG, host="A1")


def test_predict_fixture_host_advantage_helps_the_host() -> None:
    neutral, _ = predict_fixture("A1", "A2", _HISTORY, CFG)
    hosted, _ = predict_fixture("A1", "A2", _HISTORY, CFG, host="A1")
    assert hosted.home_win > neutral.home_win


def test_write_and_load_probabilities_roundtrip(tmp_path: Path) -> None:
    probs = {
        "A1": {"champion": 0.2, "advance": 0.8},
        "A2": {"champion": 0.1, "advance": 0.5},
    }
    write_probabilities(probs, tmp_path, "20260620t1200")
    assert load_latest_probabilities(tmp_path) == probs
    assert load_latest_probabilities(tmp_path / "nope") is None  # sin puntero -> None


def test_write_probabilities_is_byte_deterministic(tmp_path: Path) -> None:
    # El artefacto JSON debe ser byte-estable e independiente del orden de inserción
    # (PYTHONHASHSEED): el replay y el dashboard lo consumen. sort_keys lo garantiza.
    probs_a = {"Zambia": {"champion": 0.1}, "Brazil": {"champion": 0.2}}
    probs_b = {"Brazil": {"champion": 0.2}, "Zambia": {"champion": 0.1}}  # inverso
    text_a = write_probabilities(
        probs_a, tmp_path / "a", "20260101t0000", update_pointer=False
    ).read_text()
    text_b = write_probabilities(
        probs_b, tmp_path / "b", "20260101t0000", update_pointer=False
    ).read_text()
    assert text_a == text_b  # mismos bytes pese a distinto orden de inserción
    assert text_a.index('"Brazil"') < text_a.index('"Zambia"')  # claves ordenadas


def test_write_probabilities_can_skip_pointer(tmp_path: Path) -> None:
    probs = {"A1": {"champion": 0.5}}
    # update_pointer=False (replay/baseline): escribe el JSON pero no mueve latest.json.
    write_probabilities(probs, tmp_path, "20260101t0000", update_pointer=False)
    assert load_latest_probabilities(tmp_path) is None
    write_probabilities(
        probs, tmp_path, "20260101t0000"
    )  # el default sí mueve el puntero
    assert load_latest_probabilities(tmp_path) == probs


def test_render_outputs_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    result, _ = run_pipeline(_backbone(), [], _HISTORY, CFG, ANNEX, runs=50, seed=3)
    paths = render_outputs(result, None, tmp_path, ts="20260620t1200")
    assert paths and all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_load_latest_run_roundtrip(tmp_path: Path) -> None:
    groups = {"A": ["A1", "A2", "A3", "A4"]}
    probs = {"A1": {"champion": 0.2, "advance": 0.8}}
    write_probabilities(probs, tmp_path, "20260620t1200", groups=groups)
    run = load_latest_run(tmp_path)
    assert run is not None
    assert run.timestamp == "20260620t1200"
    assert run.groups == groups
    assert run.probabilities == probs
    assert load_latest_run(tmp_path / "nope") is None  # sin puntero -> None


def test_run_pipeline_output_feeds_dashboard_prep(tmp_path: Path) -> None:
    # Contrato del dashboard: la salida real (groups+probabilities) fluye por
    # write -> load_latest_run -> prepare_group_table / prepare_team_detail sin romper.
    from worldcup.viz.charts import prepare_group_table, prepare_team_detail

    result, _ = run_pipeline(_backbone(), [], _HISTORY, CFG, ANNEX, runs=20, seed=2)
    write_probabilities(
        result.probabilities, tmp_path, "20260620t0900", groups=result.groups
    )
    run = load_latest_run(tmp_path)
    assert run is not None
    table = prepare_group_table(run.groups, run.probabilities)
    assert len(table) == 12 and all(len(rows) == 4 for rows in table.values())
    assert prepare_team_detail(run.probabilities, next(iter(run.probabilities)))


def test_run_pipeline_history_cutoff_excludes_in_tournament() -> None:
    # El baseline excluye resultados en/posteriores al cutoff (torneo en curso):
    # quitar la goleada del torneo baja el rating de A1.
    pre = HistoricalMatch(
        date=date(2025, 1, 1),
        home_team="A1",
        away_team="A2",
        home_score=3,
        away_score=0,
        tournament="Friendly",
        neutral=True,
    )
    in_tournament = HistoricalMatch(
        date=date(2026, 6, 15),
        home_team="A1",
        away_team="A2",
        home_score=5,
        away_score=0,
        tournament="FIFA World Cup",
        neutral=True,
    )
    hist = [pre, in_tournament]
    full, _ = run_pipeline(_backbone(), [], hist, CFG, ANNEX, runs=20, seed=1)
    base, _ = run_pipeline(
        _backbone(),
        [],
        hist,
        CFG,
        ANNEX,
        runs=20,
        seed=1,
        history_cutoff=date(2026, 6, 11),
    )
    assert base.ratings["A1"] < full.ratings["A1"]


def test_outcome_from_ratings_matches_predict_fixture() -> None:
    fitted = fit_elo(_HISTORY, CFG.elo)
    direct, _ = predict_fixture("A1", "A2", _HISTORY, CFG)
    from_ratings, _ = outcome_from_ratings("A1", "A2", fitted, CFG)
    assert from_ratings == direct


def test_outcome_from_ratings_host_helps_home() -> None:
    fitted = fit_elo(_HISTORY, CFG.elo)
    neutral, _ = outcome_from_ratings("A1", "A2", fitted, CFG)
    hosted, _ = outcome_from_ratings("A1", "A2", fitted, CFG, host="A1")
    assert hosted.home_win > neutral.home_win


def test_knockout_advance_deterministic_and_bounded() -> None:
    fitted = fit_elo(_HISTORY, CFG.elo)
    p1 = knockout_advance_probability("A1", "A2", fitted, CFG, runs=500, seed=7)
    p2 = knockout_advance_probability("A1", "A2", fitted, CFG, runs=500, seed=7)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_knockout_advance_at_least_win_in_90() -> None:
    fitted = fit_elo(_HISTORY, CFG.elo)
    outcome, _ = outcome_from_ratings("A1", "A2", fitted, CFG)
    adv = knockout_advance_probability("A1", "A2", fitted, CFG, runs=3000, seed=1)
    assert adv >= outcome.home_win - 0.03


def test_knockout_stronger_team_advances_more_than_half() -> None:
    ratings = {"A": 2000.0, "B": 1200.0}
    adv = knockout_advance_probability("A", "B", ratings, CFG, runs=2000, seed=3)
    assert adv > 0.5
