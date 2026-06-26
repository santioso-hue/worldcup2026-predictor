# Dashboard redesign + match simulator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Streamlit dashboard a clean, minimalistic World-Cup look (branded theme + embedded charts + two tabs) and add a match simulator that runs only on real WC2026 fixtures.

**Architecture:** Streamlit stays. The one new piece of pure logic — `predict_fixture` (1X2 **and** the score matrix) — lives in `worldcup.pipeline` and is unit-tested; `app/dashboard.py` stays thin glue that calls it and renders the figures `viz/` already produces. No pipeline outputs change; the dashboard only reads existing artifacts.

**Tech Stack:** Python 3.11+, Streamlit, matplotlib (via `worldcup.viz`), numpy, pydantic config. Toolchain is the `.venv` (`make lint` / `make test`).

## Global Constraints

- Comments/docstrings in Spanish, code in English; 88-column lines; NumPy-style docstrings.
- Pure-core + thin-I/O: logic in `worldcup.*` (tested), glue in `app/`/`scripts/` (smoke-tested).
- Single brand accent `#185FA5` (no per-team colors). Keep it consistent with the reel charts.
- Fail-loud over silent guessing; do not gold-plate cases that essentially never happen.
- The repo must stay free of AI-assistant references; verify with a case-insensitive grep before any push.
- Run everything through the venv: `make lint`, `make test`.

---

### Task 1: `predict_fixture` helper (1X2 + score matrix), refactor `predict_match` to delegate

**Files:**
- Modify: `src/worldcup/pipeline.py` (imports + new `predict_fixture`, refactor `predict_match`)
- Test: `tests/test_pipeline.py` (add 3 tests; extend the pipeline import)

**Interfaces:**
- Consumes: `fit_elo`, `DixonColesModel`, `outcome_probabilities`, `MatchOutcome` (existing).
- Produces: `predict_fixture(home: str, away: str, history: list[HistoricalMatch], config: Config, *, host: str | None = None, reference_date: date | None = None) -> tuple[MatchOutcome, numpy.ndarray]` — the dashboard's match sim consumes this in Task 2.

- [ ] **Step 1: Add the two missing imports to `pipeline.py`**

In `src/worldcup/pipeline.py`, change the numpy/import lines. After `import json` add `import numpy as np`, and change `from .models.base import MatchOutcome` to:

```python
from .models.base import MatchOutcome, outcome_probabilities
```

So the top of the import block reads:

```python
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .config import Config
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_pipeline.py`. First extend the import from `worldcup.pipeline` to include `predict_fixture` (add it next to `predict_match`). Then append:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k predict_fixture -v`
Expected: FAIL — `ImportError`/`cannot import name 'predict_fixture'`.

- [ ] **Step 4: Implement `predict_fixture` and refactor `predict_match`**

In `src/worldcup/pipeline.py`, replace the whole existing `predict_match` function with these two functions (define `predict_fixture` first):

```python
def predict_fixture(
    home: str,
    away: str,
    history: list[HistoricalMatch],
    config: Config,
    *,
    host: str | None = None,
    reference_date: date | None = None,
) -> tuple[MatchOutcome, np.ndarray]:
    """1X2 + matriz de marcadores Dixon-Coles de un partido (puro, sin I/O).

    ``host`` aplica la ventaja de **sede del Mundial** (``elo.host_advantage``) al
    anfitrión (``home`` o ``away``); en sede neutral no hay ventaja. Equipos sin
    histórico usan ``initial_rating``. Devuelve el 1X2 y la matriz conjunta de
    marcadores (para el heatmap), calculando la matriz una sola vez.
    """
    fitted = fit_elo(history, config.elo, reference_date=reference_date)
    rating_home = fitted.get(home, config.elo.initial_rating)
    rating_away = fitted.get(away, config.elo.initial_rating)
    if host == home:
        advantage = config.elo.host_advantage
    elif host == away:
        advantage = -config.elo.host_advantage
    else:
        advantage = 0.0
    model = DixonColesModel(config.elo, config.dixon_coles)
    matrix = model.score_matrix(rating_home, rating_away, advantage)
    return outcome_probabilities(matrix), matrix


def predict_match(
    home: str,
    away: str,
    history: list[HistoricalMatch],
    config: Config,
    *,
    host: str | None = None,
    reference_date: date | None = None,
) -> MatchOutcome:
    """1X2 de un partido puntual desde los ratings Elo del histórico.

    Envoltura sobre :func:`predict_fixture` (devuelve solo el 1X2). ``host`` aplica la
    ventaja de sede del Mundial al anfitrión; sin histórico se usa ``initial_rating``.
    """
    outcome, _ = predict_fixture(
        home, away, history, config, host=host, reference_date=reference_date
    )
    return outcome
```

- [ ] **Step 5: Run the new tests + the existing predict_match tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (the new `predict_fixture` tests and the existing `test_predict_match_*` tests are all green — the refactor is behavior-preserving).

- [ ] **Step 6: Gate**

Run: `make lint` (expected: ruff + black clean, `mypy` success) and `.venv/bin/python -m pytest -q` (expected: all pass).

- [ ] **Step 7: Commit**

```bash
git add src/worldcup/pipeline.py tests/test_pipeline.py
git commit -m "Add predict_fixture (1X2 + score matrix), refactor predict_match to delegate"
```

---

### Task 2: Dashboard redesign — branded theme, tabs, embedded charts, match simulator

**Files:**
- Create: `.streamlit/config.toml` (Streamlit native theme)
- Modify: `app/dashboard.py` (full rewrite — still thin glue)
- Test: `tests/test_dashboard.py` (existing smoke test must stay green; no edit expected)

**Interfaces:**
- Consumes: `predict_fixture` (Task 1); `load_latest_run`, `load_latest_snapshot`; the `viz` `prepare_*`/`render_*` functions; `RankingRow`, `ExportSpec`; `MatchStatus`, `NormalizedMatch`, `RunArtifact`.
- Produces: nothing other tasks depend on (terminal task).

- [ ] **Step 1: Create the Streamlit theme**

Create `.streamlit/config.toml`:

```toml
# Tema de marca del dashboard (acento único, look limpio). Theming nativo, sin CSS.
[theme]
base = "light"
primaryColor = "#185FA5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F2F4F7"
textColor = "#1A1A1A"
```

- [ ] **Step 2: Rewrite `app/dashboard.py`**

Replace the entire contents of `app/dashboard.py` with:

```python
"""Dashboard interactivo (Streamlit): muestra los artefactos del pipeline.

Glue de Streamlit; la lógica vive en worldcup.pipeline/viz. Dos pestañas: "Torneo"
(probabilidades de la última corrida) y "Partido" (simula un fixture real del WC2026).
Ejecutar: streamlit run app/dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st

from worldcup.config import Config, load_config
from worldcup.data.download import load_latest_snapshot
from worldcup.data.historical import HistoricalMatch, fetch_martj42, parse_results_csv
from worldcup.data.live_results import MatchStatus, NormalizedMatch
from worldcup.models.base import MatchOutcome
from worldcup.pipeline import RunArtifact, load_latest_run, predict_fixture
from worldcup.viz.charts import (
    RankingRow,
    prepare_champion_ranking,
    prepare_group_table,
    prepare_match_bar,
    prepare_score_heatmap,
    prepare_team_detail,
    render_champion_ranking,
    render_group_table,
    render_match_bar,
    render_score_heatmap,
)
from worldcup.viz.theme import ExportSpec

_ROOT = Path(__file__).resolve().parents[1]
_TALL = ExportSpec(820, 1100, dpi=100)  # ranking / escalera (barras horizontales, alto)
_WIDE = ExportSpec(1100, 620, dpi=100)  # barra 1X2 / heatmap (panel horizontal)


def _config() -> Config:
    return load_config(_ROOT / "config" / "config.yaml")


@st.cache_data(show_spinner=False)
def _history() -> list[HistoricalMatch]:
    cfg = _config()
    dest = _ROOT / cfg.paths.data_raw / "results.csv"
    fetch_martj42(cfg.data.historical.results_url, dest)  # idempotente
    return parse_results_csv(dest.read_text())


@st.cache_data(show_spinner=False)
def _simulate(home: str, away: str, host: str | None) -> tuple[MatchOutcome, np.ndarray]:
    return predict_fixture(home, away, _history(), _config(), host=host)


def _fixtures(cfg: Config) -> list[NormalizedMatch]:
    snaps = cfg.data.snapshots
    return (
        load_latest_snapshot(
            _ROOT / snaps.dir,
            filename_pattern=snaps.filename_pattern,
            latest_pointer=snaps.latest_pointer,
        )
        or []
    )


def _rerun_pipeline() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_pipeline.py"), "--mode", "live"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _torneo_tab(run: RunArtifact) -> None:
    st.subheader("Probabilidad de campeón")
    champion = {team: probs["champion"] for team, probs in run.probabilities.items()}
    rows = prepare_champion_ranking(champion, top_n=12)
    st.pyplot(render_champion_ranking(rows, spec=_TALL, title="¿Quién gana el Mundial?"))

    if run.groups:
        st.subheader("Probabilidad de avanzar por grupo")
        table = prepare_group_table(run.groups, run.probabilities)
        st.pyplot(render_group_table(table, spec=_WIDE, title="Avance por grupo"))

    st.subheader("Camino de una selección")
    team = st.selectbox("Selección", sorted(run.probabilities), key="torneo_team")
    detail = prepare_team_detail(run.probabilities, team)
    ladder = [
        RankingRow(rank=i + 1, team=d.label, prob=d.prob, delta="flat")
        for i, d in enumerate(detail)
    ]
    st.pyplot(render_champion_ranking(ladder, spec=_TALL, title=f"{team}: ¿hasta dónde?"))


def _partido_tab(cfg: Config) -> None:
    fixtures = _fixtures(cfg)
    if not fixtures:
        st.info("Todavía no hay fixtures. Corré una simulación live primero (Re-simular).")
        return
    ordered = sorted(fixtures, key=lambda m: m.kickoff_utc)
    labels = {
        f"{m.stage} · {m.home_team} vs {m.away_team} · {m.kickoff_utc:%d %b}": m
        for m in ordered
    }
    match = labels[st.selectbox("Partido", list(labels), key="partido_fixture")]

    hosts = set(cfg.simulation.hosts)
    host_team = next((t for t in (match.home_team, match.away_team) if t in hosts), None)
    host = (
        host_team
        if host_team and st.checkbox(f"Ventaja de sede para {host_team}", value=True)
        else None
    )

    outcome, matrix = _simulate(match.home_team, match.away_team, host)
    segments = prepare_match_bar(
        match.home_team,
        match.away_team,
        outcome.home_win,
        outcome.draw,
        outcome.away_win,
    )
    st.pyplot(
        render_match_bar(
            segments, spec=_WIDE, title=f"{match.home_team} vs {match.away_team}"
        )
    )
    if match.status is MatchStatus.FINISHED and match.ft_home is not None:
        st.caption(
            f"Ya jugado: {match.home_team} {match.ft_home}–{match.ft_away} "
            f"{match.away_team}"
        )
    heatmap = prepare_score_heatmap(matrix, max_goals=5)
    st.pyplot(render_score_heatmap(heatmap, spec=_WIDE))


def main() -> None:
    st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")
    cfg = _config()
    run = load_latest_run(_ROOT / cfg.paths.data_processed)

    header, button = st.columns([4, 1])
    header.title("⚽ Predictor Mundial 2026")
    if run is not None:
        header.caption(f"Actualizado · {run.timestamp}")
    if button.button("Re-simular"):
        with st.spinner("Re-simulando…"):
            result = _rerun_pipeline()
        if result.returncode == 0:
            st.success("Listo. Recargá la página para ver la actualización.")
        else:
            st.error(result.stderr or "Falló la re-simulación.")

    torneo, partido = st.tabs(["Torneo", "Partido"])
    with torneo:
        if run is None:
            st.info("Todavía no hay corridas. Corré el pipeline primero (make run).")
        else:
            _torneo_tab(run)
    with partido:
        _partido_tab(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test (module imports clean)**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -v`
Expected: PASS (`test_dashboard_module_imports` — the module body executes; `main` exists). Streamlit is a dependency, so the module-level `import streamlit` resolves.

- [ ] **Step 4: Gate**

Run: `make lint` (ruff + black + `mypy src tests scripts app` — expected clean; streamlit has `ignore_missing_imports`) and `.venv/bin/python -m pytest -q` (all pass).

- [ ] **Step 5: Manual visual check**

Run: `PYTHONPATH=src .venv/bin/streamlit run app/dashboard.py`
Expected: branded header (⚽ + title + "Actualizado · …" + Re-simular); **Torneo** tab shows the champion-ranking bar chart, group-advance chart, and a per-team ladder (charts, not tables); **Partido** tab shows a WC-fixture dropdown → 1X2 bar + score heatmap, with a host toggle when a host nation plays. If there is no live snapshot yet, the Partido tab shows the "corré una simulación live primero" message. Confirm the accent is the brand blue. Close with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add .streamlit/config.toml app/dashboard.py
git commit -m "Redesign dashboard: branded theme, Torneo/Partido tabs, embedded charts, WC match simulator"
```

---

## Self-Review

**Spec coverage** (against `docs/specs/2026-06-24-dashboard-redesign-design.md`):
- §1 Theme & shell → Task 2 Steps 1–2 (`.streamlit/config.toml`, header, `st.tabs`). ✓
- §2 Torneo tab (embed charts) → Task 2 `_torneo_tab` (ranking, group, team ladder via `st.pyplot`). ✓
- §3 Partido tab (fixture picker from snapshot, host toggle, 1X2 + heatmap, finished shows score) → Task 2 `_partido_tab`. ✓
- §4 Pure helper `predict_fixture` + `predict_match` delegate → Task 1. ✓
- §5 Touchpoints → covered. §6 Testing (predict_fixture unit tests; dashboard smoke) → Task 1 Step 2, Task 2 Step 3. ✓
- §Caching (`@st.cache_data` for history + per-fixture sim) → `_history`, `_simulate`. ✓

**Placeholder scan:** none — every step has exact code/commands/paths.

**Type consistency:** `predict_fixture` signature/return (`tuple[MatchOutcome, np.ndarray]`) is identical in Task 1's definition and Task 2's `_simulate` consumer; `RankingRow(rank, team, prob, delta)`, `prepare_match_bar(home, away, p_home, p_draw, p_away)`, `prepare_score_heatmap(matrix, max_goals=...)`, `load_latest_snapshot(raw_dir, *, filename_pattern, latest_pointer)`, and `render_*` `spec=` kwargs all match the real signatures in `viz/charts.py`, `models/base.py`, `data/download.py`.
