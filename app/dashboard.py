"""Dashboard interactivo (Streamlit): lee los artefactos del pipeline y los muestra.

Solo glue de Streamlit; la lógica testeable vive en ``worldcup.pipeline``/``viz``.
"Re-simular" dispara el CLI del pipeline (subprocess), reusando la única orquestación.
El panel de "Próximos partidos" predice en vivo cada fixture por jugar reusando
``predict_fixture`` (vía ``outcome_from_ratings``), ajustando el Elo una sola vez.
Ejecutar: ``streamlit run app/dashboard.py``.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import streamlit as st

from worldcup.config import Config, load_config
from worldcup.data.historical import parse_results_csv
from worldcup.data.schedule import is_knockout_stage, is_predictable
from worldcup.features.elo import fit_elo
from worldcup.pipeline import (
    RunArtifact,
    knockout_advance_probability,
    load_latest_run,
    outcome_from_ratings,
)
from worldcup.viz.charts import (
    prepare_champion_ranking,
    prepare_group_table,
    prepare_team_detail,
)

_ROOT = Path(__file__).resolve().parents[1]
_ACCENT = "#185FA5"


def _config() -> Config:
    return load_config(_ROOT / "config" / "config.yaml")


@st.cache_data(show_spinner=False)
def _known_teams(results_csv: str) -> set[str]:
    """Selecciones presentes en el histórico (para detectar cruces aún sin definir)."""
    history = parse_results_csv(Path(results_csv).read_text())
    return {m.home_team for m in history} | {m.away_team for m in history}


@st.cache_data(show_spinner=False)
def _ratings(results_csv: str, ref_iso: str) -> dict[str, float]:
    """Ajusta el Elo una sola vez por carga (cacheado por histórico + fecha)."""
    history = parse_results_csv(Path(results_csv).read_text())
    return fit_elo(history, _config().elo, reference_date=date.fromisoformat(ref_iso))


@st.cache_data(show_spinner=False)
def _predict_card(
    home: str,
    away: str,
    host: str | None,
    knockout: bool,
    ref_iso: str,
    results_csv: str,
) -> dict[str, float]:
    """1X2 (y avance si es eliminatoria) de un fixture, cacheado por sus parámetros."""
    config = _config()
    ratings = _ratings(results_csv, ref_iso)
    outcome, _ = outcome_from_ratings(home, away, ratings, config, host=host)
    card = {"home": outcome.home_win, "draw": outcome.draw, "away": outcome.away_win}
    if knockout:
        card["advance"] = knockout_advance_probability(
            home, away, ratings, config, host=host
        )
    return card


def _bar_html(p_home: float, p_draw: float, p_away: float) -> str:
    """Barra 1X2 apilada (local en acento, empate y visita en grises)."""
    return (
        '<div style="display:flex;height:8px;border-radius:999px;overflow:hidden;'
        'margin:6px 0;">'
        f'<div style="width:{p_home:.0%};background:{_ACCENT};"></div>'
        f'<div style="width:{p_draw:.0%};background:#888780;"></div>'
        f'<div style="width:{p_away:.0%};background:#B4B2A9;"></div></div>'
    )


def _predicted_card_html(
    stage: str, fecha: str, home: str, away: str, card: dict[str, float]
) -> str:
    """Tarjeta de un partido con 1X2, ganador resaltado y (si aplica) avance."""
    home_fav = card["home"] >= card["away"] and card["home"] >= card["draw"]
    away_fav = card["away"] > card["home"] and card["away"] >= card["draw"]
    home_style = f"color:{_ACCENT};font-weight:600;" if home_fav else ""
    away_style = f"color:{_ACCENT};font-weight:600;" if away_fav else ""
    advance = ""
    if "advance" in card:
        leader = home if card["home"] >= card["away"] else away
        advance = (
            '<div style="font-size:13px;color:#555;border-top:1px solid #eee;'
            'padding-top:6px;margin-top:6px;">Avanza: '
            f'<b>{leader} {card["advance"]:.0%}</b></div>'
        )
    return (
        '<div style="border:1px solid #e6e6e6;border-radius:12px;padding:14px 16px;'
        'margin-bottom:12px;">'
        '<div style="display:flex;justify-content:space-between;font-size:12px;'
        f'color:#777;"><span>{stage}</span><span>{fecha}</span></div>'
        '<div style="display:flex;justify-content:space-between;font-size:15px;'
        f'margin:8px 0;"><span style="{home_style}">{home}</span>'
        f'<span style="{away_style}">{away}</span></div>'
        f"{_bar_html(card['home'], card['draw'], card['away'])}"
        '<div style="display:flex;justify-content:space-between;font-size:12px;'
        f'color:#555;"><span>{card["home"]:.0%}</span>'
        f'<span>Empate {card["draw"]:.0%}</span><span>{card["away"]:.0%}</span></div>'
        f"{advance}</div>"
    )


def _pending_card_html(stage: str, fecha: str) -> str:
    """Tarjeta de un cruce aún sin definir (no se puede predecir TBD vs TBD)."""
    return (
        '<div style="border:1px solid #e6e6e6;border-radius:12px;padding:14px 16px;'
        'margin-bottom:12px;background:#fafafa;color:#999;">'
        '<div style="display:flex;justify-content:space-between;font-size:12px;">'
        f"<span>{stage}</span><span>{fecha}</span></div>"
        '<div style="font-size:13px;margin-top:10px;">'
        "Esperando resultados previos</div>"
        "</div>"
    )


def _match_predictor_section(config: Config, run: RunArtifact) -> None:
    """Panel "Próximos partidos": predice en vivo cada fixture por jugar."""
    st.subheader("Próximos partidos")
    if not run.fixtures:
        st.info("No quedan partidos por jugar.")
        return
    results_csv = str(_ROOT / config.paths.data_raw / "results.csv")
    known = _known_teams(results_csv)
    hosts = set(config.simulation.hosts)
    ref_iso = date.today().isoformat()
    cols = st.columns(2)
    for index, fixture in enumerate(run.fixtures):
        home, away, stage = fixture["home"], fixture["away"], fixture["stage"]
        fecha = fixture["kickoff"][:10]
        with cols[index % 2]:
            if not is_predictable(home, away, known):
                st.markdown(_pending_card_html(stage, fecha), unsafe_allow_html=True)
                continue
            host = home if home in hosts else (away if away in hosts else None)
            card = _predict_card(
                home, away, host, is_knockout_stage(stage), ref_iso, results_csv
            )
            st.markdown(
                _predicted_card_html(stage, fecha, home, away, card),
                unsafe_allow_html=True,
            )


def _rerun_pipeline() -> subprocess.CompletedProcess[str]:
    """Dispara el CLI del pipeline en modo live (reusa la única orquestación)."""
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_pipeline.py"), "--mode", "live"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    st.set_page_config(page_title="Predictor Mundial 2026", layout="wide")
    config = _config()
    run = load_latest_run(_ROOT / config.paths.data_processed)

    title_col, button_col = st.columns([4, 1])
    title_col.title("Predictor Mundial 2026")
    if run is not None:
        title_col.caption(f"Actualizado {run.timestamp}")
    if button_col.button("Re-simular"):
        with st.spinner("Re-simulando…"):
            result = _rerun_pipeline()
        if result.returncode == 0:
            st.success("Recarga la página para ver la actualización.")
        else:
            st.error(result.stderr or "Falló la re-simulación.")

    if run is None:
        st.info("Todavía no hay corridas. Corre el pipeline primero (make run).")
        return

    st.subheader("Probabilidad de campeón")
    champion = {team: probs["champion"] for team, probs in run.probabilities.items()}
    rows = prepare_champion_ranking(champion, top_n=15)
    st.dataframe(
        {
            "Selección": [r.team for r in rows],
            "P(campeón)": [f"{r.prob:.1%}" for r in rows],
        },
        hide_index=True,
    )

    _match_predictor_section(config, run)

    if run.groups:
        st.subheader("Probabilidad de avance por grupo")
        table = prepare_group_table(run.groups, run.probabilities)
        group_cols = st.columns(4)
        for index, (letter, group_rows) in enumerate(sorted(table.items())):
            col = group_cols[index % 4]
            col.markdown(f"**Grupo {letter}**")
            col.dataframe(
                {
                    "Selección": [g.team for g in group_rows],
                    "P(avance)": [f"{g.p_advance:.0%}" for g in group_rows],
                },
                hide_index=True,
            )
    else:
        st.info("Esta corrida no incluye grupos; re-simula para verlas.")

    st.subheader("Detalle por selección")
    team = st.selectbox("Selección", sorted(run.probabilities))
    detail = prepare_team_detail(run.probabilities, team)
    st.dataframe(
        {
            "Ronda": [d.label for d in detail],
            "Probabilidad": [f"{d.prob:.1%}" for d in detail],
        },
        hide_index=True,
    )


if __name__ == "__main__":
    main()
