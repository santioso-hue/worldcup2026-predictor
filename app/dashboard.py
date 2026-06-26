"""Dashboard interactivo (Streamlit): lee los artefactos del pipeline y los muestra.

Solo glue de Streamlit; la lógica testeable vive en ``worldcup.pipeline``/``viz``.
"Re-simular" dispara el CLI del pipeline (subprocess), reusando la única orquestación.
Ejecutar: ``streamlit run app/dashboard.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from worldcup.config import load_config
from worldcup.pipeline import load_latest_run
from worldcup.viz.charts import (
    prepare_champion_ranking,
    prepare_group_table,
    prepare_team_detail,
)

_ROOT = Path(__file__).resolve().parents[1]


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
    import streamlit as st

    st.set_page_config(page_title="Predictor Mundial 2026", layout="wide")
    config = load_config(_ROOT / "config" / "config.yaml")
    # Resolver contra _ROOT (igual que el writer del subprocess): el dashboard no debe
    # depender del cwd desde donde se lanzó `streamlit run`.
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

    # Un artefacto sin grupos (corrida vieja) degrada con gracia, como run is None.
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
