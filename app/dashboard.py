"""Interactive dashboard (Streamlit): reads pipeline artifacts and displays them.

Just Streamlit glue; the testable logic lives in ``worldcup.pipeline``/``viz``.
"Re-simulate" triggers the pipeline CLI (subprocess), reusing the single
orchestration path. The "Upcoming matches" panel predicts each unplayed
fixture live, reusing ``predict_fixture`` (via ``outcome_from_ratings``) and
fitting Elo only once.
Run: ``streamlit run app/dashboard.py``.
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
    """Teams present in the historical data (to detect fixtures not yet defined)."""
    history = parse_results_csv(Path(results_csv).read_text())
    return {m.home_team for m in history} | {m.away_team for m in history}


@st.cache_data(show_spinner=False)
def _ratings(results_csv: str, ref_iso: str) -> dict[str, float]:
    """Fit Elo once per load (cached by history + date)."""
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
    """1X2 (and advance probability if knockout) for a fixture, cached by its params."""
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
    """Stacked 1X2 bar (home in accent color, draw and away in grays)."""
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
    """Match card with 1X2, favored side highlighted, and advance prob if knockout."""
    home_fav = card["home"] >= card["away"] and card["home"] >= card["draw"]
    away_fav = card["away"] > card["home"] and card["away"] >= card["draw"]
    home_style = f"color:{_ACCENT};font-weight:600;" if home_fav else ""
    away_style = f"color:{_ACCENT};font-weight:600;" if away_fav else ""
    advance = ""
    if "advance" in card:
        home_adv = card["advance"]
        if home_adv >= 0.5:
            leader, p_adv = home, home_adv
        else:
            leader, p_adv = away, 1.0 - home_adv
        advance = (
            '<div style="font-size:13px;color:#555;border-top:1px solid #eee;'
            'padding-top:6px;margin-top:6px;">Advances: '
            f"<b>{leader} {p_adv:.0%}</b></div>"
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
        f'<span>Draw {card["draw"]:.0%}</span><span>{card["away"]:.0%}</span></div>'
        f"{advance}</div>"
    )


def _pending_card_html(stage: str, fecha: str) -> str:
    """Card for a fixture not yet defined (can't predict TBD vs TBD)."""
    return (
        '<div style="border:1px solid #e6e6e6;border-radius:12px;padding:14px 16px;'
        'margin-bottom:12px;background:#fafafa;color:#999;">'
        '<div style="display:flex;justify-content:space-between;font-size:12px;">'
        f"<span>{stage}</span><span>{fecha}</span></div>"
        '<div style="font-size:13px;margin-top:10px;">'
        "Waiting on earlier results</div>"
        "</div>"
    )


def _host_for(home: str, away: str, hosts: set[str]) -> str | None:
    """The match's host team, or ``None``. If both are hosts it cancels out
    (neutral), matching how net host advantage works in the tournament sim."""
    if home in hosts and away not in hosts:
        return home
    if away in hosts and home not in hosts:
        return away
    return None


def _match_predictor_section(config: Config, run: RunArtifact) -> None:
    """Upcoming-matches panel: predicts each unplayed fixture live."""
    st.subheader("Upcoming matches")
    if not run.fixtures:
        st.info("No matches left to play.")
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
            host = _host_for(home, away, hosts)
            card = _predict_card(
                home, away, host, is_knockout_stage(stage), ref_iso, results_csv
            )
            st.markdown(
                _predicted_card_html(stage, fecha, home, away, card),
                unsafe_allow_html=True,
            )


def _rerun_pipeline() -> subprocess.CompletedProcess[str]:
    """Trigger the pipeline CLI in live mode (reuses the single orchestration path)."""
    return subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_pipeline.py"), "--mode", "live"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
    config = _config()
    run = load_latest_run(_ROOT / config.paths.data_processed)

    title_col, button_col = st.columns([4, 1])
    title_col.title("World Cup 2026 Predictor")
    if run is not None:
        title_col.caption(f"Updated {run.timestamp}")
    if button_col.button("Re-simulate"):
        with st.spinner("Re-simulating…"):
            result = _rerun_pipeline()
        if result.returncode == 0:
            st.cache_data.clear()
            st.success("Reload the page to see the update.")
        else:
            st.error(result.stderr or "Re-simulation failed.")

    if run is None:
        st.info("No runs yet. Run the pipeline first (make run).")
        return

    st.subheader("Championship probability")
    champion = {team: probs["champion"] for team, probs in run.probabilities.items()}
    rows = prepare_champion_ranking(champion, top_n=15)
    st.dataframe(
        {
            "Team": [r.team for r in rows],
            "P(champion)": [f"{r.prob:.1%}" for r in rows],
        },
        hide_index=True,
    )

    _match_predictor_section(config, run)

    if run.groups:
        st.subheader("Group advance probability")
        table = prepare_group_table(run.groups, run.probabilities)
        group_cols = st.columns(4)
        for index, (letter, group_rows) in enumerate(sorted(table.items())):
            col = group_cols[index % 4]
            col.markdown(f"**Group {letter}**")
            col.dataframe(
                {
                    "Team": [g.team for g in group_rows],
                    "P(advance)": [f"{g.p_advance:.0%}" for g in group_rows],
                },
                hide_index=True,
            )
    else:
        st.info("This run has no groups; re-simulate to see them.")

    st.subheader("Team detail")
    team = st.selectbox("Team", sorted(run.probabilities))
    detail = prepare_team_detail(run.probabilities, team)
    st.dataframe(
        {
            "Round": [d.label for d in detail],
            "Probability": [f"{d.prob:.1%}" for d in detail],
        },
        hide_index=True,
    )


if __name__ == "__main__":
    main()
