"""Interactive dashboard (Streamlit): reads pipeline artifacts and displays them.

Just Streamlit glue; the testable logic lives in ``worldcup.pipeline``/``viz``.
"Re-simulate" triggers the pipeline CLI (subprocess), reusing the single
orchestration path. The "Knockout bracket" panel renders the mirrored bracket
from the persisted ``run.bracket`` artifact: finished ties show the score,
scheduled ties predict the advance probability live (reusing
``knockout_advance_probability`` via the cached ``_predict_card`` path), and
TBD slots show empty boxes.
Run: ``streamlit run app/dashboard.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

# Deployment shim: Streamlit Cloud runs this file directly without an
# editable install of `worldcup`, so put `src/` on the path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from worldcup.config import Config, load_config  # noqa: E402
from worldcup.data.historical import parse_results_csv  # noqa: E402
from worldcup.features.elo import fit_elo  # noqa: E402
from worldcup.pipeline import (  # noqa: E402
    RunArtifact,
    knockout_advance_probability,
    load_latest_run,
    outcome_from_ratings,
)
from worldcup.simulation.bracket import FINAL_MATCH, KNOCKOUT_BRACKET  # noqa: E402
from worldcup.viz.bracket import (  # noqa: E402
    BracketMatch,
    prepare_bracket_mirrored,
    render_bracket_mirrored,
)
from worldcup.viz.charts import (  # noqa: E402
    prepare_champion_ranking,
    prepare_group_table,
    prepare_team_detail,
)

_ROOT = Path(__file__).resolve().parents[1]


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


def _host_for(home: str, away: str, hosts: set[str]) -> str | None:
    """The match's host team, or ``None``. If both are hosts it cancels out
    (neutral), matching how net host advantage works in the tournament sim."""
    if home in hosts and away not in hosts:
        return home
    if away in hosts and home not in hosts:
        return away
    return None


def _advance_label(home: str, away: str, p_home_adv: float) -> str:
    """Advance annotation naming the favored side with its own probability."""
    if p_home_adv >= 0.5:
        leader, p_adv = home, p_home_adv
    else:
        leader, p_adv = away, 1.0 - p_home_adv
    return f"{leader} {p_adv:.0%}"


def _bracket_round_order() -> list[list[int]]:
    """Match numbers per round, ordered depth-first from the final (skip 103).

    Walking ``KNOCKOUT_BRACKET`` depth-first from ``FINAL_MATCH`` and
    collecting leaves per depth gives each round in an order where the first
    half of the R32 list feeds the left half of the mirrored bracket and the
    second half feeds the right half (the leaf order defines the halves that
    :func:`worldcup.viz.bracket.prepare_bracket_mirrored` splits on).
    """
    rounds_by_depth: dict[int, list[int]] = {}

    def walk(match_id: int, depth: int) -> None:
        slot_a, slot_b = KNOCKOUT_BRACKET[match_id]
        children: list[int] = []
        for kind, ref in (slot_a, slot_b):
            if kind in ("MW", "ML") and ref != 103:
                assert isinstance(ref, int)  # "MW"/"ML" refs are match ids
                children.append(ref)
        if not children:
            rounds_by_depth.setdefault(depth, []).append(match_id)
            return
        for child in children:
            walk(child, depth + 1)
        rounds_by_depth.setdefault(depth, []).append(match_id)

    walk(FINAL_MATCH, 0)
    return [rounds_by_depth[d] for d in sorted(rounds_by_depth, reverse=True)]


@st.cache_data(show_spinner=False)
def _bracket_rows(
    results_csv: str, ref_iso: str, bracket: dict[str, dict]
) -> dict[int, dict]:
    """Precompute the annotated ``BracketMatch`` fields per match number.

    Cached on (results, reference date, bracket contents) so the (uncached)
    figure build downstream stays cheap. ``BracketMatch``/``Figure`` objects
    aren't hashable by ``st.cache_data``, so this returns plain dict rows and
    the figure is assembled uncached from them.
    """
    config = _config()
    hosts = set(config.simulation.hosts)
    rows: dict[int, dict] = {}
    for match_str, tie in bracket.items():
        match_id = int(match_str)
        home, away = tie["home"], tie["away"]
        status = tie["status"]
        if status == "finished":
            rows[match_id] = {
                "home": home,
                "away": away,
                "winner": tie["winner"],
                "annotation": f"{tie['ft_home']}–{tie['ft_away']}",
                "highlight": tie["winner"],
            }
        elif status == "scheduled" and home is not None and away is not None:
            host = _host_for(home, away, hosts)
            card = _predict_card(home, away, host, True, ref_iso, results_csv)
            p_home_adv = card["advance"]
            leader = home if p_home_adv >= 0.5 else away
            rows[match_id] = {
                "home": home,
                "away": away,
                "winner": None,
                "annotation": _advance_label(home, away, p_home_adv),
                "highlight": leader,
            }
        else:
            rows[match_id] = {
                "home": None,
                "away": None,
                "winner": None,
                "annotation": None,
                "highlight": None,
            }
    return rows


def _match_predictor_section(config: Config, run: RunArtifact) -> None:
    """Knockout bracket panel: mirrored bracket with live advance probabilities."""
    st.subheader("Knockout bracket")
    if not run.bracket:
        st.info("This run predates the bracket artifact — re-run the pipeline.")
        return
    results_csv = str(_ROOT / config.paths.data_raw / "results.csv")
    ref_iso = date.today().isoformat()
    rows = _bracket_rows(results_csv, ref_iso, run.bracket)
    rounds = [
        [BracketMatch(**rows[match_id]) for match_id in match_ids]
        for match_ids in _bracket_round_order()
    ]
    positioned = prepare_bracket_mirrored(rounds)
    fig = render_bracket_mirrored(positioned, title="Knockout bracket")
    st.pyplot(fig)


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
    if not os.environ.get("WC_PUBLIC_DEMO") and button_col.button("Re-simulate"):
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
