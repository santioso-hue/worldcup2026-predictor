"""Interactive dashboard (Streamlit): reads pipeline artifacts and displays them.

Just Streamlit glue; the testable logic lives in ``worldcup.pipeline``/``viz``.
"Re-simulate" triggers the pipeline CLI (subprocess), reusing the single
orchestration path. The "Knockout bracket" panel renders a two-sided bracket
of native HTML/CSS match cards (``_bracket_html``) from the persisted
``run.bracket`` artifact: finished ties show the score, scheduled ties predict
the advance probability live (reusing ``knockout_advance_probability`` via the
cached ``_predict_card`` path), and TBD slots show empty cards.
Run: ``streamlit run app/dashboard.py``.
"""

from __future__ import annotations

import html
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
from worldcup.data.historical import fetch_martj42, parse_results_csv  # noqa: E402
from worldcup.features.elo import fit_elo  # noqa: E402
from worldcup.pipeline import (  # noqa: E402
    RunArtifact,
    knockout_advance_probability,
    load_latest_run,
    outcome_from_ratings,
)
from worldcup.simulation.bracket import FINAL_MATCH, KNOCKOUT_BRACKET  # noqa: E402
from worldcup.viz.charts import (  # noqa: E402
    prepare_champion_ranking,
    prepare_group_table,
    prepare_team_detail,
)

_ROOT = Path(__file__).resolve().parents[1]


def _config() -> Config:
    return load_config(_ROOT / "config" / "config.yaml")


@st.cache_data(show_spinner="Downloading match history…")
def _ensure_history(results_csv: str) -> str:
    """Download the results history once if missing (fresh deploys have no data/)."""
    config = _config()
    fetch_martj42(config.data.historical.results_url, Path(results_csv))
    return results_csv


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


def _eliminated(probabilities: dict[str, dict[str, float]]) -> set[str]:
    """Teams with zero title probability — knocked out (or never advanced)."""
    return {t for t, p in probabilities.items() if p.get("champion", 0.0) == 0.0}


def _state_strip_stats(
    probabilities: dict[str, dict[str, float]], bracket: dict[str, dict]
) -> dict[str, str]:
    """The "right now" thesis: favorite, teams alive, ties played, next kickoff.

    Pure summary of the loaded artifact, formatted as display strings for the
    top-of-page metric row. ``bracket`` maps match number (as a string key,
    matching the persisted artifact) to a tie dict with ``status`` and
    ``kickoff`` fields.
    """
    eliminated = _eliminated(probabilities)
    alive = {t: p for t, p in probabilities.items() if t not in eliminated}
    if alive:
        favorite = max(alive, key=lambda t: alive[t].get("champion", 0.0))
        favorite_odds = _fmt_prob(alive[favorite]["champion"])
    else:
        favorite, favorite_odds = "—", ""
    played = sum(1 for tie in bracket.values() if tie["status"] == "finished")
    kickoffs = [
        tie["kickoff"]
        for tie in bracket.values()
        if tie["status"] == "scheduled" and tie["kickoff"]
    ]
    if kickoffs:
        first = date.fromisoformat(min(kickoffs)[:10])
        next_kickoff = f"{first.strftime('%b')} {first.day}"
    else:
        next_kickoff = "—"
    return {
        "favorite": favorite,
        "favorite_odds": favorite_odds,
        "alive": str(48 - len(eliminated)),
        "played": str(played),
        "next_kickoff": next_kickoff,
    }


def _fmt_prob(p: float) -> str:
    """One-decimal percentage, flooring tiny nonzero values to ``"<0.1%"``.

    A live team with a vanishingly small title chance still rounds to
    ``"0.0%"`` at one decimal, which reads as eliminated; ``"<0.1%"``
    communicates "still alive, just very unlikely" instead.
    """
    if 0.0 < p < 0.001:
        return "<0.1%"
    return f"{p:.1%}"


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


def _finished_hover(tie: dict) -> str:
    """Hover text for a decided tie: stage and final score."""
    stage, home, away = tie["stage"], tie["home"], tie["away"]
    return f"{stage} — final: {home} {tie['ft_home']}–{tie['ft_away']} {away}"


def _scheduled_hover(tie: dict, card: dict[str, float], label: str) -> str:
    """Hover text for an undecided tie: stage, kickoff date, 1X2, advance label."""
    stage, home, away = tie["stage"], tie["home"], tie["away"]
    kickoff = tie["kickoff"]
    when = kickoff[:10] if kickoff else "date TBD"
    return (
        f"{stage} — {when}<br>"
        f"{home} {card['home']:.0%} · draw {card['draw']:.0%} · "
        f"{away} {card['away']:.0%}<br>"
        f"advances: {label}"
    )


@st.cache_data(show_spinner=False)
def _bracket_rows(
    results_csv: str, ref_iso: str, bracket: dict[str, dict]
) -> dict[int, dict]:
    """Precompute the annotated match fields per match number, for card HTML.

    Cached on (results, reference date, bracket contents) so the (uncached)
    HTML build downstream stays cheap. Each row also carries a ``hover``
    string (``None`` for TBD ties) built from the same ``_predict_card`` call
    used for the advance label, so the dashboard doesn't recompute the 1X2
    separately for the tooltip.
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
                "hover": _finished_hover(tie),
            }
        elif status == "scheduled" and home is not None and away is not None:
            host = _host_for(home, away, hosts)
            card = _predict_card(home, away, host, True, ref_iso, results_csv)
            p_home_adv = card["advance"]
            leader = home if p_home_adv >= 0.5 else away
            label = _advance_label(home, away, p_home_adv)
            rows[match_id] = {
                "home": home,
                "away": away,
                "winner": None,
                "annotation": label,
                "highlight": leader,
                "hover": _scheduled_hover(tie, card, label),
            }
        else:
            rows[match_id] = {
                "home": None,
                "away": None,
                "winner": None,
                "annotation": None,
                "highlight": None,
                "hover": None,
            }
    return rows


_BRACKET_CSS = """
<style>
.bkt-wrap{overflow-x:auto;padding:8px 0;}
.bkt{display:flex;gap:14px;min-width:1180px;align-items:stretch;}
.bkt-col{display:flex;flex-direction:column;justify-content:space-around;
  gap:10px;flex:1 1 0;min-width:120px;}
.bkt-col--final{justify-content:center;}
.bkt-col-label{font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:#7C8CA3;text-align:center;margin-bottom:4px;}
.bkt-card{border:1px solid #263447;border-radius:12px;padding:8px 10px;
  background:#16202E;transition:box-shadow .15s,border-color .15s;}
.bkt-card:hover{box-shadow:0 2px 10px rgba(91,163,232,.25);border-color:#5BA3E8;}
.bkt-card--tbd{background:#111927;border-color:#1C2736;}
.bkt-row{display:flex;justify-content:space-between;align-items:center;
  font-size:13px;padding:2px 0;gap:6px;}
.bkt-team{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#E8EDF4;}
.bkt-team--win{color:#5BA3E8;font-weight:600;}
.bkt-team--tbd{color:#566577;}
.bkt-chip{margin-top:6px;padding-top:6px;border-top:1px solid #1C2736;
  font-size:11px;text-align:right;}
.bkt-chip--advance{color:#4ADBA0;background:rgba(53,194,134,.12);
  display:inline-block;padding:2px 6px;border-radius:6px;}
.bkt-chip--score{color:#C9D4E3;background:#1E2B3C;
  display:inline-block;padding:2px 6px;border-radius:6px;}
</style>
"""

_ROUND_LABELS = [
    "ROUND OF 32",
    "ROUND OF 16",
    "QUARTER-FINALS",
    "SEMI-FINALS",
    "FINAL",
]


def _bracket_card_html(row: dict) -> str:
    """One match card: two team rows, optional score/advance chip, hover title."""
    home, away = row["home"], row["away"]
    winner, highlight = row["winner"], row["highlight"]
    tbd = home is None or away is None
    hover = row["hover"]
    title_attr = ""
    if hover:
        tooltip = html.escape(hover.replace("<br>", "\n"), quote=True)
        title_attr = f' title="{tooltip}"'

    def team_row(team: str | None) -> str:
        if team is None:
            return (
                '<div class="bkt-row"><span class="bkt-team bkt-team--tbd">'
                "&mdash;</span></div>"
            )
        css = "bkt-team bkt-team--win" if team in (winner, highlight) else "bkt-team"
        return (
            f'<div class="bkt-row"><span class="{css}">'
            f"{html.escape(team)}</span></div>"
        )

    chip = ""
    if row["annotation"]:
        chip_kind = "score" if winner else "advance"
        chip = (
            f'<div class="bkt-chip bkt-chip--{chip_kind}">'
            f'{html.escape(row["annotation"])}</div>'
        )

    card_css = "bkt-card bkt-card--tbd" if tbd else "bkt-card"
    return (
        f'<div class="{card_css}"{title_attr}>'
        f"{team_row(home)}{team_row(away)}{chip}</div>"
    )


def _bracket_html(rows: dict[int, dict], round_order: list[list[int]]) -> str:
    """Two-sided bracket as linked match cards: 9 flex columns, no images.

    ``round_order`` is ``[R32(16), R16(8), QF(4), SF(2), Final(1)]``; the first
    half of each round's list is the left side of the draw, the second half is
    the right side (mirrored). Columns run left R32 -> left R16 -> left QF ->
    left SF -> Final -> right SF -> right QF -> right R16 -> right R32, each
    ``justify-content:space-around`` so cards center against their feeders —
    the classic CSS-bracket trick, no absolute positioning or connector math.
    """
    semis, final_round = round_order[:-1], round_order[-1]
    left_halves = [match_ids[: len(match_ids) // 2] for match_ids in semis]
    right_halves = [match_ids[len(match_ids) // 2 :] for match_ids in semis]
    left_labels = _ROUND_LABELS[:-1]
    right_labels = list(reversed(left_labels))

    def column(match_ids: list[int], label: str, *, extra_css: str = "") -> str:
        cards = "".join(_bracket_card_html(rows[m]) for m in match_ids)
        label_div = f'<div class="bkt-col-label">{html.escape(label)}</div>'
        return f'<div class="bkt-col{extra_css}">{label_div}{cards}</div>'

    columns = [
        column(half, label)
        for half, label in zip(left_halves, left_labels, strict=True)
    ]
    columns.append(column(final_round, _ROUND_LABELS[-1], extra_css=" bkt-col--final"))
    columns.extend(
        column(half, label)
        for half, label in zip(reversed(right_halves), right_labels, strict=True)
    )

    return (
        _BRACKET_CSS
        + '<div class="bkt-wrap"><div class="bkt">'
        + "".join(columns)
        + "</div></div>"
    )


def _match_predictor_section(config: Config, run: RunArtifact) -> None:
    """Knockout bracket panel: linked match cards with live advance probabilities."""
    st.subheader("Knockout bracket")
    if not run.bracket:
        st.info("This run predates the bracket artifact — re-run the pipeline.")
        return
    results_csv = _ensure_history(str(_ROOT / config.paths.data_raw / "results.csv"))
    ref_iso = date.today().isoformat()
    rows = _bracket_rows(results_csv, ref_iso, run.bracket)
    round_order = _bracket_round_order()
    st.markdown(_bracket_html(rows, round_order), unsafe_allow_html=True)


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
    st.set_page_config(
        page_title="World Cup 2026 predictor", page_icon="⚽", layout="wide"
    )
    config = _config()
    run = load_latest_run(_ROOT / config.paths.data_processed)

    title_col, button_col = st.columns([4, 1])
    title_col.title("World Cup 2026 predictor")
    if run is not None:
        title_col.caption(
            f"Updated {run.timestamp} · live model — Elo → Dixon-Coles → Monte Carlo"
        )
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

    eliminated = _eliminated(run.probabilities)
    stats = _state_strip_stats(run.probabilities, run.bracket)
    strip_cols = st.columns(4)
    strip_cols[0].metric(
        "Favorite", stats["favorite"], delta=stats["favorite_odds"], delta_color="off"
    )
    strip_cols[1].metric("Teams alive", f"{stats['alive']}/48")
    strip_cols[2].metric("Knockout ties played", f"{stats['played']}/32")
    strip_cols[3].metric("Next kickoff", stats["next_kickoff"])

    _match_predictor_section(config, run)

    st.subheader("Title odds")
    champion = {
        team: probs["champion"]
        for team, probs in run.probabilities.items()
        if team not in eliminated
    }
    rows = prepare_champion_ranking(champion, top_n=15)
    st.dataframe(
        {
            "Team": [r.team for r in rows],
            "P(title)": [r.prob for r in rows],
        },
        column_config={
            "P(title)": st.column_config.ProgressColumn(
                "P(title)", format="percent", min_value=0.0, max_value=1.0
            )
        },
        hide_index=True,
    )

    with st.expander("Group stage results", expanded=False):
        if run.groups:
            table = prepare_group_table(run.groups, run.probabilities)
            group_cols = st.columns(4)
            for index, (letter, group_rows) in enumerate(sorted(table.items())):
                col = group_cols[index % 4]
                col.markdown(f"**Group {letter}**")
                col.dataframe(
                    {
                        "Team": [g.team for g in group_rows],
                        "P(advance)": [g.p_advance for g in group_rows],
                    },
                    column_config={
                        "P(advance)": st.column_config.ProgressColumn(
                            "P(advance)", format="percent", min_value=0.0, max_value=1.0
                        )
                    },
                    hide_index=True,
                )
        else:
            st.info("This run has no groups; re-simulate to see them.")

    st.subheader("Team outlook")
    alive = sorted(t for t in run.probabilities if t not in eliminated)
    show_eliminated = st.toggle("Show eliminated teams")
    options = sorted(run.probabilities) if show_eliminated else alive
    team = st.selectbox("Team", options)
    if team in eliminated:
        st.caption("Eliminated")
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
