"""Dashboard smoke test: module imports clean (doesn't run Streamlit)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

_APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
_STREAMLIT_CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def test_dashboard_module_imports() -> None:
    # Run the module body under a different __name__ so it never calls st.* (main()).
    spec = importlib.util.spec_from_file_location("wc_dashboard", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "_match_predictor_section")
    assert not hasattr(module, "_predicted_card_html")
    assert not hasattr(module, "_bar_html")
    assert not hasattr(module, "_pending_card_html")


def test_dashboard_renders_bracket_with_html_cards() -> None:
    # The matplotlib/plotly bracket images were replaced by a native
    # HTML/CSS bracket of linked match cards, rendered via one st.markdown.
    source = _APP.read_text()
    assert "_bracket_html" in source
    assert "render_bracket_mirrored" not in source
    assert "st.pyplot" not in source
    assert "bracket_plotly_figure" not in source


def _load_module(name: str = "wc_dashboard_cards"):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_advance_label_follows_favored_side() -> None:
    module = _load_module()
    # Away favored: P(home advances)=0.30 -> label should show away team at 70%.
    away_fav = module._advance_label("Nigeria", "Argentina", 0.30)
    assert away_fav == "Argentina 70%"
    assert "Nigeria" not in away_fav
    # Home favored: P(home advances)=0.85 -> shows home team at 85%.
    home_fav = module._advance_label("Colombia", "Ghana", 0.85)
    assert home_fav == "Colombia 85%"


def test_bracket_round_order_shape_and_adjacency() -> None:
    module = _load_module("wc_dashboard_round_order")
    rounds = module._bracket_round_order()
    assert [len(r) for r in rounds] == [16, 8, 4, 2, 1]
    assert rounds[-1] == [104]
    # M74 and M77 are the children of M89 (first R16 slot); the depth-first
    # walk from the final must place them adjacent, first in the R32 list.
    assert rounds[0][0] == 74
    assert rounds[0][1] == 77
    from worldcup.simulation.bracket import KNOCKOUT_BRACKET

    assert KNOCKOUT_BRACKET[89] == (("MW", 74), ("MW", 77))
    # 103 (third place) must not appear anywhere in the ordering.
    assert all(103 not in r for r in rounds)


def test_host_for_cancels_when_both_teams_are_hosts() -> None:
    spec = importlib.util.spec_from_file_location("wc_dashboard_host", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = {"United States", "Mexico", "Canada"}
    assert module._host_for("United States", "Bosnia", hosts) == "United States"
    assert module._host_for("Bosnia", "Mexico", hosts) == "Mexico"
    # Two hosts cancel out (neutral), matching the tournament's net advantage.
    assert module._host_for("United States", "Mexico", hosts) is None
    assert module._host_for("Brazil", "France", hosts) is None


def test_eliminated_returns_teams_with_zero_champion_probability() -> None:
    module = _load_module("wc_dashboard_eliminated")
    probabilities = {
        "Argentina": {"champion": 0.12},
        "Nigeria": {"champion": 0.0},
        "Bosnia": {"champion": 0.0},
        "Brazil": {"champion": 0.08},
    }
    assert module._eliminated(probabilities) == {"Nigeria", "Bosnia"}


def test_eliminated_treats_missing_champion_key_as_eliminated() -> None:
    module = _load_module("wc_dashboard_eliminated_missing_key")
    probabilities = {"Argentina": {"champion": 0.12}, "Nigeria": {}}
    assert module._eliminated(probabilities) == {"Nigeria"}


def test_fmt_prob_floors_tiny_nonzero_values() -> None:
    module = _load_module("wc_dashboard_fmt_prob")
    assert module._fmt_prob(0.0) == "0.0%"
    assert module._fmt_prob(0.0001) == "<0.1%"
    assert module._fmt_prob(0.0009) == "<0.1%"
    assert module._fmt_prob(0.001) == "0.1%"
    assert module._fmt_prob(0.123) == "12.3%"


def _synthetic_bracket() -> dict[str, dict]:
    return {
        "90": {
            "status": "finished",
            "kickoff": "2026-07-01T19:00:00+00:00",
        },
        "91": {
            "status": "scheduled",
            "kickoff": "2026-07-05T23:00:00+00:00",
        },
        "92": {
            "status": "scheduled",
            "kickoff": "2026-07-03T15:00:00+00:00",
        },
        "93": {
            "status": "tbd",
            "kickoff": None,
        },
    }


def test_state_strip_stats_on_synthetic_input() -> None:
    module = _load_module("wc_dashboard_state_strip")
    probabilities = {
        "Argentina": {"champion": 0.30},
        "Brazil": {"champion": 0.12},
        "Nigeria": {"champion": 0.0},
    }
    stats = module._state_strip_stats(probabilities, _synthetic_bracket())
    assert stats["favorite"] == "Argentina"
    assert stats["favorite_odds"] == "30.0%"
    assert stats["alive"] == "47"
    assert stats["played"] == "1"
    assert stats["next_kickoff"] == "Jul 3, 15:00 UTC"


def test_state_strip_stats_no_scheduled_ties_gives_dash() -> None:
    module = _load_module("wc_dashboard_state_strip_no_sched")
    probabilities = {"Argentina": {"champion": 1.0}}
    bracket = {"90": {"status": "finished", "kickoff": "2026-07-01T19:00:00+00:00"}}
    stats = module._state_strip_stats(probabilities, bracket)
    assert stats["next_kickoff"] == "—"


def test_state_strip_stats_no_alive_teams_gives_dash_favorite() -> None:
    module = _load_module("wc_dashboard_state_strip_none_alive")
    probabilities = {"Argentina": {"champion": 0.0}}
    stats = module._state_strip_stats(probabilities, {})
    assert stats["favorite"] == "—"


def test_state_strip_stats_ignores_none_kickoff_among_scheduled_ties() -> None:
    # A scheduled tie with kickoff=None (undated fixture in the committed
    # artifact) must not crash min(...); the next kickoff should come from
    # the ties that do have a date.
    module = _load_module("wc_dashboard_state_strip_none_kickoff")
    probabilities = {"Argentina": {"champion": 0.30}}
    bracket = {
        "90": {"status": "scheduled", "kickoff": None},
        "91": {"status": "scheduled", "kickoff": "2026-07-05T23:00:00+00:00"},
    }
    stats = module._state_strip_stats(probabilities, bracket)
    assert stats["next_kickoff"] == "Jul 5, 23:00 UTC"


def test_state_strip_stats_all_none_kickoffs_gives_dash() -> None:
    module = _load_module("wc_dashboard_state_strip_all_none_kickoff")
    probabilities = {"Argentina": {"champion": 1.0}}
    bracket = {"90": {"status": "scheduled", "kickoff": None}}
    stats = module._state_strip_stats(probabilities, bracket)
    assert stats["next_kickoff"] == "—"


def test_scheduled_hover_handles_none_kickoff() -> None:
    module = _load_module("wc_dashboard_scheduled_hover_none_kickoff")
    tie = {"stage": "Round of 16", "home": "Brazil", "away": "Japan", "kickoff": None}
    card = {"home": 0.55, "draw": 0.20, "away": 0.25}
    hover = module._scheduled_hover(tie, card, "Brazil 70%")
    assert "date TBD" in hover
    assert "Brazil" in hover
    assert "advances: Brazil 70%" in hover


def test_card_header_scheduled_without_kickoff_keeps_tbd_pill() -> None:
    # The resolver pairs a tie as soon as its feeders finish, which can be
    # before the feed publishes the kickoff. The card must keep its header
    # strip (with a TBD pill) instead of rendering headerless.
    module = _load_module("wc_dashboard_header_none_kickoff")
    row = {
        "status": "scheduled",
        "kickoff": None,
        "pen_home": None,
        "et_home": None,
    }
    header = module._card_header(row)
    assert 'class="bkt-head"' in header
    assert '<span class="bkt-pill">TBD</span>' in header


def test_streamlit_config_has_exact_primary_color() -> None:
    with _STREAMLIT_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    assert config["theme"]["base"] == "dark"
    assert config["theme"]["primaryColor"] == "#5BA3E8"


def _synthetic_round_order() -> list[list[int]]:
    """Match numbers shaped like ``_bracket_round_order()``: 16/8/4/2/1."""
    r32 = list(range(1, 17))
    r16 = list(range(17, 25))
    qf = list(range(25, 29))
    sf = list(range(29, 31))
    final = [31]
    return [r32, r16, qf, sf, final]


def _synthetic_rows() -> dict[int, dict]:
    """One finished tie, one scheduled tie, and the rest TBD (31 matches)."""
    round_order = _synthetic_round_order()
    all_ids = [m for round_ids in round_order for m in round_ids]
    rows: dict[int, dict] = {}
    for match_id in all_ids:
        rows[match_id] = {
            "home": None,
            "away": None,
            "winner": None,
            "annotation": None,
            "highlight": None,
            "hover": None,
            "status": "tbd",
            "kickoff": None,
            "ft_home": None,
            "ft_away": None,
            "pen_home": None,
            "pen_away": None,
            "et_home": None,
            "et_away": None,
        }
    rows[1] = {
        "home": "Argentina",
        "away": "Nigeria",
        "winner": "Argentina",
        "annotation": None,
        "highlight": "Argentina",
        "hover": "Round of 32 — final: Argentina 2–0 Nigeria",
        "status": "finished",
        "kickoff": "2026-06-29T19:00:00+00:00",
        "ft_home": 2,
        "ft_away": 0,
        "pen_home": None,
        "pen_away": None,
        "et_home": None,
        "et_away": None,
    }
    rows[2] = {
        "home": "Brazil",
        "away": "Ghana",
        "winner": None,
        "annotation": "Brazil 68%",
        "highlight": "Brazil",
        "hover": "Round of 32 — 2026-07-05<br>Brazil 55% · draw 20% · "
        "Ghana 25%<br>advances: Brazil 68%",
        "status": "scheduled",
        "kickoff": "2026-07-05T19:00:00+00:00",
        "ft_home": None,
        "ft_away": None,
        "pen_home": None,
        "pen_away": None,
        "et_home": None,
        "et_away": None,
    }
    rows[3] = {
        "home": "Germany",
        "away": "Paraguay",
        "winner": "Paraguay",
        "annotation": None,
        "highlight": "Paraguay",
        "hover": "Round of 32 — final: Germany 1–1 (3–4 p) Paraguay",
        "status": "finished",
        "kickoff": "2026-06-29T23:00:00+00:00",
        "ft_home": 1,
        "ft_away": 1,
        "pen_home": 3,
        "pen_away": 4,
        "et_home": 1,
        "et_away": 1,
    }
    rows[4] = {
        "home": "Belgium",
        "away": "Senegal",
        "winner": "Belgium",
        "annotation": None,
        "highlight": "Belgium",
        "hover": "Round of 32 — final: Belgium 3–2 Senegal",
        "status": "finished",
        "kickoff": "2026-07-01T19:00:00+00:00",
        "ft_home": 2,
        "ft_away": 2,
        "pen_home": None,
        "pen_away": None,
        "et_home": 3,
        "et_away": 2,
    }
    return rows


def test_bracket_html_renders_all_31_cards() -> None:
    module = _load_module("wc_dashboard_bracket_html_count")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    body = html_out.split("</style>", 1)[1]
    assert body.count('class="bkt-card') == 31


def test_bracket_html_bolds_winner_and_favorite_in_accent_span() -> None:
    module = _load_module("wc_dashboard_bracket_html_winner")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert '<span class="bkt-team bkt-team--win">Argentina</span>' in html_out
    assert '<span class="bkt-team bkt-team--win">Brazil</span>' in html_out
    assert '<span class="bkt-team">Nigeria</span>' in html_out


def test_bracket_html_tbd_card_has_em_dashes() -> None:
    module = _load_module("wc_dashboard_bracket_html_tbd")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert "&mdash;" in html_out
    assert "bkt-card--tbd" in html_out


def test_bracket_html_scheduled_tie_shows_advance_chip() -> None:
    module = _load_module("wc_dashboard_bracket_html_advance_chip")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert "Brazil 68%" in html_out
    assert 'bkt-chip bkt-chip--advance">Brazil 68%' in html_out


def test_bracket_html_finished_tie_shows_per_team_scores() -> None:
    # Scoreboard style: each team row carries its own score, winner's in accent.
    module = _load_module("wc_dashboard_bracket_html_scores")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert '<span class="bkt-score bkt-score--win">2</span>' in html_out
    assert '<span class="bkt-score">0</span>' in html_out
    assert 'bkt-pill">FT</span>' in html_out


def test_bracket_html_shootout_shows_per_team_penalties_and_ft_p_pill() -> None:
    # A shootout tie reads like a scoreboard: "1 (3)" vs "1 (4)", pill "FT (P)".
    module = _load_module("wc_dashboard_bracket_html_pens")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert '<span class="bkt-score">1 (3)</span>' in html_out
    assert '<span class="bkt-score bkt-score--win">1 (4)</span>' in html_out
    assert "FT (P)" in html_out


def test_bracket_html_green_chip_only_on_scheduled_tie() -> None:
    # The advance chip (scheduled tie) is the only green element in the app;
    # the score chip (finished tie) must not use the green advance class.
    module = _load_module("wc_dashboard_bracket_html_green_chip")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert 'bkt-chip bkt-chip--advance">Brazil 68%' in html_out
    assert "#4ADBA0" in html_out
    # The advance chip is the only chip left: finished ties carry no chip at all.
    body = html_out.split("</style>", 1)[1]
    assert body.count('class="bkt-chip') == 1


def test_bracket_html_carries_hover_text_in_title_attribute() -> None:
    module = _load_module("wc_dashboard_bracket_html_title_attr")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert 'title="Round of 32 — final: Argentina 2–0 Nigeria"' in html_out
    assert "<br>" not in html_out.split("</style>", 1)[1]


def test_bracket_html_escapes_hover_text() -> None:
    module = _load_module("wc_dashboard_bracket_html_escape")
    rows = _synthetic_rows()
    rows[1]["hover"] = 'Stage <script>alert("x")</script>'
    html_out = module._bracket_html(rows, _synthetic_round_order())
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_bracket_html_has_one_style_block_scoped_under_bkt() -> None:
    module = _load_module("wc_dashboard_bracket_html_style_scope")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count("<style>") == 1
    assert ".bkt" in html_out
    assert "overflow-x:auto" in html_out


def test_bracket_html_has_fixed_row_variable_and_min_height_cards() -> None:
    # The fixed slot-height rhythm this bracket's alignment depends on: one
    # shared --row variable, and a card min-height so TBD/finished/scheduled
    # cards (chip vs no chip) don't drift a slot's content off its center.
    module = _load_module("wc_dashboard_bracket_html_row_var")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert "--row:" in html_out
    assert "min-height" in html_out


def test_bracket_html_slot_counts_per_round_match_8_4_2_1_both_sides() -> None:
    # Each round column is a flex column of equal-height .bkt-slot wrappers;
    # left+right together give 16/8/4/2/1 (i.e. 8+8, 4+4, 2+2, 1+1, 1) since
    # a parent's slot center is the midpoint of its two children's centers
    # only when both sides carry the same fixed per-round slot count.
    module = _load_module("wc_dashboard_bracket_html_slot_counts")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count('class="bkt-slot bkt-slot--r32"') == 16
    assert html_out.count('class="bkt-slot bkt-slot--r16"') == 8
    assert html_out.count('class="bkt-slot bkt-slot--qf"') == 4
    assert html_out.count('class="bkt-slot bkt-slot--sf"') == 2
    assert html_out.count('class="bkt-slot bkt-slot--final"') == 1


def test_bracket_html_has_eight_connector_columns_split_left_and_right() -> None:
    # 9 card columns (R32,R16,QF,SF,Final,SF,QF,R16,R32) need 8 connector
    # columns between them, split evenly across the mirrored halves.
    module = _load_module("wc_dashboard_bracket_html_connector_columns")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count('class="bkt-conn bkt-conn--left"') == 4
    assert html_out.count('class="bkt-conn bkt-conn--right"') == 4


def test_bracket_html_has_sixteen_elbow_connectors() -> None:
    # Per side: connectors after R32/R16/QF/SF carry 4+2+1+1 = 8 elbow slots
    # (one per parent-round card); mirrored on the right gives 16 total.
    module = _load_module("wc_dashboard_bracket_html_elbow_count")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count('<div class="bkt-elbow bkt-elbow--') == 16


def test_fmt_updated_humanizes_artifact_timestamp() -> None:
    module = _load_module("wc_dashboard_fmt_updated")
    assert module._fmt_updated("20260703t0728") == "Jul 3, 07:28 UTC"
    # Unknown formats pass through instead of crashing the page.
    assert module._fmt_updated("weird") == "weird"


def test_score_label_appends_penalties_for_shootout_ties() -> None:
    module = _load_module("wc_dashboard_score_label")
    shootout = {"ft_home": 1, "ft_away": 1, "pen_home": 3, "pen_away": 4}
    assert module._score_label(shootout) == "1–1 (3–4 p)"
    regular = {"ft_home": 2, "ft_away": 0}
    assert module._score_label(regular) == "2–0"


def test_bracket_html_extra_time_tie_shows_aet_score_and_pill() -> None:
    # An extra-time win displays the 120' score with an AET pill, not the 90'
    # draw (Belgium 2-2 Senegal after 90', 3-2 after extra time).
    module = _load_module("wc_dashboard_bracket_html_aet")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert '<span class="bkt-score bkt-score--win">3</span>' in html_out
    assert 'bkt-pill">AET</span>' in html_out


def test_score_label_uses_extra_time_score_when_present() -> None:
    module = _load_module("wc_dashboard_score_label_et")
    aet = {"ft_home": 2, "ft_away": 2, "et_home": 3, "et_away": 2}
    assert module._score_label(aet) == "3–2"


def test_bracket_html_final_connectors_are_straight_lines() -> None:
    # The Final's feeders sit on OPPOSITE sides of its column, so the two-child
    # elbow shape is wrong there: both Final-adjacent connectors must render as
    # plain horizontal lines instead.
    module = _load_module("wc_dashboard_bracket_html_final_line")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count("bkt-elbow--line") >= 3  # CSS rule + one per side
    body = html_out.split("</style>", 1)[1]
    assert body.count("bkt-elbow--line") == 2
