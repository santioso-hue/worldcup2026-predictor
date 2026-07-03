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
    assert stats["next_kickoff"] == "Jul 3"


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
    assert stats["next_kickoff"] == "Jul 5"


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


def test_streamlit_config_has_exact_primary_color() -> None:
    with _STREAMLIT_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    assert config["theme"]["primaryColor"] == "#185FA5"


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
        }
    rows[1] = {
        "home": "Argentina",
        "away": "Nigeria",
        "winner": "Argentina",
        "annotation": "2–0",
        "highlight": "Argentina",
        "hover": "Round of 32 — final: Argentina 2–0 Nigeria",
    }
    rows[2] = {
        "home": "Brazil",
        "away": "Ghana",
        "winner": None,
        "annotation": "Brazil 68%",
        "highlight": "Brazil",
        "hover": "Round of 32 — 2026-07-05<br>Brazil 55% · draw 20% · "
        "Ghana 25%<br>advances: Brazil 68%",
    }
    return rows


def test_bracket_html_renders_all_31_cards() -> None:
    module = _load_module("wc_dashboard_bracket_html_count")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert html_out.count("bkt-card") >= 31 * 2  # class + hover-open tag, at least


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


def test_bracket_html_finished_tie_shows_score_chip() -> None:
    module = _load_module("wc_dashboard_bracket_html_score_chip")
    html_out = module._bracket_html(_synthetic_rows(), _synthetic_round_order())
    assert 'bkt-chip bkt-chip--score">2–0' in html_out


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
