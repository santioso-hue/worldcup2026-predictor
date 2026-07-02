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


def test_dashboard_no_longer_uses_matplotlib_bracket_render() -> None:
    # The bracket panel now renders via plotly; the matplotlib mirrored
    # renderer and st.pyplot must no longer be referenced anywhere in the
    # module source (source-grep, like the deleted-helper checks above).
    source = _APP.read_text()
    assert "render_bracket_mirrored" not in source
    assert "st.pyplot" not in source


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
    assert stats["favorite"] == "Argentina (30.0%)"
    assert stats["alive"] == "47"
    assert stats["played"] == "1"
    assert stats["next_kickoff"] == "2026-07-03"


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


def test_streamlit_config_has_exact_primary_color() -> None:
    with _STREAMLIT_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    assert config["theme"]["primaryColor"] == "#185FA5"
