"""Dashboard smoke test: module imports clean (doesn't run Streamlit)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


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
