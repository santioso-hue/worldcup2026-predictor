"""Smoke test del dashboard: el módulo importa limpio (no ejecuta Streamlit)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


def test_dashboard_module_imports() -> None:
    # Ejecuta el cuerpo bajo otro __name__, así no llama a st.* (main()).
    spec = importlib.util.spec_from_file_location("wc_dashboard", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
    assert hasattr(module, "_match_predictor_section")


def test_predicted_card_advance_follows_favored_side() -> None:
    spec = importlib.util.spec_from_file_location("wc_dashboard_cards", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Visitante favorito: P(home avanza)=0.30 -> el label debe mostrar al visitante 70%.
    away_fav = module._predicted_card_html(
        "Round of 32",
        "2026-07-04",
        "Nigeria",
        "Argentina",
        {"home": 0.11, "draw": 0.19, "away": 0.70, "advance": 0.30},
    )
    after_label = away_fav.split("Avanza:")[1]
    assert "Argentina 70%" in after_label
    assert "Nigeria" not in after_label
    # Local favorito: P(home avanza)=0.85 -> muestra al local 85%.
    home_fav = module._predicted_card_html(
        "Round of 32",
        "2026-07-04",
        "Colombia",
        "Ghana",
        {"home": 0.66, "draw": 0.23, "away": 0.11, "advance": 0.85},
    )
    assert "Colombia 85%" in home_fav.split("Avanza:")[1]


def test_host_for_cancels_when_both_teams_are_hosts() -> None:
    spec = importlib.util.spec_from_file_location("wc_dashboard_host", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hosts = {"United States", "Mexico", "Canada"}
    assert module._host_for("United States", "Bosnia", hosts) == "United States"
    assert module._host_for("Bosnia", "Mexico", hosts) == "Mexico"
    # Dos sedes se cancelan (neutral), como la ventaja neta del torneo.
    assert module._host_for("United States", "Mexico", hosts) is None
    assert module._host_for("Brazil", "France", hosts) is None
