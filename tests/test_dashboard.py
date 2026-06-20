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
