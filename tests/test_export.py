"""Test de export: save_figure escribe un PNG del tamaño en píxeles del spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldcup.viz.charts import prepare_champion_ranking, render_champion_ranking
from worldcup.viz.export import save_figure
from worldcup.viz.theme import PORTRAIT


def test_save_figure_writes_png_of_spec_size(tmp_path: Path) -> None:
    image = pytest.importorskip("PIL.Image")
    rows = prepare_champion_ranking({"A": 0.5, "B": 0.5})
    fig = render_champion_ranking(rows)
    path = save_figure(fig, "ranking_test", PORTRAIT, outdir=tmp_path)
    assert path.exists() and path.stat().st_size > 0
    with image.open(path) as img:
        assert img.size == (PORTRAIT.width_px, PORTRAIT.height_px)
