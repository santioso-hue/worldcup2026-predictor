"""Test de export: save_figure escribe un PNG del tamaño en píxeles del spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldcup.viz.charts import (
    animate_ranking,
    prepare_champion_ranking,
    render_champion_ranking,
)
from worldcup.viz.export import save_figure
from worldcup.viz.theme import PORTRAIT

_SNAPSHOTS = [
    {"A": 0.5, "B": 0.3, "C": 0.2},
    {"A": 0.4, "B": 0.4, "C": 0.2},
    {"A": 0.3, "B": 0.5, "C": 0.2},
]


def test_save_figure_writes_png_of_spec_size(tmp_path: Path) -> None:
    image = pytest.importorskip("PIL.Image")
    rows = prepare_champion_ranking({"A": 0.5, "B": 0.5})
    fig = render_champion_ranking(rows)
    path = save_figure(fig, "ranking_test", PORTRAIT, outdir=tmp_path)
    assert path.exists() and path.stat().st_size > 0
    with image.open(path) as img:
        assert img.size == (PORTRAIT.width_px, PORTRAIT.height_px)


def test_animate_ranking_writes_gif(tmp_path: Path) -> None:
    pytest.importorskip("PIL")  # PillowWriter -> GIF, sin ffmpeg
    path = animate_ranking(
        _SNAPSHOTS, fmt="gif", outdir=tmp_path, name="drumroll", top_n=3
    )
    assert path.exists() and path.stat().st_size > 0
    assert path.suffix == ".gif"


def test_animate_ranking_writes_mp4_if_ffmpeg(tmp_path: Path) -> None:
    from matplotlib.animation import FFMpegWriter

    if not FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg no disponible")
    path = animate_ranking(
        _SNAPSHOTS, fmt="mp4", outdir=tmp_path, name="drumroll", top_n=3
    )
    assert path.exists() and path.suffix == ".mp4"
