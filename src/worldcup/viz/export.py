"""Deterministic PNG export to ``outputs/figures/``.

Sets pixel size from an ``ExportSpec`` (1080x1920 / 1920x1080), Agg backend
(headless). Output name is deterministic (``<name>.png``). matplotlib is imported
lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .theme import ExportSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DEFAULT_OUTDIR = Path("outputs/figures")
_DEFAULT_VIDEO_OUTDIR = Path("outputs/videos")


def save_figure(
    fig: Figure,
    name: str,
    spec: ExportSpec,
    *,
    outdir: Path | str = _DEFAULT_OUTDIR,
) -> Path:
    """Save ``fig`` as ``<outdir>/<name>.png`` at ``spec``'s pixel size.

    Resulting size is ``width_px x height_px`` (figsize in inches x dpi).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(spec.width_px / spec.dpi, spec.height_px / spec.dpi)
    FigureCanvasAgg(fig)  # explicit headless backend
    path = out / f"{name}.png"
    fig.savefig(path, dpi=spec.dpi, facecolor=fig.get_facecolor())
    return path


def save_animation(
    figure: Figure,
    update: Callable[[int], None],
    frames: int,
    name: str,
    *,
    fps: int = 2,
    fmt: str = "mp4",
    outdir: Path | str = _DEFAULT_VIDEO_OUTDIR,
) -> Path:
    """Animate ``figure`` (``update(i)`` draws frame ``i``) and save it as a video.

    ``fmt='mp4'`` uses ffmpeg; ``fmt='gif'`` uses Pillow (no external dependency).
    """
    from matplotlib.animation import (
        AbstractMovieWriter,
        FFMpegWriter,
        FuncAnimation,
        PillowWriter,
    )
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    writer: AbstractMovieWriter
    if fmt == "gif":
        writer = PillowWriter(fps=fps)
        ext = "gif"
    elif fmt == "mp4":
        writer = FFMpegWriter(fps=fps)
        ext = "mp4"
    else:
        raise ValueError(f"unsupported format: {fmt!r}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)  # explicit headless backend
    animation = FuncAnimation(figure, update, frames=frames)
    path = out / f"{name}.{ext}"
    animation.save(path, writer=writer)
    return path
