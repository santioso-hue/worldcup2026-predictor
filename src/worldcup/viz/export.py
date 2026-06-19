"""Export de figuras a PNG determinista en ``outputs/figures/``.

Fija el tamaño en píxeles desde un ``ExportSpec`` (1080×1920 / 1920×1080), backend Agg
(headless). El nombre es determinista (``<name>.png``). matplotlib en import perezoso.
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
    """Guarda ``fig`` como ``<outdir>/<name>.png`` al tamaño en píxeles de ``spec``.

    El tamaño resultante es ``width_px × height_px`` (figsize en pulgadas × dpi).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(spec.width_px / spec.dpi, spec.height_px / spec.dpi)
    FigureCanvasAgg(fig)  # backend headless explícito
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
    """Anima ``figure`` (``update(i)`` dibuja el frame ``i``) y la guarda como vídeo.

    ``fmt='mp4'`` usa ffmpeg; ``fmt='gif'`` usa Pillow (sin dependencia externa).
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
        raise ValueError(f"formato no soportado: {fmt!r}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)  # backend headless explícito
    animation = FuncAnimation(figure, update, frames=frames)
    path = out / f"{name}.{ext}"
    animation.save(path, writer=writer)
    return path
