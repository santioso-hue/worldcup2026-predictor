"""Export de figuras a PNG determinista en ``outputs/figures/``.

Fija el tamaño en píxeles desde un ``ExportSpec`` (1080×1920 / 1920×1080), backend Agg
(headless). El nombre es determinista (``<name>.png``). matplotlib en import perezoso.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .theme import ExportSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DEFAULT_OUTDIR = Path("outputs/figures")


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
