"""Tema visual: única fuente de marca (paleta, tipografía, tamaños, export).

No hardcodees colores/fuentes/tamaños en otros módulos de ``viz``; impórtalos de aquí.
Alto contraste, legible en móvil, acento único (sin colores por selección).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportSpec:
    """Tamaño de exportación en píxeles (con dpi para matplotlib)."""

    width_px: int
    height_px: int
    dpi: int = 150


PORTRAIT = ExportSpec(1080, 1920)  # reel / short vertical
LANDSCAPE = ExportSpec(1920, 1080)  # youtube / thumbnail horizontal


@dataclass(frozen=True)
class Theme:
    """Paleta, tipografía y tamaños de la marca."""

    accent: str = "#185FA5"  # azul principal (barras, ranking)
    background: str = "#FFFFFF"
    text_primary: str = "#1A1A1A"
    text_muted: str = "#6B6B6B"
    grid: str = "#E6E6E6"
    up: str = "#1D9E75"  # delta ↑
    down: str = "#D85A30"  # delta ↓
    draw: str = "#888780"  # empate (barra 1X2)
    away: str = "#D85A30"  # visita (barra 1X2)
    heat: str = "#1D9E75"  # base del heatmap de marcadores
    font_family: str = "DejaVu Sans"  # siempre disponible en matplotlib (determinista)
    title_size: int = 34
    label_size: int = 22
    value_size: int = 22
    stamp_size: int = 16


THEME = Theme()
