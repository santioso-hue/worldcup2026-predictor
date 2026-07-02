"""Visual theme: single source of truth for brand (palette, type, sizes, export).

Don't hardcode colors/fonts/sizes in other ``viz`` modules; import them from here.
High contrast, readable on mobile, single accent color (no per-team colors).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportSpec:
    """Export size in pixels (with dpi for matplotlib)."""

    width_px: int
    height_px: int
    dpi: int = 150


PORTRAIT = ExportSpec(1080, 1920)  # vertical
LANDSCAPE = ExportSpec(1920, 1080)  # horizontal


@dataclass(frozen=True)
class Theme:
    """Brand palette, typography, and sizes."""

    accent: str = "#185FA5"  # main blue (bars, ranking)
    background: str = "#FFFFFF"
    text_primary: str = "#1A1A1A"
    text_muted: str = "#6B6B6B"
    grid: str = "#E6E6E6"
    up: str = "#1D9E75"  # delta up
    down: str = "#D85A30"  # delta down
    draw: str = "#888780"  # draw (1X2 bar)
    away: str = "#D85A30"  # away (1X2 bar)
    heat: str = "#1D9E75"  # base color for the scoreline heatmap
    font_family: str = "DejaVu Sans"  # always available in matplotlib (deterministic)
    title_size: int = 34
    label_size: int = 22
    value_size: int = 22
    stamp_size: int = 16


THEME = Theme()
