"""Tests del tema visual: tamaños de export y validez de la paleta."""

from __future__ import annotations

import re

from worldcup.viz.theme import LANDSCAPE, PORTRAIT, THEME

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_export_specs() -> None:
    assert (PORTRAIT.width_px, PORTRAIT.height_px) == (1080, 1920)
    assert (LANDSCAPE.width_px, LANDSCAPE.height_px) == (1920, 1080)
    assert PORTRAIT.dpi == LANDSCAPE.dpi == 150


def test_palette_is_valid_hex() -> None:
    colors = [
        THEME.accent,
        THEME.background,
        THEME.text_primary,
        THEME.text_muted,
        THEME.grid,
        THEME.up,
        THEME.down,
        THEME.draw,
        THEME.away,
        THEME.heat,
    ]
    assert all(_HEX.match(c) for c in colors)
