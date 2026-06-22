"""Canonicalización de nombres de selección entre fuentes de datos.

El Elo se entrena sobre martj42 (~49k partidos); ese es el nombre **canónico** del
proyecto. Los proveedores live usan variantes ("Congo DR" vs "DR Congo", "Czechia" vs
"Czech Republic"): si no se traducen, el equipo no encuentra su histórico y cae al
rating por defecto (``initial_rating`` = 1500), distorsionando la predicción.

Mapeamos cada variante **verificada** al nombre de martj42. Solo entradas confirmadas
contra ambas fuentes: nunca adivinamos un alias. Un mapeo erróneo fusionaría dos
selecciones distintas —p.ej. ``"Congo"`` (Rep. del Congo) ≠ ``"DR Congo"``—, un error
peor que el fallback a 1500.
"""

from __future__ import annotations

# football-data.org v4 -> martj42 (base del Elo). Verificado el 22 jun 2026 contra
# ambos feeds (nombre live exacto a la izquierda, nombre martj42 con histórico real).
_FOOTBALLDATA_TO_CANONICAL: dict[str, str] = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def canonical_footballdata_team(name: str) -> str:
    """Traduce un nombre de football-data.org al canónico de martj42.

    Identidad si no hay alias (la mayoría de selecciones coinciden ya).
    """
    return _FOOTBALLDATA_TO_CANONICAL.get(name, name)
