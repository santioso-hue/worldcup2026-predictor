"""Estructura del bracket de eliminatoria + asignación de mejores terceros (Annex C).

Plantilla VERIFICADA del Reglamento FWC2026 (Art. 12.6–12.11). Cada partido de
eliminatoria se describe por dos *slots*; cada slot es una tupla ``(kind, ref)``:

- ``("W", "A")``   -> ganador del grupo A (1A)
- ``("R", "B")``   -> subcampeón del grupo B (2B)
- ``("T", "1A")``  -> mejor tercero asignado al ganador 1A (vía Annex C)
- ``("MW", 74)``   -> ganador del partido M74
- ``("ML", 101)``  -> perdedor del partido M101 (solo para el 3er puesto, M103)

``tournament.py`` resuelve estos slots con las posiciones de grupo y los resultados de
cada ronda. Annex C se carga del JSON generado por ``scripts/extract_annex_c.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- Round of 32 (M73–M88), Art. 12.6 --------------------------------------
R32_MATCHES: dict[int, tuple[tuple[str, object], tuple[str, object]]] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("T", "1E")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("T", "1I")),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("T", "1A")),
    80: (("W", "L"), ("T", "1L")),
    81: (("W", "D"), ("T", "1D")),
    82: (("W", "G"), ("T", "1G")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("T", "1B")),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("T", "1K")),
    88: (("R", "D"), ("R", "G")),
}

# --- Round of 16 (M89–M96), Art. 12.7 --------------------------------------
_R16 = {
    89: (("MW", 74), ("MW", 77)),
    90: (("MW", 73), ("MW", 75)),
    91: (("MW", 76), ("MW", 78)),
    92: (("MW", 79), ("MW", 80)),
    93: (("MW", 83), ("MW", 84)),
    94: (("MW", 81), ("MW", 82)),
    95: (("MW", 86), ("MW", 88)),
    96: (("MW", 85), ("MW", 87)),
}

# --- Quarter-finals / Semi-finals / Third place / Final (Art. 12.8–12.11) --
_QF = {
    97: (("MW", 89), ("MW", 90)),
    98: (("MW", 93), ("MW", 94)),
    99: (("MW", 91), ("MW", 92)),
    100: (("MW", 95), ("MW", 96)),
}
_SF = {101: (("MW", 97), ("MW", 98)), 102: (("MW", 99), ("MW", 100))}
_THIRD = {103: (("ML", 101), ("ML", 102))}
_FINAL = {104: (("MW", 101), ("MW", 102))}

KNOCKOUT_BRACKET: dict[int, tuple[tuple[str, object], tuple[str, object]]] = {
    **R32_MATCHES,
    **_R16,
    **_QF,
    **_SF,
    **_THIRD,
    **_FINAL,
}

FINAL_MATCH = 104

# Los 8 ganadores de grupo que enfrentan a un mejor tercero en R32 (slots "T").
THIRD_SLOT_WINNERS: tuple[str, ...] = tuple(
    str(ref) for slots in R32_MATCHES.values() for kind, ref in slots if kind == "T"
)


def load_annex_c(path: Path | str) -> dict[frozenset[str], dict[str, str]]:
    """Carga Annex C: ``frozenset(8 grupos) -> {ganador: grupo del tercero}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {frozenset(combo): dict(assign) for combo, assign in raw.items()}


def assign_best_thirds(
    qualifying_groups: set[str], annex: dict[frozenset[str], dict[str, str]]
) -> dict[str, str]:
    """Asigna los 8 mejores terceros a los ganadores de grupo según Annex C.

    Parameters
    ----------
    qualifying_groups:
        Conjunto de las 8 letras de grupo cuyos terceros clasificaron.
    annex:
        Tabla cargada con :func:`load_annex_c`.

    Returns
    -------
    dict[str, str]
        ``{ganador (p. ej. "1A"): grupo del tercero que enfrenta}``.

    Raises
    ------
    ValueError
        Si no son exactamente 8 grupos o la combinación no está en Annex C.
    """
    if len(qualifying_groups) != 8:
        raise ValueError(
            f"se esperan 8 grupos clasificados, se recibieron {len(qualifying_groups)}"
        )
    key = frozenset(qualifying_groups)
    if key not in annex:
        raise ValueError(f"combinación ausente de Annex C: {sorted(qualifying_groups)}")
    return dict(annex[key])
