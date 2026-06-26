"""Extrae y verifica la tabla Annex C (mejores terceros) del PDF de regulaciones FIFA.

Annex C del Reglamento de la FIFA World Cup 2026 mapea cada una de las 495 combinaciones
de qué grupos producen los 8 mejores terceros a la asignación de esos terceros frente a
los 8 ganadores de grupo en el Round of 32. No es reconstruible desde datos públicos
sueltos (las restricciones de slot admiten 3–214 asignaciones por combinación), así que
se extrae directamente de la fuente oficial.

Uso (one-off, requiere ``pip install pypdf``):

    python scripts/extract_annex_c.py <regulations.pdf> data/raw/annex_c_2026.json

Fuente: FIFA World Cup 2026 Regulations, Annexe C
(https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf).
La salida JSON (versionada) es el artefacto durable; este script documenta su
procedencia y permite re-derivarla; verifica y sale !=0 si falla.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

# Orden de columnas del encabezado de Annex C (ganadores de grupo).
WINNERS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
# Elegibilidad por slot, del Art. 12.6 (independiente de la tabla) — para cross-check.
ELIGIBLE = {
    "1A": set("CEFHI"),
    "1B": set("EFGIJ"),
    "1D": set("BEFIJ"),
    "1E": set("ABCDF"),
    "1G": set("AEHIJ"),
    "1I": set("CDFGH"),
    "1K": set("DEIJL"),
    "1L": set("EHIJK"),
}
_ROW_RE = re.compile(r"\b(\d{1,3})\s+" + r"\s+".join([r"3([A-L])"] * 8))


# Ancla: la fila 1 oficial de Annex C (3E 3J 3I 3F 3H 3G 3L 3K) en orden WINNERS.
# Detecta un column-scramble que respete la elegibilidad (que las invariantes
# estructurales NO detectarían — hay muchas asignaciones elegibles por combinación).
ANCHOR_ROW_1 = ["E", "J", "I", "F", "H", "G", "L", "K"]


def extract_rows(pdf_path: Path) -> dict[int, list[str]]:
    """Extrae ``{rank: [8 terceros en orden WINNERS]}`` del PDF."""
    from pypdf import PdfReader

    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf_path)).pages)
    rows: dict[int, list[str]] = {}
    for m in _ROW_RE.finditer(text):
        rows[int(m.group(1))] = [m.group(i) for i in range(2, 10)]
    return rows


def build_table(rows: dict[int, list[str]]) -> dict[str, dict[str, str]]:
    """Construye ``{combinación_ordenada: {ganador: grupo}}`` desde las filas."""
    return {
        "".join(sorted(g)): dict(zip(WINNERS, g, strict=True)) for g in rows.values()
    }


def verify(rows: dict[int, list[str]], table: dict[str, dict[str, str]]) -> list[str]:
    """Devuelve la lista de problemas (vacía si la extracción es correcta).

    Comprueba ranks 1..495 contiguos (detecta filas duplicadas/perdidas aunque la
    cobertura cuadre por casualidad), el ancla de la fila 1, y las invariantes
    estructurales (cobertura, biyección, elegibilidad).
    """
    issues: list[str] = []
    if set(rows) != set(range(1, 496)):
        issues.append("los ranks no son 1..495 contiguos")
    if rows.get(1) != ANCHOR_ROW_1:
        issues.append(f"fila 1 {rows.get(1)} != ancla oficial {ANCHOR_ROW_1}")
    if len(table) != 495:
        issues.append(f"{len(table)} entradas != 495")
    expected = {"".join(c) for c in combinations("ABCDEFGHIJKL", 8)}
    if set(table) != expected:
        issues.append("no cubre exactamente las C(12,8)=495 combinaciones")
    for combo, assign in table.items():
        if set(assign) != set(WINNERS):
            issues.append(f"{combo}: ganadores != {WINNERS}")
        if set(assign.values()) != set(combo):
            issues.append(f"{combo}: terceros asignados != combinación")
        for winner, group in assign.items():
            if group not in ELIGIBLE[winner]:
                issues.append(f"{combo}: {winner} -> 3{group} viola elegibilidad")
    return issues


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    pdf_path, out_path = Path(argv[1]), Path(argv[2])
    rows = extract_rows(pdf_path)
    table = build_table(rows)
    issues = verify(rows, table)
    if issues:
        print(f"FALLÓ la verificación ({len(issues)} problemas):")
        for msg in issues[:10]:
            print(f"  - {msg}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, sort_keys=True, indent=0), encoding="utf-8")
    print(f"OK: 495 combinaciones verificadas -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
