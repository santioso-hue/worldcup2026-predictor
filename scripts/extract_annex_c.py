"""Extract and verify the Annex C table (best third-place teams) from the FIFA regs PDF.

Annex C of the FIFA World Cup 2026 Regulations maps each of the 495 combinations
of which groups produce the 8 best third-place teams to how those teams get
assigned against the 8 group winners in the Round of 32. It can't be
reconstructed from public data alone (slot constraints allow 3-214 assignments
per combination), so we extract it directly from the official source.

Usage (one-off, requires ``pip install pypdf``):

    python scripts/extract_annex_c.py <regulations.pdf> data/raw/annex_c_2026.json

Source: FIFA World Cup 2026 Regulations, Annexe C
(https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf).
The JSON output (checked into version control) is the durable artifact; this
script documents its provenance and lets us re-derive it. It verifies the
extraction and exits non-zero on failure.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

# Column order of the Annex C header (group winners).
WINNERS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
# Per-slot eligibility from Art. 12.6 (independent of the table) — for cross-check.
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


# Anchor: the official row 1 of Annex C (3E 3J 3I 3F 3H 3G 3L 3K) in WINNERS order.
# Catches a column-scramble that still respects eligibility (which the structural
# invariants alone would NOT catch — many eligible assignments exist per combination).
ANCHOR_ROW_1 = ["E", "J", "I", "F", "H", "G", "L", "K"]


def extract_rows(pdf_path: Path) -> dict[int, list[str]]:
    """Extract ``{rank: [8 third-place teams in WINNERS order]}`` from the PDF."""
    from pypdf import PdfReader

    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf_path)).pages)
    rows: dict[int, list[str]] = {}
    for m in _ROW_RE.finditer(text):
        rows[int(m.group(1))] = [m.group(i) for i in range(2, 10)]
    return rows


def build_table(rows: dict[int, list[str]]) -> dict[str, dict[str, str]]:
    """Build ``{sorted_combo: {winner: group}}`` from the rows."""
    return {
        "".join(sorted(g)): dict(zip(WINNERS, g, strict=True)) for g in rows.values()
    }


def verify(rows: dict[int, list[str]], table: dict[str, dict[str, str]]) -> list[str]:
    """Return the list of problems found (empty if the extraction is correct).

    Checks that ranks are 1..495 contiguous (catches duplicate/missing rows even
    if coverage happens to add up), the row-1 anchor, and the structural
    invariants (coverage, bijection, eligibility).
    """
    issues: list[str] = []
    if set(rows) != set(range(1, 496)):
        issues.append("ranks are not 1..495 contiguous")
    if rows.get(1) != ANCHOR_ROW_1:
        issues.append(f"row 1 {rows.get(1)} != official anchor {ANCHOR_ROW_1}")
    if len(table) != 495:
        issues.append(f"{len(table)} entries != 495")
    expected = {"".join(c) for c in combinations("ABCDEFGHIJKL", 8)}
    if set(table) != expected:
        issues.append("does not cover exactly the C(12,8)=495 combinations")
    for combo, assign in table.items():
        if set(assign) != set(WINNERS):
            issues.append(f"{combo}: winners != {WINNERS}")
        if set(assign.values()) != set(combo):
            issues.append(f"{combo}: assigned third-place teams != combination")
        for winner, group in assign.items():
            if group not in ELIGIBLE[winner]:
                issues.append(f"{combo}: {winner} -> 3{group} violates eligibility")
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
        print(f"Verification FAILED ({len(issues)} issues):")
        for msg in issues[:10]:
            print(f"  - {msg}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, sort_keys=True, indent=0), encoding="utf-8")
    print(f"OK: 495 combinations verified -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
