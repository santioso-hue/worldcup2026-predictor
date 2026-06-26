"""Tests del bracket de eliminatoria + Annex C (best thirds)."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldcup.simulation.bracket import (
    KNOCKOUT_BRACKET,
    R32_MATCHES,
    THIRD_SLOT_WINNERS,
    assign_best_thirds,
    load_annex_c,
)

ANNEX_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "annex_c_2026.json"


def test_load_annex_c_has_495_entries() -> None:
    assert len(load_annex_c(ANNEX_PATH)) == 495


def test_assign_best_thirds_matches_official_row() -> None:
    annex = load_annex_c(ANNEX_PATH)
    # Los 8 "últimos" grupos -> fila 1 de Annex C (verificada en el PDF).
    assignment = assign_best_thirds(set("EFGHIJKL"), annex)
    assert assignment == {
        "1A": "E",
        "1B": "J",
        "1D": "I",
        "1E": "F",
        "1G": "H",
        "1I": "G",
        "1K": "L",
        "1L": "K",
    }


def test_assign_best_thirds_uses_exactly_the_qualifying_groups() -> None:
    annex = load_annex_c(ANNEX_PATH)
    combo = set("ABCDEFGH")
    assignment = assign_best_thirds(combo, annex)
    assert set(assignment.values()) == combo
    assert set(assignment) == set(THIRD_SLOT_WINNERS)


def test_assign_best_thirds_wrong_size_raises() -> None:
    annex = load_annex_c(ANNEX_PATH)
    with pytest.raises(ValueError):
        assign_best_thirds(set("ABC"), annex)


def test_knockout_bracket_has_32_matches() -> None:
    assert len(KNOCKOUT_BRACKET) == 32  # 16 + 8 + 4 + 2 + 1 (3rd) + 1 (final)
    assert len(R32_MATCHES) == 16


def test_third_slots_are_the_eight_designated_winners() -> None:
    assert set(THIRD_SLOT_WINNERS) == {"1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"}


def test_match_references_point_to_existing_matches() -> None:
    ids = set(KNOCKOUT_BRACKET)
    for slot_a, slot_b in KNOCKOUT_BRACKET.values():
        for kind, ref in (slot_a, slot_b):
            if kind in ("MW", "ML"):  # winner/loser of a prior match
                assert ref in ids
