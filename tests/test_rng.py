"""Tests for the seeded RNG: the project's reproducibility rests on this."""

from __future__ import annotations

import numpy as np

from worldcup.rng import DEFAULT_SEED, get_rng, spawn_rngs


def test_same_seed_is_reproducible() -> None:
    a = get_rng(42).random(5)
    b = get_rng(42).random(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = get_rng(1).random(5)
    b = get_rng(2).random(5)
    assert not np.array_equal(a, b)


def test_default_seed_matches_config() -> None:
    # RNG default must match project.seed in config.yaml.
    assert DEFAULT_SEED == 42


def test_spawn_streams_are_reproducible_and_independent() -> None:
    first = [g.random(3) for g in spawn_rngs(42, 4)]
    second = [g.random(3) for g in spawn_rngs(42, 4)]
    # Reproducible: same (seed, n) -> same streams.
    for x, y in zip(first, second, strict=True):
        assert np.array_equal(x, y)
    # Independent: the 4 streams aren't identical to each other.
    assert not np.array_equal(first[0], first[1])


def test_spawn_negative_raises() -> None:
    # Edge case: negative n is a programming error, not something to swallow.
    try:
        spawn_rngs(42, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("spawn_rngs(-1) should raise ValueError")
