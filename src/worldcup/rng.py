"""Central seeded RNG. **Always use this**, never the global `random`/`np.random`.

The project's reproducibility depends on all Monte Carlo randomness deriving
from a single seed. For parallel simulations, use :func:`spawn_rngs`, which
produces independent, reproducible streams via ``numpy.random.SeedSequence``
(uncorrelated, unlike hand-seeding with ``seed + i``).
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 42


def get_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a seeded, reproducible NumPy generator.

    Parameters
    ----------
    seed:
        Integer seed. The default matches ``project.seed`` in config.yaml.

    Returns
    -------
    numpy.random.Generator
        A ``PCG64`` generator. Two calls with the same seed produce the
        exact same sequence.
    """
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Create ``n`` independent, reproducible generators from a single seed.

    Useful for splitting Monte Carlo runs across workers without stream
    correlation. Deterministic: the same ``(seed, n)`` always yields the
    same streams.

    Parameters
    ----------
    seed:
        Root seed.
    n:
        Number of child generators to create (``n >= 0``).

    Returns
    -------
    list[numpy.random.Generator]
        List of ``n`` generators with independent state.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    root = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in root.spawn(n)]
