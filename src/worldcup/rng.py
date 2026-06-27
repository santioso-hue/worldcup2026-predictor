"""Central seeded RNG. **Úsalo SIEMPRE**; nunca `random`/`np.random` global.

La reproducibilidad del proyecto depende de que toda la aleatoriedad
de Monte Carlo derive de una única semilla. Para simulaciones en paralelo, usa
:func:`spawn_rngs`, que produce streams independientes y reproducibles vía
``numpy.random.SeedSequence`` (no correlacionados, a diferencia de sembrar a mano
``seed + i``).
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 42


def get_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Devuelve un generador NumPy sembrado y reproducible.

    Parameters
    ----------
    seed:
        Semilla entera. El default coincide con ``project.seed`` de config.yaml.

    Returns
    -------
    numpy.random.Generator
        Generador ``PCG64``. Dos llamadas con la misma semilla producen
        exactamente la misma secuencia.
    """
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Crea ``n`` generadores independientes y reproducibles a partir de una semilla.

    Útil para repartir las ``runs`` de Monte Carlo entre workers sin correlación
    entre streams. El resultado es determinista: misma ``(seed, n)`` -> mismos streams.

    Parameters
    ----------
    seed:
        Semilla raíz.
    n:
        Número de generadores hijos a crear (``n >= 0``).

    Returns
    -------
    list[numpy.random.Generator]
        Lista de ``n`` generadores con estados independientes.
    """
    if n < 0:
        raise ValueError(f"n debe ser >= 0, se recibió {n}")
    root = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in root.spawn(n)]
