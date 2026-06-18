"""Deterministic seeding from a single root seed via numpy ``SeedSequence``.

One root seed spawns independent child seeds for Python's ``random``, NumPy, PyTorch,
and TextAttack (``TEXTATTACK_RANDOM_SEED``), so every run is reproducible and no
component shares a stream (SPEC.md §12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    """Independent per-component seeds derived from one root seed.

    Attributes:
        python: Seed for the standard-library ``random`` module.
        numpy: Seed for NumPy's default RNG.
        torch: Seed for PyTorch (CPU and MPS).
        textattack: Seed exported as ``TEXTATTACK_RANDOM_SEED``.
    """

    python: int
    numpy: int
    torch: int
    textattack: int


def derive_seeds(root_seed: int) -> SeedBundle:
    """Derive independent per-component seeds from a single root seed.

    Uses ``numpy.random.SeedSequence`` spawning so the child streams are
    statistically independent and fully reproducible from ``root_seed``.

    Args:
        root_seed: The one seed that determines the entire run.

    Returns:
        A :class:`SeedBundle` of statistically independent child seeds.
    """
    children = np.random.SeedSequence(root_seed).spawn(4)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return SeedBundle(python=values[0], numpy=values[1], torch=values[2], textattack=values[3])
