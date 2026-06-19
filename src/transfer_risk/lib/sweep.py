"""Pure helpers for fanning the attack sweep into per-(surrogate, recipe, shard) units.

The attacks pipeline generates one Kedro node per unit (and one reduce node per cell); this
module enumerates the units and their dataset ids with no I/O, so the fan-out is unit-tested.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def shard_spans(n: int, shard_size: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` spans of at most ``shard_size`` covering ``range(n)``.

    Args:
        n: number of examples to cover.
        shard_size: maximum examples per span (coerced to ``>= 1``).

    Returns:
        Ordered ``(start, stop)`` spans whose union is ``range(n)`` with no gaps or overlaps.
    """
    step = max(1, shard_size)
    return [(start, min(start + step, n)) for start in range(0, n, step)]


def cell_key(surrogate: str, recipe: str) -> str:
    """Return the ``"<surrogate>__<recipe>"`` id of a completed-cell partition."""
    return f"{surrogate}__{recipe}"


def shard_key(surrogate: str, recipe: str, start: int) -> str:
    """Return the ``"<surrogate>__<recipe>__<start>"`` id of a shard partition."""
    return f"{surrogate}__{recipe}__{start}"


def auto_shard_size(n_examples: int, n_cells: int, cores: int, multiple: int = 2) -> int:
    """Shard size that fans the sweep out to ``~multiple * cores`` shards across ``n_cells`` cells.

    The attacks pipeline parallelises across nodes, so the sweep wants roughly a small multiple of
    the run box's cores in total shards — enough to spread an uneven (slow) cell across cores
    without the node-count blow-up of a fixed tiny shard size. Each cell is split into the same
    number of shards, ``round(multiple * cores / n_cells)`` (at least one), and the shard size is
    the eval set divided over that many shards.

    Args:
        n_examples: eval-set size attacked per cell.
        n_cells: number of ``(surrogate, recipe)`` cells.
        cores: target parallelism (the run box's vCPU count).
        multiple: target total shards as a multiple of ``cores`` (default 2).

    Returns:
        The per-cell shard size (``>= 1``); equals ``n_examples`` (one shard per cell) when the
        target collapses to a single shard, and is clamped to ``>= 1``.
    """
    if n_examples <= 0:
        return 1
    if n_cells <= 0 or cores <= 0 or multiple <= 0:
        return n_examples
    shards_per_cell = max(1, round(multiple * cores / n_cells))
    return max(1, ceil(n_examples / shards_per_cell))


def attack_units(
    surrogates: Sequence[str],
    recipes: Sequence[str],
    n_examples: int,
    shard_size: int,
) -> list[tuple[str, str, int, int]]:
    """Enumerate every ``(surrogate, recipe, start, stop)`` attack unit for the sweep.

    Args:
        surrogates: the surrogate names attacked (the whole pool).
        recipes: the TextAttack recipe keys.
        n_examples: the eval-set size each cell is attacked over.
        shard_size: examples per shard.

    Returns:
        One ``(surrogate, recipe, start, stop)`` tuple per shard, ordered by surrogate, then
        recipe, then shard start.
    """
    spans = shard_spans(n_examples, shard_size)
    return [
        (surrogate, recipe, start, stop)
        for surrogate in surrogates
        for recipe in recipes
        for start, stop in spans
    ]
