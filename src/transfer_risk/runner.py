"""A ParallelRunner that caps worker processes to fit the attack victims in RAM.

The attack sweep serves every victim from a torch checkpoint (the cloud box's aarch64 onnxruntime
cannot run transformer ONNX attention). Each ``spawn`` worker imports torch + transformers +
textattack and loads one victim model plus the recipe's offline assets — several GB resident. With
Kedro's default of one worker per vCPU, a memory-light box (the c-family is 2 GB/vCPU) exceeds RAM
and the kernel OOM-kills a worker, which surfaces as ``BrokenProcessPool`` and aborts the run. This
runner caps the worker count by available RAM as well as CPU count, so the sweep is bounded by
whichever is scarcer; a high-RAM box (the r-family is 8 GB/vCPU) still uses every core.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from kedro.runner import ParallelRunner

if TYPE_CHECKING:
    from kedro.pipeline import Pipeline

# Resident memory budgeted per worker: the torch + transformers + textattack imports, one
# transformer victim (a deberta-v3-base checkpoint is the high end), and a recipe's offline assets
# (counter-fitted embeddings + sentence encoder). Conservative, to leave headroom for the OS.
_GB_PER_WORKER = 5.0


def ram_bounded_workers(
    cpu_workers: int, ram_gb: float, gb_per_worker: float = _GB_PER_WORKER
) -> int:
    """Cap ``cpu_workers`` so the resident set of that many workers fits in ``ram_gb``.

    Args:
        cpu_workers: Worker count the CPU/pipeline would otherwise allow.
        ram_gb: Total system RAM in GB.
        gb_per_worker: Resident memory budgeted per worker.

    Returns:
        ``min(cpu_workers, floor(ram_gb / gb_per_worker))``, never below 1.
    """
    ram_workers = int(ram_gb / gb_per_worker)
    return max(1, min(cpu_workers, ram_workers))


class RamBoundedParallelRunner(ParallelRunner):
    """``ParallelRunner`` whose worker count is capped by available RAM, not just the CPU count."""

    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        """Cap the CPU-bounded worker count so concurrent torch victims fit in RAM."""
        cpu_workers = super()._get_required_workers_count(pipeline)
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        return ram_bounded_workers(cpu_workers, ram_gb)
