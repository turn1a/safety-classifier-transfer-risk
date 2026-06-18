"""Project hooks for transfer_risk.

ParallelRunner serialises the *whole* catalog with multiprocessing's ``ForkingPickler`` before
a run (kedro validates every materialised dataset, not just the ones the pipeline uses — see
kedro #3804, closed as won't-fix). kedro-mlflow's ``MlflowArtifactDataset`` builds a dynamic
per-instance subclass that ``ForkingPickler`` cannot pickle ("attribute lookup failed"), so the
attack sweep — which does not even use those datasets — would be blocked under ParallelRunner.
``cloudpickle`` *can* pickle them (by value), so before each run we register a cloudpickle-based
reducer on ``ForkingPickler`` for any materialised dataset it rejects. This keeps the catalog and
the MLflow integration unchanged while letting ParallelRunner serialise it.
"""

from __future__ import annotations

import logging
from multiprocessing.reduction import ForkingPickler
from pickle import PicklingError
from time import perf_counter
from typing import TYPE_CHECKING, Any

import cloudpickle
from kedro.framework.hooks import hook_impl

if TYPE_CHECKING:
    from kedro.io import DataCatalog
    from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)
# Per-phase timing goes to its own logger so conf/logging.yml can tune it (INFO to see every
# load/compute/save, WARNING to silence) without touching the rest of transfer_risk's logging.
_timing = logging.getLogger("transfer_risk.timing")


def _reduce_via_cloudpickle(obj: object) -> tuple[Any, tuple[bytes]]:
    """Reduce ``obj`` through cloudpickle so ``ForkingPickler`` can serialise it.

    Args:
        obj: the dataset (or any object) ForkingPickler cannot pickle directly.

    Returns:
        A ``(callable, args)`` reduction that reconstructs ``obj`` via ``cloudpickle.loads``.
    """
    return cloudpickle.loads, (cloudpickle.dumps(obj),)


class CloudpickleDatasetHook:
    """Make ForkingPickler-unfriendly datasets (e.g. kedro-mlflow's) usable with ParallelRunner."""

    @hook_impl
    def before_pipeline_run(self, catalog: DataCatalog) -> None:
        """Register a cloudpickle reducer for each materialised dataset ForkingPickler rejects.

        Runs before the runner builds and validates its (shared-memory) catalog, so the reducers
        are in place for both validation and the worker hand-off. Only materialised datasets are
        inspected; lazily-instantiated catalog entries are created per worker and never pickled.
        """
        for name, dataset in catalog._datasets.items():  # materialised datasets only
            try:
                ForkingPickler.dumps(dataset)
            except (AttributeError, PicklingError):
                ForkingPickler.register(type(dataset), _reduce_via_cloudpickle)
                logger.debug(
                    "Registered a cloudpickle reducer for %s (%s) so ParallelRunner can "
                    "serialise it",
                    name,
                    type(dataset).__name__,
                )


class NodeTimingHook:
    """Log how long each node spends loading inputs, computing, and saving outputs.

    Kedro fires its dataset hooks around every catalog load/save and its node hooks around the
    function body, so timing those deltas separates the three phases the sweep cares about: I/O
    (materialising a victim and the splits from S3) versus the greedy-search compute. Hook
    implementations declare only the spec arguments they use (pluggy matches by name). Each parallel
    worker builds its own hook-manager instance, so a node's before/after pair always lands in the
    same process and a plain dict of start times is safe; keys carry the node name so interleaved
    worker output stays attributable. See https://docs.kedro.org/en/stable/extend/hooks/.
    """

    def __init__(self) -> None:
        """Initialise the per-phase start-time maps (one process / worker each)."""
        self._t_load: dict[str, float] = {}
        self._t_save: dict[str, float] = {}
        self._t_node: dict[str, float] = {}

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None:
        """Stamp the start of a dataset load (parameters are in-memory, so skip them)."""
        if not dataset_name.startswith("params:") and dataset_name != "parameters":
            self._t_load[f"{node.name}/{dataset_name}"] = perf_counter()

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, node: Node) -> None:
        """Log how long the dataset took to load (skipped for parameters)."""
        started = self._t_load.pop(f"{node.name}/{dataset_name}", None)
        if started is not None:
            elapsed = perf_counter() - started
            _timing.info("load  %-32s %9.3fs  [%s]", dataset_name, elapsed, node.name)

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        """Stamp the start of the node's compute (inputs are already loaded by now)."""
        self._t_node[node.name] = perf_counter()

    @hook_impl
    def after_node_run(self, node: Node) -> None:
        """Log the node's compute time (load and save are timed separately)."""
        started = self._t_node.pop(node.name, None)
        if started is not None:
            _timing.info("node  %-32s %9.3fs  (compute)", node.name, perf_counter() - started)

    @hook_impl
    def before_dataset_saved(self, dataset_name: str, node: Node) -> None:
        """Stamp the start of a dataset save."""
        self._t_save[f"{node.name}/{dataset_name}"] = perf_counter()

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, node: Node) -> None:
        """Log how long the dataset took to save."""
        started = self._t_save.pop(f"{node.name}/{dataset_name}", None)
        if started is not None:
            elapsed = perf_counter() - started
            _timing.info("save  %-32s %9.3fs  [%s]", dataset_name, elapsed, node.name)
