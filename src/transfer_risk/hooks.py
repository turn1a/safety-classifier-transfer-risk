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
from typing import TYPE_CHECKING, Any

import cloudpickle
from kedro.framework.hooks import hook_impl

if TYPE_CHECKING:
    from kedro.io import DataCatalog

logger = logging.getLogger(__name__)


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
