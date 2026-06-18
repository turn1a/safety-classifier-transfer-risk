"""Register the project's Kedro pipelines.

``find_pipelines`` discovers every ``create_pipeline`` under
:mod:`transfer_risk.pipelines`. ``__default__`` is the full measurement chain
(data → reporting); the ``smoke`` wiring-check pipeline is excluded from it and
run explicitly with ``kedro run --pipeline smoke``. Two composites, ``stage`` and
``downstream``, name the split the cloud recipes run (``kedro run --pipeline`` takes a
single name, so a comma list is not a pipeline): ``cloud-stage`` builds and uploads the
boundary artifacts the box reads, and ``cloud-finish`` runs the downstream against the
partitions the box wrote to S3.
"""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

_STAGE = ("data", "models", "similarity")
_DOWNSTREAM = ("transfer", "risk", "reporting")


def register_pipelines() -> dict[str, Pipeline]:
    """Return a mapping of pipeline name to ``Pipeline`` object."""
    pipelines: dict[str, Pipeline] = find_pipelines()
    domain = [pipeline for name, pipeline in pipelines.items() if name != "smoke"]
    pipelines["__default__"] = sum(domain, Pipeline([]))
    pipelines["stage"] = sum((pipelines[name] for name in _STAGE), Pipeline([]))
    pipelines["downstream"] = sum((pipelines[name] for name in _DOWNSTREAM), Pipeline([]))
    return pipelines
