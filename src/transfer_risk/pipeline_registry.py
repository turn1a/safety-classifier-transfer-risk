"""Register the project's Kedro pipelines.

``find_pipelines`` discovers every ``create_pipeline`` under
:mod:`transfer_risk.pipelines`. ``__default__`` is the full measurement chain
(data → reporting); the ``smoke`` wiring-check pipeline is excluded from it and
run explicitly with ``kedro run --pipeline smoke``.
"""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Return a mapping of pipeline name to ``Pipeline`` object."""
    pipelines: dict[str, Pipeline] = find_pipelines()
    domain = [pipeline for name, pipeline in pipelines.items() if name != "smoke"]
    pipelines["__default__"] = sum(domain, Pipeline([]))
    return pipelines
