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

from transfer_risk.pipelines.audit.pipeline import create_finalization_pipeline
from transfer_risk.pipelines.experiment_v2_execute.pipeline import (
    create_annotation_pipeline,
    create_assets_prepare_pipeline,
    create_calibration_pipeline,
    create_evaluation_pipeline,
    create_models_pipeline,
    create_target_prepare_pipeline,
)
from transfer_risk.pipelines.retrospective_audit.pipeline import (
    create_baseline_pipeline,
    create_coverage_pipeline,
)
from transfer_risk.pipelines.semantic_audit.pipeline import (
    create_review_pipeline,
    create_tools_pipeline,
)

_EXCLUDED_FROM_DEFAULT = (
    "smoke",
    "audit",
    "audit_finalization",
    "audit_reporting",
    "similarity_audit_prepare",
    "similarity_audit",
    "paper_reporting",
    "semantic_audit",
    "semantic_audit_review",
    "semantic_review_tools",
    "semantic_screening",
    "retrospective_audit",
    "detector_baseline_audit",
    "retrospective_policy_coverage",
    "experiment_v2_preflight",
    "experiment_v2_selection",
    "experiment_v2_analysis",
    "experiment_v2_execute",
    "experiment_v2_target_prepare",
    "experiment_v2_assets_prepare",
    "experiment_v2_models",
    "experiment_v2_calibration_execute",
    "experiment_v2_evaluation_execute",
    "experiment_v2_calibration_annotations",
    "experiment_v2_evaluation_annotations",
)
_STAGE = ("data", "models", "similarity")
_DOWNSTREAM = ("transfer", "risk", "reporting")


def register_pipelines() -> dict[str, Pipeline]:
    """Return a mapping of pipeline name to ``Pipeline`` object."""
    pipelines: dict[str, Pipeline] = find_pipelines()
    pipelines["audit_finalization"] = create_finalization_pipeline()
    pipelines["semantic_audit_review"] = create_review_pipeline()
    pipelines["semantic_review_tools"] = create_tools_pipeline()
    pipelines["detector_baseline_audit"] = create_baseline_pipeline()
    pipelines["retrospective_policy_coverage"] = create_coverage_pipeline()
    pipelines.update(
        {
            "experiment_v2_target_prepare": create_target_prepare_pipeline(),
            "experiment_v2_assets_prepare": create_assets_prepare_pipeline(),
            "experiment_v2_models": create_models_pipeline(),
            "experiment_v2_calibration_execute": create_calibration_pipeline(),
            "experiment_v2_evaluation_execute": create_evaluation_pipeline(),
            "experiment_v2_calibration_annotations": create_annotation_pipeline("calibration"),
            "experiment_v2_evaluation_annotations": create_annotation_pipeline("evaluation"),
        }
    )
    domain = [
        pipeline for name, pipeline in pipelines.items() if name not in _EXCLUDED_FROM_DEFAULT
    ]
    pipelines["__default__"] = sum(domain, Pipeline([]))
    pipelines["stage"] = sum((pipelines[name] for name in _STAGE), Pipeline([]))
    pipelines["downstream"] = sum((pipelines[name] for name in _DOWNSTREAM), Pipeline([]))
    return pipelines
