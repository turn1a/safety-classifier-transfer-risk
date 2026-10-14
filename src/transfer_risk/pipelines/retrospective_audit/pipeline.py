"""Assemble explicit serial candidate audits and cached-checkpoint baseline checks."""

from functools import partial

from kedro.pipeline import Pipeline, node

from transfer_risk.lib.retrospective_audit import verify_reference_counts
from transfer_risk.pipelines._dynamic import baseline_detector_specs, retrospective_target_specs
from transfer_risk.pipelines.retrospective_audit.nodes import (
    assemble_audit_reports,
    assemble_retrospective_reports,
    build_original_coverage,
    build_source_baseline_check,
    publish_retrospective_tables,
    run_detector_baseline,
    run_retrospective_target,
)


def create_coverage_pipeline() -> Pipeline:
    """Reduce saved private predictions into public original-level union coverage."""
    return Pipeline(
        [
            node(
                build_source_baseline_check,
                inputs=[
                    "semantic_audit_sample_linkage",
                    "pub_detector_baseline_summary",
                    "pub_results_manifest",
                    "params:attacks",
                ],
                outputs="pub_source_baseline_check",
                name="verify_source_baseline_counts",
            ),
            node(
                build_original_coverage,
                inputs=[
                    "retrospective_predictions__protectai-v2",
                    "semantic_audit_sample_linkage",
                    "pub_surrogate_summary",
                    "pub_target_audit_cells",
                    "pub_results_manifest",
                    "params:attacks",
                    "params:retrospective_audit",
                ],
                outputs="pub_original_policy_coverage",
                name="summarize_original_policy_coverage",
            ),
        ],
        tags="retrospective_policy_coverage",
    )


def create_pipeline() -> Pipeline:
    """Return saved-candidate inference, separate from the original search protocol."""
    targets = retrospective_target_specs()
    nodes = [
        node(
            partial(run_retrospective_target, target_id=name),
            inputs=[
                "semantic_audit_private_pairs",
                "semantic_audit_sample_linkage",
                f"retrospective_model__{name}",
                "params:retrospective_audit",
                "params:device",
            ],
            outputs=[
                f"retrospective_predictions__{name}",
                f"retrospective_cells__{name}",
                f"retrospective_sources__{name}",
                f"retrospective_context__{name}",
            ],
            name=f"audit_saved_candidates_{name}",
        )
        for name in targets
    ]
    nodes.append(
        node(
            verify_reference_counts,
            inputs=[
                "retrospective_cells__protectai-v2",
                "pub_target_audit_cells",
                "pub_results_manifest",
                "params:retrospective_audit.targets.protectai-v2",
            ],
            outputs="pub_retrospective_reference_check",
            name="verify_retrospective_reference_counts",
        )
    )
    nodes.extend(
        node(
            publish_retrospective_tables,
            inputs=[
                "pub_retrospective_reference_check",
                "retrospective_cells__protectai-v2",
                f"retrospective_cells__{name}",
                f"retrospective_sources__{name}",
            ],
            outputs=[f"pub_retrospective_cells__{name}", f"pub_retrospective_sources__{name}"],
            name=f"publish_retrospective_{name}",
        )
        for name in targets
    )
    nodes.append(
        node(
            assemble_retrospective_reports,
            inputs=[
                "pub_retrospective_reference_check",
                *[f"retrospective_context__{name}" for name in targets],
            ],
            outputs="pub_retrospective_audit_summary",
            name="assemble_retrospective_audit_summary",
        )
    )
    return Pipeline(nodes, tags="retrospective_audit")


def create_baseline_pipeline() -> Pipeline:
    """Return a small clean-input pass over the archived transformer checkpoints."""
    models = baseline_detector_specs()
    nodes = [
        node(
            partial(run_detector_baseline, target_id=name),
            inputs=[
                "task_splits",
                f"retrospective_model__{name}",
                "params:retrospective_audit",
                "params:device",
            ],
            outputs=f"detector_baseline__{name}",
            name=f"audit_detector_baseline_{name}",
        )
        for name in models
    ]
    nodes.append(
        node(
            assemble_audit_reports,
            inputs=[f"detector_baseline__{name}" for name in models],
            outputs="pub_detector_baseline_summary",
            name="assemble_detector_baseline_summary",
        )
    )
    return Pipeline(nodes, tags="detector_baseline_audit")
