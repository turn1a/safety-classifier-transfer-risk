"""Reporting pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines._dynamic import surrogate_specs
from transfer_risk.pipelines.reporting.nodes import (
    assemble_surrogate_metadata,
    build_model_validation_summary_node,
    build_public_master_results_table,
    build_public_run_metrics_bundle,
    build_qualitative_examples,
    build_results_manifest_node,
    build_surrogate_summary_table,
    plot_cka_ranked,
    plot_transfer_scatter,
    publish_public_dataset_audit,
    recompute_master_dbs,
)


def _metadata_inputs() -> dict[str, str]:
    """Return Kedro input mapping for catalog-wired surrogate metadata fragments."""
    return {
        f"meta_{spec['name'].replace('-', '_')}": f"surrogate_meta__{spec['name']}"
        for spec in surrogate_specs()
    }


def create_pipeline() -> Pipeline:
    """Assemble the reporting pipeline."""
    nodes = [
        node(
            recompute_master_dbs,
            inputs=["master_results_table", "cka_matrices", "params:similarity"],
            outputs="master_results_corrected",
            name="recompute_master_dbs",
        ),
        node(
            build_public_master_results_table,
            inputs="master_results_corrected",
            outputs="pub_master_results_table",
            name="build_public_master_results_table",
        ),
        node(
            build_surrogate_summary_table,
            inputs=["master_results_corrected", "surrogate_selection"],
            outputs="pub_surrogate_summary",
            name="build_surrogate_summary_table",
        ),
        node(
            publish_public_dataset_audit,
            inputs="dataset_audit",
            outputs="pub_dataset_audit",
            name="publish_public_dataset_audit",
        ),
        node(
            assemble_surrogate_metadata,
            inputs=_metadata_inputs(),
            outputs="surrogate_metadata_bundle",
            name="assemble_surrogate_metadata",
        ),
        node(
            build_model_validation_summary_node,
            inputs=["surrogate_registry", "surrogate_metadata_bundle"],
            outputs="pub_model_validation_summary",
            name="build_model_validation_summary",
        ),
        node(
            build_public_run_metrics_bundle,
            inputs=[
                "master_results_corrected",
                "adversarial_examples",
                "ablation_results",
                "thresholds",
                "params:attacks",
            ],
            outputs="pub_run_metrics",
            name="build_public_run_metrics",
        ),
        node(
            build_results_manifest_node,
            inputs=[
                "params:reporting",
                "params:seed",
                "params:models",
                "params:attacks",
                "params:similarity",
                "params:transfer",
            ],
            outputs="pub_results_manifest",
            name="build_results_manifest",
        ),
        node(
            build_qualitative_examples,
            inputs=None,
            outputs="pub_qualitative_examples",
            name="build_qualitative_examples",
        ),
        node(
            plot_cka_ranked,
            inputs=["master_results_corrected", "surrogate_selection", "thresholds"],
            outputs="fig_cka_ranked",
            name="plot_cka_ranked",
        ),
        node(
            plot_transfer_scatter,
            inputs="master_results_corrected",
            outputs="fig_transfer_scatter",
            name="plot_transfer_scatter",
        ),
    ]
    return Pipeline(nodes)
