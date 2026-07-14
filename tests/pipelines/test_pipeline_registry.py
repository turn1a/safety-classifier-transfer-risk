"""The registry builds and exposes every stage (structure check for the scaffold)."""

from __future__ import annotations

from transfer_risk.pipeline_registry import register_pipelines

EXPECTED_PIPELINES = {
    "data",
    "models",
    "similarity",
    "attacks",
    "transfer",
    "risk",
    "reporting",
    "audit",
    "audit_finalization",
    "audit_reporting",
    "similarity_audit_prepare",
    "similarity_audit",
    "smoke",
    "__default__",
    "stage",
    "downstream",
}


def test_all_pipelines_register() -> None:
    pipelines = register_pipelines()
    assert set(pipelines) >= EXPECTED_PIPELINES


def test_every_pipeline_builds_nonempty() -> None:
    pipelines = register_pipelines()
    for name, pipe in pipelines.items():
        assert pipe.nodes, f"pipeline {name!r} has no nodes"


def test_default_chain_excludes_smoke() -> None:
    pipelines = register_pipelines()
    default_node_names = {node.name for node in pipelines["__default__"].nodes}
    assert "record_environment" not in default_node_names
    assert "build_canonical_dataset" in default_node_names


def test_default_chain_excludes_audit() -> None:
    pipelines = register_pipelines()
    default_node_names = {node.name for node in pipelines["__default__"].nodes}
    assert "run_target_audit" not in default_node_names
    assert "finalize_target_audit" not in default_node_names
    assert "audit" in pipelines
    assert pipelines["audit"].nodes


def test_audit_finalization_uses_saved_aggregates_without_target_model() -> None:
    """Keep DBS refresh independent of frozen-target loading or inference."""
    finalization = register_pipelines()["audit_finalization"]
    node_names = {node.name for node in finalization.nodes}

    assert node_names == {"finalize_target_audit"}
    assert "target_model" not in finalization.inputs()
    assert {
        "target_audit_raw_cells",
        "target_audit_raw_sources",
        "target_audit_raw_context",
        "master_results_table",
        "cka_matrices",
    } <= finalization.inputs()


def test_audit_reporting_is_explicit_and_target_free() -> None:
    """Keep audit publication separate from target inference and normal reporting."""
    audit_reporting = register_pipelines()["audit_reporting"]

    assert "target_model" not in audit_reporting.inputs()
    assert {
        "target_audit_cells",
        "target_audit_raw_sources",
        "target_audit_summary",
        "surrogate_selection",
        "pub_master_results_table",
    } == audit_reporting.inputs()
    assert {
        "pub_target_audit_cells",
        "pub_target_audit_sources",
        "pub_target_audit_summary",
        "pub_qualitative_examples",
        "fig_transfer_scatter",
        "fig_true_flip_scatter",
        "fig_true_flip_ablation",
    } <= audit_reporting.outputs()


def test_default_stage_and_downstream_composites_exclude_audit() -> None:
    """Keep target inference opt-in rather than part of normal pipeline composites."""
    pipelines = register_pipelines()
    for pipeline_name in ("__default__", "stage", "downstream"):
        node_names = {node.name for node in pipelines[pipeline_name].nodes}
        assert "run_target_audit" not in node_names
        assert "finalize_target_audit" not in node_names
        assert "build_public_target_audit_cells" not in node_names
        assert "build_public_target_audit_summary" not in node_names


def test_training_cka_pipelines_are_explicit_and_isolated_from_other_pipelines() -> None:
    """Keep preparation and sensitivity runs out of normal and audit flows."""
    pipelines = register_pipelines()
    for pipeline_name in ("__default__", "stage", "downstream", "audit", "audit_reporting"):
        node_names = {node.name for node in pipelines[pipeline_name].nodes}
        assert "extract_training_split" not in node_names
        assert "build_training_probe_set" not in node_names
        assert "build_training_probe_sensitivity_summary" not in node_names

    preparation = pipelines["similarity_audit_prepare"]
    assert {node.name for node in preparation.nodes} == {"extract_training_split"}
    assert preparation.inputs() == {"task_splits"}
    assert preparation.outputs() == {"training_split"}

    sensitivity = pipelines["similarity_audit"]
    assert "build_training_probe_set" in {node.name for node in sensitivity.nodes}
    assert "extract_training_split" not in {node.name for node in sensitivity.nodes}
    assert "training_split" in sensitivity.inputs()
    assert "task_splits" not in sensitivity.inputs()
    assert "adversarial_examples" not in sensitivity.inputs()
    assert "transfer_results" not in sensitivity.inputs()


def test_default_chain_is_connected_end_to_end() -> None:
    """The full chain has no dangling data inputs and produces the reporting outputs."""
    default = register_pipelines()["__default__"]
    # The only free inputs are parameters and source datasets/models nothing in the pipeline
    # produces: the raw HuggingFace datasets, the frozen target model, and the per-surrogate Hub
    # sources (hub__*). No node consumes an intermediate that nothing produces.
    sources = {"raw_deepset", "raw_jackhhao", "raw_lakera", "target_model"}
    free_inputs = default.inputs()
    assert free_inputs, "the default pipeline should declare parameter inputs"
    assert all(
        name.startswith("params:") or name.startswith("hub__") or name in sources
        for name in free_inputs
    ), free_inputs
    # The chain terminates in the public reporting bundle and figures.
    reporting_outputs = {
        "pub_master_results_table",
        "pub_surrogate_summary",
        "pub_dataset_audit",
        "pub_model_validation_summary",
        "pub_run_metrics",
        "pub_results_manifest",
        "pub_qualitative_examples",
        "fig_cka_ranked",
        "fig_transfer_scatter",
    }
    assert reporting_outputs <= default.all_outputs()
    assert "fig_selection_ablation" not in default.all_outputs()
    # ...and spans data ingestion, the generated attack shards, the transfer assembly, and metrics.
    node_names = {node.name for node in default.nodes}
    assert {"build_canonical_dataset", "assemble_adversarial", "track_run_metrics"} <= node_names
    assert any(name.startswith("attack_") for name in node_names), "expected generated attack nodes"
