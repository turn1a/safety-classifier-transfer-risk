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
    "smoke",
    "__default__",
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
    # The chain terminates in the three reporting figures.
    figures = {"fig_cka_heatmap", "fig_transfer_scatter", "fig_regression_ablation"}
    assert figures <= default.all_outputs()
    # ...and spans data ingestion, the generated attack shards, the transfer assembly, and metrics.
    node_names = {node.name for node in default.nodes}
    assert {"build_canonical_dataset", "assemble_adversarial", "track_run_metrics"} <= node_names
    assert any(name.startswith("attack_") for name in node_names), "expected generated attack nodes"
