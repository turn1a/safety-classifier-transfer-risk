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
