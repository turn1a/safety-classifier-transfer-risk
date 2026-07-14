"""Tests for the explicit training-only CKA sensitivity pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.pipelines.similarity_audit import nodes as similarity_audit_nodes
from transfer_risk.pipelines.similarity_audit.pipeline import create_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split(prefix: str, *, count_per_label: int) -> pd.DataFrame:
    """Build one synthetic canonical split without private data."""
    return pd.DataFrame(
        [
            {
                "row_id": f"{prefix}-{label}-{index}",
                "text": f"{prefix} prompt {label} {index}",
                "label": label,
                "source": f"{prefix}-source-{label}",
                "split_marker": prefix,
            }
            for label in (0, 1)
            for index in range(count_per_label)
        ]
    )


def test_build_training_probe_set_samples_the_dedicated_training_split() -> None:
    probe = similarity_audit_nodes.build_training_probe_set(
        _split("train", count_per_label=800),
        {"n_probe": 1600},
        seed=23,
    )

    assert len(probe) == 1600
    assert probe["label"].value_counts().to_dict() == {1: 800, 0: 800}
    assert set(probe["split_marker"]) == {"train"}


def test_build_training_probe_set_rejects_any_size_other_than_1600() -> None:
    with pytest.raises(ValueError, match="n_probe must be 1600"):
        similarity_audit_nodes.build_training_probe_set(
            _split("train", count_per_label=1),
            {"n_probe": 2},
            seed=23,
        )


def test_similarity_audit_pipeline_receives_rows_only_from_training_split() -> None:
    pipeline = create_pipeline()
    inputs = set(pipeline.inputs())
    node_names = {node.name for node in pipeline.nodes}

    assert {
        "training_split",
        "target_model",
        "similarity_table",
        "surrogate_selection",
        "target_audit_summary",
        "cka_matrices",
        "params:similarity",
        "params:similarity_audit",
        "params:device",
        "params:risk",
        "params:seed",
    } <= inputs
    forbidden_row_inputs = {"task_splits", "val", "test", "canonical_dataset"}
    assert forbidden_row_inputs.isdisjoint(inputs)
    assert all(forbidden_row_inputs.isdisjoint(node.inputs) for node in pipeline.nodes)
    assert "adversarial_examples" not in inputs
    assert "params:attacks" not in inputs
    assert "transfer_results" not in inputs
    assert "raw_deepset" not in inputs
    assert "hub__deberta-base-pi-v1" not in inputs
    assert not any(name.startswith("hub__") for name in inputs)
    assert len({name for name in inputs if name.startswith("surrogate__")}) == 10
    assert "build_training_probe_set" in node_names
    assert "build_training_probe_sensitivity_summary" in node_names
    assert not any(name.startswith("attack_") for name in node_names)
    assert not any(name.startswith("train_") for name in node_names)


def test_similarity_audit_catalog_keeps_internal_artifacts_local_and_wraps_public_outputs() -> None:
    loader = OmegaConfigLoader(
        conf_source=str(_PROJECT_ROOT / "conf"),
        base_env="base",
        default_run_env="local",
    )
    catalog = loader["catalog"]

    for name in (
        "training_split",
        "training_probe_set",
        "training_cka__{surrogate}",
        "training_cka_matrices",
        "training_similarity_table",
        "training_similarity_thresholds",
        "training_surrogate_selection",
    ):
        entry = catalog[name]
        assert entry["type"] != "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
        assert entry["filepath"].startswith("data/")

    public_datasets = {
        "pub_training_probe_similarity": "pandas.CSVDataset",
        "pub_training_probe_similarity_sensitivity": "json.JSONDataset",
    }
    for name, dataset_type in public_datasets.items():
        entry = catalog[name]
        assert entry["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
        assert entry["dataset"]["type"] == dataset_type
        assert entry["dataset"]["filepath"].startswith("docs/artifacts/")
        assert entry["artifact_path"] in {"tables", "results"}


def test_training_cka_recipe_forces_sequential_runner() -> None:
    justfile = (_PROJECT_ROOT / "justfile").read_text(encoding="utf-8")

    assert (
        "cka-train-sensitivity:\n"
        "    uv run kedro run --pipeline similarity_audit --runner SequentialRunner"
    ) in justfile
