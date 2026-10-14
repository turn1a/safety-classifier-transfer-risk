"""Verify isolated private checkpoint identities without loading a detector."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from kedro.io import DatasetError


def _dataset_type():
    return importlib.import_module(
        "transfer_risk.datasets.experiment_v2_model_dataset"
    ).ExperimentV2ModelDataset


@pytest.mark.parametrize(
    "filepath",
    [
        "data/06_models/legacy",
        "data/experiment_v2/other-run/models/model",
        "data/experiment_v2/test-run/../legacy",
        "s3://bucket/data/experiment_v2/test-run/model",
        "/outside/data/experiment_v2/test-run/model",
    ],
)
def test_checkpoint_boundary_rejects_shared_or_redirected_paths(filepath):
    with pytest.raises(DatasetError, match="isolated"):
        _dataset_type()(filepath=filepath, run_id="test-run")


def _write_transformer_artifact(root: Path):
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"num_labels":2}')
    (root / "model.safetensors").write_bytes(b"synthetic checkpoint bytes")
    (root / "tokenizer.json").write_text('{"model":{"vocab":{"token":0}}}')
    (root / "tokenizer_config.json").write_text('{"model_max_length":32}')


def test_descriptor_detects_weight_and_tokenizer_mutation_separately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = "data/experiment_v2/test-run/models/model"
    root = tmp_path / relative
    _write_transformer_artifact(root)
    dataset = _dataset_type()(filepath=relative, run_id="test-run")
    first = dataset.load()
    (root / "model.safetensors").write_bytes(b"different checkpoint bytes")
    second = dataset.load()
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    assert first["tokenizer_sha256"] == second["tokenizer_sha256"]
    (root / "tokenizer_config.json").write_text('{"model_max_length":64}')
    third = dataset.load()
    assert second["checkpoint_sha256"] == third["checkpoint_sha256"]
    assert second["tokenizer_sha256"] != third["tokenizer_sha256"]
    assert first["path"] == str(root)


def test_bilstm_vocabulary_is_part_of_tokenizer_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = "data/experiment_v2/test-run/models/bilstm"
    root = tmp_path / relative
    root.mkdir(parents=True)
    config = {"vocab": {"token": 0}, "max_seq_len": 32, "padding_policy": "masked"}
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.pt").write_bytes(b"synthetic checkpoint bytes")
    dataset = _dataset_type()(filepath=relative, run_id="test-run")
    first = dataset.load()
    config["vocab"] = {"other": 0}
    (root / "config.json").write_text(json.dumps(config))
    second = dataset.load()
    assert first["tokenizer_sha256"] != second["tokenizer_sha256"]


def test_loading_rejects_late_symlink_redirect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = "data/experiment_v2/test-run/models/model"
    dataset = _dataset_type()(filepath=relative, run_id="test-run")
    target = tmp_path / "legacy"
    _write_transformer_artifact(target)
    link = tmp_path / "data/experiment_v2/test-run/models"
    link.parent.mkdir(parents=True)
    link.symlink_to(target.parent)
    with pytest.raises(DatasetError, match="symlink"):
        dataset.load()


def test_absolute_model_paths_and_copy_sources_keep_the_captured_project_after_cwd_change(
    tmp_path, monkeypatch
):
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(project)
    source = Path.cwd() / "data/experiment_v2/test-run/models/prepared"
    destination = Path.cwd() / "data/experiment_v2/test-run/models/replica"
    _write_transformer_artifact(source)
    original = _dataset_type()(filepath=str(source), run_id="test-run").load()
    output = _dataset_type()(filepath=str(destination), run_id="test-run")
    monkeypatch.chdir(elsewhere)
    output.save({"copy_from": original})
    copied = output.load()
    assert copied["checkpoint_sha256"] == original["checkpoint_sha256"]
    assert copied["tokenizer_sha256"] == original["tokenizer_sha256"]
    assert copied["path"] == str(destination)
    assert not (elsewhere / "data").exists()


@pytest.mark.parametrize("source_kind", ["sibling-project", "sibling-run", "symlink"])
def test_model_copy_source_cannot_escape_the_captured_run(tmp_path, monkeypatch, source_kind):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    output = _dataset_type()(
        filepath="data/experiment_v2/test-run/models/replica", run_id="test-run"
    )
    source = project / "data/experiment_v2/test-run/models/prepared"
    if source_kind == "sibling-project":
        source = tmp_path / "sibling/data/experiment_v2/test-run/models/prepared"
    elif source_kind == "sibling-run":
        source = project / "data/experiment_v2/other-run/models/prepared"
    _write_transformer_artifact(source)
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_model_dataset")
    descriptor = module.checkpoint_descriptor(source)
    if source_kind == "symlink":
        moved = tmp_path / "saved-prepared"
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(DatasetError, match=r"isolated|symlink"):
        output.save({"copy_from": descriptor})
    assert not (project / "data/experiment_v2/test-run/models/replica").exists()
