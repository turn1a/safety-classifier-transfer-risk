"""Verify isolated private checkpoint identities without loading a detector."""

from __future__ import annotations

import hashlib
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
    descriptor = importlib.import_module(
        "transfer_risk.datasets.experiment_v2_model_dataset"
    ).checkpoint_descriptor
    first = descriptor(root)
    (root / "model.safetensors").write_bytes(b"different checkpoint bytes")
    second = descriptor(root)
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    assert first["tokenizer_sha256"] == second["tokenizer_sha256"]
    (root / "tokenizer_config.json").write_text('{"model_max_length":64}')
    third = descriptor(root)
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
    descriptor = importlib.import_module(
        "transfer_risk.datasets.experiment_v2_model_dataset"
    ).checkpoint_descriptor
    first = descriptor(root)
    config["vocab"] = {"other": 0}
    (root / "config.json").write_text(json.dumps(config))
    second = descriptor(root)
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
    original = importlib.import_module(
        "transfer_risk.datasets.experiment_v2_model_dataset"
    ).checkpoint_descriptor(source)
    output = _dataset_type()(filepath=str(destination), run_id="test-run")
    monkeypatch.chdir(elsewhere)
    output.save({"copy_from": original, "_metadata": {"preparation": "synthetic"}})
    copied = output.load()
    assert copied["checkpoint_sha256"] == original["checkpoint_sha256"]
    assert copied["tokenizer_sha256"] == original["tokenizer_sha256"]
    assert copied["path"] == str(destination / "payload")
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
        output.save({"copy_from": descriptor, "_metadata": {"preparation": "synthetic"}})
    assert not (project / "data/experiment_v2/test-run/models/replica").exists()


class _SyntheticPayload:
    def save_pretrained(self, destination):
        (destination / "config.json").write_text('{"num_labels":2}')
        (destination / "model.safetensors").write_bytes(b"synthetic weights")
        (destination / "tokenizer.json").write_text('{"vocab":{"token":0}}')


def test_bundle_recovery_verifies_staged_payload_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    dataset.save(
        {
            "kind": "transformer",
            "model": _SyntheticPayload(),
            "tokenizer": _SyntheticPayload(),
            "_metadata": {"training_seed": 123},
        }
    )
    first = dataset.load()
    assert first["_metadata"] == {"training_seed": 123}
    assert Path(first["path"]) == root / "payload"
    staged = root.with_name(f".{root.name}.staging")
    root.rename(staged)
    assert dataset.exists()
    assert dataset.load() == first
    assert not staged.exists()
    (root / "metadata.json").write_text('{"training_seed":999}')
    with pytest.raises(DatasetError, match=r"identity|corrupt"):
        dataset.exists()


def test_partial_model_directory_never_counts_as_completed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    with pytest.raises(DatasetError, match=r"complete|corrupt"):
        dataset.exists()


def test_bundle_cannot_complete_without_training_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    with pytest.raises(DatasetError, match="metadata"):
        dataset.save(
            {"kind": "transformer", "model": _SyntheticPayload(), "tokenizer": _SyntheticPayload()}
        )
    assert not root.exists()


class _FilesPayload:
    def __init__(self, files):
        self.files = files

    def save_pretrained(self, destination):
        for name, content in self.files.items():
            (destination / name).write_bytes(content)


def _write_sealed_bundle(root, model_files, tokenizer_files, metadata):
    payload = root / "payload"
    payload.mkdir(parents=True)
    for name, content in {**model_files, **tokenizer_files}.items():
        (payload / name).write_bytes(content)

    # Reproduce completion hashes from a previous bundle writer independently of its validation.
    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest(files):
        return [
            {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(files.items())
        ]

    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "complete.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest(manifest(model_files)),
                "tokenizer_sha256": digest(manifest(tokenizer_files)),
                "metadata_sha256": digest(metadata),
            }
        )
    )


@pytest.mark.parametrize("missing", ["weights", "tokenizer"])
@pytest.mark.parametrize("operation", ["save", "load", "exists"])
def test_bundle_requires_real_model_and_tokenizer_components(
    tmp_path, monkeypatch, missing, operation
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    model_files = {"config.json": b'{"num_labels":2}'}
    tokenizer_files = {"tokenizer.json": b'{"vocab":{"token":0}}'}
    if missing == "tokenizer":
        model_files["model.safetensors"] = b"synthetic weights"
        tokenizer_files = {"README.md": b"not tokenizer data", "tokenizer_config.json": b"{}"}
    metadata = {"training_seed": 123}
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    if operation != "save":
        _write_sealed_bundle(root, model_files, tokenizer_files, metadata)
    with pytest.raises(DatasetError, match=r"weights|tokenizer|components"):
        if operation == "save":
            dataset.save(
                {
                    "kind": "transformer",
                    "model": _FilesPayload(model_files),
                    "tokenizer": _FilesPayload(tokenizer_files),
                    "_metadata": metadata,
                }
            )
        else:
            getattr(dataset, operation)()
    if operation == "save":
        assert not (root / "complete.json").exists()
        assert not (root.with_name(f".{root.name}.staging") / "complete.json").exists()


@pytest.mark.parametrize(
    "weight_format", ["safetensors", "pytorch_bin", "sharded_safetensors", "sharded_bin"]
)
@pytest.mark.parametrize("tokenizer_format", ["fast", "wordpiece", "bpe", "sentencepiece"])
def test_bundle_preserves_supported_transformer_payload_formats(
    tmp_path, monkeypatch, weight_format, tokenizer_format
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    files = {"config.json": b'{"num_labels":2}'}
    if weight_format == "safetensors":
        files["model.safetensors"] = b"synthetic weights"
    elif weight_format == "pytorch_bin":
        files["pytorch_model.bin"] = b"synthetic weights"
    else:
        prefix, suffix = (
            ("model", "safetensors")
            if weight_format == "sharded_safetensors"
            else ("pytorch_model", "bin")
        )
        shards = [f"{prefix}-{index:05d}-of-00002.{suffix}" for index in (1, 2)]
        files.update(dict.fromkeys(shards, b"synthetic shard"))
        files[f"{prefix}.{suffix}.index.json"] = json.dumps(
            {"weight_map": {f"layer_{index}.weight": shard for index, shard in enumerate(shards)}}
        ).encode()
    tokenizer_files = {
        "fast": {"tokenizer.json": b'{"vocab":{"token":0}}'},
        "wordpiece": {"vocab.txt": b"[UNK]\ntoken\n"},
        "bpe": {"vocab.json": b'{"token":0}', "merges.txt": b"#version: 0.2\nt o\n"},
        "sentencepiece": {"spm.model": b"synthetic sentencepiece bytes"},
    }[tokenizer_format]
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    dataset.save(
        {
            "kind": "transformer",
            "model": _FilesPayload(files),
            "tokenizer": _FilesPayload(tokenizer_files),
            "_metadata": {"training_seed": 123},
        }
    )
    assert dataset.exists()
    assert dataset.load()["path"] == str(root / "payload")


@pytest.mark.parametrize("missing", ["all_weights", "referenced_shard", "tokenizer_payload"])
def test_index_and_tokenizer_metadata_do_not_substitute_for_required_payloads(
    tmp_path, monkeypatch, missing
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    files = {
        "config.json": b'{"num_labels":2}',
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "layer.a": "model-00001-of-00002.safetensors",
                    "layer.b": "model-00002-of-00002.safetensors",
                }
            }
        ).encode(),
    }
    if missing != "all_weights":
        files["model-00001-of-00002.safetensors"] = b"synthetic shard"
    if missing == "tokenizer_payload":
        files["model-00002-of-00002.safetensors"] = b"synthetic shard"
    tokens = (
        {"tokenizer_config.json": b"{}"}
        if missing == "tokenizer_payload"
        else {"tokenizer.json": b'{"vocab":{"token":0}}'}
    )
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    with pytest.raises(DatasetError, match=r"weights|tokenizer|shard|components"):
        dataset.save(
            {
                "kind": "transformer",
                "model": _FilesPayload(files),
                "tokenizer": _FilesPayload(tokens),
                "_metadata": {"training_seed": 123},
            }
        )
    assert not (root / "complete.json").exists()


@pytest.mark.parametrize("weight_format", ["safetensors", "pytorch_bin"])
@pytest.mark.parametrize("shape", ["orphan", "partial", "unrecognized_index"])
@pytest.mark.parametrize("operation", ["save", "load", "exists", "staged_load", "staged_exists"])
def test_shards_require_a_supported_complete_index_before_publication_or_reuse(
    tmp_path, monkeypatch, weight_format, shape, operation
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/experiment_v2/test-run/models/model"
    staged = root.with_name(f".{root.name}.staging")
    dataset = _dataset_type()(filepath=str(root), run_id="test-run")
    prefix, suffix = (
        ("model", "safetensors") if weight_format == "safetensors" else ("pytorch_model", "bin")
    )
    total = 2 if shape == "partial" else 1
    shard = f"{prefix}-00001-of-{total:05d}.{suffix}"
    model_files = {"config.json": b'{"num_labels":2}', shard: b"synthetic shard"}
    if shape == "unrecognized_index":
        model_files[f"unrecognized.{suffix}.index.json"] = json.dumps(
            {"weight_map": {"layer.weight": shard}}
        ).encode()
    tokens = {"tokenizer.json": b'{"vocab":{"token":0}}'}
    metadata = {"training_seed": 123}
    if operation != "save":
        _write_sealed_bundle(
            staged if operation.startswith("staged_") else root, model_files, tokens, metadata
        )
    with pytest.raises(DatasetError, match=r"weights|index|shard|serialization"):
        if operation == "save":
            dataset.save(
                {
                    "kind": "transformer",
                    "model": _FilesPayload(model_files),
                    "tokenizer": _FilesPayload(tokens),
                    "_metadata": metadata,
                }
            )
        else:
            getattr(dataset, operation.removeprefix("staged_"))()
    if operation == "save":
        assert not (root / "complete.json").exists()
        assert not (staged / "complete.json").exists()
    elif operation.startswith("staged_"):
        assert staged.exists()
        assert not root.exists()
