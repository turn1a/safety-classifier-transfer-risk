"""Verify pinned upstream identities and worktree provenance without downloads."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from kedro.io import DatasetError


def test_unpinned_backbone_cannot_load():
    dataset_type = importlib.import_module(
        "transfer_risk.datasets.experiment_v2_hub_dataset"
    ).ExperimentV2HubModelDataset
    dataset = dataset_type(repo_id="synthetic/backbone", revision=None)
    with pytest.raises(DatasetError, match="immutable"):
        dataset.load()


def test_upstream_model_and_tokenizer_pins_are_returned(monkeypatch):
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_hub_dataset")
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="a" * 40))
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "b" * 40})
    monkeypatch.setattr(
        module.HuggingFaceHubModelDataset,
        "load",
        lambda _self: {"kind": "transformer", "model": model, "tokenizer": tokenizer},
    )
    monkeypatch.setattr(module, "load_pinned_tokenizer", lambda *_args, **_kwargs: tokenizer)
    dataset = module.ExperimentV2HubModelDataset(
        repo_id="synthetic/backbone",
        revision="a" * 40,
        tokenizer_repo_id="synthetic/tokenizer",
        tokenizer_revision="b" * 40,
    )
    result = dataset.load()
    assert result["input_identity"] == {
        "repository": "synthetic/backbone",
        "revision": "a" * 40,
        "tokenizer_repository": "synthetic/tokenizer",
        "tokenizer_revision": "b" * 40,
    }


def test_tokenizer_revision_can_come_from_its_resolved_snapshot_path(tmp_path, monkeypatch):
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_hub_dataset")
    tokenizer_file = tmp_path / "snapshots" / ("a" * 40) / "tokenizer.json"
    tokenizer_file.parent.mkdir(parents=True)
    tokenizer_file.write_text('{"model": {}}')
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="a" * 40))
    tokenizer = SimpleNamespace(init_kwargs={"tokenizer_file": str(tokenizer_file)})
    monkeypatch.setattr(
        module.HuggingFaceHubModelDataset,
        "load",
        lambda _self: {"kind": "transformer", "model": model, "tokenizer": tokenizer},
    )
    result = module.ExperimentV2HubModelDataset(
        repo_id="synthetic/backbone", revision="a" * 40
    ).load()
    assert result["input_identity"]["tokenizer_revision"] == "a" * 40


def test_context_identifies_primary_worktree_and_dirty_source(monkeypatch):
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_context_dataset")
    results = {
        ("rev-parse", "--show-toplevel"): "/synthetic/primary",
        ("branch", "--show-current"): "codex/methodology-recompute",
        ("rev-parse", "HEAD"): "a" * 40,
        ("status", "--porcelain", "--untracked-files=normal"): " M src/runtime.py",
        ("worktree", "list", "--porcelain"): (
            "worktree /synthetic/primary\nHEAD " + "a" * 40 + "\nbranch refs/heads/main\n"
        ),
    }
    monkeypatch.setattr(module, "git_output", lambda *args: results[args])
    context = module.ExperimentV2ExecutionContextDataset().load()
    assert context["isolated_worktree"] is False
    assert context["source_clean"] is False
    assert context["execution_ref"] == "a" * 40
    assert context["branch"] == "codex/methodology-recompute"


def test_raw_execution_outputs_are_immutable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_immutable_json_dataset")
    dataset = module.ExperimentV2ImmutableJSONDataset(
        filepath="data/experiment_v2/test-run/calibration/raw_records.json", run_id="test-run"
    )
    dataset.save({"attempted": 4})
    with pytest.raises(DatasetError, match="immutable"):
        dataset.save({"attempted": 3})
    assert dataset.load() == {"attempted": 4}
