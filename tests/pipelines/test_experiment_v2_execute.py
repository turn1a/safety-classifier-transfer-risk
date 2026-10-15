"""Verify prospective execution accounting with synthetic text and mocked ML operations."""

from __future__ import annotations

import importlib
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from experiment_v2.fixtures import digest, evidence, records_for
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from transfer_risk.datasets.experiment_v2_immutable_json_dataset import (
    ExperimentV2ImmutableJSONDataset,
)
from transfer_risk.datasets.experiment_v2_model_dataset import ExperimentV2ModelDataset
from transfer_risk.lib.experiment_v2.protocol import content_digest, partition_digest
from transfer_risk.lib.experiment_v2.schemas import ExperimentRecords


def _nodes():
    return importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.nodes")


def _context(protocol):
    return {
        "project_root": "/synthetic/isolated/risk-transfer",
        "branch": "codex/methodology-recompute",
        "execution_ref": protocol.execution_ref,
        "source_clean": True,
        "isolated_worktree": True,
        "hardware": "synthetic test host; no model execution",
    }


def _execution_parameters():
    return {
        "device_policy": "cpu",
        "inference_batch_size": 4,
        "query_batch_size": 2,
        "pooling": "cls",
        "training": {
            f"source_{index}": {"lr": 0.001, "batch_size": 2, "epochs": 1, "max_seq_len": 512}
            for index in range(8)
        },
    }


def _texts(corpus, phase):
    return [{"id": row.id, "text": row.id} for row in corpus.records if row.partition == phase]


def _models(protocol, corpus):
    realizations = records_for("calibration", protocol, corpus).realizations
    registration = evidence()[2].model_copy(
        update={
            "protocol_sha256": content_digest(protocol),
            "corpus_sha256": content_digest(corpus),
        }
    )
    execution = {
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "execution_ref": protocol.execution_ref,
        "execution_spec_sha256": protocol.attack_spec_sha256,
        "registration_evidence_sha256": content_digest(registration),
        "registration_registered_at": registration.registered_at,
        "attack_assets_sha256": protocol.attack_assets_sha256,
        "started_at": "2026-10-02T00:00:00+00:00",
        "hardware": _context(protocol)["hardware"],
    }
    return {
        "_execution": _nodes().execution_header(execution),
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "realizations": [row.model_dump() for row in realizations],
        "artifacts": {
            f"{row.model_id}:r{row.replicate}": {
                "path": (
                    f"/synthetic/isolated/risk-transfer/{protocol.private_root}/"
                    f"models/{row.model_id}/r{row.replicate}"
                ),
                "checkpoint_sha256": row.checkpoint_sha256,
                "tokenizer_sha256": row.tokenizer_sha256,
            }
            for row in realizations
        },
    }


def _assets(protocol, project_root=None):
    project_root = project_root or _context(protocol)["project_root"]
    root = f"{project_root}/{protocol.private_root}/attack_assets/cache"
    directories = {
        "masked_lm": "masked-lm",
        "semantic_encoder": "semantic-encoder",
        "counterfitted_embeddings": "textattack/word_embeddings/paragramcf",
        "nltk": "nltk",
    }
    files = [
        "masked-lm/config.json",
        "masked-lm/model.safetensors",
        "masked-lm/tokenizer.json",
        "semantic-encoder/modules.json",
        "semantic-encoder/model.safetensors",
        "semantic-encoder/tokenizer.json",
        "textattack/post_install_check_3",
        *[
            f"textattack/word_embeddings/paragramcf/{name}"
            for name in (
                "paragram.npy",
                "wordlist.pickle",
                "mse_dist.p",
                "cos_sim.p",
                "nn.npy",
            )
        ],
        "nltk/corpora/wordnet.zip",
        "nltk/corpora/omw-1.4.zip",
        "nltk/corpora/stopwords/english",
        *[
            f"nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{name}.json"
            for name in ("weights", "tagdict", "classes")
        ],
        "nltk/taggers/universal_tagset/en-ptb.map",
    ]
    manifest = {
        "schema_version": 2,
        "run_id": protocol.run_id,
        "cache_paths": {name: name for name in ("textattack", "nltk", "huggingface")},
        "roles": {
            name: {"repository": f"synthetic/{name}", "revision": "a" * 40, "path": path}
            for name, path in directories.items()
        },
        "files": [{"path": path, "sha256": digest(path)} for path in sorted(files)],
    }
    return {
        "manifest": manifest,
        "manifest_sha256": content_digest(manifest),
        "cache_root": root,
        "paths": {
            name: f"{root}/{path}"
            for name, path in {
                **directories,
                **manifest["cache_paths"],
            }.items()
        },
    }


def _execution_evidence():
    protocol, corpus, registration = evidence()
    settings = _execution_parameters()
    protocol = protocol.model_copy(
        update={
            "attack_assets_sha256": _assets(protocol)["manifest_sha256"],
            "models": [
                model.model_copy(
                    update={
                        "training_recipe_sha256": content_digest(settings["training"][model.id])
                    }
                )
                if model.kind != "frozen"
                else model
                for model in protocol.models
            ],
        }
    )
    protocol = protocol.model_copy(
        update={
            "attack_spec_sha256": _nodes().execution_spec_digest(protocol, settings),
            "input_policy_sha256": _nodes().input_policy_digest(protocol),
        }
    )
    registration = registration.model_copy(update={"protocol_sha256": content_digest(protocol)})
    return protocol, corpus, registration


@pytest.mark.filterwarnings("ignore:The node producing outputs.*partial:UserWarning")
@pytest.mark.parametrize(
    "registered_at", ["2026-10-14T12:00:00+00:00", "2099-01-01T00:00:00+00:00"]
)
def test_model_gate_rejects_unregistered_start_before_private_or_hub_inputs(
    monkeypatch, registered_at
):
    module = _nodes()
    now = datetime(2026, 10, 14, 12, tzinfo=UTC)
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda _tz: now))
    protocol, corpus, registration = _execution_evidence()
    registration = registration.model_copy(update={"registered_at": registered_at})
    factory = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.pipeline")
    pipeline = factory.create_models_pipeline(protocol.model_dump())
    datasets = {
        "params:experiment_v2": MemoryDataset(protocol.model_dump()),
        "v2_corpus_evidence": MemoryDataset(corpus.model_dump()),
        "v2_registration_evidence": MemoryDataset(registration.model_dump()),
        "v2_execution_context": MemoryDataset(_context(protocol)),
        "params:seed": MemoryDataset(protocol.root_seed),
        "params:experiment_v2_execution": MemoryDataset(_execution_parameters()),
        "v2_attack_asset_manifest": MemoryDataset(_assets(protocol)),
    }
    protected_load = Mock(side_effect=AssertionError("private text or Hub input was opened"))
    for name in pipeline.inputs() - datasets.keys():
        dataset = MemoryDataset({})
        monkeypatch.setattr(dataset, "load", protected_load)
        datasets[name] = dataset
    catalog = DataCatalog(datasets)
    with pytest.raises(ValueError, match=r"registration.*precede"):
        SequentialRunner().run(pipeline, catalog)
    protected_load.assert_not_called()
    assert not catalog.exists("v2_models_receipt")


def test_execution_headers_retain_initial_registration_timestamp():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    receipt = module.gate_models(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
        _assets(protocol),
    )
    header = module.execution_header(receipt)
    assert receipt["registration_registered_at"] == registration.registered_at
    assert header["registration_registered_at"] == registration.registered_at


@pytest.mark.parametrize("started_at", ["2026-09-30T23:59:59+00:00", "2026-10-01T00:00:00+00:00"])
def test_resumed_calibration_rejects_models_started_before_initial_registration(started_at):
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    manifest["_execution"]["started_at"] = started_at
    with pytest.raises(ValueError, match=r"registration"):
        _nodes().gate_calibration(
            protocol.model_dump(),
            corpus.model_dump(),
            registration.model_dump(),
            _context(protocol),
            manifest,
            _execution_parameters(),
            _assets(protocol),
        )


@pytest.mark.parametrize("registered_at", [None, "2026-09-30T00:00:00+00:00"])
def test_resumed_headers_reject_missing_or_changed_initial_registration_timestamp(registered_at):
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    receipt = module.gate_models(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
        _assets(protocol),
    )
    inherited = module.execution_header(receipt)
    inherited["registration_registered_at"] = registered_at
    with pytest.raises(ValueError, match=r"stale.*identity"):
        module.execution_header(receipt, inherited)


@pytest.mark.parametrize(
    "changed", [{"isolated_worktree": False}, {"source_clean": False}, {"branch": "main"}]
)
def test_heavy_gate_rejects_wrong_worktree_before_execution(changed):
    protocol, corpus, registration = evidence()
    context = {**_context(protocol), **changed}
    with pytest.raises(ValueError, match=r"isolated|clean|branch"):
        _nodes().gate_models(
            protocol.model_dump(),
            corpus.model_dump(),
            registration.model_dump(),
            context,
            protocol.root_seed,
        )


@pytest.mark.parametrize("axis", ["models", "replicates", "recipes", "target_roles"])
def test_first_heavy_gate_rejects_registered_axes_that_differ_from_built_graph(axis):
    protocol, corpus, registration = _execution_evidence()
    structure = protocol.model_dump()
    if axis == "models":
        structure[axis][2]["kind"] = "bilstm"
    elif axis == "target_roles":
        structure[axis][0]["candidate_ids"] = structure[axis][0]["candidate_ids"][:-1]
    else:
        structure[axis] = structure[axis][:-1]
    with pytest.raises(ValueError, match=r"graph|axes|structure"):
        _nodes().gate_models(
            protocol.model_dump(),
            corpus.model_dump(),
            registration.model_dump(),
            _context(protocol),
            protocol.root_seed,
            _execution_parameters(),
            _assets(protocol),
            structure=structure,
        )


def test_graph_binding_allows_runtime_execution_ref_and_registration_pins():
    protocol, corpus, registration = _execution_evidence()
    structure = protocol.model_dump()
    structure["execution_ref"] = None
    structure["attack_spec_sha256"] = None
    structure["models"][2]["revision"] = None
    receipt = _nodes().gate_models(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
        _assets(protocol),
        structure=structure,
    )
    assert receipt["protocol_sha256"] == content_digest(protocol)


def test_model_gate_requires_the_exact_verified_registered_auxiliary_manifest():
    protocol, corpus, registration = _execution_evidence()
    arguments = [
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
    ]
    with pytest.raises(ValueError, match=r"offline|assets|absent"):
        _nodes().gate_models(*arguments)
    assets = _assets(protocol)
    assets["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"manifest|registered|identity"):
        _nodes().gate_models(*arguments, assets)


def test_resumed_calibration_rejects_a_model_manifest_from_other_auxiliary_assets():
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    manifest["_execution"]["attack_assets_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"stale|execution|identity"):
        _nodes().gate_calibration(
            protocol.model_dump(),
            corpus.model_dump(),
            registration.model_dump(),
            _context(protocol),
            manifest,
            _execution_parameters(),
            _assets(protocol),
        )


def test_resumed_model_manifest_retains_original_training_time_and_hardware():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    receipt = module.gate_models(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
        _assets(protocol),
    )
    saved = _models(protocol, corpus)
    inputs = []
    for realization in saved["realizations"]:
        metadata = {
            key: value
            for key, value in realization.items()
            if key not in {"checkpoint_sha256", "tokenizer_sha256"}
        }
        metadata["_execution"] = module.execution_header(receipt)
        inputs.extend(
            [saved["artifacts"][f"{realization['model_id']}:r{realization['replicate']}"], metadata]
        )
    inputs[1]["_execution"]["started_at"] = "2026-10-02T00:00:00+00:00"
    inputs[1]["_execution"]["hardware"] = "earlier synthetic training host"
    manifest = module.collect_models(receipt, protocol.model_dump(), corpus.model_dump(), *inputs)
    assert manifest["_execution"]["started_at"] == "2026-10-02T00:00:00+00:00"
    assert "earlier synthetic training host" in manifest["_execution"]["hardware"]


def test_private_texts_require_exact_registered_partition_and_content():
    _protocol, corpus, _registration = evidence()
    texts = _texts(corpus, "calibration")
    validated = _nodes().phase_frame(texts, corpus, "calibration")
    assert validated["id"].tolist() == [row["id"] for row in texts]
    tampered = deepcopy(texts)
    tampered[0]["text"] = "altered synthetic original"
    with pytest.raises(ValueError, match="content"):
        _nodes().phase_frame(tampered, corpus, "calibration")
    wrong_partition = _texts(corpus, "evaluation")
    with pytest.raises(ValueError, match=r"partition|coverage"):
        _nodes().phase_frame(wrong_partition, corpus, "calibration")


def test_attempt_shard_retains_failures_skips_errors_and_private_candidates():
    protocol, corpus, registration = _execution_evidence()
    receipt = _nodes().gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        _models(protocol, corpus),
        _execution_parameters(),
        _assets(protocol),
    )
    statuses = ["success", "failed", "skipped_source_benign", "error"]

    def fake_attack(_model, original, _cfg, _seed, _budget):
        index = int(original["id"].rsplit("_", 1)[1])
        return {
            "status": statuses[index],
            "source_queries": [4, 3, 1, 2][index],
            "candidate_text": "synthetic candidate" if index < 2 else None,
            "source_raw_perturbed": "0" if index == 0 else "1" if index == 1 else None,
            "error_type": "RuntimeError" if index == 3 else None,
        }

    cell = {"phase": "calibration", "source_id": "source_0", "replicate": 0, "recipe": "textfooler"}
    result = _nodes().run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        _models(protocol, corpus),
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell=cell,
        attack_fn=fake_attack,
    )
    assert [row["attempt"]["status"] for row in result] == statuses
    assert [row["attempt"]["source_queries"] for row in result] == [4, 3, 1, 2]
    assert all(row["attempt"]["target_search_queries"] == 0 for row in result)
    assert [row["candidate_text"] for row in result] == [
        "synthetic candidate",
        "synthetic candidate",
        None,
        None,
    ]
    assert result[0]["attempt"]["candidate_sha256"] == digest("synthetic candidate")
    assert all(row["attempt"]["mechanical_screen"] == "unassessable" for row in result)


def test_target_audit_includes_every_failure_candidate_without_publishing_text():
    protocol, corpus, registration = _execution_evidence()
    receipt = _nodes().gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        _models(protocol, corpus),
        _execution_parameters(),
        _assets(protocol),
    )
    cell = {"phase": "calibration", "source_id": "source_0", "replicate": 0, "recipe": "textfooler"}
    private = _nodes().run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        _models(protocol, corpus),
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell=cell,
        attack_fn=lambda *_args: {
            "status": "failed",
            "source_queries": 3,
            "candidate_text": "synthetic failed candidate",
            "source_raw_perturbed": "1",
            "error_type": None,
        },
    )
    audited = _nodes().audit_attack_shard(
        receipt,
        protocol.model_dump(),
        _models(protocol, corpus),
        private,
        _execution_parameters(),
        predict_fn=lambda _asset, _descriptor, texts, _cfg: ["0"] * len(texts),
        corpus_evidence=corpus.model_dump(),
        clean_predictions=_clean_rows(protocol, corpus, receipt),
    )
    assert len(audited) == 4
    assert all(len(row["target_predictions"]) == 2 for row in audited)
    assert all(row["status"] == "failed" for row in audited)
    assert all("candidate_text" not in row and "text" not in row for row in audited)


def test_evaluation_gate_requires_registered_selection_before_unsealing():
    protocol, corpus, registration = evidence()
    with pytest.raises(ValueError, match=r"selection|lock"):
        _nodes().gate_evaluation(
            protocol.model_dump(),
            corpus.model_dump(),
            registration.model_dump(),
            _context(protocol),
            _models(protocol, corpus),
            {},
            {},
            {},
        )


def test_query_ledger_counts_actual_calls_and_rejects_over_budget():
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    ledger = module.QueryLedger(lambda texts: np.tile([0.1, 0.9], (len(texts), 1)), 3)
    assert ledger(["synthetic original", "candidate one"]).shape == (2, 2)
    assert ledger.source_queries == 2
    ledger(["synthetic original"])
    assert ledger.source_queries == 2
    ledger(["candidate two"])
    assert ledger.source_queries == 3
    with pytest.raises(module.SourceQueryBudgetError):
        ledger(["candidate three"])
    assert ledger.source_queries == 3


def test_query_ledger_moves_device_logits_to_cpu_before_cache_accounting():
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    observed = []

    class DeviceLogits:
        def detach(self):
            observed.append("detach")
            return self

        def cpu(self):
            observed.append("cpu")
            return SimpleNamespace(numpy=lambda: np.array([[0.1, 0.9]]))

        def __iter__(self):
            message = "device logits cannot be iterated as host arrays"
            raise AssertionError(message)

    ledger = module.QueryLedger(lambda _texts: DeviceLogits(), 2)
    assert ledger(["synthetic original"]).tolist() == [[0.1, 0.9]]
    assert ledger(["synthetic original"]).tolist() == [[0.1, 0.9]]
    assert ledger.source_queries == 1 and observed == ["detach", "cpu"]


def _target_plan(protocol, corpus):
    return {
        "run_id": protocol.run_id,
        "target_role": "replication",
        "model_id": "target_b",
        "repository": "synthetic/roberta-backbone",
        "revision": "a" * 40,
        "tokenizer_repository": "synthetic/roberta-backbone",
        "tokenizer_revision": "a" * 40,
        "backbone_family": "roberta",
        "root_seed": protocol.root_seed,
        "execution_ref": protocol.execution_ref,
        "max_tokens": 512,
        "output_labels": ["0", "1"],
        "label_map": {"0": "benign", "1": "prompt_injection"},
        "label_review_sha256": digest("target_labels"),
        "training": {"lr": 0.001, "batch_size": 2, "epochs": 1, "max_seq_len": 512},
        "min_train_groups": 2,
        "min_validation_groups": 2,
        "core_corpus_sha256": content_digest(corpus),
    }


def _target_evidence(protocol, core):
    rows = [
        row.model_copy(
            update={
                "id": "target:" + row.id,
                "group_id": "target:" + row.group_id,
                "origin_id": "target:" + row.origin_id,
                "attack_family_id": "target:" + row.attack_family_id,
                "content_sha256": digest("target:" + row.id),
            }
        )
        for row in core.records
        if row.partition in {"train", "validation"}
    ]
    corpus = core.model_copy(update={"records": rows, "target_preparations": []})
    plan = _target_plan(protocol, core)
    return plan, {
        "corpus": corpus.model_dump(),
        "source_revisions": {"training": "b" * 40},
        "annotation_evidence_sha256": digest("target_human_review"),
        "registration": {
            "registry_uri": "https://example.invalid/target-plan",
            "registered_at": "2026-10-01T00:00:00+00:00",
            "protocol_sha256": content_digest(plan),
            "corpus_sha256": content_digest(corpus),
            "evidence_sha256": digest("target_registration"),
        },
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("core_content", "content"),
        ("duplicate_train_content", "duplicate content"),
        ("duplicate_validation_content", "content"),
        ("origin_id", "group closure"),
        ("attack_family_id", "group closure"),
    ],
)
def test_target_gate_rejects_content_leakage_and_inflated_group_counts(change, message):
    protocol, core, _registration = evidence()
    plan, target_evidence = _target_evidence(protocol, core)
    rows = target_evidence["corpus"]["records"]
    if change == "core_content":
        heldout = next(
            row for row in core.records if row.partition == "evaluation" and row.label == 0
        )
        rows[0]["content_sha256"] = heldout.content_sha256
    elif change == "duplicate_train_content":
        rows[1]["content_sha256"] = rows[0]["content_sha256"]
    elif change == "duplicate_validation_content":
        rows[4]["content_sha256"] = rows[0]["content_sha256"]
    else:
        rows[1][change] = rows[0][change]
    target_evidence["registration"]["corpus_sha256"] = content_digest(target_evidence["corpus"])
    module = importlib.import_module(
        "transfer_risk.pipelines.experiment_v2_execute.target_preparation"
    )
    with pytest.raises(ValueError, match=message):
        module.gate_target_preparation(
            plan,
            target_evidence,
            protocol.model_dump(),
            core.model_dump(),
            _context(protocol),
            protocol.root_seed,
        )


def test_disjoint_target_evidence_opens_only_the_registered_preparation_gate():
    protocol, core, _registration = evidence()
    plan, target_evidence = _target_evidence(protocol, core)
    module = importlib.import_module(
        "transfer_risk.pipelines.experiment_v2_execute.target_preparation"
    )
    receipt = module.gate_target_preparation(
        plan,
        target_evidence,
        protocol.model_dump(),
        core.model_dump(),
        _context(protocol),
        protocol.root_seed,
    )
    assert receipt["target_corpus_sha256"] == content_digest(target_evidence["corpus"])


def test_prepared_target_export_retains_registered_nontext_corpus_inventory(monkeypatch):
    protocol, core, _registration = evidence()
    plan, target_evidence = _target_evidence(protocol, core)
    module = importlib.import_module(
        "transfer_risk.pipelines.experiment_v2_execute.target_preparation"
    )
    receipt = module.gate_target_preparation(
        plan,
        target_evidence,
        protocol.model_dump(),
        core.model_dump(),
        _context(protocol),
        protocol.root_seed,
    )
    corpus = module.CorpusEvidence.model_validate(target_evidence["corpus"])
    monkeypatch.setattr(module.runtime, "train_model", lambda *_args: ({"mock_bundle": True}, {}))
    backbone = {
        "input_identity": {
            key: plan[key]
            for key in ("repository", "revision", "tokenizer_repository", "tokenizer_revision")
        }
    }
    _bundle, metadata = module.prepare_target(
        receipt,
        plan,
        target_evidence,
        _texts(corpus, "train"),
        _texts(corpus, "validation"),
        backbone,
        {"device_policy": "cpu"},
    )
    descriptor = {
        "checkpoint_sha256": digest("checkpoint:target_b"),
        "tokenizer_sha256": digest("tokenizer:target_b"),
    }
    exported = module.export_target_preparation(receipt, descriptor, metadata)
    assert exported["target_corpus_sha256"] == receipt["target_corpus_sha256"]
    assert exported["corpus_records"] == target_evidence["corpus"]["records"]
    assert exported["grouping_method"] == corpus.grouping_method
    assert all("text" not in row for row in exported["corpus_records"])


def test_target_preparation_gate_rejects_shared_target_training_groups():
    protocol, corpus, _registration = evidence()
    plan = _target_plan(protocol, corpus)
    target_corpus = corpus.model_copy(
        update={
            "records": [row for row in corpus.records if row.partition in {"train", "validation"}],
            "target_preparations": [],
        }
    )
    target_evidence = {
        "corpus": target_corpus.model_dump(),
        "source_revisions": {"training": "b" * 40},
        "annotation_evidence_sha256": digest("target_human_review"),
        "registration": {
            "registry_uri": "https://example.invalid/target-plan",
            "registered_at": "2026-10-01T00:00:00+00:00",
            "protocol_sha256": content_digest(plan),
            "evidence_sha256": digest("registration"),
        },
    }
    module = importlib.import_module(
        "transfer_risk.pipelines.experiment_v2_execute.target_preparation"
    )
    with pytest.raises(ValueError, match=r"disjoint|overlap"):
        module.gate_target_preparation(
            plan,
            target_evidence,
            protocol.model_dump(),
            corpus.model_dump(),
            _context(protocol),
            protocol.root_seed,
        )


def test_factories_keep_evaluation_probe_and_human_review_out_of_heavy_work():
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.pipeline")
    protocol, _corpus, _registration = evidence()
    evaluation = module.create_evaluation_pipeline(protocol.model_dump())
    assert "v2_private_evaluation" in evaluation.inputs()
    assert "v2_private_probe" not in evaluation.inputs()
    assert "v2_calibration_records" in evaluation.inputs()
    assert "v2_selection_registration" in evaluation.inputs()
    assert "v2_evaluation_raw_records" in evaluation.outputs()
    annotations = module.create_annotation_pipeline("evaluation")
    assert "v2_evaluation_annotations" in annotations.inputs()
    assert "v2_private_evaluation" not in annotations.inputs()
    assert "v2_evaluation_records" in annotations.outputs()


def test_counted_recipe_preserves_actual_original_and_failed_candidate_queries(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    protocol, _corpus, _registration = evidence()
    source = next(model for model in protocol.models if model.id == "source_0")

    def builder(counted):
        def attack(original, ground_truth):
            counted([original])
            counted(["synthetic failed candidate", "synthetic other candidate"])
            return SimpleNamespace(
                perturbed_result=SimpleNamespace(
                    attacked_text=SimpleNamespace(text="synthetic failed candidate"),
                    output=ground_truth,
                    num_queries=999,
                )
            )

        return SimpleNamespace(
            goal_function=SimpleNamespace(query_budget=None, batch_size=None), attack=attack
        )

    recipe = SimpleNamespace(build=builder)
    monkeypatch.setattr(
        module, "load_recipe_backend", lambda: ({"textfooler": recipe}, lambda _seed: None)
    )
    monkeypatch.setattr(module, "_ledger_wrapper", lambda ledger, _source: ledger)
    result = module.attack_original(
        {
            "identity": source.model_dump(),
            "artifact": {},
            "runtime_wrapper": lambda texts: np.tile([0.1, 0.9], (len(texts), 1)),
        },
        {"id": "original", "text": "synthetic original", "label": 1},
        {**_execution_parameters(), "recipe": "textfooler"},
        123,
        5,
    )
    assert result["source_queries"] == 3
    assert result["status"] == "failed"
    assert result["candidate_text"] == "synthetic failed candidate"
    assert result["source_raw_perturbed"] == "1"


def test_recipe_setup_failure_preserves_an_actual_zero_query_error(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    protocol, _corpus, _registration = evidence()

    def unavailable_backend():
        message = "synthetic setup error"
        raise RuntimeError(message)

    monkeypatch.setattr(module, "load_recipe_backend", unavailable_backend)
    result = module.attack_original(
        {"identity": protocol.models[2].model_dump(), "artifact": {}, "runtime_wrapper": object()},
        {"id": "original", "text": "synthetic original", "label": 1},
        {**_execution_parameters(), "recipe": "textfooler"},
        123,
        5,
    )
    assert result["source_queries"] == 0
    assert result["status"] == "error"
    assert result["candidate_text"] is None
    assert result["error_type"] == "RuntimeError"


def test_auxiliary_recipe_setup_failure_precedes_any_source_search_query(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    protocol, _corpus, _registration = evidence()

    def missing_asset(_counted):
        message = "synthetic missing auxiliary asset"
        raise RuntimeError(message)

    monkeypatch.setattr(
        module,
        "load_recipe_backend",
        lambda: (
            {
                "textfooler": SimpleNamespace(build=missing_asset),
            },
            lambda _seed: None,
        ),
    )
    monkeypatch.setattr(module, "_ledger_wrapper", lambda ledger, _source: ledger)
    result = module.attack_original(
        {
            "identity": protocol.models[2].model_dump(),
            "artifact": {},
            "runtime_wrapper": lambda texts: np.tile([0.1, 0.9], (len(texts), 1)),
        },
        {"id": "original", "text": "synthetic original", "label": 1},
        {**_execution_parameters(), "recipe": "textfooler"},
        123,
        5,
    )
    assert result["source_queries"] == 0
    assert result["status"] == "error" and result["candidate_text"] is None


def test_shard_reuses_pinned_recipe_assets_and_clears_cross_original_caches(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    protocol, _corpus, _registration = evidence()
    observed = {"builds": 0, "clears": 0, "queries": []}

    class Router:
        def __init__(self, ledger):
            self.ledger = ledger

        def __call__(self, texts):
            return self.ledger(texts)

    def classifier(texts):
        observed["queries"].extend(texts)
        return np.tile([0.1, 0.9], (len(texts), 1))

    def builder(counted):
        observed["builds"] += 1

        def attack(original, output):
            counted([original, "synthetic candidate"])
            return SimpleNamespace(
                perturbed_result=SimpleNamespace(
                    attacked_text=SimpleNamespace(text="synthetic candidate"),
                    output=output,
                )
            )

        def clear_cache():
            observed["clears"] += 1

        return SimpleNamespace(
            goal_function=SimpleNamespace(), attack=attack, clear_cache=clear_cache
        )

    monkeypatch.setattr(
        module,
        "load_recipe_backend",
        lambda: (
            {
                "textfooler": SimpleNamespace(build=builder),
            },
            lambda _seed: None,
        ),
    )
    monkeypatch.setattr(module, "_ledger_wrapper", lambda ledger, _source: Router(ledger))
    model_input = {
        "identity": protocol.models[2].model_dump(),
        "artifact": {},
        "runtime_wrapper": classifier,
    }
    for index in range(2):
        result = module.attack_original(
            model_input,
            {"id": str(index), "text": f"original {index}", "label": 1},
            {**_execution_parameters(), "recipe": "textfooler"},
            123,
            5,
        )
        assert result["source_queries"] == 2 and result["status"] == "failed"
    assert observed == {
        "builds": 1,
        "clears": 1,
        "queries": ["original 0", "synthetic candidate", "original 1", "synthetic candidate"],
    }


def test_training_uses_exact_registered_membership_and_replica_seed(monkeypatch):
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    receipt = module.gate_models(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        protocol.root_seed,
        _execution_parameters(),
        _assets(protocol),
    )
    model = next(model for model in protocol.models if model.id == "source_0")
    observed = {}

    def synthetic_training(identity, _backbone, train, validation, recipe, _policy, seed):
        observed.update(
            {
                "id": identity.id,
                "train": train["id"].tolist(),
                "validation": validation["id"].tolist(),
                "recipe": recipe,
                "seed": seed,
            }
        )
        return {
            "kind": "transformer",
            "model": "synthetic weights",
            "tokenizer": "synthetic tokenizer",
        }, {
            "device": "synthetic device; no training",
        }

    monkeypatch.setattr(module.runtime, "train_model", synthetic_training)
    _bundle, metadata = module.prepare_model(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        _texts(corpus, "train"),
        _texts(corpus, "validation"),
        {
            "input_identity": {
                "repository": model.repository,
                "revision": model.revision,
                "tokenizer_repository": model.tokenizer_repository,
                "tokenizer_revision": model.tokenizer_revision,
            }
        },
        _execution_parameters(),
        cell={"model_id": "source_0", "replicate": 2},
    )
    assert observed["train"] == ["train_0_0", "train_0_1", "train_1_0", "train_1_1"]
    assert observed["validation"] == [
        "validation_0_0",
        "validation_0_1",
        "validation_1_0",
        "validation_1_1",
    ]
    expected = next(
        row
        for row in records_for("calibration", protocol, corpus).realizations
        if row.model_id == "source_0" and row.replicate == 2
    )
    assert observed["seed"] == metadata["training_seed"] == expected.training_seed
    assert metadata["train_partition_sha256"] == partition_digest(corpus, "train")
    assert observed["recipe"] == _execution_parameters()["training"]["source_0"]


def test_full_probe_cka_uses_registered_probe_and_realized_identities(monkeypatch):
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    modeling = importlib.import_module("transfer_risk.modeling")

    def synthetic_representations(_spec, texts, **kwargs):
        assert texts == ["probe_0_0", "probe_0_1", "probe_1_0", "probe_1_1"]
        assert kwargs["pooling"] == "cls"
        assert kwargs["max_seq_len"] == 512
        return [np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])]

    monkeypatch.setattr(modeling, "layer_representations", synthetic_representations)
    result = module.compute_probe_similarities(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "probe"),
        _execution_parameters(),
    )
    assert len(result) == 48
    assert all(row["mean_cka"] == pytest.approx(1.0) for row in result)
    assert all(row["probe_partition_sha256"] == partition_digest(corpus, "probe") for row in result)
    with pytest.raises(ValueError, match="evaluation"):
        module.compute_probe_similarities(
            {**receipt, "phase": "evaluation"},
            protocol.model_dump(),
            corpus.model_dump(),
            manifest,
            _texts(corpus, "probe"),
            _execution_parameters(),
        )


def test_complete_synthetic_phase_emits_non_text_full_grid_without_human_verdicts():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    clean = module.score_phase_originals(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        predict_fn=lambda _asset, _descriptor, texts, _cfg: [
            "1" if "_1_" in text else "0" for text in texts
        ],
    )
    audited = []
    for source_index in range(8):
        for replicate in range(3):
            cell = {
                "phase": "calibration",
                "source_id": f"source_{source_index}",
                "replicate": replicate,
                "recipe": "textfooler",
            }
            private = module.run_attack_shard(
                receipt,
                protocol.model_dump(),
                corpus.model_dump(),
                manifest,
                _texts(corpus, "calibration"),
                _execution_parameters(),
                cell=cell,
                attack_fn=lambda *_args: {
                    "status": "failed",
                    "source_queries": 3,
                    "candidate_text": "synthetic candidate",
                    "source_raw_perturbed": "1",
                    "error_type": None,
                },
            )
            audited.append(
                module.audit_attack_shard(
                    receipt,
                    protocol.model_dump(),
                    manifest,
                    private,
                    _execution_parameters(),
                    predict_fn=lambda _asset, _descriptor, texts, _cfg: ["0"] * len(texts),
                    corpus_evidence=corpus.model_dump(),
                    clean_predictions=_clean_rows(protocol, corpus, receipt),
                )
            )
    similarities = [
        row.model_dump() for row in records_for("calibration", protocol, corpus).similarities
    ]
    for row in similarities:
        row["_execution"] = module.execution_header(receipt)
    clean[0]["_execution"]["started_at"] = "2026-10-02T00:00:00+00:00"
    records = module.collect_phase_records(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        manifest,
        clean,
        similarities,
        *audited,
    )
    parsed = ExperimentRecords.model_validate(records)
    assert len(parsed.clean_predictions) == 240
    assert len(parsed.similarities) == 48
    assert len(parsed.attempts) == 96
    assert all(row.status == "failed" for row in parsed.attempts)
    assert all(len(row.target_predictions) == 2 for row in parsed.attempts)
    assert parsed.annotations == []
    assert parsed.application_records == []
    assert parsed.started_at == "2026-10-02T00:00:00+00:00"
    assert "synthetic candidate" not in str(records)


def test_resumed_target_audit_rejects_stale_shard_execution_identity():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    private = module.run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell={
            "phase": "calibration",
            "source_id": "source_0",
            "replicate": 0,
            "recipe": "textfooler",
        },
        attack_fn=lambda *_args: {
            "status": "failed",
            "source_queries": 3,
            "candidate_text": "synthetic candidate",
            "source_raw_perturbed": "1",
            "error_type": None,
        },
    )
    private[0]["_execution"] = {"protocol_sha256": "0" * 64}
    with pytest.raises(ValueError, match=r"stale|execution|identity"):
        module.audit_attack_shard(
            receipt,
            protocol.model_dump(),
            manifest,
            private,
            _execution_parameters(),
            predict_fn=lambda _asset, _descriptor, texts, _cfg: ["0"] * len(texts),
            corpus_evidence=corpus.model_dump(),
            clean_predictions=_clean_rows(protocol, corpus, receipt),
        )


def _realized_synthetic_assets(tmp_path):
    protocol, _corpus, _registration = evidence()
    descriptor = _assets(protocol, tmp_path)
    for row in descriptor["manifest"]["files"]:
        path = Path(descriptor["cache_root"]) / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "" if path.name == "post_install_check_3" else row["path"]
        path.write_text(data, encoding="utf-8")
        row["sha256"] = digest(data)
    for path in descriptor["manifest"]["cache_paths"].values():
        (Path(descriptor["cache_root"]) / path).mkdir(parents=True, exist_ok=True)
    descriptor["manifest_sha256"] = content_digest(descriptor["manifest"])
    return descriptor


def test_offline_recipe_assets_bind_actual_cache_and_local_model_constructors(
    tmp_path, monkeypatch
):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.offline")
    descriptor = _realized_synthetic_assets(tmp_path)
    monkeypatch.setattr(module, "_STATE", None)
    for key in module.OFFLINE_ENVIRONMENT:
        monkeypatch.setenv(key, "synthetic previous environment")
    queries = []

    def find_nltk(name, paths):
        queries.append((name, paths))
        assert paths == [descriptor["paths"]["nltk"]]
        return f"{paths[0]}/{name}"

    nltk = SimpleNamespace(
        data=SimpleNamespace(path=[], _resource_cache={}, find=find_nltk),
        corpus=SimpleNamespace(),
        tag=SimpleNamespace(),
    )
    modules = {
        "nltk": nltk,
        "sentence_transformers": SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: (args, kwargs)
        ),
        (
            "textattack.constraints.semantics.sentence_encoders.sentence_transformer_encoder"
        ): SimpleNamespace(sentence_transformers=None),
        "huggingface_hub.constants": SimpleNamespace(HF_HUB_OFFLINE=False),
        "textattack.shared.utils.install": SimpleNamespace(TEXTATTACK_CACHE_DIR="host cache"),
        "textattack.shared.utils": SimpleNamespace(
            GLOBAL_OBJECTS={
                "textattack_counterfitted_GLOVE_embedding": "unregistered old embedding",
            }
        ),
        "textattack.attack_recipes.bae_garg_2019": SimpleNamespace(
            WordSwapMaskedLM=lambda **kwargs: kwargs,
        ),
        "textattack.constraints.semantics.sentence_encoders": SimpleNamespace(
            SentenceTransformerEncoder=lambda **kwargs: kwargs,
        ),
    }
    monkeypatch.setattr(module, "import_module", lambda name: modules[name])
    module.configure_offline_assets(descriptor, "cpu")
    assert os.environ["TA_CACHE_DIR"] == descriptor["paths"]["textattack"]
    assert os.environ["NLTK_DATA"] == descriptor["paths"]["nltk"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert modules["huggingface_hub.constants"].HF_HUB_OFFLINE is True
    bae = modules["textattack.attack_recipes.bae_garg_2019"].WordSwapMaskedLM(
        method="bae", max_candidates=50
    )
    assert bae["masked_language_model"] == descriptor["paths"]["masked_lm"]
    assert bae["method"] == "bae" and bae["max_candidates"] == 50
    encoder = modules[
        "textattack.constraints.semantics.sentence_encoders"
    ].SentenceTransformerEncoder(
        threshold=0.840845057,
        metric="angular",
    )
    backend = modules[
        "textattack.constraints.semantics.sentence_encoders.sentence_transformer_encoder"
    ].sentence_transformers
    assert backend is not None
    assert backend.SentenceTransformer("local-encoder")[1] == {
        "device": "cpu",
        "local_files_only": True,
    }
    assert encoder["model_name"] == descriptor["paths"]["semantic_encoder"]
    assert encoder["threshold"] == 0.840845057 and encoder["metric"] == "angular"
    install = modules["textattack.shared.utils.install"]
    assert (
        install.download_from_s3("word_embeddings/paragramcf")
        == descriptor["paths"]["counterfitted_embeddings"]
    )
    with pytest.raises(module.OfflineAssetError, match=r"offline|cache|registered"):
        install.download_from_s3("unregistered/resource")
    with pytest.raises(module.OfflineAssetError, match=r"offline|network"):
        install.http_get("https://synthetic.invalid/asset", object())
    assert nltk.download("omw-1.4") is True
    assert queries == [("corpora/omw-1.4/", [descriptor["paths"]["nltk"]])]
    assert (
        "textattack_counterfitted_GLOVE_embedding"
        not in modules["textattack.shared.utils"].GLOBAL_OBJECTS
    )


def test_changed_offline_asset_fails_before_loading_any_recipe_backend(tmp_path, monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.offline")
    descriptor = _realized_synthetic_assets(tmp_path)
    (Path(descriptor["cache_root"]) / "masked-lm/model.safetensors").write_bytes(b"changed")
    imported = []
    monkeypatch.setattr(module, "_STATE", None)
    monkeypatch.setattr(module, "import_module", imported.append)
    with pytest.raises(Exception, match=r"hash|bytes|identity"):
        module.configure_offline_assets(descriptor, "cpu")
    assert imported == []


def test_recipe_backend_refuses_an_unconfigured_offline_asset_boundary(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.offline")
    monkeypatch.setattr(module, "_STATE", None)
    runtime = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    with pytest.raises(module.OfflineAssetError, match=r"offline|configured|registered"):
        runtime.load_recipe_backend()


@pytest.mark.filterwarnings("ignore:The node producing outputs.*partial:UserWarning")
@pytest.mark.parametrize("target", [False, True])
def test_production_model_graph_resumes_missing_metadata_without_retraining(
    tmp_path, monkeypatch, target
):

    monkeypatch.chdir(tmp_path)
    graph = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.pipeline")
    target_module = graph.target_preparation
    protocol, corpus, registration = _execution_evidence()
    context = {**_context(protocol), "project_root": str(tmp_path)}
    calls = {"trainer": 0, "gate": 0}

    class Payload:
        def save_pretrained(self, path):
            (path / "config.json").write_text('{"num_labels":2}')
            (path / "model.safetensors").write_bytes(b"synthetic model")
            (path / "tokenizer.json").write_text('{"vocab":{"token":0}}')

    def train(*_args):
        calls["trainer"] += 1
        return {"kind": "transformer", "model": Payload(), "tokenizer": Payload()}, {}

    monkeypatch.setattr(graph.nodes.runtime, "train_model", train)
    if target:
        plan, target_evidence = _target_evidence(protocol, corpus)
        original_gate = target_module.gate_target_preparation

        def gate(*args, **kwargs):
            calls["gate"] += 1
            return original_gate(*args, **kwargs)

        monkeypatch.setattr(target_module, "gate_target_preparation", gate)
        pipeline = graph.create_target_prepare_pipeline(protocol.model_dump())
        checkpoint, metadata = "v2_prepared_target", "v2_prepared_target_metadata"
        target_corpus = target_module.CorpusEvidence.model_validate(target_evidence["corpus"])
        values = {
            "params:experiment_v2_target_preparation": plan,
            "v2_target_preparation_evidence": target_evidence,
            "v2_target_private_train": _texts(target_corpus, "train"),
            "v2_target_private_validation": _texts(target_corpus, "validation"),
            "v2_target_backbone": {
                "input_identity": {
                    key: plan[key]
                    for key in (
                        "repository",
                        "revision",
                        "tokenizer_repository",
                        "tokenizer_revision",
                    )
                }
            },
        }
    else:
        original_gate = graph.nodes.gate_models

        def gate(*args, **kwargs):
            calls["gate"] += 1
            return original_gate(*args, **kwargs)

        monkeypatch.setattr(graph.nodes, "gate_models", gate)
        pipeline = graph.create_models_pipeline(protocol.model_dump())
        checkpoint, metadata = "v2_model__source_0__r0", "v2_model_meta__source_0__r0"
        # Retain the production gate and one complete training/projection branch.
        names = [
            row.name
            for row in pipeline.nodes
            if row.name
            in {
                "v2_gate_models",
                "v2_verify_models_gate",
                "v2_prepare_source_0_r0",
                "v2_project_source_0_r0_metadata",
            }
        ]
        pipeline = pipeline.only_nodes(*names)
        model = next(row for row in protocol.models if row.id == "source_0")
        values = {
            "v2_registration_evidence": registration.model_dump(),
            "v2_attack_asset_manifest": _assets(protocol, str(tmp_path)),
            "v2_private_train": _texts(corpus, "train"),
            "v2_private_validation": _texts(corpus, "validation"),
            "v2_hub__source_0": {
                "input_identity": {
                    key: getattr(model, key)
                    for key in (
                        "repository",
                        "revision",
                        "tokenizer_repository",
                        "tokenizer_revision",
                    )
                }
            },
        }
    values.update(
        {
            "params:experiment_v2": protocol.model_dump(),
            "v2_corpus_evidence": corpus.model_dump(),
            "v2_execution_context": context,
            "params:seed": protocol.root_seed,
            "params:experiment_v2_execution": _execution_parameters(),
        }
    )
    root = tmp_path / protocol.private_root
    output = ExperimentV2ModelDataset(filepath=str(root / "models/model"), run_id=protocol.run_id)
    meta_path = root / "model_metadata/model.json"

    class InterruptedMetadata(ExperimentV2ImmutableJSONDataset):
        def save(self, _data):
            msg = "synthetic interruption before metadata projection"
            raise RuntimeError(msg)

    def catalog(interrupt):
        result = {name: MemoryDataset(value) for name, value in values.items()}
        for name in pipeline.datasets() - set(result):
            result[name] = MemoryDataset()
        result[checkpoint] = output
        cls = InterruptedMetadata if interrupt else ExperimentV2ImmutableJSONDataset
        result[metadata] = cls(filepath=str(meta_path), run_id=protocol.run_id)
        if target:
            result["v2_target_preparation_record"] = ExperimentV2ImmutableJSONDataset(
                filepath=str(root / "target_preparation/record.json"), run_id=protocol.run_id
            )
        return DataCatalog(result)

    with pytest.raises(Exception, match="synthetic interruption"):
        SequentialRunner().run(pipeline, catalog(interrupt=True), only_missing_outputs=True)
    assert output.exists() and not meta_path.exists()
    SequentialRunner().run(pipeline, catalog(interrupt=False), only_missing_outputs=True)
    assert meta_path.exists()
    assert calls == {"trainer": 1, "gate": 2}
    SequentialRunner().run(pipeline, catalog(interrupt=False), only_missing_outputs=True)
    assert calls == {"trainer": 1, "gate": 3}


@pytest.mark.parametrize(
    "corruption",
    ["hash", "budget", "source_identity", "target_queries", "source_window", "clean_window"],
)
def test_candidate_corruption_is_rejected_before_any_target_prediction(corruption):
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    private = module.run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell={
            "phase": "calibration",
            "source_id": "source_0",
            "replicate": 0,
            "recipe": "textfooler",
        },
        attack_fn=lambda *_args: {
            "status": "failed",
            "source_queries": 3,
            "candidate_text": "synthetic candidate",
            "source_raw_perturbed": "1",
        },
    )
    clean = _clean_rows(protocol, corpus, receipt)
    if corruption == "hash":
        private[-1]["candidate_text"] = "corrupted synthetic candidate"
    elif corruption == "budget":
        private[-1]["attempt"]["source_queries"] = protocol.source_query_budget + 1
    elif corruption == "source_identity":
        private[-1]["attempt"]["source_checkpoint_sha256"] = digest("wrong")
    elif corruption == "source_window":
        private[-1]["attempt"]["source_candidate_tokenization"] = {
            "input_tokens": 513,
            "max_tokens": 522,
            "truncated": False,
        }
    elif corruption == "clean_window":
        clean[-1]["tokenization"] = {"input_tokens": 513, "max_tokens": 522, "truncated": False}
    else:
        private[-1]["attempt"]["target_search_queries"] = 1
    predictor = Mock(return_value=["0"] * len(private))
    with pytest.raises(ValueError):
        module.audit_attack_shard(
            receipt,
            protocol.model_dump(),
            manifest,
            private,
            _execution_parameters(),
            corpus.model_dump(),
            clean,
            predict_fn=predictor,
        )
    predictor.assert_not_called()


def _clean_rows(protocol, corpus, receipt):
    return [
        {**row.model_dump(), "_execution": _nodes().execution_header(receipt)}
        for row in records_for(receipt["phase"], protocol, corpus).clean_predictions
    ]


def test_classifier_predictor_records_untruncated_token_counts_without_changing_text(monkeypatch):
    module = importlib.import_module("transfer_risk.pipelines.experiment_v2_execute.runtime")
    modeling = importlib.import_module("transfer_risk.modeling")
    protocol, _corpus, _registration = evidence()
    identity = protocol.models[0].model_copy(update={"max_tokens": 4})
    texts = ["one two", "one two three four five"]
    seen = []

    def tokenizer(batch, **kwargs):
        seen.append((batch, kwargs))
        return {"input_ids": [[0, *range(len(text.split())), 9] for text in batch]}

    monkeypatch.setattr(modeling, "load_transformer", lambda *_: (object(), tokenizer))
    monkeypatch.setattr(
        modeling, "predict", lambda _model, _tokenizer, batch, **_kwargs: [1] * len(batch)
    )
    diagnostics = []
    labels = module.predict_model(
        identity,
        {"path": "/synthetic/payload"},
        texts,
        {**_execution_parameters(), "_tokenization_diagnostics": diagnostics},
    )
    assert labels == [identity.output_labels[1]] * 2
    assert diagnostics == [
        {"input_tokens": 4, "max_tokens": 4, "truncated": False},
        {"input_tokens": 7, "max_tokens": 4, "truncated": True},
    ]
    assert seen == [(texts, {"truncation": False, "padding": False, "add_special_tokens": True})]


def test_input_policy_explicitly_serializes_only_registered_lower_trust_text():
    protocol, _corpus, _registration = evidence()
    expected = {
        "text_encoding": "utf-8",
        "content_hash": "sha256",
        "normalization": "none",
        "character_clipping": None,
        "tokenizer_truncation": "registered_max_tokens",
        "model_windows": {model.id: model.max_tokens for model in protocol.models},
        "grouping": "separate_registered_evidence",
        "classifier_input": "registered_lower_trust_text_only",
        "trusted_context": "human_ground_truth_and_optional_application_only",
    }
    assert _nodes().input_policy_digest(protocol) == content_digest(expected)


def test_clean_and_candidate_nodes_persist_tokenizer_diagnostics():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    diagnostic = {"input_tokens": 600, "max_tokens": 512, "truncated": True}

    def predict(_model, _descriptor, texts, settings):
        settings["_tokenization_diagnostics"].extend([diagnostic.copy() for _ in texts])
        return ["1"] * len(texts)

    clean = module.score_phase_originals(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        predict_fn=predict,
    )
    assert all(row["tokenization"] == diagnostic for row in clean)
    private = module.run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell={
            "phase": "calibration",
            "source_id": "source_0",
            "replicate": 0,
            "recipe": "textfooler",
        },
        attack_fn=lambda *_: {
            "status": "failed",
            "source_queries": 3,
            "candidate_text": "synthetic candidate",
            "source_raw_perturbed": "1",
            "original_tokenization": diagnostic,
            "candidate_tokenization": diagnostic,
        },
    )
    assert private[0]["attempt"]["source_candidate_tokenization"] == diagnostic
    audited = module.audit_attack_shard(
        receipt,
        protocol.model_dump(),
        manifest,
        private,
        _execution_parameters(),
        corpus.model_dump(),
        clean,
        predict_fn=predict,
    )
    assert all(
        prediction["tokenization"] == diagnostic
        for row in audited
        for prediction in row["target_predictions"]
    )


def test_audit_rejects_conflicting_predictions_for_duplicate_candidate_content():
    module = _nodes()
    protocol, corpus, registration = _execution_evidence()
    manifest = _models(protocol, corpus)
    receipt = module.gate_calibration(
        protocol.model_dump(),
        corpus.model_dump(),
        registration.model_dump(),
        _context(protocol),
        manifest,
        _execution_parameters(),
        _assets(protocol),
    )
    private = module.run_attack_shard(
        receipt,
        protocol.model_dump(),
        corpus.model_dump(),
        manifest,
        _texts(corpus, "calibration"),
        _execution_parameters(),
        cell={
            "phase": "calibration",
            "source_id": "source_0",
            "replicate": 0,
            "recipe": "textfooler",
        },
        attack_fn=lambda *_: {
            "status": "failed",
            "source_queries": 3,
            "candidate_text": "same synthetic candidate",
            "source_raw_perturbed": "1",
        },
    )
    with pytest.raises(ValueError, match="identical content"):
        module.audit_attack_shard(
            receipt,
            protocol.model_dump(),
            manifest,
            private,
            _execution_parameters(),
            corpus.model_dump(),
            _clean_rows(protocol, corpus, receipt),
            predict_fn=lambda _model, _descriptor, texts, _settings: ["0", "1"] * (len(texts) // 2),
        )
