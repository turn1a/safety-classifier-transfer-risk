"""Behavioral gates for prospective v2 evidence, independent of model execution."""

from __future__ import annotations

import pytest

from transfer_risk.lib.experiment_v2.protocol import (
    build_execution_manifest,
    content_digest,
    require_valid_preflight,
    validate_preflight,
)
from transfer_risk.lib.experiment_v2.schemas import CorpusEvidence, Protocol

from .fixtures import corpus_dict, evidence, protocol_dict


def test_complete_contract_validates_without_claiming_runtime_readiness():
    protocol, corpus, registration = evidence()
    report = validate_preflight(protocol, corpus, registration)
    assert report["status"] == "validated"
    assert report["execution_wiring"] == "implemented_gated"
    assert report["execution_context_checked"] is False
    assert report["rerun_ready"] is False
    manifest = build_execution_manifest(protocol, corpus, 20260616, report)
    assert manifest["execution_wiring"] == "implemented_gated"
    assert manifest["phases"]["acquisition_and_grouping"]["status"] == "external_inputs_required"
    assert manifest["phases"]["target_preparation"]["pipeline"] == "experiment_v2_target_prepare"
    assert manifest["phases"]["analysis"]["status"] == "implemented"
    assert len(manifest["replicate_seeds"]) == 3
    assert all(
        path.startswith("data/experiment_v2/synthetic-contract/")
        for path in manifest["private_artifacts"].values()
    )


@pytest.mark.parametrize("field", ["revision", "tokenizer_revision", "label_review_sha256"])
def test_missing_model_identity_or_label_review_blocks_preflight(field):
    protocol, corpus, registration = evidence()
    models = [
        asset.model_copy(update={field: None}) if i == 0 else asset
        for i, asset in enumerate(protocol.models)
    ]
    report = validate_preflight(
        protocol.model_copy(update={"models": models}), corpus, registration
    )
    assert report["status"] == "blocked"
    assert any(field in issue for issue in report["issues"])
    with pytest.raises(ValueError, match="blocked"):
        require_valid_preflight(report)


@pytest.mark.parametrize("revision", ["main", "v2", "a1b2c3d", "refs/pr/12"])
def test_mutable_or_short_model_revision_is_rejected(revision):
    protocol, corpus, registration = evidence()
    model = protocol.models[0].model_copy(update={"revision": revision})
    report = validate_preflight(
        protocol.model_copy(update={"models": [model, *protocol.models[1:]]}), corpus, registration
    )
    assert any("immutable" in issue and "revision" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "root",
    [
        "data/06_models",
        "data/experiment_v2/shared",
        "/absolute/run",
        "data/experiment_v2/synthetic-contract/../historical",
    ],
)
def test_historical_shared_absolute_or_traversal_output_path_is_rejected(root):
    protocol, corpus, registration = evidence()
    report = validate_preflight(
        protocol.model_copy(update={"private_root": root}), corpus, registration
    )
    assert any("private_root" in issue for issue in report["issues"])


def test_source_revision_group_evidence_and_human_original_evidence_are_required():
    protocol, corpus, registration = evidence()
    source = protocol.sources[0].model_copy(update={"revision": None})
    protocol = protocol.model_copy(update={"sources": [source, *protocol.sources[1:]]})
    row = corpus.records[-1].model_copy(update={"reviewers": [], "adjudication": "pending"})
    corpus = corpus.model_copy(
        update={"records": [*corpus.records[:-1], row], "grouping_evidence_sha256": None}
    )
    report = validate_preflight(protocol, corpus, registration)
    assert any("source" in issue and "revision" in issue for issue in report["issues"])
    assert any("grouping" in issue for issue in report["issues"])
    assert any("human" in issue for issue in report["issues"])


@pytest.mark.parametrize("key", ["group_id", "content_sha256"])
def test_no_family_or_exact_content_crosses_partition_boundaries(key):
    protocol, corpus, registration = evidence()
    original = corpus.records[0]
    heldout = corpus.records[-1].model_copy(update={key: getattr(original, key)})
    corpus = corpus.model_copy(update={"records": [*corpus.records[:-1], heldout]})
    report = validate_preflight(protocol, corpus, registration)
    assert any("overlap" in issue for issue in report["issues"])


def test_jailbreak_cannot_be_relabelled_as_prompt_injection():
    protocol, corpus, registration = evidence()
    row = corpus.records[-1].model_copy(update={"task": "jailbreak"})
    report = validate_preflight(
        protocol, corpus.model_copy(update={"records": [*corpus.records[:-1], row]}), registration
    )
    assert any("jailbreak" in issue for issue in report["issues"])


def test_target_or_identical_alias_cannot_be_in_own_surrogate_pool():
    protocol, corpus, registration = evidence()
    alias = protocol.models[2].model_copy(
        update={
            "repository": protocol.models[0].repository,
            "revision": protocol.models[0].revision,
        }
    )
    report = validate_preflight(
        protocol.model_copy(update={"models": [*protocol.models[:2], alias, *protocol.models[3:]]}),
        corpus,
        registration,
    )
    assert any("identical" in issue for issue in report["issues"])


def test_protocol_or_partition_changes_invalidate_registration():
    protocol, corpus, registration = evidence()
    protocol = protocol.model_copy(update={"source_query_budget": 101})
    report = validate_preflight(protocol, corpus, registration)
    assert any("registration" in issue and "digest" in issue for issue in report["issues"])
    assert registration.protocol_sha256 != content_digest(protocol)


def test_model_label_map_cannot_hide_an_unmapped_class():
    raw = protocol_dict()
    raw["models"][0]["output_labels"] = ["0", "1", "2"]
    _protocol, corpus, registration = evidence()
    report = validate_preflight(Protocol.model_validate(raw), corpus, registration)
    assert any("label_map" in issue for issue in report["issues"])


def test_empty_corpus_is_blocked_and_does_not_claim_completed_annotations():
    protocol, corpus, registration = evidence()
    corpus = corpus.model_copy(update={"records": [], "grouping_evidence_sha256": None})
    report = validate_preflight(protocol, corpus, registration)
    assert report["status"] == "blocked"
    assert report["corpus_records"] == 0
    assert report["human_original_review_complete"] is False
    assert corpus_dict()["records"]


def test_replication_requires_independently_prepared_target_evidence():
    protocol, corpus, registration = evidence()
    corpus = corpus.model_copy(update={"target_preparations": []})
    report = validate_preflight(protocol, corpus, registration)
    assert any("target preparation" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    "missing", [{"target_corpus_sha256": None}, {"grouping_method": None}, {"corpus_records": []}]
)
def test_prepared_target_requires_registered_private_content_inventory(missing):
    protocol, corpus, registration = evidence()
    preparation = corpus.target_preparations[0].model_copy(update=missing)
    corpus = corpus.model_copy(update={"target_preparations": [preparation]})
    registration = registration.model_copy(update={"corpus_sha256": content_digest(corpus)})
    report = validate_preflight(protocol, corpus, registration)
    assert report["status"] == "blocked"
    assert any("target preparation" in issue for issue in report["issues"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("core_content", "content overlap"),
        ("duplicate_train_content", "duplicate content"),
        ("duplicate_validation_content", "content"),
        ("origin_id", "group closure"),
        ("attack_family_id", "group closure"),
    ],
)
def test_prepared_inventory_rechecks_isolation_with_correct_registered_digests(change, message):
    protocol, corpus, registration = evidence()
    preparation = corpus.target_preparations[0]
    rows = list(preparation.corpus_records)
    if change == "core_content":
        heldout = next(
            row for row in corpus.records if row.partition == "evaluation" and row.label == 0
        )
        rows[0] = rows[0].model_copy(update={"content_sha256": heldout.content_sha256})
    elif change == "duplicate_train_content":
        rows[1] = rows[1].model_copy(update={"content_sha256": rows[0].content_sha256})
    elif change == "duplicate_validation_content":
        rows[4] = rows[4].model_copy(update={"content_sha256": rows[0].content_sha256})
    else:
        rows[1] = rows[1].model_copy(update={change: getattr(rows[0], change)})
    target_corpus = CorpusEvidence(
        grouping_method=preparation.grouping_method,
        grouping_evidence_sha256=preparation.grouping_evidence_sha256,
        records=rows,
    )
    preparation = preparation.model_copy(
        update={
            "corpus_records": rows,
            "target_corpus_sha256": content_digest(target_corpus),
            "origin_ids": sorted({row.origin_id for row in rows}),
            "attack_family_ids": sorted({row.attack_family_id for row in rows}),
        }
    )
    corpus = corpus.model_copy(update={"target_preparations": [preparation]})
    registration = registration.model_copy(update={"corpus_sha256": content_digest(corpus)})
    report = validate_preflight(protocol, corpus, registration)
    assert report["status"] == "blocked"
    assert any("target preparation" in issue and message in issue for issue in report["issues"])


def test_prepared_inventory_and_summary_ids_are_bound_to_target_corpus_digest():
    protocol, corpus, registration = evidence()
    preparation = corpus.target_preparations[0]
    rows = list(preparation.corpus_records)
    rows[0] = rows[0].model_copy(update={"content_sha256": "f" * 64})
    preparation = preparation.model_copy(
        update={"corpus_records": rows, "training_group_ids": ["unrecorded_training_group"]}
    )
    corpus = corpus.model_copy(update={"target_preparations": [preparation]})
    registration = registration.model_copy(update={"corpus_sha256": content_digest(corpus)})
    report = validate_preflight(protocol, corpus, registration)
    assert report["status"] == "blocked"
    assert any("target corpus digest" in issue for issue in report["issues"])
    assert any("inventory summary" in issue for issue in report["issues"])


def test_replication_target_must_use_another_backbone_family():
    protocol, corpus, registration = evidence()
    target = protocol.models[1].model_copy(update={"backbone_family": "deberta"})
    protocol = protocol.model_copy(
        update={"models": [protocol.models[0], target, *protocol.models[2:]]}
    )
    report = validate_preflight(protocol, corpus, registration)
    assert any("backbone" in issue for issue in report["issues"])


@pytest.mark.parametrize("field", ["origin_id", "attack_family_id"])
def test_origin_and_attack_family_cannot_cross_partitions_with_different_group_ids(field):
    protocol, corpus, registration = evidence()
    row = corpus.records[-1].model_copy(update={field: corpus.records[0].id})
    corpus = corpus.model_copy(update={"records": [*corpus.records[:-1], row]})
    report = validate_preflight(protocol, corpus, registration)
    assert any(field in issue and "overlap" in issue for issue in report["issues"])


def test_quoted_and_neutral_instruction_hard_negatives_are_required():
    protocol, corpus, registration = evidence()
    records = [
        row.model_copy(update={"negative_subtype": "ordinary"})
        if row.partition == "evaluation" and row.label == 0
        else row
        for row in corpus.records
    ]
    report = validate_preflight(
        protocol, corpus.model_copy(update={"records": records}), registration
    )
    assert any("hard negative" in issue for issue in report["issues"])


@pytest.mark.parametrize("field", ["origin_id", "attack_family_id", "content_sha256"])
def test_shared_identity_cannot_be_counted_as_multiple_independent_groups(field):
    protocol, corpus, registration = evidence()
    records = list(corpus.records)
    first, second = records[-2:]
    second = second.model_copy(update={field: getattr(first, field)})
    records[-1] = second
    report = validate_preflight(
        protocol, corpus.model_copy(update={"records": records}), registration
    )
    assert any(
        "group closure" in issue or "duplicate content" in issue for issue in report["issues"]
    )


def test_multiple_output_classes_cannot_silently_collapse_a_jailbreak_class():
    protocol, corpus, registration = evidence()
    model = protocol.models[0].model_copy(
        update={
            "output_labels": ["0", "1", "2"],
            "label_map": {"0": "benign", "1": "prompt_injection", "2": "prompt_injection"},
        }
    )
    report = validate_preflight(
        protocol.model_copy(update={"models": [model, *protocol.models[1:]]}), corpus, registration
    )
    assert any("label_map" in issue for issue in report["issues"])


def test_unpinned_auxiliary_attack_assets_block_the_core_protocol():
    protocol, corpus, registration = evidence()
    report = validate_preflight(
        protocol.model_copy(update={"attack_assets_sha256": None}), corpus, registration
    )
    assert report["status"] == "blocked"
    assert any("attack_assets_sha256" in issue for issue in report["issues"])


@pytest.mark.parametrize("reviewer", ["", "reviewer-with-trailing-space "])
def test_original_review_cannot_count_an_absent_or_noncanonical_reviewer(reviewer):
    protocol, corpus, registration = evidence()
    row = corpus.records[-1].model_copy(update={"reviewers": ["reviewer-a", reviewer]})
    corpus = corpus.model_copy(update={"records": [*corpus.records[:-1], row]})
    report = validate_preflight(protocol, corpus, registration)
    assert any("human original review" in issue for issue in report["issues"])
    assert report["human_original_review_complete"] is False
