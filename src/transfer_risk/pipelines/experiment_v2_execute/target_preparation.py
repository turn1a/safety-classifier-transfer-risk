"""Prepare an independent frozen target under an earlier, separate registered gate."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from transfer_risk.lib.experiment_v2.protocol import (
    component_seed,
    content_digest,
    is_digest,
    original_review_complete,
    parse_timestamp,
    split_identity_issues,
)
from transfer_risk.lib.experiment_v2.schemas import (
    CanonicalLabel,
    CorpusEvidence,
    CorpusRecord,
    EvidenceRecord,
    ModelIdentity,
    Protocol,
    RegistrationEvidence,
    TargetPreparation,
)
from transfer_risk.pipelines.experiment_v2_execute import runtime
from transfer_risk.pipelines.experiment_v2_execute.nodes import (
    _check_context,
    _check_structure,
    _fail,
    phase_frame,
)

_BINARY_LABEL_COUNT = 2


class TargetPreparationPlan(EvidenceRecord):
    """Pin independent-target inputs and intended core membership before target training."""

    run_id: str
    target_role: str
    model_id: str
    repository: str
    revision: str | None
    tokenizer_repository: str
    tokenizer_revision: str | None
    backbone_family: str
    root_seed: int
    execution_ref: str | None
    max_tokens: int
    output_labels: list[str]
    label_map: dict[str, CanonicalLabel]
    label_review_sha256: str | None
    training: dict[str, Any]
    min_train_groups: int
    min_validation_groups: int
    core_corpus_sha256: str | None


class TargetPreparationEvidence(EvidenceRecord):
    """Seal target-only grouping, human task review, source revisions, and registration."""

    corpus: CorpusEvidence
    source_revisions: dict[str, str]
    annotation_evidence_sha256: str | None
    registration: RegistrationEvidence


def _validate_target_isolation(records: list[CorpusRecord], core: CorpusEvidence) -> None:
    """Reject repeated target originals and target/core content or identity overlap.

    Args:
        records: Target-only registered train and validation original identities.
        core: Intended core corpus excluded from all target preparation data.

    Raises:
        ValueError: If content, group, origin, or family independence is violated.
    """
    if len({row.id for row in records}) != len(records):
        _fail("target preparation has duplicate original ids")
    identity_issues = split_identity_issues(records)
    if identity_issues:
        _fail("target preparation: " + "; ".join(identity_issues))
    if {row.content_sha256 for row in records} & {row.content_sha256 for row in core.records}:
        _fail("target and core exact content must be disjoint")
    for dimension in ("group_id", "origin_id", "attack_family_id"):
        core_ids = {getattr(row, dimension) for row in core.records}
        target_ids = {getattr(row, dimension) for row in records}
        train_ids = {getattr(row, dimension) for row in records if row.partition == "train"}
        validation_ids = {
            getattr(row, dimension) for row in records if row.partition == "validation"
        }
        if core_ids & target_ids or train_ids & validation_ids or "" in target_ids:
            _fail("target and core train/validation groups, origins and families must be disjoint")


def _validate_target_data(
    plan: TargetPreparationPlan, evidence: TargetPreparationEvidence, core: CorpusEvidence
) -> None:
    """Require separate target/core content and closed target train/validation identities.

    Args:
        plan: Registered target training sample-size requirements.
        evidence: Target-only grouping and original review evidence.
        core: Intended future core-study metadata, without prepared-target hashes.

    Raises:
        ValueError: If any overlap, missing review, or insufficient partition occurs.
    """
    records = evidence.corpus.records
    if (
        not records
        or evidence.corpus.target_preparations
        or any(row.partition not in {"train", "validation"} for row in records)
    ):
        _fail("target preparation requires only its separate train and validation originals")
    _validate_target_isolation(records, core)
    if (
        not evidence.corpus.grouping_method
        or not is_digest(evidence.corpus.grouping_evidence_sha256)
        or not is_digest(evidence.annotation_evidence_sha256)
    ):
        _fail("target preparation requires grouping and human task-review evidence")
    for row in records:
        if (
            not original_review_complete(row)
            or not is_digest(row.content_sha256)
            or not is_digest(row.task_context_sha256)
            or not is_digest(row.trusted_instruction_sha256)
            or row.trust_boundary != "untrusted_content"
            or not row.origin_id
            or not row.attack_family_id
            or (row.label == 1 and (row.task != "prompt_injection" or not row.objective_id))
            or (row.label == 0 and (row.task != "benign" or row.negative_subtype is None))
        ):
            _fail("target originals require complete independent human task and context review")
        if row.source_id not in evidence.source_revisions:
            _fail("target original source lacks a pinned source revision")
    for partition, minimum in (
        ("train", plan.min_train_groups),
        ("validation", plan.min_validation_groups),
    ):
        rows = [row for row in records if row.partition == partition]
        if minimum <= 0 or any(
            len({row.group_id for row in rows if row.label == label}) < minimum for label in (0, 1)
        ):
            _fail("target train/validation require the registered positive and benign group counts")
        if not {"quoted_instruction", "neutral_instruction"}.issubset(
            {row.negative_subtype for row in rows if row.label == 0}
        ):
            _fail("target train/validation require quoted and neutral instruction hard negatives")
    if not evidence.source_revisions or any(
        not is_digest(revision, lengths=(40, 64)) for revision in evidence.source_revisions.values()
    ):
        _fail("target preparation requires immutable source revisions")


def gate_target_preparation(
    parameters: dict[str, Any],
    target_evidence: dict[str, Any],
    core_parameters: dict[str, Any],
    core_corpus_evidence: dict[str, Any],
    context: dict[str, Any],
    root_seed: int,
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate independent-target training before its weights enter the core registration.

    Args:
        parameters: Separately registered target-preparation plan.
        target_evidence: Target-only metadata and external registration.
        core_parameters: Intended future protocol, with unrealized target hashes allowed.
        core_corpus_evidence: Intended core membership used for target-data exclusions.
        context: Fresh actual clean isolated source and host facts.
        root_seed: Shared project root seed.
        structure: Intended committed core axes bound by the target-preparation factory.

    Returns:
        Fresh target-only receipt without reading text or loading the backbone.
    """
    plan = TargetPreparationPlan.model_validate(parameters)
    evidence = TargetPreparationEvidence.model_validate(target_evidence)
    protocol = Protocol.model_validate(core_parameters)
    core = CorpusEvidence.model_validate(core_corpus_evidence)
    _check_context(context, plan.execution_ref)
    _check_structure(protocol, structure)
    if (
        plan.root_seed != root_seed
        or root_seed != protocol.root_seed
        or plan.run_id != protocol.run_id
    ):
        _fail("target preparation uses a different run or project root seed")
    if plan.core_corpus_sha256 != content_digest(core):
        _fail("target preparation requires the sealed intended core corpus identity")
    roles = {role.id: role for role in protocol.target_roles}
    if (
        plan.target_role not in roles
        or roles[plan.target_role].model_id != plan.model_id
        or (roles[plan.target_role].preparation != "independently_prepared")
    ):
        _fail("target preparation does not name the intended independent target role")
    primary = next(role for role in protocol.target_roles if role.purpose == "primary")
    primary_model = next(model for model in protocol.models if model.id == primary.model_id)
    if plan.backbone_family == primary_model.backbone_family:
        _fail("independent replication target must use a distinct backbone family")
    if (
        not is_digest(plan.revision, lengths=(40,))
        or not is_digest(plan.tokenizer_revision, lengths=(40,))
        or not is_digest(plan.label_review_sha256)
        or set(plan.output_labels) != set(plan.label_map)
        or len(plan.output_labels) != _BINARY_LABEL_COUNT
        or len(set(plan.output_labels)) != _BINARY_LABEL_COUNT
        or set(plan.label_map.values()) != {"benign", "prompt_injection"}
        or plan.max_tokens != plan.training.get("max_seq_len")
        or not plan.training
    ):
        _fail("target preparation requires pinned inputs, reviewed labels, and fixed training")
    _validate_target_data(plan, evidence, core)
    registration = evidence.registration
    if (
        registration.protocol_sha256 != content_digest(plan)
        or registration.corpus_sha256 != content_digest(evidence.corpus)
        or not str(registration.registry_uri).startswith("https://")
        or not is_digest(registration.evidence_sha256)
        or parse_timestamp(registration.registered_at) >= datetime.now(UTC)
    ):
        _fail("target plan and target-only corpus require external registration before preparation")
    return {
        "preparation_protocol_sha256": content_digest(plan),
        "target_corpus_sha256": content_digest(evidence.corpus),
        "execution_ref": plan.execution_ref,
        "training_seed": component_seed(root_seed, 0, f"model:{plan.model_id}"),
        "hardware": context["hardware"],
        "started_at": datetime.now(UTC).isoformat(),
    }


def prepare_target(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    target_evidence: dict[str, Any],
    train_texts: list[dict[str, Any]],
    validation_texts: list[dict[str, Any]],
    backbone: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the target using existing corrected loops and target-only sealed originals.

    Args:
        receipt: Fresh earlier target-preparation gate.
        parameters: Registered preparation plan.
        target_evidence: Sealed target-only grouping and task-review evidence.
        train_texts: Private target-only training originals.
        validation_texts: Private target-only validation originals.
        backbone: Catalog-loaded pinned upstream target backbone.
        settings: Device policy; fixed training settings come from the registered plan.

    Returns:
        A catalog-owned checkpoint bundle and non-text preparation metadata.
    """
    plan = TargetPreparationPlan.model_validate(parameters)
    evidence = TargetPreparationEvidence.model_validate(target_evidence)
    if receipt["preparation_protocol_sha256"] != content_digest(plan) or (
        receipt["target_corpus_sha256"] != content_digest(evidence.corpus)
    ):
        _fail("target preparation receipt is stale")
    expected = {
        "repository": plan.repository,
        "revision": plan.revision,
        "tokenizer_repository": plan.tokenizer_repository,
        "tokenizer_revision": plan.tokenizer_revision,
    }
    if backbone.get("input_identity") != expected:
        _fail("target backbone differs from separately registered immutable pins")
    identity = ModelIdentity(
        id=plan.model_id,
        repository=plan.repository,
        backbone_family=plan.backbone_family,
        revision=plan.revision,
        tokenizer_repository=plan.tokenizer_repository,
        tokenizer_revision=plan.tokenizer_revision,
        kind="finetune",
        output_labels=plan.output_labels,
        label_map=plan.label_map,
        label_review_sha256=plan.label_review_sha256,
        max_tokens=plan.max_tokens,
        decision_rule="argmax",
        comparison_block="independent-target-preparation",
        training_recipe_sha256=content_digest(plan.training),
    )
    train = phase_frame(train_texts, evidence.corpus, "train")
    validation = phase_frame(validation_texts, evidence.corpus, "validation")
    bundle, training_metadata = runtime.train_model(
        identity,
        deepcopy(backbone),
        train,
        validation,
        plan.training,
        settings["device_policy"],
        receipt["training_seed"],
    )
    rows = evidence.corpus.records
    return bundle, {
        "target_role": plan.target_role,
        "target_corpus_sha256": receipt["target_corpus_sha256"],
        "grouping_method": evidence.corpus.grouping_method,
        "corpus_records": [row.model_dump(mode="json") for row in rows],
        "training_group_ids": sorted({row.group_id for row in rows if row.partition == "train"}),
        "validation_group_ids": sorted(
            {row.group_id for row in rows if row.partition == "validation"}
        ),
        "origin_ids": sorted({row.origin_id for row in rows}),
        "attack_family_ids": sorted({row.attack_family_id for row in rows}),
        "source_revisions": evidence.source_revisions,
        "preparation_protocol_sha256": content_digest(plan),
        "grouping_evidence_sha256": evidence.corpus.grouping_evidence_sha256,
        "annotation_evidence_sha256": evidence.annotation_evidence_sha256,
        "training_metadata": training_metadata,
        "training_seed": receipt["training_seed"],
        "execution_ref": receipt["execution_ref"],
        "completed_at": datetime.now(UTC).isoformat(),
    }


def export_target_preparation(
    receipt: dict[str, Any], descriptor: dict[str, str], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Export realized target fingerprints for the later frozen core protocol.

    Args:
        receipt: Fresh earlier target-preparation gate.
        descriptor: Fingerprints of the exact catalog-saved target checkpoint.
        metadata: Non-text completed target training and grouping evidence.

    Returns:
        A strict TargetPreparation record with no raw data or model object.
    """
    if (
        metadata["preparation_protocol_sha256"] != receipt["preparation_protocol_sha256"]
        or metadata["target_corpus_sha256"] != receipt["target_corpus_sha256"]
    ):
        _fail("target preparation metadata has a different registered plan or corpus")
    fields = set(TargetPreparation.model_fields) - {"checkpoint_sha256", "tokenizer_sha256"}
    return TargetPreparation.model_validate(
        {
            **{key: metadata[key] for key in fields},
            "checkpoint_sha256": descriptor["checkpoint_sha256"],
            "tokenizer_sha256": descriptor["tokenizer_sha256"],
        }
    ).model_dump(mode="json")
