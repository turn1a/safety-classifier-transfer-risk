"""Validate prospective v2 evidence and describe gated execution phases.

These pure functions validate supplied records. They perform no I/O, model loading,
training, attacks, inference, or authentication of human and registration attestations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from transfer_risk.lib.seeds import derive_seeds

if TYPE_CHECKING:
    from transfer_risk.lib.experiment_v2.schemas import (
        CorpusEvidence,
        CorpusRecord,
        Partition,
        Protocol,
        RegistrationEvidence,
        TargetPreparation,
    )

_PARTITIONS: tuple[Partition, ...] = ("train", "validation", "probe", "calibration", "evaluation")
_RECIPES = {"textfooler", "bae", "pwws", "deepwordbug", "textbugger"}
_MIN_REPLICATES = 3
_BINARY_CLASSES = 2
HUMAN_REVIEWERS = 2
_MIN_BOOTSTRAP_REPLICATES = 100


def content_digest(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    """Hash a JSON-compatible evidence object with deterministic serialization.

    Args:
        value: Evidence model, mapping, or list without non-finite numbers.

    Returns:
        The lowercase SHA256 digest of canonical JSON bytes.
    """
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def partition_digest(corpus: CorpusEvidence, partition: str) -> str:
    """Hash the ordered membership and review evidence of one independent partition.

    Args:
        corpus: Non-text corpus and group evidence.
        partition: One preregistered partition name.

    Returns:
        A SHA256 digest that changes with membership, order, or task evidence.
    """
    return content_digest(
        [row.model_dump(mode="json") for row in corpus.records if row.partition == partition]
    )


def component_seed(root_seed: int, replicate: int, component: str) -> int:
    """Derive a stable component seed through the project's root-seed helper.

    Args:
        root_seed: Root seed shared by the registered protocol.
        replicate: Nonnegative preregistered replicate index.
        component: Stable model, attack-cell, or bootstrap identifier.

    Returns:
        An independent uint32 seed, unaffected by iteration order.
    """
    component_key = int.from_bytes(hashlib.sha256(component.encode()).digest()[:4], "big")
    sequence = np.random.SeedSequence(
        derive_seeds(root_seed).numpy, spawn_key=(replicate, component_key)
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def is_digest(value: str | None, *, lengths: tuple[int, ...] = (64,)) -> bool:
    """Check whether an evidence identity is a full lowercase hexadecimal digest.

    Args:
        value: Supplied revision or artifact hash.
        lengths: Full digest lengths accepted by this evidence boundary.

    Returns:
        Whether the identity has an accepted length and hexadecimal form.
    """
    return value is not None and len(value) in lengths and bool(re.fullmatch("[0-9a-f]+", value))


def parse_timestamp(value: str | None) -> datetime:
    """Parse a timezone-aware evidence timestamp without assuming a local timezone.

    Args:
        value: An ISO8601 timestamp supplied in an evidence record.

    Returns:
        The timezone-aware timestamp.

    Raises:
        ValueError: If the timestamp is absent, malformed, or timezone-naive.
    """
    if value is None:
        msg = "evidence timestamp is absent"
        raise ValueError(msg)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        msg = "evidence timestamp must include a timezone"
        raise ValueError(msg)
    return parsed


def _model_issues(protocol: Protocol) -> list[str]:
    """Check immutable inputs, label semantics, and matched detector settings.

    Args:
        protocol: Prospective model and task specification.

    Returns:
        Human-readable blockers without model or prompt content.
    """
    issues: list[str] = []
    ids = [model.id for model in protocol.models]
    if len(set(ids)) != len(ids) or not ids:
        issues.append("models require unique, nonempty ids")
    for model in protocol.models:
        for field in ("revision", "tokenizer_revision"):
            if not is_digest(getattr(model, field), lengths=(40, 64)):
                issues.append(f"model {model.id}: {field} requires an immutable full revision")
        evidence_fields = ["label_review_sha256"]
        if model.kind != "frozen":
            evidence_fields.append("training_recipe_sha256")
        for field in evidence_fields:
            if not is_digest(getattr(model, field)):
                issues.append(f"model {model.id}: {field} evidence is absent or invalid")
        if (
            len(model.output_labels) != _BINARY_CLASSES
            or set(model.label_map) != set(model.output_labels)
            or set(model.label_map.values()) != {"benign", "prompt_injection"}
            or any("jailbreak" in name.casefold() for name in model.output_labels)
        ):
            issues.append(f"model {model.id}: label_map must cover the reviewed pure task outputs")
        if (
            not model.repository
            or not model.tokenizer_repository
            or not model.backbone_family
            or model.max_tokens <= 0
        ):
            issues.append(f"model {model.id}: model/tokenizer identity or token limit is absent")
    return issues


def _role_issues(protocol: Protocol) -> list[str]:
    """Reject target self-selection and unmatched candidate comparison blocks.

    Args:
        protocol: Fixed target roles and candidate pool.

    Returns:
        Role and selection-resolution blockers.
    """
    issues: list[str] = []
    models = {model.id: model for model in protocol.models}
    purposes = {role.purpose for role in protocol.target_roles}
    role_ids = [role.id for role in protocol.target_roles]
    if purposes != {"primary", "replication"} or len(set(role_ids)) != len(role_ids):
        issues.append("target_roles require distinct primary and replication roles")
    target_identities: set[tuple[str, str | None]] = set()
    for role in protocol.target_roles:
        if role.model_id not in models:
            issues.append(f"target role {role.id}: unknown target model")
            continue
        target = models[role.model_id]
        identity = (target.repository, target.revision)
        if identity in target_identities:
            issues.append(f"target role {role.id}: target roles require independent identities")
        target_identities.add(identity)
        candidates = [models[name] for name in role.candidate_ids if name in models]
        if len(candidates) != len(set(role.candidate_ids)) or not candidates:
            issues.append(f"target role {role.id}: candidate ids must be unique known models")
        if any((item.repository, item.revision) == identity for item in candidates):
            issues.append(f"target role {role.id}: identical target is in its own candidate pool")
        settings = {
            (item.max_tokens, item.decision_rule, item.comparison_block)
            for item in [target, *candidates]
        }
        if len(settings) != 1:
            issues.append(f"target role {role.id}: detector windows and comparison blocks differ")
        if not 0 < protocol.selection_k < len(candidates):
            issues.append(f"target role {role.id}: selection_k must be smaller than the pool")
        if not protocol.selection_k < protocol.minimum_eligible_sources <= len(candidates):
            issues.append(f"target role {role.id}: minimum eligible source count is infeasible")
    return issues


def _context_issues(row: CorpusRecord) -> list[str]:
    """Require the context needed to distinguish lower-trust injection from jailbreak.

    Args:
        row: Human-reviewed original with fixed task and trust-boundary metadata.

    Returns:
        Missing-context and task-validity blockers.
    """
    valid = (
        is_digest(row.task_context_sha256)
        and is_digest(row.trusted_instruction_sha256)
        and row.trust_boundary == "untrusted_content"
        and (row.label == 0 or bool(row.objective_id))
    )
    return (
        []
        if valid
        else [f"corpus record {row.id}: task context and trust-boundary evidence required"]
    )


def split_identity_issues(records: list[CorpusRecord]) -> list[str]:
    """Check shared group, origin, family, and exact-content isolation invariants.

    Args:
        records: Non-text identities from any independently partitioned corpus.

    Returns:
        Cross-partition overlaps, repeated content, or inflated independent-group counts.
    """
    issues = []
    for field, label in (
        ("group_id", "group"),
        ("content_sha256", "exact content"),
        ("origin_id", "origin_id"),
        ("attack_family_id", "attack_family_id"),
    ):
        partitions: dict[str, set[str]] = defaultdict(set)
        groups_by_identity: dict[str, set[str]] = defaultdict(set)
        counts: Counter[str] = Counter()
        for row in records:
            identity = getattr(row, field)
            partitions[identity].add(row.partition)
            groups_by_identity[identity].add(row.group_id)
            counts[identity] += 1
            if not identity:
                issues.append(f"corpus record {row.id}: {field} is absent")
        if any(len(values) > 1 for values in partitions.values()):
            issues.append(f"{label} overlap crosses partition boundaries")
        if any(len(groups) > 1 for groups in groups_by_identity.values()):
            issues.append(f"{label} violates independent group closure")
        if field == "content_sha256" and any(count > 1 for count in counts.values()):
            issues.append("duplicate content cannot supply repeated independent originals")
    return issues


def _split_identity_issues(protocol: Protocol, corpus: CorpusEvidence) -> list[str]:
    """Require closed identities and reserved calibration/evaluation hard negatives.

    Args:
        protocol: Preregistered minimum hard-negative group count.
        corpus: Reviewed corpus identities and independent partition assignments.

    Returns:
        Shared identity blockers and insufficient hard-negative group counts.
    """
    issues = split_identity_issues(corpus.records)
    for partition in ("calibration", "evaluation"):
        for subtype in ("quoted_instruction", "neutral_instruction"):
            groups = {
                row.group_id
                for row in corpus.records
                if row.partition == partition and row.label == 0 and row.negative_subtype == subtype
            }
            if len(groups) < protocol.min_hard_negative_groups:
                issues.append(f"partition {partition}: {subtype} hard negatives are absent")
    return issues


def _preparation_inventory_issues(item: TargetPreparation, corpus: CorpusEvidence) -> list[str]:
    """Recheck a prepared target's registered private corpus against the core study.

    Args:
        item: Retained target-only records, summary identities, and corpus digest.
        corpus: Core partitions whose identities and content must remain independent.

    Returns:
        Missing, stale, inconsistent, or overlapping target-preparation evidence.
    """
    prefix = f"target preparation for {item.target_role}"
    rows = item.corpus_records
    issues = []
    if (
        not item.grouping_method
        or not rows
        or any(row.partition not in {"train", "validation"} for row in rows)
    ):
        issues.append(f"{prefix} requires a complete target-only content inventory")
    target_corpus = {
        "grouping_method": item.grouping_method,
        "grouping_evidence_sha256": item.grouping_evidence_sha256,
        "records": [row.model_dump(mode="json") for row in rows],
        "target_preparations": [],
    }
    if item.target_corpus_sha256 != content_digest(target_corpus):
        issues.append(f"{prefix} target corpus digest differs from retained metadata")
    actual = {
        "training_group_ids": {row.group_id for row in rows if row.partition == "train"},
        "validation_group_ids": {row.group_id for row in rows if row.partition == "validation"},
        "origin_ids": {row.origin_id for row in rows},
        "attack_family_ids": {row.attack_family_id for row in rows},
    }
    if any(
        set(getattr(item, field)) != identities or len(getattr(item, field)) != len(identities)
        for field, identities in actual.items()
    ):
        issues.append(f"{prefix} inventory summary differs from its retained records")
    if len({row.id for row in rows}) != len(rows):
        issues.append(f"{prefix} has duplicate target original ids")
    if {row.content_sha256 for row in rows} & {row.content_sha256 for row in corpus.records}:
        issues.append(f"{prefix} has exact content overlap with the core study")
    issues.extend(f"{prefix}: {issue}" for issue in split_identity_issues(rows))
    if any(
        not original_review_complete(row)
        or row.source_id not in item.source_revisions
        or not is_digest(row.content_sha256)
        or row.task != ("prompt_injection" if row.label else "benign")
        or (row.label == 0 and row.negative_subtype is None)
        or _context_issues(row)
        for row in rows
    ):
        issues.append(f"{prefix} requires reviewed task/context and pinned original identities")
    return issues


def _preparation_issues(protocol: Protocol, corpus: CorpusEvidence) -> list[str]:
    """Require a separately prepared target with another backbone and disjoint origins.

    Args:
        protocol: Fixed historical and independently prepared target roles.
        corpus: Preparation hashes and group metadata separate from core study partitions.

    Returns:
        Target-preparation and backbone-diversity blockers.
    """
    issues = []
    models = {model.id: model for model in protocol.models}
    preparations = {item.target_role: item for item in corpus.target_preparations}
    primary_families = {
        models[role.model_id].backbone_family
        for role in protocol.target_roles
        if role.purpose == "primary" and role.model_id in models
    }
    study_groups = {row.group_id for row in corpus.records}
    study_origins = {row.origin_id for row in corpus.records}
    study_families = {row.attack_family_id for row in corpus.records}
    independent = [
        role for role in protocol.target_roles if role.preparation == "independently_prepared"
    ]
    if not independent:
        issues.append("target preparation requires an independently prepared replication target")
    if len(preparations) != len(corpus.target_preparations):
        issues.append("target preparation evidence is duplicated")
    for role in independent:
        if role.model_id not in models:
            continue
        model = models[role.model_id]
        if model.backbone_family in primary_families:
            issues.append("independent replication requires another target backbone family")
        if role.id not in preparations:
            issues.append(f"target preparation for {role.id} is absent")
            continue
        item = preparations[role.id]
        issues.extend(_preparation_inventory_issues(item, corpus))
        hashes = (
            item.preparation_protocol_sha256,
            item.target_corpus_sha256,
            item.grouping_evidence_sha256,
            item.annotation_evidence_sha256,
            item.checkpoint_sha256,
            item.tokenizer_sha256,
        )
        if not all(is_digest(value) for value in hashes):
            issues.append(f"target preparation for {role.id} lacks immutable preparation evidence")
        if not item.source_revisions or not all(
            is_digest(value, lengths=(40, 64)) for value in item.source_revisions.values()
        ):
            issues.append(f"target preparation for {role.id} lacks pinned training sources")
        train, validation = set(item.training_group_ids), set(item.validation_group_ids)
        if not train or not validation or train & validation or (train | validation) & study_groups:
            issues.append(f"target preparation for {role.id} has absent or overlapping groups")
        if (
            not item.origin_ids
            or not item.attack_family_ids
            or (
                set(item.origin_ids) & study_origins or set(item.attack_family_ids) & study_families
            )
        ):
            issues.append(
                f"target preparation for {role.id} has absent or overlapping origins/families"
            )
        if (
            model.revision != item.checkpoint_sha256
            or model.tokenizer_revision != item.tokenizer_sha256
        ):
            issues.append(
                f"target preparation for {role.id} does not pin the frozen realized target"
            )
    return issues


def original_review_complete(row: CorpusRecord) -> bool:
    """Check completed original-review metadata and canonical human identifiers.

    Args:
        row: Non-text original review record, including the adjudication state.

    Returns:
        Whether review is complete with two distinct nonempty canonical reviewer ids.
        This structural check does not authenticate the people or their judgments.
    """
    return (
        row.adjudication == "complete"
        and len(set(row.reviewers)) >= HUMAN_REVIEWERS
        and all(reviewer and reviewer == reviewer.strip() for reviewer in row.reviewers)
    )


def _corpus_issues(protocol: Protocol, corpus: CorpusEvidence) -> list[str]:
    """Check task purity, human originals, and leakage across independent groups.

    Args:
        protocol: Source identities and minimum partition sizes.
        corpus: Supplied group assignments and original review records.

    Returns:
        Corpus and annotation-evidence blockers.
    """
    issues: list[str] = []
    sources = {source.id for source in protocol.sources}
    roles = {role.id for role in protocol.target_roles}
    if not corpus.grouping_method or not is_digest(corpus.grouping_evidence_sha256):
        issues.append("grouping method and grouping_evidence_sha256 evidence are required")
    group_partitions: dict[str, set[str]] = defaultdict(set)
    hash_partitions: dict[str, set[str]] = defaultdict(set)
    ids = [row.id for row in corpus.records]
    if len(set(ids)) != len(ids) or not ids:
        issues.append("corpus records require unique, nonempty ids")
    for row in corpus.records:
        group_partitions[row.group_id].add(row.partition)
        hash_partitions[row.content_sha256].add(row.partition)
        if row.source_id not in sources or not row.group_id or not is_digest(row.content_sha256):
            issues.append(f"corpus record {row.id}: source, group or content identity is invalid")
        issues.extend(_context_issues(row))
        expected_task = "prompt_injection" if row.label == 1 else "benign"
        if row.task != expected_task:
            issues.append(f"corpus record {row.id}: {row.task} cannot map to the pure task label")
        if not original_review_complete(row):
            issues.append(
                f"corpus record {row.id}: independent human original review is incomplete"
            )
        if row.partition in {"probe", "calibration", "evaluation"} and (
            set(row.known_target_training_roles) & roles
        ):
            issues.append(f"corpus record {row.id}: known target-training overlap in held-out data")
    if any(len(parts) > 1 for parts in group_partitions.values()):
        issues.append("group overlap crosses partition boundaries")
    if any(len(parts) > 1 for parts in hash_partitions.values()):
        issues.append("exact content overlap crosses partition boundaries")
    for partition in _PARTITIONS:
        rows = [row for row in corpus.records if row.partition == partition]
        class_groups = [{row.group_id for row in rows if row.label == label} for label in (0, 1)]
        if min(map(len, class_groups)) < protocol.min_groups.get(partition, 1):
            issues.append(f"partition {partition}: insufficient independent groups")
        if {row.label for row in rows} != {0, 1}:
            issues.append(
                f"partition {partition}: reviewed benign and prompt-injection rows required"
            )
    return issues


def _source_issues(protocol: Protocol) -> list[str]:
    """Require immutable data, task reviews, and documented contamination limits.

    Args:
        protocol: Prospective source identities.

    Returns:
        Source pinning and task-review blockers.
    """
    issues: list[str] = []
    ids = [source.id for source in protocol.sources]
    if not ids or len(set(ids)) != len(ids):
        issues.append("sources require unique, nonempty ids")
    for source in protocol.sources:
        if not is_digest(source.revision, lengths=(40, 64)):
            issues.append(f"source {source.id}: revision requires an immutable full identity")
        for field in ("content_sha256", "task_review_sha256", "exposure_review_sha256"):
            if not is_digest(getattr(source, field)):
                issues.append(f"source {source.id}: {field} evidence is absent or invalid")
        if "jailbreak" in source.id.casefold():
            issues.append(f"source {source.id}: jailbreak sources belong to a separate study")
    return issues


def _design_issues(protocol: Protocol) -> list[str]:
    """Check registered budgets, decision rules, and isolated output roots.

    Args:
        protocol: Prospective run specification.

    Returns:
        Invalid design and output-path blockers.
    """
    issues: list[str] = []
    if not re.fullmatch("[a-z][a-z0-9-]{3,63}", protocol.run_id):
        issues.append("run_id must be a dedicated lowercase run identity")
    for field, prefix in (
        ("private_root", "data/experiment_v2"),
        ("public_root", "docs/artifacts/experiment_v2"),
    ):
        if getattr(protocol, field) != f"{prefix}/{protocol.run_id}":
            issues.append(f"{field} must be the isolated path {prefix}/{protocol.run_id}")
    if not is_digest(protocol.execution_ref, lengths=(40,)):
        issues.append("execution_ref requires a full frozen Git commit")
    for field in (
        "attack_spec_sha256",
        "attack_assets_sha256",
        "input_policy_sha256",
        "annotation_protocol_sha256",
        "power_analysis_sha256",
    ):
        if not is_digest(getattr(protocol, field)):
            issues.append(f"{field} evidence is absent or invalid")
    if (
        len(set(protocol.replicates)) < _MIN_REPLICATES
        or len(set(protocol.replicates)) != len(protocol.replicates)
        or any(index < 0 for index in protocol.replicates)
    ):
        issues.append("replicates require at least three distinct nonnegative indices")
    if (
        not protocol.recipes
        or len(set(protocol.recipes)) != len(protocol.recipes)
        or (set(protocol.recipes) - _RECIPES)
    ):
        issues.append("recipes must be a unique subset of the five supported single-turn recipes")
    if protocol.source_query_budget <= 1:
        issues.append("source_query_budget must include the clean prediction and search queries")
    if (
        protocol.min_hard_negative_groups <= 0 or set(protocol.min_groups) != set(_PARTITIONS)
    ) or any(size <= 0 for size in protocol.min_groups.values()):
        issues.append("min_groups must preregister a positive size for every independent partition")
    bounds = (protocol.utility_min_tpr, protocol.utility_max_fpr, protocol.minimum_effect)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in bounds):
        issues.append("utility operating points and minimum_effect must be probabilities")
    if not 0 < protocol.alpha < 1 or protocol.bootstrap_replicates < _MIN_BOOTSTRAP_REPLICATES:
        issues.append(
            "alpha and bootstrap_replicates do not define the registered interval procedure"
        )
    return issues


def _registration_issues(
    protocol: Protocol, corpus: CorpusEvidence, registration: RegistrationEvidence
) -> list[str]:
    """Require a timestamped external attestation matching the current protocol and corpus.

    Args:
        protocol: Frozen prospective run specification.
        corpus: Reviewed partition and group evidence.
        registration: Supplied external registration metadata.

    Returns:
        Missing, stale, or malformed registration blockers.
    """
    issues: list[str] = []
    if not registration.registry_uri or not registration.registry_uri.startswith("https://"):
        issues.append("registration requires an external HTTPS registry identity")
    if not is_digest(registration.evidence_sha256):
        issues.append("registration evidence_sha256 is absent or invalid")
    try:
        parse_timestamp(registration.registered_at)
    except ValueError:
        issues.append("registration registered_at must be a timezone-aware timestamp")
    if registration.protocol_sha256 != content_digest(protocol):
        issues.append("registration protocol digest does not match the current specification")
    if registration.corpus_sha256 != content_digest(corpus):
        issues.append(
            "registration corpus digest does not match partition and human review evidence"
        )
    return issues


def validate_preflight(
    protocol: Protocol, corpus: CorpusEvidence, registration: RegistrationEvidence
) -> dict[str, Any]:
    """Return all validation blockers without claiming that heavy execution is available.

    Args:
        protocol: Prospective run identities, rules, budgets, and output roots.
        corpus: Human-reviewed grouped corpus metadata, without prompts.
        registration: External registration attestation for exact protocol and corpus hashes.

    Returns:
        A persistable report. Validation success is distinct from rerun readiness.
    """
    issues = [
        *_design_issues(protocol),
        *_model_issues(protocol),
        *_source_issues(protocol),
        *_role_issues(protocol),
        *_corpus_issues(protocol, corpus),
        *_registration_issues(protocol, corpus, registration),
        *_split_identity_issues(protocol, corpus),
        *_preparation_issues(protocol, corpus),
    ]
    reviewed = bool(corpus.records) and all(original_review_complete(row) for row in corpus.records)
    return {
        "schema_version": 2,
        "status": "blocked" if issues else "validated",
        "issues": issues,
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "corpus_records": len(corpus.records),
        "partition_counts": dict(Counter(row.partition for row in corpus.records)),
        "human_original_review_complete": reviewed,
        "execution_wiring": "implemented_gated",
        "rerun_ready": False,
        "execution_context_checked": False,
        "attestation_limit": "Human identities and external attestations remain unauthenticated.",
    }


def require_valid_preflight(report: dict[str, Any]) -> dict[str, str]:
    """Fail closed after the preflight report has been persisted by Kedro.

    Args:
        report: Current report from validate_preflight, including all blockers.

    Returns:
        Hashes binding downstream target-free analysis to validated evidence.

    Raises:
        ValueError: If any preflight blocker remains.
    """
    if report.get("status") != "validated" or report.get("issues"):
        msg = "v2 preflight blocked: " + "; ".join(report.get("issues", []))
        raise ValueError(msg)
    return {key: str(report[key]) for key in ("protocol_sha256", "corpus_sha256")}


def build_execution_manifest(
    protocol: Protocol, corpus: CorpusEvidence, root_seed: int, report: dict[str, Any]
) -> dict[str, Any]:
    """Describe exact catalog phase boundaries and their separate execution gates.

    Args:
        protocol: Prospective run, fixed targets, matched candidate pool, and budgets.
        corpus: Reviewed partition metadata.
        root_seed: Registered project root seed.
        report: Current preflight report, including an honestly blocked state.

    Returns:
        A structured contract identifying phase executors, gates, and external evidence.
    """
    root = protocol.private_root
    replication_model = next(
        (role.model_id for role in protocol.target_roles if role.purpose == "replication"),
        "{replication_model_id}",
    )
    paths: dict[str, str] = {
        **{name: f"{root}/corpus/{name}.json" for name in _PARTITIONS},
        "attack_asset_manifest": f"{root}/attack_assets/manifest.json",
        "attack_asset_cache": f"{root}/attack_assets/cache",
        "target_train": f"{root}/target_preparation/corpus/train.json",
        "target_validation": f"{root}/target_preparation/corpus/validation.json",
        "prepared_target": f"{root}/models/{replication_model}/prepared",
        "target_preparation_record": f"{root}/target_preparation/preparation_record.json",
        "checkpoints": f"{root}/models/{{model_id}}/replicate-{{replicate}}",
        "model_metadata": f"{root}/model_metadata/{{model_id}}/replicate-{{replicate}}.json",
        "models_manifest": f"{root}/analysis/models_manifest.json",
        "clean_predictions": f"{root}/{{phase}}/clean_predictions.json",
        "probe_similarities": f"{root}/calibration/similarities.json",
        "attempt_shards": (
            f"{root}/{{phase}}/attempts/{{source_id}}/replicate-{{replicate}}/{{recipe}}.json"
        ),
        "audited_shards": (
            f"{root}/{{phase}}/audited/{{source_id}}/replicate-{{replicate}}/{{recipe}}.json"
        ),
        "raw_phase_records": f"{root}/{{phase}}/raw_records.json",
        "reviewed_phase_records": f"{root}/analysis/{{phase}}_reviewed_records.json",
        "selection_lock": f"{root}/analysis/selection_lock.json",
    }
    source_ids = sorted({source for role in protocol.target_roles for source in role.candidate_ids})
    phase_counts = {
        phase: {
            "originals": sum(row.partition == phase for row in corpus.records),
            "positive_originals": sum(
                row.partition == phase and row.label == 1 for row in corpus.records
            ),
            "planned_attempts": sum(
                row.partition == phase and row.label == 1 for row in corpus.records
            )
            * len(source_ids)
            * len(protocol.replicates)
            * len(protocol.recipes),
        }
        for phase in ("calibration", "evaluation")
    }
    return {
        "schema_version": 2,
        "run_id": protocol.run_id,
        "validation_status": report["status"],
        "execution_wiring": "implemented_gated",
        "rerun_ready": False,
        "execution_context_checked": False,
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "root_seed": root_seed,
        "replicate_seeds": {
            str(index): component_seed(root_seed, index, "replicate")
            for index in protocol.replicates
        },
        "model_seeds": {
            f"{model.id}:{index}": component_seed(root_seed, index, f"model:{model.id}")
            for model in protocol.models
            if model.kind != "frozen"
            for index in protocol.replicates
        },
        "attack_seeds": {
            f"{source}:{index}:{recipe}": component_seed(
                root_seed, index, f"attack:{source}:{recipe}"
            )
            for source in source_ids
            for index in protocol.replicates
            for recipe in protocol.recipes
        },
        "target_roles": [role.model_dump() for role in protocol.target_roles],
        "source_ids": source_ids,
        "private_artifacts": paths,
        "external_input_datasets": {
            "v2_corpus_evidence": "Grouped non-text identities and original human task review.",
            "v2_registration_evidence": "External core protocol and corpus registration.",
            "v2_target_preparation_evidence": "Separate target corpus and prior registration.",
            "v2_calibration_annotations": "Exact-pair blinded human judgments; never synthesized.",
            "v2_evaluation_annotations": "Exact-pair blinded human judgments; pending permitted.",
            "v2_selection_registration": "External selection-lock registration before evaluation.",
            "v2_attack_asset_source": "Pinned local auxiliary bundle with complete file manifest.",
        },
        "partition_digests": {name: partition_digest(corpus, name) for name in _PARTITIONS},
        "phase_grid": phase_counts,
        "fixed_budget": {
            "source_queries_per_original_recipe_source": protocol.source_query_budget,
            "selected_sources": protocol.selection_k,
            "recipes": protocol.recipes,
            "target_search_queries": 0,
            "early_stopping": "Permitted within the fixed cap; actual queries are reported.",
        },
        "runtime_gate": {
            "context_dataset": "v2_execution_context",
            "required_branch": "codex/methodology-recompute",
            "execution_ref": protocol.execution_ref,
            "attack_assets_sha256": protocol.attack_assets_sha256,
            "constraints": "Clean isolated worktree; pinned inputs; recipe/input hashes checked.",
            "receipts": "Ephemeral MemoryDataset outputs force fresh checks on resumption.",
        },
        "classifier_input": (
            "Registered lower-trust UTF-8 text only; trusted context supports "
            "human ground truth and optional application evaluation."
        ),
        "tokenization_diagnostics": (
            "Original and returned-candidate token counts and truncation per saved "
            "tokenizer; every attempt remains in its denominator."
        ),
        "inference_scope": (
            "Group resampling conditional on the three realized training "
            "replicates, fixed targets, pool, and recipes."
        ),
        "checkpoint_bundle": {
            "payload": "payload/ contains only model and tokenizer artifacts.",
            "metadata": "metadata.json has original non-text training evidence.",
            "completion": "complete.json verifies payload and metadata hashes before publication.",
            "recovery": (
                "Verified complete staging bundles recover without training; "
                "metadata projections can be reconstructed."
            ),
        },
        "phases": {
            "attack_asset_preparation": {
                "status": "implemented",
                "pipeline": "experiment_v2_assets_prepare",
                "inputs": ["v2_attack_asset_source", "v2_execution_context"],
                "outputs": ["v2_attack_asset_manifest"],
                "constraints": "Pin/copy supplied files; isolated cache; no asset downloads.",
            },
            "target_preparation": {
                "status": "implemented",
                "pipeline": "experiment_v2_target_prepare",
                "inputs": [
                    "params:experiment_v2_target_preparation",
                    "v2_target_preparation_evidence",
                    "v2_corpus_evidence",
                    "v2_target_private_train",
                    "v2_target_private_validation",
                    "v2_target_backbone",
                ],
                "outputs": [
                    "v2_prepared_target",
                    "v2_prepared_target_metadata",
                    "v2_target_preparation_record",
                ],
                "constraints": "Register target-only groups; freeze before core registration.",
            },
            "acquisition_and_grouping": {
                "status": "external_inputs_required",
                "inputs": ["pinned source records", "joint origin/family/near-duplicate review"],
                "outputs": [
                    "v2_corpus_evidence",
                    *[f"v2_private_{partition}" for partition in _PARTITIONS],
                ],
                "constraints": "Independent splits/probe; original human reviews are external.",
            },
            "models": {
                "status": "implemented",
                "pipeline": "experiment_v2_models",
                "reuse": "modeling.finetune and pipelines.models.bilstm.train_bilstm.",
                "inputs": [
                    "v2_private_train",
                    "v2_private_validation",
                    "v2_hub__{model_id}",
                    "params:experiment_v2_execution",
                ],
                "outputs": [
                    "v2_model__{model_id}__r{replicate}",
                    "v2_model_meta__{model_id}__r{replicate}",
                    "v2_models_manifest",
                ],
                "constraints": "Train/validation only; seeded heads; masked BiLSTM.",
            },
            "similarity": {
                "status": "implemented",
                "pipeline": "experiment_v2_calibration_execute",
                "reuse": "Existing representation helpers and globally centered linear CKA.",
                "inputs": ["v2_private_probe", "v2_calibration_current_models"],
                "outputs": ["v2_calibration_similarities"],
                "constraints": "CLS transformer/masked BiLSTM pairs; full probe; calibration only.",
            },
            "source_attacks": {
                "status": "implemented",
                "pipelines": [
                    "experiment_v2_calibration_execute",
                    "experiment_v2_evaluation_execute",
                ],
                "reuse": "Existing TextAttack recipes and wrappers with a capped query ledger.",
                "inputs": ["v2_private_{phase}", "v2_{phase}_current_models"],
                "outputs": ["v2_{phase}_attempts__{source_id}__r{replicate}__{recipe}"],
                "constraints": "Complete cells; retain candidates, skips, errors and query counts.",
            },
            "target_audit": {
                "status": "implemented",
                "pipelines": [
                    "experiment_v2_calibration_execute",
                    "experiment_v2_evaluation_execute",
                ],
                "reuse": "Existing classifier prediction helpers.",
                "inputs": [
                    "v2_private_{phase}",
                    "v2_{phase}_current_models",
                    "v2_{phase}_attempts__{source_id}__r{replicate}__{recipe}",
                ],
                "outputs": [
                    "v2_{phase}_clean_predictions",
                    "v2_{phase}_audited__{source_id}__r{replicate}__{recipe}",
                    "v2_{phase}_raw_records",
                ],
                "constraints": "All originals/candidates; pinned models; zero search feedback.",
            },
            "human_pair_adjudication": {
                "status": "external_evidence_required",
                "inputs": ["private exact original-candidate pairs", "blinded review allocation"],
                "outputs": ["v2_calibration_annotations", "v2_evaluation_annotations"],
                "scope": "Calibration source transitions; evaluation eligible target flips.",
                "reuse": "One exact-pair adjudication covers matching source/recipe/seed cells.",
            },
            "annotation_merge": {
                "status": "implemented",
                "pipelines": [
                    "experiment_v2_calibration_annotations",
                    "experiment_v2_evaluation_annotations",
                ],
                "inputs": ["v2_{phase}_raw_records", "v2_{phase}_annotations"],
                "outputs": ["v2_{phase}_records"],
                "constraints": "Target-free merge; empty labels remain unresolved.",
            },
            "freeze_selection": {
                "status": "implemented",
                "pipeline": "experiment_v2_selection",
                "inputs": [
                    "v2_calibration_records",
                    "resolved eligible source-transition validity",
                ],
                "outputs": ["v2_selection_lock"],
                "constraints": "Externally register lock digest before any evaluation text load.",
            },
            "analysis": {
                "status": "implemented",
                "pipeline": "experiment_v2_analysis",
                "inputs": ["v2_calibration_records", "v2_evaluation_records", "v2_selection_lock"],
                "outputs": ["v2_public_summary"],
                "public_path": f"{protocol.public_root}/summary.json",
                "constraints": "Pending contributing-flip labels give bounds; nonflips add zero.",
            },
            "application_study": {
                "status": "optional_executor_not_implemented",
                "constraints": "Separate single-turn local decoder with fixed controls and checks.",
            },
        },
    }
