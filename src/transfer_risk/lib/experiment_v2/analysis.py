"""Target-free v2 selection freezing, held-out analysis, and public aggregate reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transfer_risk.lib.experiment_v2.application import analyse_application
from transfer_risk.lib.experiment_v2.protocol import (
    content_digest,
    is_digest,
    parse_timestamp,
    require_valid_preflight,
    validate_preflight,
)
from transfer_risk.lib.experiment_v2.records import (
    EvidenceIndex,
    human_review_summary,
    source_flips,
    target_flips,
    validate_records,
)
from transfer_risk.lib.experiment_v2.selection import (
    calibration_selections,
    evaluate_selection,
    utility,
)

if TYPE_CHECKING:
    from transfer_risk.lib.experiment_v2.schemas import (
        CorpusEvidence,
        ExperimentRecords,
        Protocol,
        RegistrationEvidence,
    )


def freeze_selection(
    protocol: Protocol,
    corpus: CorpusEvidence,
    registration: RegistrationEvidence,
    calibration: ExperimentRecords,
    root_seed: int,
    frozen_at: str,
) -> dict[str, Any]:
    """Freeze equal-budget CKA and source-only policies before held-out evaluation.

    Args:
        protocol: Registered pure-task detector, seed, partition, and budget rules.
        corpus: Human-reviewed group and partition metadata.
        registration: External protocol/corpus registration attestation.
        calibration: Complete calibration attempts and independent probe CKA evidence.
        root_seed: Registered project root seed.
        frozen_at: Explicit timezone-aware timestamp from the orchestration boundary.

    Returns:
        A private lock with chosen sources and exact calibration evidence hashes.

    Raises:
        ValueError: If preflight, human review, provenance, or temporal ordering fails.
    """
    require_valid_preflight(validate_preflight(protocol, corpus, registration))
    if calibration.phase != "calibration":
        msg = "selection freezing requires calibration evidence only"
        raise ValueError(msg)
    index = validate_records(protocol, corpus, registration, calibration, root_seed)
    if parse_timestamp(frozen_at) <= parse_timestamp(calibration.completed_at):
        msg = "selection lock must follow completed calibration and human review"
        raise ValueError(msg)
    lock = {
        "schema_version": 2,
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "calibration_sha256": content_digest(calibration),
        "root_seed": root_seed,
        "frozen_at": frozen_at,
        "selections": calibration_selections(protocol, index),
        "external_registration_required": True,
    }
    return {**lock, "lock_sha256": content_digest(lock)}


def _outcomes(protocol: Protocol, index: EvidenceIndex) -> dict[str, Any]:
    """Keep all-attempt yields, validity states, and raw classifier transitions separate.

    Args:
        protocol: Registered target roles and source pools.
        index: Validated complete-grid held-out evidence.

    Returns:
        Aggregate counts and validity bounds without private row or pair identities.
    """
    result = {}
    for role in protocol.target_roles:
        attempts = [
            item for item in index.attempts.values() if item.source_id in role.candidate_ids
        ]
        counts = dict.fromkeys(
            [
                "attempted",
                "returned_candidates",
                "raw_source_flips",
                "raw_target_flips",
                "joint_source_target_flips",
                "human_valid",
                "human_invalid",
                "human_unresolved",
                "valid_source_flips",
                "valid_target_flips",
                "valid_joint_source_target_flips",
                "unresolved_source_flips",
                "unresolved_target_flips",
                "unresolved_joint_source_target_flips",
                "mechanical_passed",
                "mechanical_failed",
                "mechanical_unassessable",
                "technical_errors",
                "skipped_source_benign",
                "target_original_positive",
                "source_original_positive",
                "source_queries",
            ],
            0,
        )
        for attempt in attempts:
            state = index.human.get(attempt.id)
            source = source_flips(protocol, index, attempt)
            target = target_flips(protocol, index, attempt, role.model_id)
            counts["attempted"] += 1
            counts["returned_candidates"] += int(attempt.candidate_sha256 is not None)
            counts["raw_source_flips"] += int(source)
            counts["raw_target_flips"] += int(target)
            counts["joint_source_target_flips"] += int(source and target)
            counts["target_original_positive"] += index.clean[
                role.model_id, attempt.replicate, attempt.original_id
            ]
            counts["source_original_positive"] += index.clean[
                attempt.source_id, attempt.replicate, attempt.original_id
            ]
            counts["source_queries"] += attempt.source_queries
            counts["technical_errors"] += int(attempt.status == "error")
            counts["skipped_source_benign"] += int(attempt.status == "skipped_source_benign")
            if state is not None:
                counts[f"human_{state}"] += 1
                counts[f"mechanical_{attempt.mechanical_screen}"] += 1
                if state in {"valid", "unresolved"}:
                    counts[f"{state}_source_flips"] += int(source)
                    counts[f"{state}_target_flips"] += int(target)
                    counts[f"{state}_joint_source_target_flips"] += int(source and target)
        total = counts["attempted"]
        result[role.id] = {
            **counts,
            "valid_source_yield_lower": counts["valid_source_flips"] / total,
            "valid_source_yield_upper": (
                counts["valid_source_flips"] + counts["unresolved_source_flips"]
            )
            / total,
            "valid_target_yield_lower": counts["valid_target_flips"] / total,
            "valid_target_yield_upper": (
                counts["valid_target_flips"] + counts["unresolved_target_flips"]
            )
            / total,
            "source_query_cap_total": total * protocol.source_query_budget,
        }
    return result


def _validate_lock(
    protocol: Protocol,
    corpus: CorpusEvidence,
    registration: RegistrationEvidence,
    calibration: ExperimentRecords,
    evaluation: ExperimentRecords,
    lock: dict[str, Any],
    root_seed: int,
) -> None:
    """Reject stale choices, changed calibration, or evaluation opened before lock registration.

    Args:
        protocol: Registered experiment specification.
        corpus: Registered independent group membership.
        registration: Initial external registration attestation.
        calibration: Current calibration evidence, which must match its frozen digest.
        evaluation: Held-out evaluation envelope and selection-registration evidence.
        lock: Supplied private selection lock.
        root_seed: Registered root seed.

    Raises:
        ValueError: If lock content or externally attested temporal ordering is inconsistent.
    """
    expected = freeze_selection(
        protocol, corpus, registration, calibration, root_seed, str(lock.get("frozen_at"))
    )
    if lock != expected or evaluation.selection_lock_sha256 != expected["lock_sha256"]:
        msg = "selection lock is stale, tampered, or not bound to evaluation records"
        raise ValueError(msg)
    uri = evaluation.selection_registry_uri
    if (
        not uri
        or not uri.startswith("https://")
        or not is_digest(evaluation.selection_registry_evidence_sha256)
        or parse_timestamp(evaluation.selection_registered_at) < parse_timestamp(lock["frozen_at"])
        or parse_timestamp(evaluation.selection_registered_at)
        >= parse_timestamp(evaluation.started_at)
    ):
        msg = "selection lock requires external registration before evaluation is unsealed"
        raise ValueError(msg)


def analyse_experiment(
    protocol: Protocol,
    corpus: CorpusEvidence,
    registration: RegistrationEvidence,
    calibration: ExperimentRecords,
    evaluation: ExperimentRecords,
    lock: dict[str, Any],
    root_seed: int,
) -> dict[str, Any]:
    """Analyse supplied rerun records with complete denominators and no heavy execution.

    Args:
        protocol: Frozen pure prompt-injection design and isolated output roots.
        corpus: Human original task review and group-aware independent partitions.
        registration: Initial external protocol/corpus registration attestation.
        calibration: Frozen calibration source yields, clean utility, and independent probe CKA.
        evaluation: Complete held-out attempt and target-prediction accounting.
        lock: Source selections externally registered before evaluation was opened.
        root_seed: Registered root seed for model, attack, and cluster-bootstrap streams.

    Returns:
        Public aggregate-only outcomes, selector comparisons, and optional application checks.

    Raises:
        ValueError: If any preflight, lock, identity, budget, or record-coverage invariant fails.
    """
    require_valid_preflight(validate_preflight(protocol, corpus, registration))
    if evaluation.phase != "evaluation":
        msg = "held-out analysis requires evaluation evidence"
        raise ValueError(msg)
    _validate_lock(protocol, corpus, registration, calibration, evaluation, lock, root_seed)
    index = validate_records(protocol, corpus, registration, evaluation, root_seed)
    calibration_index = validate_records(protocol, corpus, registration, calibration, root_seed)
    if index.identities != calibration_index.identities:
        msg = "model/tokenizer identity changes between calibration and evaluation"
        raise ValueError(msg)
    selection = evaluate_selection(protocol, index, lock["selections"], root_seed)
    return {
        "schema_version": 2,
        "run_id": protocol.run_id,
        "human_review": {
            "calibration": human_review_summary(calibration, calibration_index),
            "evaluation": human_review_summary(evaluation, index),
        },
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "selection_lock_sha256": lock["lock_sha256"],
        "execution_ref": protocol.execution_ref,
        "hardware": evaluation.hardware,
        "analysis_only": True,
        "outcomes": _outcomes(protocol, index),
        "selection": selection,
        "protocol_decision": (
            all(item["decision"] for item in selection.values())
            if all(item["status"] == "complete" for item in selection.values())
            else None
        ),
        "clean_evaluation_utility": {
            f"{model.id}:{replicate}": utility(index, model.id, replicate)
            for model in protocol.models
            for replicate in protocol.replicates
        },
        "application_study": analyse_application(protocol, evaluation, index),
        "evidence_limit": "Human identities, attestations and predictions require upstream audit.",
    }
