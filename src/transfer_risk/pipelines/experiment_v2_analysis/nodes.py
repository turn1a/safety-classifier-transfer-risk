"""Reduce supplied v2 rerun records into aggregate-only public evidence."""

from __future__ import annotations

from typing import Any

from transfer_risk.lib.experiment_v2.analysis import analyse_experiment
from transfer_risk.lib.experiment_v2.protocol import content_digest
from transfer_risk.lib.experiment_v2.schemas import (
    CorpusEvidence,
    ExperimentRecords,
    Protocol,
    RegistrationEvidence,
)


def analyse_v2_records(
    receipt: dict[str, str],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    calibration_records: dict[str, Any],
    evaluation_records: dict[str, Any],
    selection_lock: dict[str, Any],
    root_seed: int,
) -> dict[str, Any]:
    """Validate every accounting boundary and publish aggregate outcomes only.

    Args:
        receipt: Fresh preflight digests produced in this pipeline run.
        parameters: Registered prospective v2 protocol and isolated run roots.
        corpus_evidence: Independent grouped original task-review metadata.
        registration_evidence: External protocol/corpus registration attestation.
        calibration_records: Exact frozen calibration and probe evidence.
        evaluation_records: Full-grid held-out non-text inference and attempt evidence.
        selection_lock: Selection rules and identities registered before evaluation.
        root_seed: Registered root seed for deterministic group resampling.

    Returns:
        Aggregate outcomes, honest human-review states, and held-out policy comparisons.

    Raises:
        ValueError: If fresh preflight hashes differ from the current input evidence.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    if receipt != {
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
    }:
        msg = "v2 analysis preflight digests differ from current evidence"
        raise ValueError(msg)
    return analyse_experiment(
        protocol,
        corpus,
        RegistrationEvidence.model_validate(registration_evidence),
        ExperimentRecords.model_validate(calibration_records),
        ExperimentRecords.model_validate(evaluation_records),
        selection_lock,
        root_seed,
    )
