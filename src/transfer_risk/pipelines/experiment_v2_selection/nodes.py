"""Freeze source selectors from supplied calibration and independent probe evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from transfer_risk.lib.experiment_v2.analysis import freeze_selection
from transfer_risk.lib.experiment_v2.protocol import content_digest
from transfer_risk.lib.experiment_v2.schemas import (
    CorpusEvidence,
    ExperimentRecords,
    Protocol,
    RegistrationEvidence,
)


def freeze_v2_selectors(
    receipt: dict[str, str],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    calibration_records: dict[str, Any],
    root_seed: int,
) -> dict[str, Any]:
    """Create the local lock that must be externally registered before evaluation.

    Args:
        receipt: Fresh preflight digests produced in this pipeline run.
        parameters: Registered prospective v2 protocol.
        corpus_evidence: Grouped original task-review metadata.
        registration_evidence: External protocol/corpus registration attestation.
        calibration_records: Full calibration grid, contributing-source validity, and probe CKA.
        root_seed: Registered root seed.

    Returns:
        A private selection lock with exact evidence hashes and a current UTC timestamp.

    Raises:
        ValueError: If the fresh preflight digests do not match this node's inputs.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    if receipt != {
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
    }:
        msg = "v2 selection preflight digests differ from current evidence"
        raise ValueError(msg)
    return freeze_selection(
        protocol,
        corpus,
        RegistrationEvidence.model_validate(registration_evidence),
        ExperimentRecords.model_validate(calibration_records),
        root_seed,
        datetime.now(UTC).isoformat(),
    )
