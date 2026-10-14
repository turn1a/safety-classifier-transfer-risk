"""Parse catalog evidence and expose pure prospective v2 gates to Kedro."""

from __future__ import annotations

from typing import Any

from transfer_risk.lib.experiment_v2.protocol import (
    build_execution_manifest,
    require_valid_preflight,
    validate_preflight,
)
from transfer_risk.lib.experiment_v2.schemas import (
    CorpusEvidence,
    Protocol,
    RegistrationEvidence,
)


def inspect_v2_contract(
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    root_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a persistable blocker report and the catalog execution contract.

    Args:
        parameters: Prospective v2 parameters loaded by the dedicated environment.
        corpus_evidence: Non-text grouped corpus and original annotation evidence.
        registration_evidence: External protocol/corpus registration attestation.
        root_seed: Shared registered project root seed.

    Returns:
        The blocker report and the structured future execution manifest.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    registration = RegistrationEvidence.model_validate(registration_evidence)
    report = validate_preflight(protocol, corpus, registration)
    if root_seed != protocol.root_seed:
        report["issues"].append("project root seed differs from the registered v2 protocol")
        report["status"] = "blocked"
    return report, build_execution_manifest(protocol, corpus, root_seed, report)


def gate_v2_contract(report: dict[str, Any]) -> dict[str, str]:
    """Stop dependent nodes only after the blocker report is saved through the catalog.

    Args:
        report: Current v2 preflight report.

    Returns:
        Digests binding target-free downstream nodes to validated protocol and corpus.

    Raises:
        ValueError: If preflight evidence is absent, stale, or invalid.
    """
    return require_valid_preflight(report)
