"""Validate complete non-text rerun records and resolve human evidence honestly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from transfer_risk.lib.experiment_v2.protocol import (
    HUMAN_REVIEWERS,
    component_seed,
    content_digest,
    is_digest,
    parse_timestamp,
    partition_digest,
)

if TYPE_CHECKING:
    from transfer_risk.lib.experiment_v2.schemas import (
        AttackAttempt,
        CorpusEvidence,
        CorpusRecord,
        ExperimentRecords,
        HumanAnnotation,
        ModelRealization,
        Protocol,
        RegistrationEvidence,
        TokenizationDiagnostic,
    )


@dataclass(frozen=True)
class EvidenceIndex:
    """Index validated records without performing inference or reading private text.

    Attributes:
        originals: Reviewed originals in this independent partition.
        clean: Canonical clean labels keyed by model, replicate, and original.
        identities: Realized checkpoint/tokenizer fingerprints by model and replicate.
        attempts: Every planned attempt keyed by its unique id.
        human: Human validity state for every returned candidate.
        cka: Probe-only mean CKA keyed by role, replicate, and source.
    """

    originals: dict[str, CorpusRecord]
    clean: dict[tuple[str, int, str], int]
    identities: dict[tuple[str, int], tuple[str, str]]
    attempts: dict[str, AttackAttempt]
    human: dict[str, str]
    cka: dict[tuple[str, int, str], float]


def _fail(message: str) -> None:
    """Raise a consistent evidence-contract error without prompt content.

    Args:
        message: The violated evidence invariant.

    Raises:
        ValueError: Always, with the violated invariant.
    """
    raise ValueError(message)


def canonical_label(protocol: Protocol, model_id: str, raw_label: str) -> int:
    """Map an actual raw output through the model's reviewed complete label map.

    Args:
        protocol: Pinned detector task and label semantics.
        model_id: The model producing the prediction.
        raw_label: Unmodified output label from the detector.

    Returns:
        Zero for benign and one for prompt injection.

    Raises:
        ValueError: If the model or raw output is absent from the reviewed map.
    """
    models = {model.id: model for model in protocol.models}
    if model_id not in models or raw_label not in models[model_id].label_map:
        _fail("prediction uses an unknown model or unmapped raw label")
    return int(models[model_id].label_map[raw_label] == "prompt_injection")


def _check_prepared_target_identities(
    protocol: Protocol,
    corpus: CorpusEvidence,
    indexed: dict[tuple[str, int], ModelRealization],
) -> None:
    """Bind every replication-target use to its independently prepared artifacts.

    Args:
        protocol: Fixed target roles and seed replicates.
        corpus: Target preparation fingerprints established before core registration.
        indexed: Complete realized model coverage for the current phase.

    Raises:
        ValueError: If a realized target differs from its prepared weights or tokenizer.
    """
    for preparation in corpus.target_preparations:
        roles = {role.id: role for role in protocol.target_roles}
        if preparation.target_role in roles:
            model_id = roles[preparation.target_role].model_id
            for replicate in protocol.replicates:
                realization = indexed[model_id, replicate]
                if (
                    realization.checkpoint_sha256 != preparation.checkpoint_sha256
                    or realization.tokenizer_sha256 != preparation.tokenizer_sha256
                ):
                    _fail(
                        "independently prepared target identity differs from its frozen preparation"
                    )


def _realizations(
    protocol: Protocol, corpus: CorpusEvidence, records: ExperimentRecords, root_seed: int
) -> dict[tuple[str, int], tuple[str, str]]:
    """Check full model coverage, seeded training, and realized target exclusions.

    Args:
        protocol: Registered detector inputs and seed replicates.
        corpus: Registered train partition membership.
        records: Supplied phase evidence.
        root_seed: Registered project root seed.

    Returns:
        Exact realized checkpoint and tokenizer identity pairs.

    Raises:
        ValueError: If coverage, seed provenance, or identity constraints fail.
    """
    expected = {(model.id, index) for model in protocol.models for index in protocol.replicates}
    indexed: dict[tuple[str, int], ModelRealization] = {
        (item.model_id, item.replicate): item for item in records.realizations
    }
    if set(indexed) != expected or len(indexed) != len(records.realizations):
        _fail("model realization coverage is incomplete or duplicated")
    for model in protocol.models:
        frozen_identities = set()
        for index in protocol.replicates:
            item = indexed[model.id, index]
            if not is_digest(item.checkpoint_sha256) or not is_digest(item.tokenizer_sha256):
                _fail("model realization identity requires checkpoint and tokenizer SHA256")
            if item.execution_ref != protocol.execution_ref:
                _fail("model realization execution_ref differs from the registered source")
            if model.kind == "frozen":
                frozen_identities.add((item.checkpoint_sha256, item.tokenizer_sha256))
                if item.training_seed is not None or item.train_partition_sha256 is not None:
                    _fail("frozen model realization cannot claim v2 training")
            elif item.training_seed != component_seed(
                root_seed, index, f"model:{model.id}"
            ) or item.train_partition_sha256 != partition_digest(corpus, "train"):
                _fail("trained model has the wrong seed or train partition identity")
        if model.kind == "frozen" and len(frozen_identities) != 1:
            _fail("frozen model identity changes across replicates")
    _check_prepared_target_identities(protocol, corpus, indexed)
    identities = {
        key: (item.checkpoint_sha256, item.tokenizer_sha256) for key, item in indexed.items()
    }
    for role in protocol.target_roles:
        for index in protocol.replicates:
            target_weights = identities[role.model_id, index][0]
            if any(identities[name, index][0] == target_weights for name in role.candidate_ids):
                _fail("realized identical target is in its own surrogate pool")
    return identities


def validate_tokenization_window(
    protocol: Protocol, model_id: str, diagnostic: TokenizationDiagnostic | None
) -> None:
    """Bind any supplied tokenization evidence to its registered model window.

    Args:
        protocol: Registered model and tokenizer inference policy.
        model_id: Model whose tokenizer produced the measurement.
        diagnostic: Optional saved diagnostic; absence remains missing evidence.

    Raises:
        ValueError: If the recorded token window differs from the registered model.
    """
    if diagnostic is not None:
        model = next(item for item in protocol.models if item.id == model_id)
        if diagnostic.max_tokens != model.max_tokens:
            _fail("registered model token window differs from supplied tokenization evidence")


def _clean_predictions(
    protocol: Protocol,
    records: ExperimentRecords,
    originals: dict[str, CorpusRecord],
    identities: dict[tuple[str, int], tuple[str, str]],
) -> dict[tuple[str, int, str], int]:
    """Require matched clean controls for every original, model, and replicate.

    Args:
        protocol: Registered model pool.
        records: Supplied phase evidence.
        originals: Originals belonging to this partition only.
        identities: Validated realized fingerprints.

    Returns:
        Canonical labels for all required clean controls.

    Raises:
        ValueError: If clean baseline coverage or identities differ.
    """
    expected = {(*key, original) for key in identities for original in originals}
    clean: dict[tuple[str, int, str], int] = {}
    for item in records.clean_predictions:
        key = (item.model_id, item.replicate, item.original_id)
        identity = (item.checkpoint_sha256, item.tokenizer_sha256)
        if key in clean or key not in expected or identities[key[:2]] != identity:
            _fail("clean prediction coverage or identity is invalid")
        validate_tokenization_window(protocol, item.model_id, item.tokenization)
        clean[key] = canonical_label(protocol, item.model_id, item.raw_label)
    if set(clean) != expected:
        _fail("clean prediction coverage is incomplete")
    return clean


def _similarities(
    protocol: Protocol,
    corpus: CorpusEvidence,
    records: ExperimentRecords,
    identities: dict[tuple[str, int], tuple[str, str]],
) -> dict[tuple[str, int, str], float]:
    """Require exactly the independent representation-probe CKA evidence.

    Args:
        protocol: Registered target roles and candidate pools.
        corpus: Independently grouped probe membership.
        records: Calibration evidence or held-out evidence with no new CKA.
        identities: Realized model and tokenizer hashes.

    Returns:
        Mean CKA scores by target role, replicate, and source.

    Raises:
        ValueError: If probe identity, numeric range, coverage, or model identity differs.
    """
    expected = (
        {
            (role.id, index, source)
            for role in protocol.target_roles
            for index in protocol.replicates
            for source in role.candidate_ids
        }
        if records.phase == "calibration"
        else set()
    )
    roles = {role.id: role for role in protocol.target_roles}
    cka = {}
    for item in records.similarities:
        key = (item.target_role, item.replicate, item.source_id)
        if key in cka or key not in expected:
            _fail("similarity coverage is invalid or evaluation attempted to recompute selection")
        if item.probe_partition_sha256 != partition_digest(corpus, "probe"):
            _fail("similarity probe partition identity differs")
        source_identity = (item.source_checkpoint_sha256, item.source_tokenizer_sha256)
        target_identity = (item.target_checkpoint_sha256, item.target_tokenizer_sha256)
        if (
            identities[item.source_id, item.replicate] != source_identity
            or identities[roles[item.target_role].model_id, item.replicate] != target_identity
        ):
            _fail("similarity model/tokenizer identity differs from realized checkpoints")
        if not 0 <= item.mean_cka <= 1:
            _fail("mean CKA must be a finite probability-sized score")
        cka[key] = item.mean_cka
    if set(cka) != expected:
        _fail("similarity coverage is incomplete")
    return cka


def _candidate_predictions(
    protocol: Protocol,
    attempt: AttackAttempt,
    identities: dict[tuple[str, int], tuple[str, str]],
    clean: dict[tuple[str, int, str], int],
    *,
    require_target_audit: bool = True,
) -> None:
    """Validate output accounting and target audit coverage for all returned candidates.

    Args:
        protocol: Reviewed model labels and frozen target roles.
        attempt: One source attempt, including failures and skips.
        identities: Exact realized model/tokenizer fingerprints.
        clean: Complete original controls.
        require_target_audit: Require completed target coverage after generation.

    Raises:
        ValueError: If output accounting, mechanical evidence, or prediction identity differs.
    """
    source_original = clean[attempt.source_id, attempt.replicate, attempt.original_id]
    validate_tokenization_window(protocol, attempt.source_id, attempt.source_original_tokenization)
    validate_tokenization_window(protocol, attempt.source_id, attempt.source_candidate_tokenization)
    if attempt.candidate_sha256 is None:
        if attempt.source_raw_perturbed is not None or attempt.target_predictions:
            _fail("a no-candidate attempt contains perturbed predictions")
        if attempt.status == "success":
            _fail("source success requires a returned candidate")
        if attempt.mechanical_screen != "unassessable":
            _fail("a no-candidate attempt cannot pass a mechanical screen")
        if attempt.status == "skipped_source_benign" and source_original != 0:
            _fail("source-benign skip has an injection baseline")
        return
    if not is_digest(attempt.candidate_sha256) or attempt.source_raw_perturbed is None:
        _fail("returned candidate requires a content identity and source prediction")
    if source_original != 1 or attempt.status in {"skipped_source_benign", "error"}:
        _fail("returned candidate has an inconsistent source status or baseline")
    perturbed = canonical_label(protocol, attempt.source_id, str(attempt.source_raw_perturbed))
    if (attempt.status == "success") != (perturbed == 0):
        _fail("source success does not match its actual classifier transition")
    if attempt.mechanical_screen != "unassessable" and not is_digest(
        attempt.mechanical_evidence_sha256
    ):
        _fail("mechanical screening verdict requires separate evidence")
    if not require_target_audit:
        if attempt.target_predictions:
            _fail("generation output already contains target predictions")
        return
    _target_predictions(protocol, attempt, identities)


def _target_predictions(
    protocol: Protocol,
    attempt: AttackAttempt,
    identities: dict[tuple[str, int], tuple[str, str]],
) -> None:
    """Require every target audit under its realized model and tokenizer identity.

    Args:
        protocol: Registered target roles and raw label maps.
        attempt: Source candidate with completed target predictions.
        identities: Realized model/tokenizer hashes.

    Raises:
        ValueError: If target coverage or identity differs.
    """
    expected_targets = {role.model_id for role in protocol.target_roles}
    if {item.model_id for item in attempt.target_predictions} != expected_targets or (
        len(attempt.target_predictions) != len(expected_targets)
    ):
        _fail("target candidate prediction coverage must include every returned candidate")
    for item in attempt.target_predictions:
        if identities[item.model_id, attempt.replicate] != (
            item.checkpoint_sha256,
            item.tokenizer_sha256,
        ):
            _fail("perturbed target identity differs from its clean baseline")
        canonical_label(protocol, item.model_id, item.raw_label)
        validate_tokenization_window(protocol, item.model_id, item.tokenization)


def _attempts(
    protocol: Protocol,
    records: ExperimentRecords,
    originals: dict[str, CorpusRecord],
    identities: dict[tuple[str, int], tuple[str, str]],
    clean: dict[tuple[str, int, str], int],
    root_seed: int,
) -> dict[str, AttackAttempt]:
    """Require the full fixed-budget attempt grid, with failures retained.

    Args:
        protocol: Registered candidates, recipes, budgets, and replicates.
        records: Supplied phase evidence.
        originals: Independent partition originals.
        identities: Realized model/tokenizer hashes.
        clean: Complete baseline predictions.
        root_seed: Registered project root seed.

    Returns:
        Attempts keyed by their unique evidence ids.

    Raises:
        ValueError: If coverage, query accounting, source seed, or predictions differ.
    """
    candidates = {name for role in protocol.target_roles for name in role.candidate_ids}
    expected = {
        (source, index, original.id, recipe)
        for source in candidates
        for index in protocol.replicates
        for original in originals.values()
        if original.label == 1
        for recipe in protocol.recipes
    }
    seen = set()
    attempts = {}
    for attempt in records.attempts:
        key = (attempt.source_id, attempt.replicate, attempt.original_id, attempt.recipe)
        if key not in expected or key in seen or attempt.id in attempts:
            _fail("attempt coverage contains an unexpected or duplicate cell")
        seen.add(key)
        minimum_queries = (
            0 if (attempt.status == "error" and attempt.candidate_sha256 is None) else 1
        )
        if not minimum_queries <= attempt.source_queries <= protocol.source_query_budget:
            _fail("source query budget is exceeded or its baseline query is missing")
        if attempt.target_search_queries != 0:
            _fail("target search queries violate the zero-feedback generation protocol")
        seed = component_seed(
            root_seed, attempt.replicate, f"attack:{attempt.source_id}:{attempt.recipe}"
        )
        if attempt.attack_seed != seed:
            _fail("source attack cell seed differs from the registered root seed")
        if identities[attempt.source_id, attempt.replicate] != (
            attempt.source_checkpoint_sha256,
            attempt.source_tokenizer_sha256,
        ):
            _fail("source attack identity differs from clean baseline and similarity")
        _candidate_predictions(protocol, attempt, identities, clean)
        attempts[attempt.id] = attempt
    if seen != expected:
        _fail("attempt coverage is incomplete; skips, failures and errors must remain")
    return attempts


def remember_prediction(
    seen: dict[tuple[str, str, str], str],
    content: str,
    checkpoint: str,
    tokenizer: str,
    label: str,
) -> None:
    """Require one label for identical content under identical inference identities.

    Args:
        seen: Prior predictions in this registered execution specification.
        content: Exact UTF-8 content hash.
        checkpoint: Saved model fingerprint.
        tokenizer: Saved tokenizer fingerprint.
        label: Actual raw classifier output.

    Raises:
        ValueError: If identical content already has a different raw prediction.
    """
    key = content, checkpoint, tokenizer
    if key in seen and seen[key] != label:
        _fail("identical content has contradictory predictions under identical inference identity")
    seen[key] = label


def consistent_content_predictions(
    records: ExperimentRecords, originals: dict[str, CorpusRecord]
) -> None:
    """Reject contradictory labels for the same bytes and model/tokenizer identities.

    Args:
        records: Complete predictions under one registered inference specification.
        originals: Registered original hashes.

    Raises:
        ValueError: If identical content has conflicting raw labels under the same identity.
    """
    seen: dict[tuple[str, str, str], str] = {}

    def add(content: str, checkpoint: str, tokenizer: str, label: str) -> None:
        """Require one label for a deterministic classifier input and identity.

        Args:
            content: Exact UTF-8 content hash.
            checkpoint: Saved model fingerprint.
            tokenizer: Saved tokenizer fingerprint.
            label: Actual raw classifier output.
        """
        remember_prediction(seen, content, checkpoint, tokenizer, label)

    for row in records.clean_predictions:
        add(
            originals[row.original_id].content_sha256,
            row.checkpoint_sha256,
            row.tokenizer_sha256,
            row.raw_label,
        )
    for attempt in records.attempts:
        if attempt.candidate_sha256 is not None:
            add(
                attempt.candidate_sha256,
                attempt.source_checkpoint_sha256,
                attempt.source_tokenizer_sha256,
                str(attempt.source_raw_perturbed),
            )
            for prediction in attempt.target_predictions:
                add(
                    attempt.candidate_sha256,
                    prediction.checkpoint_sha256,
                    prediction.tokenizer_sha256,
                    prediction.raw_label,
                )


def validate_generation_shard(
    protocol: Protocol,
    attempts: list[AttackAttempt],
    originals: dict[str, CorpusRecord],
    identities: dict[tuple[str, int], tuple[str, str]],
    clean: dict[tuple[str, int, str], int],
) -> None:
    """Check every saved source attempt before any target inference.

    Args:
        protocol: Registered model, recipe, seed, and budget axes.
        attempts: One saved private generation shard.
        originals: Registered phase originals.
        identities: Verified realization fingerprints.
        clean: Validated original predictions.

    Raises:
        ValueError: If membership, identities, status, or query accounting differs.
    """
    sources = {name for role in protocol.target_roles for name in role.candidate_ids}
    seen: set[tuple[str, int, str, str]] = set()
    ids: set[str] = set()
    for attempt in attempts:
        key = attempt.source_id, attempt.replicate, attempt.original_id, attempt.recipe
        if (
            attempt.source_id not in sources
            or attempt.replicate not in protocol.replicates
            or attempt.recipe not in protocol.recipes
            or attempt.original_id not in originals
            or originals[attempt.original_id].label != 1
            or key in seen
            or attempt.id in ids
        ):
            _fail("generation attempt coverage contains an invalid or duplicate cell")
        seen.add(key)
        ids.add(attempt.id)
        minimum = 0 if attempt.status == "error" and attempt.candidate_sha256 is None else 1
        if not minimum <= attempt.source_queries <= protocol.source_query_budget:
            _fail("generation attempt source query budget is invalid")
        if attempt.target_search_queries != 0:
            _fail("generation attempt used target search queries")
        if attempt.attack_seed != component_seed(
            protocol.root_seed, attempt.replicate, f"attack:{attempt.source_id}:{attempt.recipe}"
        ):
            _fail("generation attempt attack seed differs")
        if identities[attempt.source_id, attempt.replicate] != (
            attempt.source_checkpoint_sha256,
            attempt.source_tokenizer_sha256,
        ):
            _fail("generation attempt source identity differs")
        _candidate_predictions(protocol, attempt, identities, clean, require_target_audit=False)


def resolved_criteria(annotation: HumanAnnotation) -> dict[str, str | bool | None]:
    """Resolve each criterion while preserving definite independent agreements.

    Args:
        annotation: Blinded pair judgments and an optional independent adjudicator.

    Returns:
        Definite consensus or adjudication per criterion, with None for unresolved evidence.

    Raises:
        ValueError: If identities are absent, repeated, noncanonical, or unblinded.
    """
    if not annotation.blinded:
        _fail("human pair review must be blinded to model identities and outcomes")
    reviewers = [vote.reviewer_id for vote in annotation.votes]
    if (
        len(reviewers) > HUMAN_REVIEWERS
        or len(set(reviewers)) != len(reviewers)
        or any(not name or name != name.strip() for name in reviewers)
    ):
        _fail("human pair review requires independent reviewer identities")
    adjudicator = annotation.adjudicator
    if adjudicator is not None:
        if adjudicator.reviewer_id in reviewers:
            _fail("human adjudicator must be independent of the original reviewers")
        if (
            not adjudicator.reviewer_id
            or adjudicator.reviewer_id != adjudicator.reviewer_id.strip()
        ):
            _fail("human adjudicator identity is absent or noncanonical")
    result: dict[str, str | bool | None] = {}
    for criterion, unknown in (
        ("semantic", "uncertain"),
        ("intent", "uncertain"),
        ("is_prompt_injection", None),
    ):
        value = None
        if len(reviewers) == HUMAN_REVIEWERS:
            first, second = (getattr(vote, criterion) for vote in annotation.votes)
            if first == second and first != unknown:
                value = first
            elif adjudicator is not None:
                final = getattr(adjudicator, criterion)
                value = final if final != unknown else None
        result[criterion] = value
    return result


def human_state(annotation: HumanAnnotation) -> str:
    """Resolve conjunctive validity using the shared criterion-level resolution.

    Args:
        annotation: Non-text blinded human judgments for an exact pair.

    Returns:
        Invalid if any criterion fails, valid if all pass, otherwise unresolved.
    """
    final = resolved_criteria(annotation)
    if (
        final["semantic"] == "changed"
        or final["intent"] == "lost"
        or final["is_prompt_injection"] is False
    ):
        return "invalid"
    return "valid" if all(value is not None for value in final.values()) else "unresolved"


def human_review_complete(annotation: HumanAnnotation) -> bool:
    """Check whether every criterion has a definite consensus or adjudicated judgment.

    Args:
        annotation: Structurally valid blinded human judgments for an exact pair.

    Returns:
        Whether all criteria are resolved, separately from conjunction validity.
    """
    return all(value is not None for value in resolved_criteria(annotation).values())


def _annotations(
    protocol: Protocol,
    records: ExperimentRecords,
    originals: dict[str, CorpusRecord],
    attempts: dict[str, AttackAttempt],
) -> dict[str, str]:
    """Bind human judgments to exact pairs and reuse them across duplicate cells.

    Args:
        protocol: Registered human adjudication protocol hash.
        records: Supplied judgments, which may be incomplete.
        originals: Reviewed original content hashes.
        attempts: Full attempt grid and candidate hashes.

    Returns:
        Human validity for every returned candidate, sharing exact-pair adjudication.

    Raises:
        ValueError: If identities differ or an exact pair has contradictory final states.
    """
    human = {item.id: "unresolved" for item in attempts.values() if item.candidate_sha256}
    seen: set[str] = set()
    pairs: dict[tuple[str, str], str] = {}
    for annotation in records.annotations:
        if annotation.attempt_id not in human or annotation.attempt_id in seen:
            _fail("human annotation coverage contains unknown or duplicate candidate ids")
        seen.add(annotation.attempt_id)
        attempt = attempts[annotation.attempt_id]
        if (
            annotation.original_sha256 != originals[attempt.original_id].content_sha256
            or annotation.candidate_sha256 != attempt.candidate_sha256
            or annotation.protocol_sha256 != protocol.annotation_protocol_sha256
        ):
            _fail("human annotation pair or protocol identity differs")
        pair = (annotation.original_sha256, annotation.candidate_sha256)
        state = human_state(annotation)
        previous = pairs.get(pair, "unresolved")
        if previous != "unresolved" and state not in {"unresolved", previous}:
            _fail("exact pair has contradictory resolved human adjudications")
        pairs[pair] = state if state != "unresolved" else previous
    for identifier in human:
        attempt = attempts[identifier]
        pair = (
            originals[attempt.original_id].content_sha256,
            str(attempt.candidate_sha256),
        )
        human[identifier] = pairs.get(pair, "unresolved")
    return human


def pair_review_summary(index: EvidenceIndex, attempts: list[AttackAttempt]) -> dict[str, int]:
    """Count logical validity states for a specified set of exact candidate pairs.

    Args:
        index: Validated original identities and shared human validity states.
        attempts: Cells relevant to the caller's outcome or policy, possibly duplicated.

    Returns:
        Distinct pair counts without exposing original, candidate, or reviewer identities.
    """
    pairs = {
        (
            index.originals[attempt.original_id].content_sha256,
            attempt.candidate_sha256,
        ): index.human.get(attempt.id, "unresolved")
        for attempt in attempts
        if attempt.candidate_sha256 is not None
    }
    return {
        "pairs": len(pairs),
        "valid_pairs": sum(state == "valid" for state in pairs.values()),
        "invalid_pairs": sum(state == "invalid" for state in pairs.values()),
        "unresolved_pairs": sum(state == "unresolved" for state in pairs.values()),
    }


def human_review_summary(records: ExperimentRecords, index: EvidenceIndex) -> dict[str, int | bool]:
    """Separate overall full-review coverage from logically resolved pair validity.

    Args:
        records: Validated human annotations for a complete phase grid.
        index: Indexed exact-pair identities and logical validity states.

    Returns:
        Observed pair coverage without estimating overall fidelity from selective review.
    """
    summary = pair_review_summary(index, list(index.attempts.values()))
    complete = {
        (annotation.original_sha256, annotation.candidate_sha256)
        for annotation in records.annotations
        if human_review_complete(annotation)
    }
    return {
        **summary,
        "fully_reviewed_pairs": len(complete),
        "full_review_complete": len(complete) == summary["pairs"],
    }


def validate_records(
    protocol: Protocol,
    corpus: CorpusEvidence,
    registration: RegistrationEvidence,
    records: ExperimentRecords,
    root_seed: int,
) -> EvidenceIndex:
    """Validate supplied full-grid evidence without running any model or attack.

    Args:
        protocol: Registered model, corpus, budget, and selection specification.
        corpus: Grouped human original review evidence.
        registration: External registration metadata.
        records: One phase's complete non-text accounting records.
        root_seed: Registered root seed, included in the protocol digest.

    Returns:
        Indexed, validated evidence for deterministic analysis.

    Raises:
        ValueError: If phase provenance, coverage, identity, or query constraints fail.
    """
    if (
        records.protocol_sha256 != content_digest(protocol)
        or records.corpus_sha256 != content_digest(corpus)
        or records.execution_ref != protocol.execution_ref
        or root_seed != protocol.root_seed
    ):
        _fail("phase evidence protocol, corpus, source identity or root seed differs")
    if not records.hardware:
        _fail("phase evidence requires actual hardware provenance")
    if parse_timestamp(records.started_at) < parse_timestamp(registration.registered_at):
        _fail("phase execution precedes prospective registration")
    if parse_timestamp(records.completed_at) < parse_timestamp(records.started_at):
        _fail("phase completion precedes its start")
    originals = {row.id: row for row in corpus.records if row.partition == records.phase}
    identities = _realizations(protocol, corpus, records, root_seed)
    clean = _clean_predictions(protocol, records, originals, identities)
    cka = _similarities(protocol, corpus, records, identities)
    attempts = _attempts(protocol, records, originals, identities, clean, root_seed)
    consistent_content_predictions(records, originals)
    human = _annotations(protocol, records, originals, attempts)
    return EvidenceIndex(originals, clean, identities, attempts, human, cka)


def target_flips(
    protocol: Protocol, index: EvidenceIndex, attempt: AttackAttempt, target: str
) -> bool:
    """Check a raw target injection-to-benign transition independently of source success.

    Args:
        protocol: Reviewed target label map.
        index: Validated clean controls.
        attempt: A generated attempt, including a possible absent candidate.
        target: Frozen target model id.

    Returns:
        Whether the target is injection on the original and benign on the candidate.
    """
    if index.clean[target, attempt.replicate, attempt.original_id] != 1:
        return False
    return any(
        item.model_id == target and canonical_label(protocol, target, item.raw_label) == 0
        for item in attempt.target_predictions
    )


def source_flips(protocol: Protocol, index: EvidenceIndex, attempt: AttackAttempt) -> bool:
    """Check the source's actual baseline and candidate prediction transition.

    Args:
        protocol: Reviewed source label map.
        index: Validated clean controls.
        attempt: One complete attempt record.

    Returns:
        Whether a baseline injection becomes benign on the returned candidate.
    """
    return bool(
        attempt.source_raw_perturbed is not None
        and index.clean[attempt.source_id, attempt.replicate, attempt.original_id] == 1
        and canonical_label(protocol, attempt.source_id, attempt.source_raw_perturbed) == 0
    )
