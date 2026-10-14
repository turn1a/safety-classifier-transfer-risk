"""Strict non-text evidence schemas for the prospective prompt-injection study.

Absent identities and human evidence can be represented without being accepted by a
preflight gate. Unknown fields are rejected so raw prompts cannot enter these boundaries.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Partition = Literal["train", "validation", "probe", "calibration", "evaluation"]
CanonicalLabel = Literal["benign", "prompt_injection"]


class EvidenceRecord(BaseModel):
    """Reject undeclared fields and coercion at non-text evidence boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ModelIdentity(EvidenceRecord):
    """Describe an input model, tokenizer, decision rule, and matched training block."""

    id: str
    repository: str
    backbone_family: str
    revision: str | None
    tokenizer_repository: str
    tokenizer_revision: str | None
    kind: Literal["frozen", "finetune", "bilstm"]
    output_labels: list[str]
    label_map: dict[str, CanonicalLabel]
    label_review_sha256: str | None
    max_tokens: int
    decision_rule: Literal["argmax"]
    comparison_block: str
    training_recipe_sha256: str | None


class SourceIdentity(EvidenceRecord):
    """Describe a pinned corpus source and separate task and exposure reviews."""

    id: str
    revision: str | None
    content_sha256: str | None
    task_review_sha256: str | None
    exposure_review_sha256: str | None


class TargetRole(EvidenceRecord):
    """Fix one target and its eligible designed candidate pool before evaluation."""

    id: str
    model_id: str
    purpose: Literal["primary", "replication"]
    candidate_ids: list[str]
    preparation: Literal["historical_anchor", "independently_prepared"]


class ApplicationStudy(EvidenceRecord):
    """Pin an optional local, single-turn application study independently of detectors."""

    enabled: bool = False
    decoder_repository: str | None = None
    decoder_revision: str | None = None
    decoder_sha256: str | None = None
    tokenizer_sha256: str | None = None
    decoding_spec_sha256: str | None = None
    task_sha256: str | None = None
    context_sha256: str | None = None
    objective_id: str | None = None
    checker_sha256: str | None = None


class Protocol(EvidenceRecord):
    """Fix v2 estimands, pools, budgets, operating points, partitions, and decisions."""

    schema_version: Literal[2]
    run_id: str
    task: Literal["prompt_injection"]
    root_seed: int = 20260616
    execution_ref: str | None
    private_root: str
    public_root: str
    replicates: list[int]
    models: list[ModelIdentity]
    sources: list[SourceIdentity]
    target_roles: list[TargetRole]
    recipes: list[str]
    source_query_budget: int
    attack_spec_sha256: str | None
    attack_assets_sha256: str | None = None
    input_policy_sha256: str | None
    annotation_protocol_sha256: str | None
    power_analysis_sha256: str | None
    selection_k: int
    minimum_eligible_sources: int = 7
    min_groups: dict[Partition, int]
    min_hard_negative_groups: int = 1
    utility_min_tpr: float
    utility_max_fpr: float
    bootstrap_replicates: int
    alpha: float
    minimum_effect: float
    application_study: ApplicationStudy = Field(default_factory=ApplicationStudy)


class CorpusRecord(EvidenceRecord):
    """Identify one grouped, human-reviewed original without publishing its text."""

    id: str
    group_id: str
    origin_id: str
    attack_family_id: str
    source_id: str
    content_sha256: str
    partition: Partition
    task: str
    label: Literal[0, 1]
    reviewers: list[str]
    adjudication: Literal["pending", "complete", "disputed"]
    known_target_training_roles: list[str]
    task_context_sha256: str | None
    trusted_instruction_sha256: str | None
    trust_boundary: Literal["untrusted_content", "trusted_user", "unknown"]
    objective_id: str | None
    negative_subtype: Literal["ordinary", "quoted_instruction", "neutral_instruction"] | None


class TargetPreparation(EvidenceRecord):
    """Bind an independently prepared target to separate data, code, and realized artifacts."""

    target_role: str
    training_group_ids: list[str]
    validation_group_ids: list[str]
    origin_ids: list[str]
    attack_family_ids: list[str]
    source_revisions: dict[str, str]
    target_corpus_sha256: str | None = None
    grouping_method: str | None = None
    corpus_records: list[CorpusRecord] = Field(default_factory=list)
    preparation_protocol_sha256: str | None
    grouping_evidence_sha256: str | None
    annotation_evidence_sha256: str | None
    checkpoint_sha256: str | None
    tokenizer_sha256: str | None


class CorpusEvidence(EvidenceRecord):
    """Record group closure, partition membership, and original task adjudication."""

    grouping_method: str | None
    grouping_evidence_sha256: str | None
    records: list[CorpusRecord]
    target_preparations: list[TargetPreparation] = Field(default_factory=list)


class RegistrationEvidence(EvidenceRecord):
    """Record an external registration attestation; software cannot authenticate it."""

    registry_uri: str | None = None
    registered_at: str | None = None
    protocol_sha256: str | None = None
    corpus_sha256: str | None = None
    evidence_sha256: str | None = None


class ModelRealization(EvidenceRecord):
    """Bind realized weights and tokenizer to a training split, seed, and code ref."""

    model_id: str
    replicate: int
    checkpoint_sha256: str
    tokenizer_sha256: str
    training_seed: int | None
    train_partition_sha256: str | None
    execution_ref: str


class CleanPrediction(EvidenceRecord):
    """Record a clean original prediction under one realized model identity."""

    model_id: str
    replicate: int
    original_id: str
    raw_label: str
    checkpoint_sha256: str
    tokenizer_sha256: str


class SimilarityRecord(EvidenceRecord):
    """Bind full-probe, globally centered mean CKA to exact model identities."""

    target_role: str
    source_id: str
    replicate: int
    mean_cka: float
    probe_partition_sha256: str
    source_checkpoint_sha256: str
    source_tokenizer_sha256: str
    target_checkpoint_sha256: str
    target_tokenizer_sha256: str


class CandidatePrediction(EvidenceRecord):
    """Record a perturbed target prediction under the same frozen identity."""

    model_id: str
    raw_label: str
    checkpoint_sha256: str
    tokenizer_sha256: str


class AttackAttempt(EvidenceRecord):
    """Account for every planned attempt, including skips, failures, and errors."""

    id: str
    source_id: str
    replicate: int
    original_id: str
    recipe: str
    status: Literal["success", "failed", "skipped_source_benign", "error"]
    source_queries: int
    target_search_queries: int
    attack_seed: int
    candidate_sha256: str | None
    source_raw_perturbed: str | None
    source_checkpoint_sha256: str
    source_tokenizer_sha256: str
    target_predictions: list[CandidatePrediction]
    mechanical_screen: Literal["passed", "failed", "unassessable"]
    mechanical_evidence_sha256: str | None


class HumanVote(EvidenceRecord):
    """Record a person's separate semantic, intent, and task judgments."""

    reviewer_id: str
    semantic: Literal["preserved", "changed", "uncertain"]
    intent: Literal["retained", "lost", "uncertain"]
    is_prompt_injection: bool | None


class HumanAnnotation(EvidenceRecord):
    """Bind blinded independent judgments to one exact original-candidate pair."""

    attempt_id: str
    original_sha256: str
    candidate_sha256: str
    protocol_sha256: str
    blinded: bool
    votes: list[HumanVote]
    adjudicator: HumanVote | None = None


class ApplicationControl(EvidenceRecord):
    """Store objective-specific deterministic checks and an execution artifact hash."""

    task_pass: bool | None
    attacker_objective: bool | None
    execution_sha256: str | None


class ApplicationRecord(EvidenceRecord):
    """Record benign, original, and perturbed controls in one fixed local task."""

    attempt_id: str
    decoder_sha256: str | None
    tokenizer_sha256: str | None
    decoding_spec_sha256: str | None
    task_sha256: str | None
    context_sha256: str | None
    objective_id: str | None
    checker_sha256: str | None
    benign: ApplicationControl
    original: ApplicationControl
    perturbed: ApplicationControl


class ExperimentRecords(EvidenceRecord):
    """Supply calibration or held-out evidence without loading a model or prompt."""

    phase: Literal["calibration", "evaluation"]
    protocol_sha256: str
    corpus_sha256: str
    execution_ref: str
    started_at: str
    completed_at: str
    hardware: str
    selection_lock_sha256: str | None = None
    selection_registry_uri: str | None = None
    selection_registry_evidence_sha256: str | None = None
    selection_registered_at: str | None = None
    realizations: list[ModelRealization]
    clean_predictions: list[CleanPrediction]
    similarities: list[SimilarityRecord]
    attempts: list[AttackAttempt]
    annotations: list[HumanAnnotation]
    application_records: list[ApplicationRecord]
