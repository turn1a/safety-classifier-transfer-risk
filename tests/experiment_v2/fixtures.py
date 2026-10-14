"""Build fully synthetic, non-text evidence for v2 contract tests."""

from __future__ import annotations

import hashlib

from transfer_risk.lib.experiment_v2.protocol import (
    component_seed,
    content_digest,
    partition_digest,
)
from transfer_risk.lib.experiment_v2.schemas import (
    CorpusEvidence,
    ExperimentRecords,
    Protocol,
    RegistrationEvidence,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def protocol_dict() -> dict:
    assets = []
    for name in ["target_a", "target_b", *[f"source_{i}" for i in range(8)]]:
        assets.append(
            {
                "id": name,
                "repository": f"synthetic/{name}",
                "backbone_family": "deberta"
                if name == "target_a"
                else "roberta"
                if name == "target_b"
                else name,
                "revision": digest("checkpoint:target_b")
                if name == "target_b"
                else digest(name)[:40],
                "tokenizer_repository": f"synthetic/{name}",
                "tokenizer_revision": digest("tokenizer:target_b")
                if name == "target_b"
                else digest(name)[:40],
                "kind": "frozen" if name.startswith("target") else "finetune",
                "output_labels": ["0", "1"],
                "label_map": {"0": "benign", "1": "prompt_injection"},
                "label_review_sha256": digest(f"labels:{name}"),
                "max_tokens": 512,
                "decision_rule": "argmax",
                "comparison_block": "matched_pi_detectors",
                "training_recipe_sha256": digest("same_training_recipe"),
            }
        )
    return {
        "schema_version": 2,
        "run_id": "synthetic-contract",
        "task": "prompt_injection",
        "execution_ref": "e" * 40,
        "private_root": "data/experiment_v2/synthetic-contract",
        "public_root": "docs/artifacts/experiment_v2/synthetic-contract",
        "replicates": [0, 1, 2],
        "models": assets,
        "sources": [
            {
                "id": name,
                "revision": digest(name),
                "content_sha256": digest(f"corpus:{name}"),
                "task_review_sha256": digest(f"task:{name}"),
                "exposure_review_sha256": digest(f"exposure:{name}"),
            }
            for name in ["training", "probe", "calibration", "evaluation"]
        ],
        "target_roles": [
            {
                "id": role,
                "model_id": target,
                "purpose": purpose,
                "preparation": "historical_anchor"
                if purpose == "primary"
                else "independently_prepared",
                "candidate_ids": [f"source_{i}" for i in range(8)],
            }
            for role, target, purpose in [
                ("primary", "target_a", "primary"),
                ("replication", "target_b", "replication"),
            ]
        ],
        "recipes": ["textfooler"],
        "source_query_budget": 100,
        "attack_spec_sha256": digest("attack_spec"),
        "attack_assets_sha256": digest("synthetic_attack_assets"),
        "input_policy_sha256": digest("input_policy"),
        "annotation_protocol_sha256": digest("human_annotation_protocol"),
        "power_analysis_sha256": digest("prospective_power_analysis"),
        "selection_k": 2,
        "min_groups": {
            "train": 2,
            "validation": 2,
            "probe": 2,
            "calibration": 2,
            "evaluation": 4,
        },
        "utility_min_tpr": 0.8,
        "utility_max_fpr": 0.1,
        "bootstrap_replicates": 199,
        "alpha": 0.05,
        "minimum_effect": 0.05,
        "application_study": {"enabled": False},
    }


def corpus_dict() -> dict:
    records = []
    for partition in ["train", "validation", "probe", "calibration", "evaluation"]:
        count = 4 if partition in {"calibration", "evaluation"} else 2
        for label in [0, 1]:
            for i in range(count):
                name = f"{partition}_{label}_{i}"
                records.append(
                    {
                        "id": name,
                        "group_id": name,
                        "origin_id": name,
                        "attack_family_id": name,
                        "source_id": "training"
                        if partition in {"train", "validation"}
                        else partition,
                        "content_sha256": digest(name),
                        "partition": partition,
                        "task": "prompt_injection" if label else "benign",
                        "label": label,
                        "reviewers": ["human_one", "human_two"],
                        "adjudication": "complete",
                        "known_target_training_roles": [],
                        "task_context_sha256": digest(f"context:{name}"),
                        "trusted_instruction_sha256": digest("trusted_task_instruction"),
                        "trust_boundary": "untrusted_content",
                        "objective_id": "synthetic-injection-objective" if label else None,
                        "negative_subtype": None
                        if label
                        else "quoted_instruction"
                        if i % 2
                        else "neutral_instruction",
                    }
                )
    target_records = [
        {
            **row,
            "id": "independent_target:" + row["id"],
            "group_id": "independent_target:" + row["group_id"],
            "origin_id": "independent_target:" + row["origin_id"],
            "attack_family_id": "independent_target:" + row["attack_family_id"],
            "source_id": "independent_target_corpus",
            "content_sha256": digest("independent_target:" + row["id"]),
        }
        for row in records
        if row["partition"] in {"train", "validation"}
    ]
    target_corpus = {
        "grouping_method": "synthetic target exact, near-duplicate and template-family closure",
        "grouping_evidence_sha256": digest("target_group_review"),
        "records": target_records,
        "target_preparations": [],
    }
    return {
        "grouping_method": "synthetic exact, near-duplicate and template-family closure",
        "grouping_evidence_sha256": digest("grouping_evidence"),
        "records": records,
        "target_preparations": [
            {
                "target_role": "replication",
                "training_group_ids": sorted(
                    row["group_id"] for row in target_records if row["partition"] == "train"
                ),
                "validation_group_ids": sorted(
                    row["group_id"] for row in target_records if row["partition"] == "validation"
                ),
                "origin_ids": sorted(row["origin_id"] for row in target_records),
                "attack_family_ids": sorted(row["attack_family_id"] for row in target_records),
                "source_revisions": {"independent_target_corpus": digest("target_data_revision")},
                "target_corpus_sha256": content_digest(target_corpus),
                "grouping_method": target_corpus["grouping_method"],
                "corpus_records": target_records,
                "preparation_protocol_sha256": digest("target_preparation_protocol"),
                "grouping_evidence_sha256": digest("target_group_review"),
                "annotation_evidence_sha256": digest("target_human_task_review"),
                "checkpoint_sha256": digest("checkpoint:target_b"),
                "tokenizer_sha256": digest("tokenizer:target_b"),
            }
        ],
    }


def evidence():

    protocol = Protocol.model_validate(protocol_dict())
    corpus = CorpusEvidence.model_validate(corpus_dict())
    registration = RegistrationEvidence.model_validate(
        {
            "registry_uri": "https://example.invalid/synthetic-registration",
            "registered_at": "2026-10-01T00:00:00+00:00",
            "protocol_sha256": content_digest(protocol),
            "corpus_sha256": content_digest(corpus),
            "evidence_sha256": digest("registration_evidence"),
        }
    )
    return protocol, corpus, registration


def records_for(phase: str, protocol, corpus):

    realizations = []
    predictions = []
    similarities = []
    attempts = []
    annotations = []
    selected_records = [row for row in corpus.records if row.partition == phase]
    identities = {}
    for asset in protocol.models:
        for replicate in protocol.replicates:
            key = f"{asset.id}:{replicate}" if asset.kind != "frozen" else asset.id
            checkpoint = digest(f"checkpoint:{key}")
            tokenizer = digest(f"tokenizer:{key}")
            identities[asset.id, replicate] = (checkpoint, tokenizer)
            realizations.append(
                {
                    "model_id": asset.id,
                    "replicate": replicate,
                    "checkpoint_sha256": checkpoint,
                    "tokenizer_sha256": tokenizer,
                    "training_seed": (
                        component_seed(20260616, replicate, f"model:{asset.id}")
                        if asset.kind != "frozen"
                        else None
                    ),
                    "train_partition_sha256": (
                        partition_digest(corpus, "train") if asset.kind != "frozen" else None
                    ),
                    "execution_ref": protocol.execution_ref,
                }
            )
            for row in selected_records:
                predictions.append(
                    {
                        "model_id": asset.id,
                        "replicate": replicate,
                        "original_id": row.id,
                        "raw_label": str(row.label),
                        "checkpoint_sha256": checkpoint,
                        "tokenizer_sha256": tokenizer,
                    }
                )
    for role in protocol.target_roles:
        for replicate in protocol.replicates:
            for i, candidate in enumerate(role.candidate_ids):
                checkpoint, tokenizer = identities[candidate, replicate]
                target_checkpoint, target_tokenizer = identities[role.model_id, replicate]
                similarities.append(
                    {
                        "target_role": role.id,
                        "source_id": candidate,
                        "replicate": replicate,
                        "mean_cka": 0.9 - i * 0.1,
                        "probe_partition_sha256": partition_digest(corpus, "probe"),
                        "source_checkpoint_sha256": checkpoint,
                        "source_tokenizer_sha256": tokenizer,
                        "target_checkpoint_sha256": target_checkpoint,
                        "target_tokenizer_sha256": target_tokenizer,
                    }
                )
    for replicate in protocol.replicates:
        for source_index in range(8):
            source = f"source_{source_index}"
            for row in selected_records:
                if row.label != 1:
                    continue
                index = int(row.id.rsplit("_", 1)[1])
                for recipe in protocol.recipes:
                    name = f"{phase}:{replicate}:{source}:{row.id}:{recipe}"
                    candidate_hash = digest(f"candidate:{name}")
                    source_flips = (source_index == 7 and index < 3) or (
                        source_index == 6 and index == 3
                    )
                    target_flips = (
                        (source_index == 0 and index < 2)
                        or (source_index == 1 and index >= 2)
                        or (source_index == 7 and index == 0)
                    )
                    source_checkpoint, source_tokenizer = identities[source, replicate]
                    attempts.append(
                        {
                            "id": name,
                            "source_id": source,
                            "replicate": replicate,
                            "original_id": row.id,
                            "recipe": recipe,
                            "status": "success" if source_flips else "failed",
                            "source_queries": 50,
                            "target_search_queries": 0,
                            "attack_seed": component_seed(
                                20260616, replicate, f"attack:{source}:{recipe}"
                            ),
                            "candidate_sha256": candidate_hash,
                            "source_raw_perturbed": "0" if source_flips else "1",
                            "source_checkpoint_sha256": source_checkpoint,
                            "source_tokenizer_sha256": source_tokenizer,
                            "target_predictions": [
                                {
                                    "model_id": role.model_id,
                                    "raw_label": "0" if target_flips else "1",
                                    "checkpoint_sha256": identities[role.model_id, replicate][0],
                                    "tokenizer_sha256": identities[role.model_id, replicate][1],
                                }
                                for role in protocol.target_roles
                            ],
                            "mechanical_screen": "passed",
                            "mechanical_evidence_sha256": digest(f"screen:{name}"),
                        }
                    )
                    annotations.append(
                        {
                            "attempt_id": name,
                            "original_sha256": row.content_sha256,
                            "candidate_sha256": candidate_hash,
                            "protocol_sha256": protocol.annotation_protocol_sha256,
                            "blinded": True,
                            "votes": [
                                {
                                    "reviewer_id": reviewer,
                                    "semantic": "preserved",
                                    "intent": "retained",
                                    "is_prompt_injection": True,
                                }
                                for reviewer in ["human_one", "human_two"]
                            ],
                            "adjudicator": None,
                        }
                    )
    return ExperimentRecords.model_validate(
        {
            "phase": phase,
            "protocol_sha256": content_digest(protocol),
            "corpus_sha256": content_digest(corpus),
            "execution_ref": protocol.execution_ref,
            "started_at": "2026-10-02T00:00:00+00:00"
            if phase == "calibration"
            else "2026-10-04T00:00:00+00:00",
            "completed_at": "2026-10-02T01:00:00+00:00"
            if phase == "calibration"
            else "2026-10-04T01:00:00+00:00",
            "hardware": "synthetic test fixture; no model execution",
            "realizations": realizations,
            "clean_predictions": predictions,
            "similarities": similarities if phase == "calibration" else [],
            "attempts": attempts,
            "annotations": annotations,
            "application_records": [],
        }
    )
