"""Conflict-refusing human export reuse with exact-text and fixed-role provenance."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from transfer_risk.lib.semantic_flip_audit import fingerprint
from transfer_risk.lib.semantic_review_tool import REVIEW_CSV_COLUMNS, combine_reviewer_exports

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _validated_rows(
    dataset: Mapping[str, Any],
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate both fixed roles and return nonpending export rows."""
    return cast(
        "list[dict[str, Any]]", combine_reviewer_exports(dataset, first, second).to_dict("records")
    )


def _existing_annotations(
    dataset: Mapping[str, Any], existing: pd.DataFrame
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate saved rows and their private role/text provenance before merging."""
    current_pairs = {row["pair_id"]: row for row in dataset["pairs"]}
    accumulated: dict[tuple[str, str], dict[str, Any]] = {}
    if not existing.empty:
        required = {*REVIEW_CSV_COLUMNS, "provenance"}
        if set(existing.columns) != required:
            msg = "existing annotations require the role-bound schema and provenance"
            raise ValueError(msg)
        if not existing["reviewer_id"].isin(["reviewer_1", "reviewer_2"]).all():
            msg = "existing annotations contain an unknown reviewer role"
            raise ValueError(msg)
        records = cast("list[dict[str, Any]]", existing.to_dict("records"))
        _validated_rows(
            dataset,
            [
                {key: row[key] for key in REVIEW_CSV_COLUMNS}
                for row in records
                if row["reviewer_id"] == "reviewer_1"
            ],
            [
                {key: row[key] for key in REVIEW_CSV_COLUMNS}
                for row in records
                if row["reviewer_id"] == "reviewer_2"
            ],
        )
        for row in records:
            proof = json.loads(row["provenance"])
            expected = fingerprint(current_pairs[row["pair_id"]])
            if not proof or any(
                entry.get("reviewer_id") != row["reviewer_id"]
                or entry.get("exact_pair_fingerprint") != expected
                or not entry.get("source_dataset_fingerprint")
                for entry in proof
            ):
                msg = "existing annotation provenance does not certify exact text and reviewer role"
                raise ValueError(msg)
            accumulated[(row["pair_id"], row["reviewer_id"])] = {**row, "provenance": proof}
    return accumulated


def merge_flip_exports(
    dataset: Mapping[str, Any],
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    existing: pd.DataFrame,
    *,
    reuse_dataset: Mapping[str, Any] | None = None,
    reuse_first: Sequence[Mapping[str, Any]] = (),
    reuse_second: Sequence[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Merge explicit answers without erasing old answers or accepting contradictions.

    Args:
        dataset: New frozen staged dataset, including exact queue texts.
        first: New reviewer-one CSV export rows.
        second: New reviewer-two CSV export rows.
        existing: Previously imported new-cohort annotation rows with provenance.
        reuse_dataset: Optional older exact dataset validating exported old answers.
        reuse_first: Explicit old reviewer-one CSV rows, never inferred from browser state.
        reuse_second: Explicit old reviewer-two CSV rows.

    Returns:
        Private role-bound annotations with source fingerprint, role, and exact-text
        provenance retained per answer. Blank fields cannot erase prior responses.

    Raises:
        ValueError: If fingerprints, roles, exact pairs, existing provenance or answers conflict.
    """
    current_pairs = {row["pair_id"]: row for row in dataset["pairs"]}
    accumulated = _existing_annotations(dataset, existing)
    sources = [(dataset, _validated_rows(dataset, first, second))]
    if reuse_dataset is not None:
        sources.append((reuse_dataset, _validated_rows(reuse_dataset, reuse_first, reuse_second)))
    elif reuse_first or reuse_second:
        msg = "reused exports require their exact source dataset"
        raise ValueError(msg)
    for source, rows in sources:
        source_pairs = {pair["pair_id"]: pair for pair in source["pairs"]}
        for row in rows:
            identity = row["pair_id"]
            if identity not in current_pairs:
                continue
            if source_pairs[identity] != current_pairs[identity]:
                msg = "reused review exact texts differ from the current pair"
                raise ValueError(msg)
            proof = {
                "source_dataset_fingerprint": source["dataset_fingerprint"],
                "source_schema_version": source["schema_version"],
                "reviewer_id": row["reviewer_id"],
                "exact_pair_fingerprint": fingerprint(current_pairs[identity]),
            }
            key = (identity, row["reviewer_id"])
            old = accumulated.get(key)
            incoming = {
                **row,
                "dataset_fingerprint": dataset["dataset_fingerprint"],
                "provenance": [proof],
            }
            if old is not None:
                for field in (
                    "independent_review",
                    "semantic_preserved",
                    "intent_preserved",
                    "original_is_injection",
                    "notes",
                ):
                    if old[field] and incoming[field] and old[field] != incoming[field]:
                        msg = "review import conflicts with a previously retained answer"
                        raise ValueError(msg)
                    incoming[field] = old[field] or incoming[field]
                incoming["provenance"] = old["provenance"] + (
                    [proof] if proof not in old["provenance"] else []
                )
                incoming["review_status"] = (
                    "complete"
                    if incoming["independent_review"] == "yes"
                    and all(
                        incoming[field]
                        for field in (
                            "semantic_preserved",
                            "intent_preserved",
                            "original_is_injection",
                        )
                    )
                    else "in_progress"
                )
            accumulated[key] = incoming
    output = [
        {**row, "provenance": json.dumps(row["provenance"], sort_keys=True)}
        for _, row in sorted(accumulated.items())
    ]
    return pd.DataFrame(output, columns=[*REVIEW_CSV_COLUMNS, "provenance"])


def progress_for_role(
    dataset: Mapping[str, Any], annotations: pd.DataFrame, reviewer: str
) -> dict[str, Any]:
    """Create a role-only browser restore file from validated genuine imported answers.

    Args:
        dataset: New staged exact-pair dataset.
        annotations: Imported annotations retaining exact-text and role provenance.
        reviewer: Fixed role whose own answers may enter this restore file.

    Returns:
        Private progress JSON accepted by the browser's matching-role import control.
        It includes no other role's answers or source metadata.

    Raises:
        ValueError: If the role, dataset or saved answer provenance is invalid.
    """
    if reviewer not in {"reviewer_1", "reviewer_2"}:
        msg = "progress requires a fixed reviewer role"
        raise ValueError(msg)
    checked = merge_flip_exports(dataset, [], [], annotations)
    fields = [
        "semantic_preserved",
        "intent_preserved",
        "original_is_injection",
        "independent_review",
        "notes",
    ]
    rows = cast(
        "list[dict[str, Any]]", checked.loc[checked["reviewer_id"] == reviewer].to_dict("records")
    )
    return {
        "schema_version": "semantic-review-progress-v1",
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "reviewer_id": reviewer,
        "current_pair_id": dataset["pairs"][0]["pair_id"],
        "reviews": {row["pair_id"]: {key: row[key] for key in fields} for row in rows},
    }
