"""Catalog-facing preparation and human-only finalization of the frozen flip cohort."""

from __future__ import annotations

from typing import Any

import pandas as pd

from transfer_risk.lib.retrospective_audit import (
    build_retrospective_cells,
    reference_cells_digest,
    verify_reference_counts,
)
from transfer_risk.lib.semantic_audit import resolve_annotations
from transfer_risk.lib.semantic_flip_audit import (
    build_flip_cohort,
    fingerprint,
    summarize_flip_review,
    validate_flip_cohort,
)
from transfer_risk.lib.semantic_flip_reviews import merge_flip_exports
from transfer_risk.lib.semantic_review_tool import REVIEWER_IDS, render_review_tool_html


def prepare_flip_cohort(  # noqa: PLR0913
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_cells: pd.DataFrame,
    historical_cells: pd.DataFrame,
    manifest: dict[str, Any],
    checked_public_context: dict[str, Any],
    parameters: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Validate current saved predictions against independently checked public evidence."""
    current, _, _ = build_retrospective_cells(
        predictions, linkage, target_id="protectai-v2", excluded_surrogates=["bilstm-attention"]
    )
    try:
        rates = [
            "source_gated_yield",
            "all_candidate_yield",
            "source_conditional_flip_rate",
            "all_candidate_conditional_flip_rate",
        ]
        keys = ["surrogate", "recipe"]
        reduced = current.sort_values(keys).reset_index(drop=True)
        saved = reference_cells.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            reduced.drop(columns=rates), saved.drop(columns=rates), check_exact=True
        )
        pd.testing.assert_frame_equal(
            reduced[rates],
            saved[rates],
            check_exact=False,
            rtol=0,
            atol=1e-15,
        )
    except AssertionError as error:
        msg = "current saved predictions differ from checked reference cells"
        raise ValueError(msg) from error
    receipt = verify_reference_counts(
        current, historical_cells, manifest, {"excluded_surrogates": ["bilstm-attention"]}
    )
    if (
        checked_public_context.get("inference_context_check", {}).get("identity_and_settings_agree")
        is not True
    ):
        msg = "semantic flip preparation requires independent saved-context validation"
        raise ValueError(msg)
    cohort = build_flip_cohort(
        pairs, linkage, predictions, seed=seed, sample_size=parameters["sample_trials"]
    )
    expected = parameters["expected_population"]
    actual = {
        "attempted_trials": cohort["attempted_trials"],
        "changed_trials": len(cohort["trials"]),
        "unique_pairs": cohort["review_dataset"]["pair_count"],
    }
    if actual != expected:
        msg = "saved semantic flip population differs from the approved cohort counts"
        raise ValueError(msg)
    cohort["reference_check"] = receipt
    cohort["saved_reference_cells_sha256"] = reference_cells_digest(reference_cells)
    cohort["checked_public_context_fingerprint"] = fingerprint(checked_public_context)
    cohort["cohort_fingerprint"] = fingerprint(
        {key: value for key, value in cohort.items() if key != "cohort_fingerprint"}
    )
    return cohort


def pending_flip_review(cohort: dict[str, Any], similarity: pd.DataFrame) -> dict[str, Any]:
    """Publish pending aggregate-only bounds without loading or modifying human answers."""
    return summarize_flip_review(cohort, pd.DataFrame(columns=["pair_id", "status"]), similarity)


def render_flip_tools(cohort: dict[str, Any], template: str) -> tuple[str, str]:
    """Render the frozen stage order into separate private fixed-role browser tools."""
    validate_flip_cohort(cohort)
    return (
        render_review_tool_html(cohort["review_dataset"], template, REVIEWER_IDS[0]),
        render_review_tool_html(cohort["review_dataset"], template, REVIEWER_IDS[1]),
    )


def finalize_flip_review(
    cohort: dict[str, Any], annotations: pd.DataFrame, similarity: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate imported provenance and resolve only genuine independent human labels."""
    validate_flip_cohort(cohort)
    checked = merge_flip_exports(cohort["review_dataset"], [], [], annotations)
    consensus = resolve_annotations(
        checked, [pair["pair_id"] for pair in cohort["review_dataset"]["pairs"]], REVIEWER_IDS
    )
    return consensus, summarize_flip_review(cohort, consensus, similarity)
