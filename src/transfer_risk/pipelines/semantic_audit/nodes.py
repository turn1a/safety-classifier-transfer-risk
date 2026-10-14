"""Catalog-facing semantic review from saved attacks, without model access."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd

from transfer_risk.lib import semantic_audit, target_audit
from transfer_risk.lib.retrospective_audit import build_retrospective_cells, verify_reference_counts
from transfer_risk.lib.semantic_review_tool import (
    REVIEWER_IDS,
    build_review_dataset,
    render_review_tool_html,
)
from transfer_risk.lib.semantic_yield import summarize_valid_flip_yield

logger = logging.getLogger(__name__)


def prepare_semantic_audit(
    adversarial_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    task_splits: Mapping[str, pd.DataFrame],
    semantic_params: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare private blinded exact-pair sampling from every saved attack outcome.

    Args:
        adversarial_examples: Saved all-outcome surrogate/recipe attempt records.
        task_splits: Saved split mapping used for historical attack-source attribution.
        semantic_params: Evaluation clipping and per-stratum review sample parameters.
        seed: Root seed passed to the deterministic project seed derivation.

    Returns:
        Private full pair table, private blinded candidate template, private occurrence
        and sampling linkage, and public aggregate-only fidelity summary. These tables
        contain no fabricated human labels. Raw text outputs require private local catalog entries.
    """
    source_map = target_audit.build_eval_original_source_map(
        task_splits["test"],
        eval_set_size=int(semantic_params["eval_set_size"]),
        max_prompt_chars=int(semantic_params["max_prompt_chars"]),
    )
    pairs, linkage = semantic_audit.build_pair_population(adversarial_examples, source_map)
    candidates, sampled_linkage = semantic_audit.sample_annotation_candidates(
        pairs,
        linkage,
        pairs_per_stratum=semantic_params["pairs_per_stratum"],
        seed=seed,
    )
    summary = semantic_audit.fidelity_summary(pairs, sampled_linkage)
    logger.info(
        "Prepared semantic review: %d saved attempts, %d unique pairs, %d blinded candidates",
        summary["population"]["attempts"],
        len(pairs),
        len(candidates),
    )
    return pairs, candidates, sampled_linkage, summary


def pending_semantic_review(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Describe the pending sample without reading or replacing submitted annotations.

    Args:
        pairs: Private full pair population.
        linkage: Private full occurrence and sampling linkage.

    Returns:
        Private pending consensus and public pending coverage/bounds. Catalog outputs
        must differ from completed-review outputs so preparation reruns preserve review evidence.
    """
    return review_semantic_audit(pairs, linkage, pd.DataFrame(), {"reviewer_ids": []})


def review_semantic_audit(
    pairs: pd.DataFrame,
    linkage: pd.DataFrame,
    annotations: pd.DataFrame,
    semantic_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate private independent human reviews and publish aggregate coverage.

    Args:
        pairs: Private full pair population from preparation.
        linkage: Private full occurrence and sampling linkage from the same preparation.
        annotations: Private annotation CSV with reviewer identities and explicit labels.
        semantic_params: Registered reviewer identities, separate from the annotation template.

    Returns:
        Private pair consensus and public aggregate-only review summary. Missing or
        disputed labels remain unresolved; no inference or automatic semantic label is used.

    Raises:
        ValueError: If reviewer configuration is not a list of identity strings.
    """
    reviewer_ids = semantic_params.get("reviewer_ids", [])
    if not isinstance(reviewer_ids, list) or not all(
        isinstance(value, str) for value in reviewer_ids
    ):
        msg = "semantic_audit.reviewer_ids must be a list of reviewer identity strings"
        raise ValueError(msg)
    selected = sorted(set(linkage.loc[linkage["selected_for_review"], "pair_id"]))
    consensus = semantic_audit.resolve_annotations(annotations, selected, reviewer_ids)
    summary = semantic_audit.review_summary(pairs, linkage, consensus)
    logger.info(
        "Semantic review: %d resolved of %d selected pairs; %d population pairs unresolved",
        summary["resolved_pairs"],
        summary["sampled_pairs"],
        summary["unresolved_population_pairs"],
    )
    return consensus, summary


def render_semantic_review_tools(
    candidates: pd.DataFrame,
    html_template: str,
    semantic_params: Mapping[str, Any],
) -> tuple[str, str]:
    """Render two private role-bound tools from the already saved blinded sample.

    Args:
        candidates: Catalog-owned blinded annotation sample from preparation.
        html_template: Source-controlled static template loaded through the catalog.
        semantic_params: Confirmed ``reviewer_1`` and ``reviewer_2`` identity configuration.

    Returns:
        Private self-contained HTML for the user and partner, respectively. The node
        does not resample, load model scores, read submitted labels, or update consensus.

    Raises:
        ValueError: If the two fixed reviewer roles have not been configured.
    """
    if semantic_params.get("reviewer_ids") != list(REVIEWER_IDS):
        msg = "semantic_audit.reviewer_ids must be [reviewer_1, reviewer_2] for the private tool"
        raise ValueError(msg)
    dataset = build_review_dataset(cast("list[dict[str, Any]]", candidates.to_dict("records")))
    first = render_review_tool_html(dataset, html_template, REVIEWER_IDS[0])
    second = render_review_tool_html(dataset, html_template, REVIEWER_IDS[1])
    logger.info("Rendered two fixed-role private tools for %d saved sample pairs", len(candidates))
    return first, second


def summarize_semantic_valid_flip_yield(
    consensus: pd.DataFrame,
    predictions: pd.DataFrame,
    linkage: pd.DataFrame,
    reference_params: Mapping[str, Any],
    historical_cells: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Connect private human consensus to saved reference-target classifier changes.

    Args:
        consensus: Private validated consensus from the preceding human-review node.
        predictions: Saved reference-target exact-pair predictions, without new inference.
        linkage: Private full trial and sampling linkage.
        reference_params: ``retrospective_audit.targets.protectai-v2`` configuration.
        historical_cells: Published source-success reference cells from the original audit.
        manifest: Historical grid and trial-count configuration.

    Returns:
        Aggregate-only literal valid-flip count/yield bounds for both saved policies.
        Trial-yield point estimates and original-union validity estimates are withheld.
    """
    cells, _, _ = build_retrospective_cells(
        predictions,
        linkage,
        target_id="protectai-v2",
        positive_label=int(reference_params["positive_label"]),
        benign_label=int(reference_params["benign_label"]),
        excluded_surrogates=reference_params.get("excluded_surrogates", []),
    )
    receipt = verify_reference_counts(cells, historical_cells, manifest, dict(reference_params))
    result = summarize_valid_flip_yield(
        consensus,
        predictions,
        linkage,
        target_id="protectai-v2",
        positive_label=int(reference_params["positive_label"]),
        benign_label=int(reference_params["benign_label"]),
        excluded_surrogates=reference_params.get("excluded_surrogates", []),
    )
    result["reference_check"] = receipt
    return result
