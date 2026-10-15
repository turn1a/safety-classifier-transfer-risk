"""Freeze CKA and source-only selectors and compare them on held-out groups."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.experiment_v2.protocol import component_seed
from transfer_risk.lib.experiment_v2.records import (
    EvidenceIndex,
    pair_review_summary,
    source_flips,
    target_flips,
)

if TYPE_CHECKING:
    from transfer_risk.lib.experiment_v2.schemas import Protocol, TargetRole


def positive_groups(index: EvidenceIndex) -> dict[str, list[str]]:
    """Collect independent positive-original groups in a stable order.

    Args:
        index: Validated evidence for one partition.

    Returns:
        Positive original ids grouped by preregistered near-duplicate/template families.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for original in index.originals.values():
        if original.label == 1:
            groups[original.group_id].append(original.id)
    return {name: groups[name] for name in sorted(groups)}


def group_vector(index: EvidenceIndex, covered: set[str]) -> list[float]:
    """Compute one within-group original yield per independent group.

    Args:
        index: Validated positive-original group membership.
        covered: Originals with at least one qualifying returned candidate.

    Returns:
        Group yields, giving every family equal weight regardless of its row count.
    """
    return [
        sum(original in covered for original in originals) / len(originals)
        for originals in positive_groups(index).values()
    ]


def utility(index: EvidenceIndex, model: str, replicate: int) -> dict[str, float]:
    """Measure matched clean detector utility on the complete calibration partition.

    Args:
        index: Validated clean controls, including benign originals.
        model: Reviewed detector id.
        replicate: Registered seed replicate.

    Returns:
        TPR and FPR at the frozen argmax decision rule, with their denominators.
    """
    positives = [row for row in index.originals.values() if row.label == 1]
    negatives = [row for row in index.originals.values() if row.label == 0]
    result = {
        "tpr": sum(index.clean[model, replicate, row.id] == 1 for row in positives)
        / len(positives),
        "fpr": sum(index.clean[model, replicate, row.id] == 1 for row in negatives)
        / len(negatives),
        "positives": float(len(positives)),
        "benign": float(len(negatives)),
    }
    for subtype in ("quoted_instruction", "neutral_instruction"):
        rows = [row for row in negatives if row.negative_subtype == subtype]
        if rows:
            result[f"{subtype}_fpr"] = sum(
                index.clean[model, replicate, row.id] == 1 for row in rows
            ) / len(rows)
            result[f"{subtype}_benign"] = float(len(rows))
    return result


def coverage_maps(
    protocol: Protocol,
    index: EvidenceIndex,
    role: TargetRole,
    *,
    outcome: str,
    state: str = "valid",
) -> dict[tuple[int, str], set[str]]:
    """Collect valid or unresolved original coverage without success conditioning.

    Args:
        protocol: Fixed target and source label maps.
        index: Full-grid phase evidence.
        role: Target and eligible source pool.
        outcome: Source transition for calibration or target transition for evaluation.
        state: Required human evidence state.

    Returns:
        Covered originals by source and replicate.
    """
    result: dict[tuple[int, str], set[str]] = {
        (replicate, source): set()
        for replicate in protocol.replicates
        for source in role.candidate_ids
    }
    for attempt in index.attempts.values():
        if attempt.source_id not in role.candidate_ids or index.human.get(attempt.id) != state:
            continue
        flipped = (
            source_flips(protocol, index, attempt)
            if outcome == "source"
            else target_flips(protocol, index, attempt, role.model_id)
        )
        if flipped:
            result[attempt.replicate, attempt.source_id].add(attempt.original_id)
    return result


def _union_coverage(
    maps: dict[tuple[int, str], set[str]], replicate: int, sources: list[str] | tuple[str, ...]
) -> set[str]:
    """Union recipe and source successes for one selected pool.

    Args:
        maps: Per-source original coverage.
        replicate: Registered seed replicate.
        sources: Equal-budget selected candidate ids.

    Returns:
        Distinct originals covered by any selected source and recipe.
    """
    return set().union(*(maps[replicate, source] for source in sources))


def _greedy_source_selection(
    index: EvidenceIndex,
    maps: dict[tuple[int, str], set[str]],
    replicate: int,
    candidates: list[str],
    selected_count: int,
) -> list[str]:
    """Select complementary calibration source successes without target outcome feedback.

    Args:
        index: Calibration group membership only.
        maps: Human-valid source-transition coverage only.
        replicate: Seed replicate.
        candidates: Pool satisfying the matched clean utility gate.
        selected_count: Fixed number of sources, identical to CKA and random selection.

    Returns:
        Greedy maximum group-yield sources, with model-id tie breaking.
    """
    selected: list[str] = []
    covered: set[str] = set()
    for _ in range(selected_count):
        remaining = sorted(set(candidates) - set(selected))
        scores = {
            source: float(np.mean(group_vector(index, covered | maps[replicate, source])))
            for source in remaining
        }
        best = min(remaining, key=lambda source: (-scores[source], source))
        selected.append(best)
        covered |= maps[replicate, best]
    return selected


def calibration_selections(protocol: Protocol, index: EvidenceIndex) -> dict[str, Any]:
    """Freeze CKA and source-only choices using independent probe and calibration evidence.

    Args:
        protocol: Preregistered candidate pools, utility limits, and source count.
        index: Complete calibration evidence and independent probe CKA.

    Returns:
        Deterministic source choices and utility evidence for each role and replicate.

    Raises:
        ValueError: If human calibration review or matched detector eligibility is insufficient.
    """
    selections: dict[str, Any] = {}
    for role in protocol.target_roles:
        clean_utility = {
            f"{model}:{replicate}": utility(index, model, replicate)
            for model in [role.model_id, *role.candidate_ids]
            for replicate in protocol.replicates
        }
        qualified = {
            model
            for model in [role.model_id, *role.candidate_ids]
            if all(
                clean_utility[f"{model}:{replicate}"]["tpr"] >= protocol.utility_min_tpr
                and all(
                    clean_utility[f"{model}:{replicate}"][metric] <= protocol.utility_max_fpr
                    for metric in ("fpr", "quoted_instruction_fpr", "neutral_instruction_fpr")
                )
                for replicate in protocol.replicates
            )
        }
        candidates = sorted(set(role.candidate_ids) & qualified)
        if (
            role.model_id not in qualified
            or len(candidates) <= protocol.selection_k
            or (len(candidates) < protocol.minimum_eligible_sources)
        ):
            msg = "matched calibration utility leaves too few eligible detectors"
            raise ValueError(msg)
        required = [
            attempt
            for attempt in index.attempts.values()
            if attempt.source_id in candidates and source_flips(protocol, index, attempt)
        ]
        review = pair_review_summary(index, required)
        if review["unresolved_pairs"]:
            msg = "calibration human validity is unresolved for eligible source transitions"
            raise ValueError(msg)
        maps = coverage_maps(protocol, index, role, outcome="source")
        for replicate in protocol.replicates:
            ranked = sorted(
                candidates, key=lambda source: (-index.cka[role.id, replicate, source], source)
            )
            selections[f"{role.id}:{replicate}"] = {
                "eligible_pool": candidates,
                "cka": ranked[: protocol.selection_k],
                "source_only": _greedy_source_selection(
                    index, maps, replicate, candidates, protocol.selection_k
                ),
                "calibration_clean_utility": clean_utility,
                "source_outcome_review": review,
            }
    return selections


def _strategy_vectors(
    protocol: Protocol,
    index: EvidenceIndex,
    maps: dict[tuple[int, str], set[str]],
    role: TargetRole,
    selections: dict[str, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], list[float]]:
    """Evaluate equal-budget frozen and random selectors over independent held-out groups.

    Args:
        protocol: Fixed replicate and source-count rules.
        index: Held-out evaluation originals and groups.
        maps: Human-valid or conservatively bounded target-transition coverage.
        role: Fixed target role.
        selections: Frozen calibration/probe decisions without evaluation outcomes.

    Returns:
        Group vectors averaged over seeds for CKA, source-only, and expected random
        selection, and descriptive within-replicate random selection ranks.
    """
    vectors: dict[str, list[list[float]]] = defaultdict(list)
    ranks = []
    for replicate in protocol.replicates:
        chosen = selections[f"{role.id}:{replicate}"]
        cka = group_vector(index, _union_coverage(maps, replicate, chosen["cka"]))
        source = group_vector(index, _union_coverage(maps, replicate, chosen["source_only"]))
        random = [
            group_vector(index, _union_coverage(maps, replicate, subset))
            for subset in itertools.combinations(chosen["eligible_pool"], protocol.selection_k)
        ]
        random_means = np.mean(random, axis=1)
        ranks.append(float(np.mean(random_means >= float(np.mean(cka)) - 1e-12)))
        vectors["cka"].append(cka)
        vectors["source_only"].append(source)
        vectors["random"].append(np.mean(random, axis=0).tolist())
    return (
        np.mean(vectors["cka"], axis=0),
        np.mean(vectors["source_only"], axis=0),
        np.mean(vectors["random"], axis=0),
        ranks,
    )


def _comparison(
    protocol: Protocol, differences: np.ndarray[Any, Any], seed: int
) -> dict[str, float]:
    """Compute a paired cluster-bootstrap lower bound with planned multiplicity adjustment.

    Args:
        protocol: Registered bootstrap, alpha, target roles, and meaningful effect.
        differences: One seed-averaged paired effect per independent evaluation group.
        seed: Derived bootstrap seed.

    Returns:
        The observed effect and one-sided Bonferroni-adjusted percentile lower bound.
    """
    rng = np.random.default_rng(seed)
    bootstrap = [
        float(np.mean(differences[rng.integers(0, len(differences), len(differences))]))
        for _ in range(protocol.bootstrap_replicates)
    ]
    comparisons_per_target = 2
    adjusted_alpha = protocol.alpha / (comparisons_per_target * len(protocol.target_roles))
    return {
        "effect": float(np.mean(differences)),
        "lower_bound": float(np.quantile(bootstrap, adjusted_alpha)),
        "one_sided_confidence": 1 - adjusted_alpha,
        "minimum_effect": protocol.minimum_effect,
    }


def evaluate_selection(
    protocol: Protocol, index: EvidenceIndex, selections: dict[str, Any], root_seed: int
) -> dict[str, Any]:
    """Compare frozen selectors on held-out data and withhold inference for missing labels.

    Args:
        protocol: Registered estimands, pools, and inference procedure.
        index: Complete held-out accounting with possibly unresolved human judgments.
        selections: Frozen choices made before evaluation was unsealed.
        root_seed: Registered root seed for deterministic paired group resampling.

    Returns:
        Aggregate policy yields, honest incomplete states, and conditional group intervals.
    """
    results = {}
    for role in protocol.target_roles:
        maps = coverage_maps(protocol, index, role, outcome="target")
        cka, source, random, ranks = _strategy_vectors(protocol, index, maps, role, selections)
        required = [
            attempt
            for attempt in index.attempts.values()
            if attempt.source_id in selections[f"{role.id}:{attempt.replicate}"]["eligible_pool"]
            and target_flips(protocol, index, attempt, role.model_id)
        ]
        review = pair_review_summary(index, required)
        pending = review["unresolved_pairs"] > 0
        common = {
            "outcome_review": review,
            "evaluation_groups": len(positive_groups(index)),
            "evaluation_positive_originals": sum(
                row.label == 1 for row in index.originals.values()
            ),
            "replicates": len(protocol.replicates),
            "per_original_source_query_cap": (
                protocol.selection_k * len(protocol.recipes) * protocol.source_query_budget
            ),
            "selections": {
                str(replicate): {
                    name: selections[f"{role.id}:{replicate}"][name]
                    for name in ("cka", "source_only")
                }
                for replicate in protocol.replicates
            },
        }
        if pending:
            unresolved = coverage_maps(protocol, index, role, outcome="target", state="unresolved")
            upper_maps = {key: covered | unresolved[key] for key, covered in maps.items()}
            upper_cka, upper_source, upper_random, _ = _strategy_vectors(
                protocol, index, upper_maps, role, selections
            )
            results[role.id] = {
                **common,
                "status": "incomplete_human_review",
                "decision": None,
                "cka_yield_lower": float(np.mean(cka)),
                "cka_yield_upper": float(np.mean(upper_cka)),
                "source_only_yield_lower": float(np.mean(source)),
                "source_only_yield_upper": float(np.mean(upper_source)),
                "random_yield_lower": float(np.mean(random)),
                "random_yield_upper": float(np.mean(upper_random)),
            }
            continue
        comparisons = {
            name: _comparison(
                protocol,
                cka - baseline,
                component_seed(root_seed, 0, f"bootstrap:{role.id}:{name}"),
            )
            for name, baseline in (("random", random), ("source_only", source))
        }
        pool_size = len(selections[f"{role.id}:{protocol.replicates[0]}"]["eligible_pool"])
        results[role.id] = {
            **common,
            "status": "complete",
            "cka_yield": float(np.mean(cka)),
            "source_only_yield": float(np.mean(source)),
            "random_expected_yield": float(np.mean(random)),
            "random_subset_count": math.comb(pool_size, protocol.selection_k),
            "finite_pool_random_rank_probability": float(np.mean(ranks)),
            "random_rank_definition": "Mean within-seed random-subset rank; descriptive.",
            "comparisons": comparisons,
            "decision": all(
                comparison["lower_bound"] > protocol.minimum_effect
                for comparison in comparisons.values()
            ),
            "inference_scope": (
                "Group resampling conditional on the three realized training "
                "replicates, fixed targets, pool, and recipes."
            ),
        }
    return results
