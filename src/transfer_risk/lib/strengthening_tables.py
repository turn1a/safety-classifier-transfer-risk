"""Render rounded manuscript includes for local saved-data strengthening."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str) -> str:
    """Render one-line Markdown rows and a Quarto caption."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n\n: " + caption + "\n"


def _model_link(identifier: str) -> str:
    """Link a checkpoint identifier to its model explanation in the study guide."""
    anchor = "target-model" if identifier == "protectai-v2" else f"model-{identifier}"
    return f"[`{identifier}`](guide.qmd#{anchor})"


def render_baseline_table(summary: Mapping[str, Any]) -> str:
    """Display clean-input confusion counts with one-decimal percentages."""
    rows = []
    for report in summary["reports"]:
        counts = report["pooled"]
        positive, negative = counts["canonical_positive_count"], counts["canonical_negative_count"]
        rows.append(
            [
                _model_link(report["target_id"]),
                f"{counts['canonical_positive_predicted_positive_count']}/{positive} "
                f"({100 * counts['canonical_positive_recall']:.1f}%)",
                f"{counts['canonical_negative_predicted_positive_count']}/{negative} "
                f"({100 * counts['canonical_negative_false_positive_rate']:.1f}%)",
                f"{100 * counts['canonical_label_accuracy']:.1f}%",
            ]
        )
    return _table(
        [
            "Detector",
            "[Canonical-positive recall](guide.qmd#clean-recall){.term-help}",
            "[Canonical-benign flagged](guide.qmd#clean-false-positives){.term-help}",
            "[Label accuracy](guide.qmd#clean-accuracy){.term-help}",
        ],
        rows,
        "Clean-input predictions on the saved 295-row test split. Canonical labels mix "
        "prompt injection and jailbreaks; these counts describe the sample. "
        "The affected BiLSTM is omitted. {#tbl-clean-baselines}",
    )


def render_policy_table(comparison: Mapping[str, Any]) -> str:
    """Display common-source candidate outcomes with one shared attempted budget."""
    rows = []
    for row in comparison["common_pool"]:
        originals = row["original_target_positive"] + row["original_target_benign"]
        rows.append(
            [
                _model_link(row["target"]),
                f"{row['original_target_positive']}/{originals}",
                f"{row['source_gated_flips']:,}",
                f"{row['all_candidate_flips']:,}",
                f"{100 * row['all_candidate_yield']:.1f}%",
            ]
        )
    budget = comparison["common_pool"][0]["attempted"]
    sources = len(comparison["common_surrogates"])
    return _table(
        [
            "Target checkpoint",
            "Originally positive",
            "[Source-success flips](guide.qmd#verified-flip){.term-help}",
            "All returned flips",
            "[All returned yield](guide.qmd#all-candidate-yield){.term-help}",
        ],
        rows,
        f"Common-source checkpoint check: each row uses {budget:,} attempted trials over "
        f"the same {sources} surrogates and {comparison['n_recipes']} recipes. "
        "Returned candidates include failed source searches. Different target objectives "
        "and operating points prevent a robustness ranking. {#tbl-checkpoint-policies}",
    )


def render_objective_table(objectives: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    """Display conditional and fixed-budget associations as distinct outcomes."""
    rows = []
    for pool_name, label in (
        ("all_surrogates", "Ten-model historical pool"),
        ("surrogate_excluded", "Nine transformers"),
    ):
        pool = objectives["pools"][pool_name]
        for outcome, outcome_label in (
            ("macro_conditional_verified_flip_rate", "Conditional macro flip rate"),
            ("source_success_gated_verified_flip_yield", "Source-success-gated trial yield"),
        ):
            stats = pool["associations"][outcome]
            rows.append(
                [label, outcome_label, f"{stats['rho']:.3f}", f"{stats['two_sided_p']:.2g}"]
            )
    stats = comparison["reference_pool"]["association"]
    if stats is not None:
        rows.append(
            [
                "Nine transformers",
                "All-returned-candidate trial yield",
                f"{stats['rho']:.3f}",
                f"{stats['two_sided_p']:.2g}",
            ]
        )
    return _table(
        [
            "Pool",
            "Outcome",
            "[Spearman $\\rho$](guide.qmd#rho){.term-help}",
            "[Two-sided p](guide.qmd#exact-p){.term-help}",
        ],
        rows,
        "CKA association depends on the outcome definition. Trial yields retain the same "
        "attempted denominator for each surrogate. Enumerated p-values condition on the "
        "designed-pool exchangeability null; post-hoc rows provide no external selection "
        "validation. {#tbl-objective-associations}",
    )


def render_coverage_table(coverage: Mapping[str, Any]) -> str:
    """Compare selected and observed-best coverage without counting an input twice."""
    denominator = coverage["original_target_positive"]
    rows = []
    policies = (
        ("source_gated", "Source-success gated"),
        ("all_candidate", "All returned candidates"),
    )
    for policy, label in policies:
        chosen = coverage["selected"][f"{policy}_originals"]
        best = coverage["comparisons"][policy]["best_originals"]
        chosen_rate = f"{100 * chosen / denominator:.1f}%" if denominator else "undefined"
        best_rate = f"{100 * best / denominator:.1f}%" if denominator else "undefined"
        rows.append(
            [
                label,
                f"{chosen}/{denominator} ({chosen_rate})",
                f"{best}/{denominator} ({best_rate})",
                str(coverage["comparisons"][policy]["n_strictly_higher"]),
            ]
        )
    return _table(
        [
            "Candidate policy",
            "[CKA-selected coverage](guide.qmd#original-coverage){.term-help}",
            "[Observed-best coverage](guide.qmd#original-coverage){.term-help}",
            "Higher subsets",
        ],
        rows,
        f"Distinct target-positive originals affected by at least one candidate. Each of "
        f"{coverage['n_subsets']} three-source choices uses "
        f"{coverage['attempted_trials_per_subset']:,} "
        "saved trials. Observed-best selections are post-hoc comparators on the same data; "
        "meaning and intent remain unresolved. {#tbl-original-coverage}",
    )
