"""Aggregate-only tables and sensitivity calculations for the manuscript.

Display values are rounded here. The source CSV and JSON artifacts retain their full
precision, and all sensitivity outcomes are derived from saved attack aggregates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.statistic_display import association_cells, format_number, format_percent

_MIN_SURROGATES = 2

_COUNTS = (
    "source_successful",
    "target_original_benign",
    "target_original_injection",
    "target_perturbed_benign",
    "true_target_flips",
)


def audit_transition_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    """Recover the target's joint prediction counts from saved audit marginals.

    Args:
        summary: Finalized aggregate target audit.

    Returns:
        Four transition counts over successful source-attack records.

    Raises:
        ValueError: If a marginal is negative or the joint counts are inconsistent.
    """
    cohort = summary["full_cohort"]
    counts = {name: int(cohort[name]) for name in _COUNTS}
    flips = counts["true_target_flips"]
    result = {
        "injection_to_injection": counts["target_original_injection"] - flips,
        "injection_to_benign": flips,
        "benign_to_injection": (
            counts["target_original_benign"] - counts["target_perturbed_benign"] + flips
        ),
        "benign_to_benign": counts["target_perturbed_benign"] - flips,
    }
    if (
        min(counts.values()) < 0
        or min(result.values()) < 0
        or sum(result.values()) != counts["source_successful"]
    ):
        msg = "inconsistent target-audit transition counts"
        raise ValueError(msg)
    return result


def exclude_surrogates(
    summary: Mapping[str, Any],
    excluded: Sequence[str],
) -> dict[str, Any]:
    """Recalculate a post-hoc pool sensitivity from saved surrogate aggregates.

    Args:
        summary: Finalized aggregate target audit.
        excluded: Exact surrogate identifiers to omit.

    Returns:
        Aggregate counts, pooled verified rate, and surrogate-level CKA association.

    Raises:
        ValueError: If an identifier is unknown or fewer than two surrogates remain.
    """
    rows = summary["full_cohort"]["surrogate_macro"]["rows"]
    names = {row["surrogate"] for row in rows}
    unknown = set(excluded) - names
    if unknown:
        msg = f"unknown excluded surrogates: {sorted(unknown)}"
        raise ValueError(msg)
    retained = [row for row in rows if row["surrogate"] not in excluded]
    if len(retained) < _MIN_SURROGATES:
        msg = "surrogate sensitivity needs at least two retained models"
        raise ValueError(msg)
    counts = {name: sum(int(row[name]) for row in retained) for name in _COUNTS}
    denominator = counts["target_original_injection"]
    complete = [row for row in retained if row["true_target_flip_rate_macro_mean"] is not None]
    reason = None
    if len(complete) < _MIN_SURROGATES:
        reason = "fewer than two surrogates with defined outcomes"
    elif (
        len({row["mean_cka"] for row in complete}) < _MIN_SURROGATES
        or len({row["true_target_flip_rate_macro_mean"] for row in complete}) < _MIN_SURROGATES
    ):
        reason = "Spearman association is undefined for constant input sequences"
    association = (
        {"rho": None, "two_sided_p": None, "n": len(complete), "exact": False, "reason": reason}
        if reason is not None
        else spearman_association(
            [row["mean_cka"] for row in complete],
            [row["true_target_flip_rate_macro_mean"] for row in complete],
        )
    )
    return {
        "analysis_label": "post-hoc surrogate-excluded sensitivity on saved outcomes",
        "excluded_surrogates": sorted(set(excluded)),
        "n_surrogates": len(retained),
        **counts,
        "true_target_flip_rate": counts["true_target_flips"] / denominator if denominator else None,
        "mean_cka_association": association,
        "interpretation": (
            "Saved predictions and attacks are unchanged. Removing a model does not correct "
            "its historical implementation or provide independent validation. The enumerated "
            "p-value conditions on an exchangeability null for the retained designed pool."
        ),
    }


def _markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, left_columns: int = 1
) -> str:
    """Render a Markdown table with one physical line per row."""
    alignment = [":---" if position < left_columns else "---:" for position in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(alignment) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_transition_table(summary: Mapping[str, Any]) -> str:
    """Render the aggregate target transition table with explicit record totals."""
    transitions = audit_transition_counts(summary)
    cohort = summary["full_cohort"]
    rows = [
        [
            "Injection",
            f"{transitions['injection_to_benign']:,}",
            f"{transitions['injection_to_injection']:,}",
            f"{cohort['target_original_injection']:,}",
        ],
        [
            "Benign",
            f"{transitions['benign_to_benign']:,}",
            f"{transitions['benign_to_injection']:,}",
            f"{cohort['target_original_benign']:,}",
        ],
        [
            "Total",
            f"{cohort['target_perturbed_benign']:,}",
            f"{cohort['source_successful'] - cohort['target_perturbed_benign']:,}",
            f"{cohort['source_successful']:,}",
        ],
    ]
    return (
        _markdown_table(
            ["Original prediction", "Perturbed benign", "Perturbed injection", "Total"], rows
        )
        + "\n: Target transitions over successful source-attack records. {#tbl-transitions}\n"
    )


def render_surrogate_table(summary: Mapping[str, Any], attempts_per_surrogate: int) -> str:
    """Render similarity, counts, and both macro and pooled verified rates."""
    membership = summary["membership"]
    rows = sorted(
        summary["full_cohort"]["surrogate_macro"]["rows"],
        key=lambda row: row["mean_cka"],
        reverse=True,
    )
    formatted = []
    for row in rows:
        name = row["surrogate"]
        group = "M1" if name in membership["M1"] else "M2" if name in membership["M2"] else "Middle"
        denominator = row["target_original_injection"]
        pooled = 100 * row["true_target_flips"] / denominator if denominator else None
        formatted.append(
            [
                f"[`{name}`](guide.qmd#model-{name})",
                group,
                f"{row['mean_cka']:.3f}",
                f"{row['dbs']:.3f}",
                f"{row['source_successful']:,}",
                f"{row['true_target_flips']:,} / {denominator:,}",
                format_percent(
                    row["true_target_flip_rate_macro_mean"], reason="no defined recipe rates"
                ),
                format_number(
                    pooled,
                    ".1f",
                    reason="no original target injections",
                ),
                format_percent(
                    row["true_target_flips"] / attempts_per_surrogate
                    if attempts_per_surrogate
                    else None,
                    reason="no attempted trials",
                ),
            ]
        )
    return (
        _markdown_table(
            [
                "Surrogate",
                "[Group](guide.qmd#groups){.term-help}",
                "[CKA](guide.qmd#cka){.term-help}",
                "[DBS](guide.qmd#dbs){.term-help}",
                "[Source successes](guide.qmd#source-success){.term-help}",
                "[Flips / target injections](guide.qmd#target-baseline){.term-help}",
                "[Macro (%)](guide.qmd#macro){.term-help}",
                "[Pooled (%)](guide.qmd#pooled){.term-help}",
                "[Yield (%)](guide.qmd#yield){.term-help}",
            ],
            formatted,
            left_columns=2,
        )
        + "\n: Surrogate-level results, ordered by original mean CKA. "
        '{#tbl-surrogate-results tbl-colwidths="[24,7,7,7,12,16,9,9,9]"}\n'
    )


def render_recipe_table(cells: pd.DataFrame) -> str:
    """Render recipe totals over successful source-attack records."""
    rows = []
    for recipe, frame in cells.groupby("recipe", sort=True):
        successes = int(frame["source_successful"].sum())
        denominator = int(frame["target_original_injection"].sum())
        flips = int(frame["true_target_flips"].sum())
        rows.append(
            [
                str(recipe),
                f"{successes:,}",
                f"{flips:,} / {denominator:,}",
                format_percent(
                    flips / denominator if denominator else None,
                    reason="no original target injections",
                ),
            ]
        )
    return (
        _markdown_table(
            [
                "Recipe",
                "[Source successes](guide.qmd#source-success){.term-help}",
                "[Flips / target injections](guide.qmd#target-baseline){.term-help}",
                "[Pooled verified rate (%)](guide.qmd#pooled){.term-help}",
            ],
            rows,
        )
        + "\n: Audited outcome totals by attack recipe. "
        '{#tbl-recipe-results tbl-colwidths="[20,20,30,30]"}\n'
    )


def render_sensitivity_table(
    summary: Mapping[str, Any],
    training: Mapping[str, Any],
    reduced: Mapping[str, Any],
) -> str:
    """Compare the original result with post-hoc source, probe, and pool sensitivities."""
    rows = []
    for label, cohort in (
        ("Original probe, full cohort", summary["full_cohort"]),
        ("Known source excluded", summary["known_source_excluded_sensitivity"]),
    ):
        association = cohort["cka_dbs_association"]["true_target_flip_rate"]["mean_cka"][
            "exchangeability_null"
        ]
        rows.append(
            [
                label,
                "10",
                f"{cohort['true_target_flips']:,} / {cohort['target_original_injection']:,}",
                format_percent(
                    cohort["true_target_flip_rate"], reason="no original target injections"
                ),
                *association_cells(association),
            ]
        )
    association = training["true_target_flip_sensitivity"][
        "training_probe_mean_cka_vs_full_cohort_macro_true_target_flip_rate"
    ]["exchangeability_null"]
    full = summary["full_cohort"]
    rows.append(
        [
            "Training-only probe, full cohort",
            "10",
            f"{full['true_target_flips']:,} / {full['target_original_injection']:,}",
            format_percent(full["true_target_flip_rate"], reason="no original target injections"),
            *association_cells(association),
        ]
    )
    association = reduced["mean_cka_association"]
    excluded = ", ".join(
        "BiLSTM" if name == "bilstm-attention" else f"`{name}`"
        for name in reduced["excluded_surrogates"]
    )
    rows.append(
        [
            f"{excluded} excluded, saved outcomes" if excluded else "Full pool, saved outcomes",
            str(reduced["n_surrogates"]),
            f"{reduced['true_target_flips']:,} / {reduced['target_original_injection']:,}",
            format_percent(
                reduced["true_target_flip_rate"], reason="no original target injections"
            ),
            *association_cells(association),
        ]
    )
    return (
        _markdown_table(
            [
                "Analysis",
                "Models",
                "[Flips / target injections](guide.qmd#target-baseline){.term-help}",
                "[Pooled (%)](guide.qmd#pooled){.term-help}",
                "[CKA versus macro flip rho](guide.qmd#rho){.term-help}",
                "[Exact p](guide.qmd#exact-p){.term-help}",
            ],
            rows,
        )
        + "\n: Original association and post-hoc sensitivities on saved outcomes. "
        '{#tbl-sensitivities tbl-colwidths="[30,7,23,11,17,12]"}\n'
    )
