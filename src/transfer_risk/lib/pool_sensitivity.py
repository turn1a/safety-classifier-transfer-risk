"""Pure saved-pool sensitivity for conditional verified flips and attempted-trial yield.

The yield retains failed and skipped source trials in its fixed attempted denominator.
It counts verified target changes only among successful source attacks that were audited.
Associations condition on the designed surrogate pool. Subset enumeration describes saved
outcomes at equal attempted budgets, without a selection test or an equal-compute claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from numbers import Integral, Real
from statistics import fmean, median
from typing import TYPE_CHECKING, Any

from transfer_risk.lib.association import spearman_association

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import pandas as pd

_COUNTS = (
    "source_successful",
    "target_original_benign",
    "target_original_injection",
    "target_perturbed_benign",
    "true_target_flips",
)
_MACRO = "macro_conditional_verified_flip_rate"
_YIELD = "source_success_gated_verified_flip_yield"
_OUTCOMES = (_MACRO, _YIELD)
_MIN_SURROGATES = 2
_TOLERANCE = 1e-12
_ENUMERATION_CAP = math.factorial(10)


def _require_columns(frame: pd.DataFrame, fields: Sequence[str], *, artifact: str) -> None:
    """Reject missing or ambiguous dataframe columns before reducing an artifact.

    Args:
        frame: Input aggregate dataframe.
        fields: Required column identifiers.
        artifact: Artifact label for errors.

    Raises:
        ValueError: If columns are duplicated or a required field is missing.
    """
    if not frame.columns.is_unique:
        msg = f"{artifact} has duplicate dataframe columns"
        raise ValueError(msg)
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        msg = f"{artifact} is missing required columns: {missing}"
        raise ValueError(msg)


def _identifier(value: Any, *, field: str) -> str:
    """Validate a nonempty scalar identifier without coercing missing values.

    Args:
        value: Candidate surrogate or recipe identifier.
        field: Field label for errors.

    Returns:
        The unchanged identifier.

    Raises:
        ValueError: If the value is not a nonempty string without surrounding spaces.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        msg = f"{field} must be a nonempty string identifier without surrounding spaces"
        raise ValueError(msg)
    return value


def _identifiers(values: Sequence[str], *, field: str, allow_empty: bool = False) -> list[str]:
    """Validate and canonically order a configured set of identifiers.

    Args:
        values: Configured recipe or surrogate identifiers.
        field: Configuration field label for errors.
        allow_empty: Whether an empty configured set is valid.

    Returns:
        Sorted unique identifiers.

    Raises:
        ValueError: If the input is a scalar string, contains duplicates, or is invalid.
    """
    if isinstance(values, str):
        msg = f"{field} must be a sequence of identifiers"
        raise ValueError(msg)
    names = [_identifier(value, field=field) for value in values]
    if len(set(names)) != len(names) or (not names and not allow_empty):
        msg = f"{field} must contain unique identifiers" + (
            "" if allow_empty else " and be nonempty"
        )
        raise ValueError(msg)
    return sorted(names)


def _finite_number(value: Any, *, field: str) -> float:
    """Convert numeric scalars to native finite floats without accepting booleans.

    Args:
        value: Candidate numeric scalar.
        field: Field label for errors.

    Returns:
        The value as a finite native Python float.

    Raises:
        ValueError: If a value is boolean, nonnumeric, or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        msg = f"{field} must be a finite numeric value"
        raise ValueError(msg)
    return float(value)


def _count(value: Any, *, field: str) -> int:
    """Validate a nonnegative integer count without rounding fractional values.

    Args:
        value: Candidate aggregate count or discrete parameter.
        field: Field label for errors.

    Returns:
        The count as a native Python integer.

    Raises:
        ValueError: If a count is negative, fractional, nonnumeric, or non-finite.
    """
    if isinstance(value, bool):
        msg = f"{field} count cannot be boolean"
        raise ValueError(msg)
    if isinstance(value, Integral):
        result = int(value)
    else:
        number = _finite_number(value, field=f"{field} count")
        if not number.is_integer():
            msg = f"{field} count must be an integer"
            raise ValueError(msg)
        result = int(number)
    if result < 0:
        msg = f"{field} count must be nonnegative"
        raise ValueError(msg)
    return result


def _positive_count(value: Any, *, field: str) -> int:
    """Validate a positive integer parameter.

    Args:
        value: Candidate discrete parameter.
        field: Parameter label for errors.

    Returns:
        A native positive integer.

    Raises:
        ValueError: If a value is not a positive integer.
    """
    result = _count(value, field=field)
    if result == 0:
        msg = f"{field} must be positive"
        raise ValueError(msg)
    return result


def _similarities(frame: pd.DataFrame) -> dict[str, float]:
    """Validate one bounded mean CKA value per unique surrogate.

    Args:
        frame: Public similarity rows containing surrogate and mean CKA.

    Returns:
        Canonically ordered surrogate-to-CKA mapping.

    Raises:
        ValueError: If names or scores are invalid, duplicated, or insufficient.
    """
    _require_columns(frame, ("surrogate", "mean_cka"), artifact="similarity rows")
    values = {}
    for row in frame.to_dict("records"):
        name = _identifier(row["surrogate"], field="similarity surrogate")
        if name in values:
            msg = f"similarity rows contain duplicate surrogate {name!r}"
            raise ValueError(msg)
        score = _finite_number(row["mean_cka"], field=f"mean_cka for {name!r}")
        if not 0.0 <= score <= 1.0:
            msg = f"mean_cka for {name!r} must be within [0, 1]"
            raise ValueError(msg)
        values[name] = score
    if len(values) < _MIN_SURROGATES:
        msg = f"saved-pool sensitivity needs at least {_MIN_SURROGATES} surrogates"
        raise ValueError(msg)
    return dict(sorted(values.items()))


def _validate_counts(counts: Mapping[str, int], *, attempted: int, cell: tuple[str, str]) -> None:
    """Reconcile source counts and target joint-transition marginals.

    Args:
        counts: Validated nonnegative audit counts.
        attempted: Fixed attempted trial budget for the cell.
        cell: Surrogate and recipe identifiers for errors.

    Raises:
        ValueError: If source successes exceed attempts or target marginals conflict.
    """
    successful = counts["source_successful"]
    if successful > attempted:
        msg = f"source-successful counts exceed the attempted budget for cell {cell!r}"
        raise ValueError(msg)
    original_benign = counts["target_original_benign"]
    original_injection = counts["target_original_injection"]
    perturbed_benign = counts["target_perturbed_benign"]
    flips = counts["true_target_flips"]
    joint = (
        original_injection - flips,
        flips,
        original_benign - perturbed_benign + flips,
        perturbed_benign - flips,
    )
    if original_benign + original_injection != successful or min(joint) < 0:
        msg = f"inconsistent target-audit counts for cell {cell!r}"
        raise ValueError(msg)


def _validate_rate(value: Any, expected: float | None, *, field: str) -> None:
    """Reject optional saved rates that contradict their aggregate counts.

    Args:
        value: Optional saved rate value.
        expected: Rate derived from validated counts, or undefined for a zero baseline.
        field: Rate and cell label for errors.

    Raises:
        ValueError: If the value is missing when defined or contradicts the counts.
    """
    missing = value is None or (isinstance(value, Real) and math.isnan(value))
    if expected is None and missing:
        return
    if missing or expected is None:
        msg = f"saved rate {field} is inconsistent with its counts"
        raise ValueError(msg)
    number = _finite_number(value, field=f"saved rate {field}")
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=_TOLERANCE):
        msg = f"saved rate {field} is inconsistent with its counts"
        raise ValueError(msg)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a count ratio or an undefined rate for a zero denominator.

    Args:
        numerator: Nonnegative numerator count.
        denominator: Nonnegative denominator count.

    Returns:
        A native rate, or ``None`` when no relevant baseline exists.
    """
    return numerator / denominator if denominator else None


def _normalized_cell(raw: Mapping[Hashable, Any], *, attempts_per_cell: int) -> dict[str, Any]:
    """Extract permitted scalar cell fields and validate target-transition accounting.

    Args:
        raw: One input aggregate cell, possibly with unrelated columns.
        attempts_per_cell: Fixed attempted trial budget including skips and failures.

    Returns:
        A normalized aggregate-only cell with reconstructed rates and trial yield.

    Raises:
        ValueError: If any key, count, optional saved rate, or attempted budget conflicts.
    """
    name = _identifier(raw["surrogate"], field="cell surrogate")
    recipe = _identifier(raw["recipe"], field="cell recipe")
    key = (name, recipe)
    counts = {field: _count(raw[field], field=f"{field} in cell {key!r}") for field in _COUNTS}
    _validate_counts(counts, attempted=attempts_per_cell, cell=key)
    rates = {
        "true_target_flip_rate": _safe_rate(
            counts["true_target_flips"], counts["target_original_injection"]
        ),
        "conditional_target_benign_rate": _safe_rate(
            counts["target_perturbed_benign"], counts["source_successful"]
        ),
    }
    for field, expected in rates.items():
        if field in raw:
            _validate_rate(raw[field], expected, field=f"{field} for cell {key!r}")
    if "attempted" in raw and _count(raw["attempted"], field="attempted") != attempts_per_cell:
        msg = f"saved attempted count conflicts with attempts_per_cell for cell {key!r}"
        raise ValueError(msg)
    return {
        "surrogate": name,
        "recipe": recipe,
        **counts,
        **rates,
        "attempted": attempts_per_cell,
        "not_source_successful": attempts_per_cell - counts["source_successful"],
        _YIELD: counts["true_target_flips"] / attempts_per_cell,
    }


def _cell_rows(
    frame: pd.DataFrame,
    similarity: Mapping[str, float],
    *,
    attempts_per_cell: int,
    recipes: Sequence[str],
) -> list[dict[str, Any]]:
    """Require the complete configured model-recipe grid without zero imputation.

    Args:
        frame: Saved public audit cells.
        similarity: Validated CKA values defining the full surrogate pool.
        attempts_per_cell: Fixed attempted trial budget including skips and failures.
        recipes: Complete configured recipe identifiers.

    Returns:
        Canonically ordered validated aggregate cells.

    Raises:
        ValueError: If cells are absent, duplicated, extra, or use inconsistent CKA.
    """
    _require_columns(frame, ("surrogate", "recipe", *_COUNTS), artifact="audit cells")
    cells = []
    keys = set()
    for raw in frame.to_dict("records"):
        row = _normalized_cell(raw, attempts_per_cell=attempts_per_cell)
        key = (row["surrogate"], row["recipe"])
        if key in keys:
            msg = f"audit grid contains duplicate cell {key!r}"
            raise ValueError(msg)
        keys.add(key)
        if "mean_cka" in raw and key[0] in similarity:
            score = _finite_number(raw["mean_cka"], field=f"mean_cka for cell {key!r}")
            if not math.isclose(score, similarity[key[0]], rel_tol=0.0, abs_tol=_TOLERANCE):
                msg = f"cell mean_cka is inconsistent with similarity rows for {key!r}"
                raise ValueError(msg)
        cells.append(row)
    if {key[0] for key in keys} != set(similarity):
        msg = "audit cells and similarity rows must use the same surrogate pool"
        raise ValueError(msg)
    expected = {(name, recipe) for name in similarity for recipe in recipes}
    if keys != expected:
        msg = (
            "audit grid must contain every configured surrogate-recipe cell; "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
        raise ValueError(msg)
    return sorted(cells, key=lambda row: (row["surrogate"], row["recipe"]))


def _pool_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Sum normalized counts without introducing a different denominator.

    Args:
        rows: Normalized cell or surrogate aggregate rows.

    Returns:
        Native integer counts for all target marginals and attempted trials.
    """
    return {
        field: sum(int(row[field]) for row in rows)
        for field in (*_COUNTS, "attempted", "not_source_successful")
    }


def _surrogate_rows(
    cells: Sequence[Mapping[str, Any]], similarity: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Keep recipe, conditional-pool, and attempted-trial weights separate.

    Args:
        cells: Complete normalized target-audit cells.
        similarity: One validated mean CKA value per surrogate.

    Returns:
        Canonically ordered model counts and three explicitly denominated outcomes.
    """
    rows = []
    for name, score in similarity.items():
        selected = [row for row in cells if row["surrogate"] == name]
        counts = _pool_counts(selected)
        rates = [
            float(row["true_target_flip_rate"])
            for row in selected
            if row["true_target_flip_rate"] is not None
        ]
        rows.append(
            {
                "surrogate": name,
                "mean_cka": score,
                **counts,
                "n_recipes": len(selected),
                "defined_conditional_recipe_count": len(rates),
                _MACRO: fmean(rates) if rates else None,
                "pooled_conditional_verified_flip_rate": _safe_rate(
                    counts["true_target_flips"], counts["target_original_injection"]
                ),
                _YIELD: counts["true_target_flips"] / counts["attempted"],
            }
        )
    return rows


def _association(
    rows: Sequence[Mapping[str, Any]], outcome: str, *, enumerate_null: bool
) -> dict[str, Any]:
    """Describe aligned complete cases and optionally calculate their null p-value.

    Args:
        rows: Designed-pool surrogate summaries.
        outcome: Outcome field to associate with mean CKA.
        enumerate_null: Whether to include the existing routine's p-value result.

    Returns:
        Native rho, complete-case metadata, and optional labelled null statistics.
    """
    included = [row for row in rows if row[outcome] is not None]
    first = [float(row["mean_cka"]) for row in included]
    second = [float(row[outcome]) for row in included]
    result: dict[str, Any] = {
        "n": len(included),
        "included_surrogates": [str(row["surrogate"]) for row in included],
        "undefined_surrogates": [str(row["surrogate"]) for row in rows if row[outcome] is None],
        "complete_case_pool": len(included) == len(rows),
    }
    reason = None
    if len(included) < _MIN_SURROGATES:
        reason = "fewer than two surrogates with defined outcomes"
    elif len(set(first)) < _MIN_SURROGATES or len(set(second)) < _MIN_SURROGATES:
        reason = "Spearman association is undefined for constant input sequences"
    if reason is not None:
        result.update(status="not_estimable", reason=reason, rho=None)
        if enumerate_null:
            result.update(two_sided_p=None, exact=False, p_value_method=None)
        return result
    statistics = spearman_association(
        first, second, enumeration_cap=_ENUMERATION_CAP if enumerate_null else 1
    )
    result.update(status="estimated", rho=float(statistics["rho"]))
    if enumerate_null:
        exact = bool(statistics["exact"])
        result.update(
            two_sided_p=float(statistics["two_sided_p"]),
            exact=exact,
            p_value_method="exact_enumeration" if exact else "normal_approximation",
        )
    return result


def _leave_one_out(
    rows: Sequence[Mapping[str, Any]], associations: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Recompute ranks after each model omission and report its rho change.

    Args:
        rows: Full or excluded designed-pool model summaries.
        associations: Correlations in the current complete designed pool.

    Returns:
        One descriptive omission result per pool member, without additional p-values.
    """
    result = []
    for omitted in rows:
        retained = [row for row in rows if row["surrogate"] != omitted["surrogate"]]
        outcomes = {}
        for outcome in _OUTCOMES:
            stats = _association(retained, outcome, enumerate_null=False)
            before = associations[outcome]["rho"]
            after = stats["rho"]
            stats["rho_change_from_pool"] = (
                float(after) - float(before) if after is not None and before is not None else None
            )
            outcomes[outcome] = stats
        result.append(
            {
                "omitted_surrogate": str(omitted["surrogate"]),
                "n_surrogates": len(retained),
                "counts": _pool_counts(retained),
                "associations": outcomes,
            }
        )
    return result


def _influence_summary(rows: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, Any]:
    """Summarize descriptive omission ranges and the largest absolute rho change.

    Args:
        rows: Leave-one-surrogate-out results for one designed pool.
        outcome: Outcome whose rho range and influence are summarized.

    Returns:
        The range, estimable omission count, and all tied most influential models.
    """
    rho_values = [
        float(row["associations"][outcome]["rho"])
        for row in rows
        if row["associations"][outcome]["rho"] is not None
    ]
    changes = [
        (
            str(row["omitted_surrogate"]),
            abs(float(row["associations"][outcome]["rho_change_from_pool"])),
        )
        for row in rows
        if row["associations"][outcome]["rho_change_from_pool"] is not None
    ]
    largest = max((value for _, value in changes), default=None)
    return {
        "n_estimable": len(rho_values),
        "rho_min": min(rho_values, default=None),
        "rho_max": max(rho_values, default=None),
        "max_absolute_rho_change": largest,
        "most_influential_surrogates": [
            name
            for name, value in changes
            if largest is not None and math.isclose(value, largest, abs_tol=_TOLERANCE)
        ],
    }


def _subset_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe a selection's counts and equal-model conditional macro mean.

    Args:
        rows: Model summaries belonging to one fixed-size selection.

    Returns:
        Native selection counts and outcomes. The conditional macro is undefined when
        any selected model has no defined conditional rate.
    """
    counts = _pool_counts(rows)
    rates = [float(row[_MACRO]) for row in rows if row[_MACRO] is not None]
    return {
        "surrogates": [str(row["surrogate"]) for row in rows],
        "n_surrogates": len(rows),
        **counts,
        "n_defined_conditional_surrogates": len(rates),
        _MACRO: fmean(rates) if len(rates) == len(rows) else None,
        "pooled_conditional_verified_flip_rate": _safe_rate(
            counts["true_target_flips"], counts["target_original_injection"]
        ),
        _YIELD: counts["true_target_flips"] / counts["attempted"],
    }


def _selection_comparison(
    subsets: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], outcome: str
) -> dict[str, Any]:
    """Locate the CKA selection within the descriptive saved-subset distribution.

    Args:
        subsets: Every fixed-size subset under the same attempted trial budget.
        selected: The subset chosen by descending mean CKA and the disclosed tie rule.
        outcome: Conditional macro or attempted-trial yield objective.

    Returns:
        Distribution summaries, counts, ranks, and descriptive fractions. No p-value
        or claim of external selection validation is produced.
    """
    defined = [row for row in subsets if row[outcome] is not None]
    values = [float(row[outcome]) for row in defined]
    chosen = float(selected[outcome]) if selected[outcome] is not None else None
    average = fmean(values) if values else None
    maximum = max(values, default=None)
    higher = equal = None
    if chosen is not None:
        equal = sum(
            math.isclose(value, chosen, rel_tol=0.0, abs_tol=_TOLERANCE) for value in values
        )
        higher = sum(
            value > chosen and not math.isclose(value, chosen, rel_tol=0.0, abs_tol=_TOLERANCE)
            for value in values
        )
    return {
        "status": "descriptive" if chosen is not None else "selected_outcome_undefined",
        "n_subsets_with_defined_outcome": len(defined),
        "selected_value": chosen,
        "mean": average,
        "median": float(median(values)) if values else None,
        "min": min(values, default=None),
        "max": maximum,
        "n_strictly_higher": higher,
        "n_equal": equal,
        "n_strictly_lower": len(values) - higher - equal
        if higher is not None and equal is not None
        else None,
        "selected_descending_rank_min": higher + 1 if higher is not None else None,
        "selected_descending_rank_max": higher + equal
        if higher is not None and equal is not None
        else None,
        "fraction_at_or_above": (higher + equal) / len(values)
        if higher is not None and equal is not None
        else None,
        "selected_minus_mean": chosen - average
        if chosen is not None and average is not None
        else None,
        "selected_minus_best": chosen - maximum
        if chosen is not None and maximum is not None
        else None,
        "best_selections": [
            list(row["surrogates"])
            for row in defined
            if maximum is not None
            and math.isclose(float(row[outcome]), maximum, rel_tol=0.0, abs_tol=_TOLERANCE)
        ],
    }


def _selection(rows: Sequence[Mapping[str, Any]], *, size: int) -> dict[str, Any]:
    """Enumerate every designed-pool subset at the supplied attempted budget.

    Args:
        rows: Canonically ordered surrogate summaries.
        size: Number of models selected under each compared trial budget.

    Returns:
        All subset counts, outcomes, the CKA choice, and descriptive comparisons.
    """
    ordered = sorted(rows, key=lambda row: (-float(row["mean_cka"]), str(row["surrogate"])))
    selected_names = sorted(str(row["surrogate"]) for row in ordered[:size])
    subsets = [_subset_row(subset) for subset in combinations(rows, size)]
    selected = next(row for row in subsets if row["surrogates"] == selected_names)
    cutoff = float(ordered[size - 1]["mean_cka"])
    tied = [str(row["surrogate"]) for row in rows if float(row["mean_cka"]) == cutoff]
    needed_at_cutoff = size - sum(float(row["mean_cka"]) > cutoff for row in rows)
    return {
        "selection_size": size,
        "n_subsets": len(subsets),
        "attempts_per_subset": int(selected["attempted"]),
        "selected_surrogates": selected_names,
        "selected_subset": selected,
        "cka_cutoff": cutoff,
        "cka_cutoff_tied_surrogates": tied if len(tied) > 1 else [],
        "cka_selection_is_unique": len(tied) == needed_at_cutoff,
        "tie_break": "ascending surrogate identifier",
        "subsets": subsets,
        "comparisons": {
            outcome: _selection_comparison(subsets, selected, outcome) for outcome in _OUTCOMES
        },
    }


def _pool(
    rows: Sequence[Mapping[str, Any]], *, excluded: Sequence[str], selection_size: int
) -> dict[str, Any]:
    """Assemble one designed-pool comparison from normalized model aggregates.

    Args:
        rows: Complete or excluded designed-pool summaries.
        excluded: Explicit model identifiers absent from this pool.
        selection_size: Fixed subset size for descriptive selection comparisons.

    Returns:
        Counts, full correlations, omission sensitivity, and exhaustive subsets.
    """
    counts = _pool_counts(rows)
    associations = {
        outcome: _association(rows, outcome, enumerate_null=True) for outcome in _OUTCOMES
    }
    omissions = _leave_one_out(rows, associations)
    return {
        "surrogates": [str(row["surrogate"]) for row in rows],
        "n_surrogates": len(rows),
        "excluded_surrogates": list(excluded),
        "counts": counts,
        "pooled_conditional_verified_flip_rate": _safe_rate(
            counts["true_target_flips"], counts["target_original_injection"]
        ),
        _YIELD: counts["true_target_flips"] / counts["attempted"],
        "associations": associations,
        "leave_one_out": omissions,
        "leave_one_out_summary": {
            outcome: _influence_summary(omissions, outcome) for outcome in _OUTCOMES
        },
        "selection": _selection(rows, size=selection_size),
    }


def _input_hash(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Hash canonical permitted aggregate values without forwarding arbitrary fields.

    Args:
        rows: Canonically ordered normalized input rows.
        fields: Explicit aggregate fields included in the provenance digest.

    Returns:
        SHA256 of compact sorted-key JSON containing native input values.
    """
    payload = [{field: row[field] for field in fields} for row in rows]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_pool_sensitivity(
    cells: pd.DataFrame,
    similarity_rows: pd.DataFrame,
    *,
    attempts_per_cell: int,
    recipes: Sequence[str],
    excluded_surrogates: Sequence[str] = ("bilstm-attention",),
    selection_size: int = 3,
) -> dict[str, Any]:
    """Build saved-pool conditional, trial-yield, and selection sensitivities.

    Args:
        cells: Exactly one cell per surrogate and configured recipe, containing
            ``source_successful``, both original target label counts,
            ``target_perturbed_benign``, and ``true_target_flips``. Optional saved rates,
            mean CKA, and attempted counts must agree with the validated inputs.
        similarity_rows: Exactly one row per pool member containing ``surrogate`` and
            finite ``mean_cka``. Unrelated columns are ignored and never published.
        attempts_per_cell: Historical attempted trials per model-recipe cell, including
            source baseline skips and source failures. This budget is supplied from saved
            run metadata, because successful-source audit cells cannot reconstruct it.
        recipes: The complete configured recipe list, used to detect missing whole recipes.
        excluded_surrogates: Identifiers for the secondary pool; empty omits that comparison.
        selection_size: Number of models in every exhaustively enumerated selection.

    Returns:
        Native JSON-compatible provenance, normalized cell and surrogate rows, outcome
        definitions, interpretation limits, and ``pools`` keyed by ``all_surrogates``
        and optionally ``surrogate_excluded``. Each pool contains explicit counts,
        complete-case CKA associations, descriptive leave-one-out rho changes, and all
        equal-attempted-budget subsets. Association p-values use the existing exact
        tie-free enumeration when feasible and label its approximation otherwise.

    Raises:
        ValueError: If inputs contain missing or malformed cells, inconsistent counts,
            invalid identifiers or CKA values, or an invalid budget, exclusion, or size.
    """
    budget = _positive_count(attempts_per_cell, field="attempts_per_cell")
    size = _positive_count(selection_size, field="selection_size")
    recipe_names = _identifiers(recipes, field="recipes")
    excluded = _identifiers(excluded_surrogates, field="excluded_surrogates", allow_empty=True)
    similarity = _similarities(similarity_rows)
    unknown = sorted(set(excluded).difference(similarity))
    if unknown:
        msg = f"unknown excluded surrogates: {unknown}"
        raise ValueError(msg)
    retained_count = len(similarity) - len(excluded)
    if retained_count < _MIN_SURROGATES:
        msg = f"excluded-pool sensitivity needs at least {_MIN_SURROGATES} retained surrogates"
        raise ValueError(msg)
    if size > retained_count:
        msg = "selection_size cannot exceed the number of retained surrogates"
        raise ValueError(msg)
    normalized = _cell_rows(cells, similarity, attempts_per_cell=budget, recipes=recipe_names)
    rows = _surrogate_rows(normalized, similarity)
    pools = {"all_surrogates": _pool(rows, excluded=[], selection_size=size)}
    if excluded:
        retained = [row for row in rows if row["surrogate"] not in excluded]
        pools["surrogate_excluded"] = _pool(retained, excluded=excluded, selection_size=size)
    return {
        "schema_version": 1,
        "analysis_label": "post-hoc sensitivity of the saved designed surrogate pool",
        "provenance": {
            "audit_counts_sha256": _input_hash(normalized, ("surrogate", "recipe", *_COUNTS)),
            "similarity_values_sha256": _input_hash(rows, ("surrogate", "mean_cka")),
            "n_surrogates": len(rows),
            "n_recipes": len(recipe_names),
            "n_cells": len(normalized),
            "recipes": recipe_names,
            "attempts_per_cell": budget,
            "attempts_per_surrogate": budget * len(recipe_names),
            "attempted": budget * len(normalized),
            "budget_basis": (
                "supplied fixed attempted trial count, including source skips and failures"
            ),
        },
        "outcome_definitions": {
            _MACRO: (
                "Per model, the equal-recipe mean of true_target_flips divided by "
                "source-successful target_original_injection counts. Zero-baseline recipes "
                "are undefined and omitted; a model with no defined recipes is undefined."
            ),
            _YIELD: (
                "Verified target flips after source success divided by all attempted "
                "model-recipe trials, including source failures and skipped baselines. "
                "Failed-source perturbations were not target-audited."
            ),
        },
        "statistical_interpretation": (
            "Model rows are units of a designed pool. Association p-values condition on "
            "an exchangeability null for complete-case model rows. These post-hoc, multiple "
            "analyses are exploratory and do not validate a population predictor or "
            "selection rule. Leave-one-out ranges and subset fractions are descriptive."
        ),
        "limitations": [
            (
                "Repeated originals and perturbations across models and recipes "
                "are overlapping records."
            ),
            "Successful-source and target-positive cohorts differ between models and recipes.",
            (
                "Equal attempted trial counts do not imply equal queries, runtime, or compute; "
                "actual query use is unrecorded."
            ),
            (
                "Subset outcomes reuse saved attacks and audited predictions; "
                "they are not held-out selection outcomes."
            ),
            (
                "Conditional subset means require every selected model "
                "to have a defined conditional macro rate."
            ),
        ],
        "cell_rows": normalized,
        "surrogate_rows": rows,
        "pools": pools,
    }
