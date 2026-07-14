"""Pure helpers for publishing the finalized target-outcome audit.

These helpers select aggregate-only public tables and prepare deterministic plot data from
saved audit aggregates. They perform no I/O and never load models, prompts, or attack records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import pandas as pd

PUBLIC_AUDIT_CELL_COLUMNS = (
    "surrogate",
    "recipe",
    "source_successful",
    "target_original_benign",
    "target_original_injection",
    "target_perturbed_benign",
    "true_target_flips",
    "conditional_target_benign_rate",
    "true_target_flip_rate",
    "mean_cka",
    "dbs",
)
PUBLIC_AUDIT_SOURCE_COLUMNS = (
    "surrogate",
    "recipe",
    "source",
    "source_successful",
    "target_original_benign",
    "target_original_injection",
    "target_perturbed_benign",
    "true_target_flips",
    "conditional_target_benign_rate",
    "true_target_flip_rate",
)
_EXPECTED_CELL_COUNT = 50
_EXPECTED_SURROGATE_COUNT = 10
_EXPECTED_ABLATION_GROUP_SIZE = 3
_RAW_TEXT_FIELD_NAMES = frozenset({"original", "perturbed", "text", "prompt"})


def export_public_target_audit_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed, aggregate-only 50-cell public audit table.

    Args:
        cells: Finalized corrected audit cells from the stable catalog dataset.

    Returns:
        The ordered 50-cell public table without raw prompt or perturbation columns.

    Raises:
        ValueError: If the finalized audit does not have the expected schema or cell count.
    """
    _require_columns(cells, PUBLIC_AUDIT_CELL_COLUMNS, artifact="target audit cells")
    if len(cells) != _EXPECTED_CELL_COUNT:
        msg = (
            f"target audit cells must contain exactly {_EXPECTED_CELL_COUNT} rows, got {len(cells)}"
        )
        raise ValueError(msg)
    return _public_frame(
        cells,
        PUBLIC_AUDIT_CELL_COLUMNS,
        sort_keys=("surrogate", "recipe"),
    )


def export_public_target_audit_sources(sources: pd.DataFrame) -> pd.DataFrame:
    """Select aggregate-only target-audit source rows for public release.

    Args:
        sources: Stable target-audit source aggregates from the catalog.

    Returns:
        Source aggregate rows without raw prompt or perturbation columns.

    Raises:
        ValueError: If the source aggregate schema is incomplete.
    """
    _require_columns(sources, PUBLIC_AUDIT_SOURCE_COLUMNS, artifact="target audit sources")
    return _public_frame(
        sources,
        PUBLIC_AUDIT_SOURCE_COLUMNS,
        sort_keys=("surrogate", "recipe", "source"),
    )


def publish_safe_audit_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict JSON-safe copy of the finalized non-text audit summary.

    Args:
        summary: Finalized corrected audit summary from the stable catalog dataset.

    Returns:
        A deep JSON-compatible copy that preserves all cohort, macro-analysis, and
        exchangeability-label content.

    Raises:
        ValueError: If the summary contains a non-finite scalar or a raw-text field.
        TypeError: If the summary contains a value that JSON cannot represent safely.
    """
    _require_summary_sections(summary)
    published = _json_safe_value(summary)
    if not isinstance(published, dict):
        msg = "target audit summary must serialize to a mapping"
        raise TypeError(msg)
    try:
        json.dumps(published, allow_nan=False)
    except (TypeError, ValueError) as error:
        msg = "target audit summary is not strict JSON serializable"
        raise ValueError(msg) from error
    return published


def prepare_true_flip_scatter_data(
    cells: pd.DataFrame,
    summary: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Prepare validated macro, recipe, and association data for the true-flip scatter.

    Args:
        cells: Finalized corrected target-audit cells.
        summary: Finalized corrected target-audit summary.

    Returns:
        Tuple of surrogate macro rows, recipe-level rows, and exact CKA association values.

    Raises:
        ValueError: If saved audit cells or summary statistics are incomplete or non-finite.
    """
    recipe_rows = export_public_target_audit_cells(cells)
    recipe_rows = _finite_numeric_frame(
        recipe_rows,
        ("mean_cka", "true_target_flip_rate"),
        artifact="target audit recipe rows",
    )
    macro_rows = _true_flip_macro_rows(summary)
    association = _true_flip_association(summary)
    return macro_rows, recipe_rows, association


def prepare_true_flip_ablation_data(
    summary: Mapping[str, Any],
    selection: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Prepare validated M1/M2 macro true-flip rows and exact ablation statistics.

    Args:
        summary: Finalized corrected target-audit summary.
        selection: Saved original-CKA M1/M2 surrogate membership.

    Returns:
        Tuple of six group-labelled surrogate rows and exact one-sided ablation values.

    Raises:
        ValueError: If membership changed, required macro rows are absent, or statistics
            cannot support the required exact-enumeration annotation.
    """
    macro_rows = _true_flip_macro_rows(summary)
    m1 = _selection_members(selection, "M1")
    m2 = _selection_members(selection, "M2")
    _validate_summary_membership(summary, m1, m2)
    if len(m1) != _EXPECTED_ABLATION_GROUP_SIZE or len(m2) != _EXPECTED_ABLATION_GROUP_SIZE:
        msg = "target-audit true-flip ablation requires exactly three M1 and three M2 surrogates"
        raise ValueError(msg)

    rows_by_surrogate = macro_rows.set_index("surrogate", drop=False)
    selected_rows: list[dict[str, Any]] = []
    for group, names in (("M1", m1), ("M2", m2)):
        for surrogate in names:
            if surrogate not in rows_by_surrogate.index:
                msg = f"target audit summary has no macro row for selected surrogate {surrogate!r}"
                raise ValueError(msg)
            row = rows_by_surrogate.loc[surrogate]
            if not isinstance(row, pd.Series):
                msg = f"target audit summary has duplicate macro rows for {surrogate!r}"
                raise ValueError(msg)
            selected_rows.append(
                {
                    "surrogate": surrogate,
                    "similarity_group": group,
                    "mean_cka": _finite_float(
                        row["mean_cka"],
                        field=f"{surrogate} mean CKA",
                    ),
                    "true_target_flip_rate": _finite_float(
                        row["true_target_flip_rate"],
                        field=f"{surrogate} macro true-target-flip rate",
                    ),
                }
            )

    stats = _true_flip_ablation_stats(summary)
    if int(stats["n_m1"]) != len(m1) or int(stats["n_m2"]) != len(m2):
        msg = "target audit ablation group sizes do not match saved M1/M2 membership"
        raise ValueError(msg)
    return pd.DataFrame.from_records(selected_rows), stats


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact: str) -> None:
    """Raise when a saved aggregate table lacks a required public column."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        msg = f"{artifact} are missing required columns: {missing}"
        raise ValueError(msg)


def _public_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_keys: Sequence[str],
) -> pd.DataFrame:
    """Select fixed columns and return a deterministic row order."""
    return (
        frame.loc[:, list(columns)]
        .copy()
        .sort_values(list(sort_keys), kind="stable")
        .reset_index(drop=True)
    )


def _require_summary_sections(summary: Mapping[str, Any]) -> None:
    """Require the final audit summary's full, sensitivity, and membership sections."""
    required = {
        "full_cohort",
        "known_source_excluded_sensitivity",
        "membership",
    }
    missing = sorted(required.difference(summary))
    if missing:
        msg = f"target audit summary is missing required sections: {missing}"
        raise ValueError(msg)


def _is_raw_text_field_name(key: str) -> bool:
    """Return whether a summary field name could carry raw prompt text."""
    normalized = key.strip().lower()
    return (
        normalized in _RAW_TEXT_FIELD_NAMES
        or normalized.endswith("_text")
        or normalized.endswith("_prompt")
    )


def _json_safe_value(value: Any) -> Any:
    """Recursively normalize JSON values while rejecting raw-text fields and non-finite scalars."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, nested_value in value.items():
            key = str(raw_key)
            if _is_raw_text_field_name(key):
                msg = f"target audit public summary cannot include raw-text field {key!r}"
                raise ValueError(msg)
            output[key] = _json_safe_value(nested_value)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            msg = "target audit summary contains a non-finite numeric value"
            raise ValueError(msg)
        return number
    msg = f"target audit summary contains unsupported JSON value {type(value).__name__!r}"
    raise TypeError(msg)


def _mapping_at(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    """Return a nested summary mapping or raise a path-specific validation error."""
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            joined_path = ".".join(path)
            msg = f"target audit summary is missing mapping at {joined_path!r}"
            raise ValueError(msg)
        current = current[key]
    if not isinstance(current, Mapping):
        joined_path = ".".join(path)
        msg = f"target audit summary field {joined_path!r} must be a mapping"
        raise ValueError(msg)
    return current


def _sequence_at(root: Mapping[str, Any], path: Sequence[str]) -> list[Any]:
    """Return a nested non-string sequence as a list or raise a validation error."""
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            joined_path = ".".join(path)
            msg = f"target audit summary is missing sequence at {joined_path!r}"
            raise ValueError(msg)
        current = current[key]
    if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
        joined_path = ".".join(path)
        msg = f"target audit summary field {joined_path!r} must be a sequence"
        raise ValueError(msg)
    return list(current)


def _finite_float(value: Any, *, field: str) -> float:
    """Return a finite numeric scalar or raise a field-specific validation error."""
    if isinstance(value, bool):
        msg = f"target audit field {field!r} must be a finite number"
        raise ValueError(msg)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        msg = f"target audit field {field!r} must be a finite number"
        raise ValueError(msg) from error
    if not math.isfinite(number):
        msg = f"target audit field {field!r} must be a finite number"
        raise ValueError(msg)
    return number


def _finite_int(value: Any, *, field: str) -> int:
    """Return a finite integral scalar or raise a field-specific validation error."""
    number = _finite_float(value, field=field)
    if not number.is_integer():
        msg = f"target audit field {field!r} must be an integer"
        raise ValueError(msg)
    return int(number)


def _finite_numeric_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    artifact: str,
) -> pd.DataFrame:
    """Return a copy with named columns coerced to finite floats."""
    prepared = frame.copy()
    for column in columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        if values.isna().any() or not all(math.isfinite(float(value)) for value in values):
            msg = f"{artifact} have non-finite {column!r} values"
            raise ValueError(msg)
        prepared[column] = values.astype(float)
    return prepared


def _true_flip_macro_rows(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Extract the ten full-cohort surrogate macro true-target-flip rows."""
    rows = _sequence_at(summary, ("full_cohort", "surrogate_macro", "rows"))
    macro_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            msg = "target audit surrogate macro rows must be mappings"
            raise ValueError(msg)
        macro_rows.append(dict(item))
    frame = pd.DataFrame.from_records(macro_rows)
    required = ("surrogate", "mean_cka", "true_target_flip_rate_macro_mean")
    _require_columns(frame, required, artifact="target audit surrogate macro rows")
    if len(frame) != _EXPECTED_SURROGATE_COUNT or frame["surrogate"].duplicated().any():
        msg = "target audit summary must contain exactly ten unique surrogate macro rows"
        raise ValueError(msg)
    prepared = _finite_numeric_frame(
        frame.loc[:, list(required)].rename(
            columns={"true_target_flip_rate_macro_mean": "true_target_flip_rate"}
        ),
        ("mean_cka", "true_target_flip_rate"),
        artifact="target audit surrogate macro rows",
    )
    return prepared.sort_values("mean_cka", kind="stable").reset_index(drop=True)


def _true_flip_association(summary: Mapping[str, Any]) -> dict[str, float | int]:
    """Extract exact full-cohort mean-CKA association values for true-target-flip rate."""
    exchangeability = _mapping_at(
        summary,
        (
            "full_cohort",
            "cka_dbs_association",
            "true_target_flip_rate",
            "mean_cka",
            "exchangeability_null",
        ),
    )
    if exchangeability.get("exact") is not True:
        msg = "target audit true-flip CKA association must use exact enumeration"
        raise ValueError(msg)
    return {
        "rho": _finite_float(exchangeability.get("rho"), field="true-flip CKA rho"),
        "two_sided_p": _finite_float(
            exchangeability.get("two_sided_p"),
            field="true-flip CKA two-sided p",
        ),
        "n": _finite_int(exchangeability.get("n"), field="true-flip CKA n"),
    }


def _selection_members(selection: Mapping[str, Sequence[str]], group: str) -> list[str]:
    """Return non-empty string surrogate names for one saved similarity group."""
    if group not in selection:
        msg = f"saved surrogate selection is missing {group!r}"
        raise ValueError(msg)
    members = [str(name) for name in selection[group]]
    if not members or len(members) != len(set(members)):
        msg = f"saved surrogate selection {group!r} must contain unique names"
        raise ValueError(msg)
    return members


def _validate_summary_membership(
    summary: Mapping[str, Any],
    m1: Sequence[str],
    m2: Sequence[str],
) -> None:
    """Reject audit summaries whose saved membership diverges from the catalog selection."""
    membership = _mapping_at(summary, ("membership",))
    summary_m1 = _sequence_at(membership, ("M1",))
    summary_m2 = _sequence_at(membership, ("M2",))
    if [str(name) for name in summary_m1] != list(m1) or [str(name) for name in summary_m2] != list(
        m2
    ):
        msg = "target audit summary membership differs from saved original CKA selection"
        raise ValueError(msg)


def _true_flip_ablation_stats(summary: Mapping[str, Any]) -> dict[str, float | int]:
    """Extract exact full-cohort M1/M2 true-target-flip ablation statistics."""
    exchangeability = _mapping_at(
        summary,
        (
            "full_cohort",
            "selection_ablation",
            "true_target_flip_rate",
            "one_sided_exchangeability_null",
        ),
    )
    if exchangeability.get("exact") is not True:
        msg = "target audit true-flip ablation must use exact enumeration"
        raise ValueError(msg)
    return {
        "m1_mean": _finite_float(exchangeability.get("m1_mean"), field="true-flip M1 mean"),
        "m2_mean": _finite_float(exchangeability.get("m2_mean"), field="true-flip M2 mean"),
        "mean_diff_pp": _finite_float(
            exchangeability.get("mean_diff_pp"),
            field="true-flip M1/M2 mean difference",
        ),
        "mean_p_value": _finite_float(
            exchangeability.get("mean_p_value"),
            field="true-flip M1/M2 one-sided p",
        ),
        "n_m1": _finite_int(exchangeability.get("n_m1"), field="true-flip M1 count"),
        "n_m2": _finite_int(exchangeability.get("n_m2"), field="true-flip M2 count"),
    }
