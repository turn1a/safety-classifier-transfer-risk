"""Pure helpers for the post-hoc target-outcome audit.

Deterministic grouping, source mapping, count aggregation, sensitivity analysis, and
CKA/M1-M2 association summaries. No I/O and no imports of kedro, mlflow, transformers,
or textattack — nodes own target predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from transfer_risk.lib.ablation import selection_ablation
from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.public_bundle import corrected_dbs_by_surrogate
from transfer_risk.lib.seeds import derive_seeds

KNOWN_TARGET_TRAINING_SOURCE = "jackhhao/jailbreak-classification"
_INJECTION_LABEL = 1
_MIN_ASSOCIATION_OBS = 2
_KEY_COLUMNS = ("surrogate", "recipe")
_COUNT_COLUMNS = (
    "source_successful",
    "target_original_benign",
    "target_original_injection",
    "target_perturbed_benign",
    "true_target_flips",
)
_RATE_COLUMNS = (
    "conditional_target_benign_rate",
    "true_target_flip_rate",
)
_CELL_COLUMNS = (*_KEY_COLUMNS, *_COUNT_COLUMNS, *_RATE_COLUMNS)
_ASSOCIATION_FEATURES = ("mean_cka", "dbs")
_AUDIT_OUTCOMES = ("conditional_target_benign_rate", "true_target_flip_rate")


def truncate_prompt(text: str, max_prompt_chars: int) -> str:
    """Truncate a prompt to the attack-time character limit.

    Args:
        text: Raw prompt text.
        max_prompt_chars: Maximum characters retained before attack records are written.

    Returns:
        The truncated prompt string.
    """
    return text[:max_prompt_chars]


def build_eval_original_source_map(
    test_df: pd.DataFrame,
    *,
    eval_set_size: int,
    max_prompt_chars: int,
) -> dict[str, str]:
    """Map attack-time truncated originals to their canonical test-split source id.

    The eval set matches the attacks stage: front-loaded test injections capped at
    ``eval_set_size``, truncated to ``max_prompt_chars``.

    Args:
        test_df: The ``test`` split with ``text``, ``label``, and ``source`` columns.
        eval_set_size: Maximum number of test injections in the attack eval set.
        max_prompt_chars: Character truncation applied before attacks run.

    Returns:
        Mapping from truncated original text to canonical ``source`` id.

    Raises:
        ValueError: If two eval originals truncate to the same key with different sources.
    """
    injections = test_df.loc[test_df["label"] == _INJECTION_LABEL, "text"].head(eval_set_size)
    mapping: dict[str, str] = {}
    for text, source in zip(injections, test_df.loc[injections.index, "source"], strict=True):
        key = truncate_prompt(str(text), max_prompt_chars)
        source_id = str(source)
        if key in mapping and mapping[key] != source_id:
            msg = f"ambiguous source for truncated original key {key[:40]!r}"
            raise ValueError(msg)
        mapping[key] = source_id
    return mapping


def assign_sources_to_records(
    records: Sequence[Mapping[str, Any]],
    source_map: Mapping[str, str],
    *,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    """Attach ``source`` to successful attack records using truncated originals.

    Args:
        records: Attack records with ``original``, ``perturbed``, and ``success`` keys.
        source_map: Mapping built by :func:`build_eval_original_source_map`.
        max_prompt_chars: Character truncation applied before lookup.

    Returns:
        Annotated records with ``source`` and no additional text fields.

    Raises:
        ValueError: If a successful record's truncated original is missing from ``source_map``.
    """
    annotated: list[dict[str, Any]] = []
    for record in records:
        if not record.get("success"):
            continue
        original = truncate_prompt(str(record["original"]), max_prompt_chars)
        if original not in source_map:
            msg = f"unmapped successful attack original {original[:40]!r}"
            raise ValueError(msg)
        annotated.append(
            {
                "surrogate": str(record["surrogate"]),
                "recipe": str(record["recipe"]),
                "source": source_map[original],
                "original": original,
                "perturbed": str(record["perturbed"]),
            }
        )
    return annotated


def dedupe_texts(texts: Sequence[str]) -> list[str]:
    """Return unique texts in deterministic first-seen order.

    Args:
        texts: Input text sequence.

    Returns:
        Deduplicated texts preserving first occurrence order.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def collect_unique_texts(records: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Count unique originals, perturbations, and their union across annotated records.

    Args:
        records: Annotated records with ``original`` and ``perturbed`` text keys.

    Returns:
        Dict with ``unique_originals``, ``unique_perturbations``, and ``unique_texts``.
    """
    originals = dedupe_texts([str(record["original"]) for record in records])
    perturbations = dedupe_texts([str(record["perturbed"]) for record in records])
    union = dedupe_texts([*originals, *perturbations])
    return {
        "unique_originals": len(originals),
        "unique_perturbations": len(perturbations),
        "unique_texts": len(union),
    }


def is_true_target_flip(
    original_pred: int,
    perturbed_pred: int,
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> bool:
    """Return whether the target flipped injection -> benign on a perturbed prompt.

    Args:
        original_pred: Target prediction on the original prompt.
        perturbed_pred: Target prediction on the perturbed prompt.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        ``True`` when the target predicted injection on the original and benign on the perturbed.
    """
    return original_pred == injection_label and perturbed_pred == benign_label


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a fraction or ``None`` when the denominator is zero."""
    return numerator / denominator if denominator else None


def _finite_optional_float(value: Any) -> float | None:
    """Convert a scalar to a finite float or return ``None`` for missing values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_optional_float_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Store nullable numeric columns as Python ``float`` or ``None`` values."""
    normalised = frame.copy()
    for column in columns:
        if column not in normalised.columns:
            continue
        values = [_finite_optional_float(value) for value in normalised[column].tolist()]
        normalised[column] = pd.Series(values, index=normalised.index, dtype="object")
    return normalised


def aggregate_audit_counts(
    records: Sequence[Mapping[str, Any]],
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> dict[str, int | float | None]:
    """Aggregate target-outcome counts and rates for one audit group.

    Args:
        records: Annotated records with ``original_pred`` and ``perturbed_pred``.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        Counts plus ``conditional_target_benign_rate`` and ``true_target_flip_rate``.
        Rates are ``None`` whenever their respective denominator is zero.
    """
    source_successful = len(records)
    target_original_benign = sum(
        1 for record in records if int(record["original_pred"]) == benign_label
    )
    target_original_injection = source_successful - target_original_benign
    target_perturbed_benign = sum(
        1 for record in records if int(record["perturbed_pred"]) == benign_label
    )
    true_target_flips = sum(
        1
        for record in records
        if is_true_target_flip(
            int(record["original_pred"]),
            int(record["perturbed_pred"]),
            benign_label=benign_label,
            injection_label=injection_label,
        )
    )
    return {
        "source_successful": source_successful,
        "target_original_benign": target_original_benign,
        "target_original_injection": target_original_injection,
        "target_perturbed_benign": target_perturbed_benign,
        "true_target_flips": true_target_flips,
        "conditional_target_benign_rate": _safe_rate(target_perturbed_benign, source_successful),
        "true_target_flip_rate": _safe_rate(true_target_flips, target_original_injection),
    }


def _aggregate_grouped(
    records: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    *,
    benign_label: int,
    injection_label: int,
) -> pd.DataFrame:
    """Aggregate audit counts for arbitrary grouping keys."""
    output_columns = [*group_keys, *_COUNT_COLUMNS, *_RATE_COLUMNS]
    if not records:
        return _normalise_optional_float_columns(
            pd.DataFrame(columns=output_columns),
            _RATE_COLUMNS,
        )
    frame = pd.DataFrame.from_records(records)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_keys), sort=True):
        key_tuple = (keys,) if len(group_keys) == 1 else keys
        row = dict(zip(group_keys, key_tuple, strict=True))
        row.update(
            aggregate_audit_counts(
                cast("list[dict[str, Any]]", group.to_dict("records")),
                benign_label=benign_label,
                injection_label=injection_label,
            )
        )
        rows.append(row)
    return _normalise_optional_float_columns(
        pd.DataFrame.from_records(rows, columns=output_columns),
        _RATE_COLUMNS,
    )


def aggregate_by_cell(
    records: Sequence[Mapping[str, Any]],
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> pd.DataFrame:
    """Aggregate audit counts per ``(surrogate, recipe)`` cell.

    Args:
        records: Annotated records with predictions and grouping keys.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        One record-derived row per observed cell with aggregate counts and rates; no raw text.
    """
    return _aggregate_grouped(
        records,
        _KEY_COLUMNS,
        benign_label=benign_label,
        injection_label=injection_label,
    )


def aggregate_by_source(
    records: Sequence[Mapping[str, Any]],
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> pd.DataFrame:
    """Aggregate audit counts per ``(surrogate, recipe, source)`` group.

    Args:
        records: Annotated records with predictions and grouping keys.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        One row per source group with aggregate counts and rates; no raw text columns.
    """
    return _aggregate_grouped(
        records,
        (*_KEY_COLUMNS, "source"),
        benign_label=benign_label,
        injection_label=injection_label,
    )


def _validated_aggregate_frame(
    aggregates: pd.DataFrame,
    group_keys: Sequence[str],
) -> pd.DataFrame:
    """Validate aggregate keys/counts and return the columns needed for rebuilding cells."""
    required = {*group_keys, *_COUNT_COLUMNS}
    missing = sorted(required.difference(aggregates.columns))
    if missing:
        msg = f"audit aggregates are missing required columns: {missing}"
        raise ValueError(msg)
    output = aggregates.loc[:, [*group_keys, *_COUNT_COLUMNS]].copy()
    if output.duplicated(list(group_keys)).any():
        msg = f"audit aggregates must be unique by {list(group_keys)!r}"
        raise ValueError(msg)
    for column in _COUNT_COLUMNS:
        values = pd.to_numeric(output[column], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if (
            values.isna().any()
            or not np.isfinite(numeric).all()
            or (numeric < 0.0).any()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            msg = f"audit aggregate column {column!r} must contain non-negative integers"
            raise ValueError(msg)
        output[column] = values.astype(int)
    return output


def validate_source_aggregate_rollups(
    raw_cells: pd.DataFrame,
    raw_sources: pd.DataFrame,
) -> None:
    """Reject source aggregates that do not exactly reconcile to raw full-cell counts.

    Args:
        raw_cells: Raw target-inference aggregates keyed by surrogate and recipe.
        raw_sources: Raw target-inference aggregates keyed by surrogate, recipe, and source.

    Returns:
        ``None`` when every source rollup exactly matches its full-cell count fields.

    Raises:
        ValueError: If source keys are missing, unknown, or sum to different cell counts.
    """
    cells = _validated_aggregate_frame(raw_cells, _KEY_COLUMNS)
    sources = _validated_aggregate_frame(raw_sources, (*_KEY_COLUMNS, "source"))
    if sources["source"].isna().any():
        msg = "audit source aggregates must define a source for every row"
        raise ValueError(msg)
    cell_counts = cells.set_index(list(_KEY_COLUMNS))[list(_COUNT_COLUMNS)].sort_index()
    source_counts = (
        sources.groupby(list(_KEY_COLUMNS), sort=True)[list(_COUNT_COLUMNS)].sum().sort_index()
    )
    unknown = source_counts.index.difference(cell_counts.index)
    if not unknown.empty:
        msg = "source aggregate rollups include cells absent from raw full-cell counts"
        raise ValueError(msg)
    aligned_source_counts = source_counts.reindex(cell_counts.index, fill_value=0)
    if bool(aligned_source_counts.ne(cell_counts).to_numpy().any()):
        msg = "source aggregate rollups do not reconcile to raw full-cell counts"
        raise ValueError(msg)


def _corrected_master_audit_grid(
    master: pd.DataFrame,
    cka_matrices: Mapping[str, Any],
    *,
    dbs_box: int,
) -> pd.DataFrame:
    """Return the master audit grid with DBS recomputed from saved CKA matrices."""
    grid = _master_audit_grid(master)
    matrices: dict[str, npt.NDArray[np.float64]] = {
        str(name): np.asarray(matrix, dtype=np.float64) for name, matrix in cka_matrices.items()
    }
    corrected = corrected_dbs_by_surrogate(matrices, box=dbs_box)
    missing = sorted(set(grid["surrogate"]).difference(corrected))
    if missing:
        msg = f"CKA matrices are missing audit surrogates: {missing}"
        raise ValueError(msg)
    grid["dbs"] = [corrected[str(surrogate)] for surrogate in grid["surrogate"]]
    return grid


def build_complete_grid_from_aggregates(
    raw_cells: pd.DataFrame,
    master: pd.DataFrame,
    cka_matrices: Mapping[str, Any],
    *,
    dbs_box: int,
) -> pd.DataFrame:
    """Rebuild a complete audit grid from target-inference aggregates and corrected DBS.

    Args:
        raw_cells: One aggregate row per observed ``(surrogate, recipe)`` cell. Legacy
            full-grid files may include additional stale similarity/rate columns; they are
            ignored and count fields are recomputed on the corrected grid.
        master: Saved master table whose keys and mean CKA values define the audit grid.
        cka_matrices: Saved target-vs-surrogate CKA matrices keyed by surrogate.
        dbs_box: Bresenham diagonal-box half-width.

    Returns:
        Complete master-keyed cells with corrected DBS and nullable outcome rates.
    """
    grid = _corrected_master_audit_grid(master, cka_matrices, dbs_box=dbs_box)
    aggregate_cells = _validated_aggregate_frame(raw_cells, _KEY_COLUMNS)
    _validate_record_cell_keys(aggregate_cells, grid)
    merged = grid.merge(
        aggregate_cells,
        how="left",
        on=list(_KEY_COLUMNS),
        sort=False,
        validate="one_to_one",
    )
    completed = _complete_cell_counts(merged)
    return completed.loc[:, [*_CELL_COLUMNS, *_ASSOCIATION_FEATURES]]


def _master_audit_grid(master: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate the unique master-result keys and fixed similarity fields."""
    required = {*_KEY_COLUMNS, *_ASSOCIATION_FEATURES}
    missing = sorted(required.difference(master.columns))
    if missing:
        msg = f"master results table is missing required audit columns: {missing}"
        raise ValueError(msg)
    grid = master.loc[:, [*_KEY_COLUMNS, *_ASSOCIATION_FEATURES]].copy()
    if grid.duplicated(list(_KEY_COLUMNS)).any():
        msg = "master results table must have one row per surrogate and recipe"
        raise ValueError(msg)
    for feature in _ASSOCIATION_FEATURES:
        values = pd.to_numeric(grid[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            msg = f"master results table has non-finite {feature!r} values"
            raise ValueError(msg)
        grid[feature] = values.astype(float)
    return grid.reset_index(drop=True)


def _validate_record_cell_keys(record_cells: pd.DataFrame, grid: pd.DataFrame) -> None:
    """Reject aggregate records whose cell keys are absent from the saved master grid."""
    if record_cells.empty:
        return
    record_index = pd.MultiIndex.from_frame(record_cells.loc[:, list(_KEY_COLUMNS)])
    grid_index = pd.MultiIndex.from_frame(grid.loc[:, list(_KEY_COLUMNS)])
    unknown = record_index.difference(grid_index)
    if not unknown.empty:
        msg = f"audit records include keys absent from master results: {unknown.tolist()}"
        raise ValueError(msg)


def _complete_cell_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill unobserved cell counts with zero and recompute nullable rates from counts."""
    completed = frame.copy()
    for column in _COUNT_COLUMNS:
        completed[column] = pd.to_numeric(completed[column], errors="coerce").fillna(0).astype(int)
    conditional_rates = [
        _safe_rate(int(numerator), int(denominator))
        for numerator, denominator in zip(
            completed["target_perturbed_benign"],
            completed["source_successful"],
            strict=True,
        )
    ]
    true_flip_rates = [
        _safe_rate(int(numerator), int(denominator))
        for numerator, denominator in zip(
            completed["true_target_flips"],
            completed["target_original_injection"],
            strict=True,
        )
    ]
    completed["conditional_target_benign_rate"] = pd.Series(
        conditional_rates,
        index=completed.index,
        dtype="object",
    )
    completed["true_target_flip_rate"] = pd.Series(
        true_flip_rates,
        index=completed.index,
        dtype="object",
    )
    return completed


def build_full_grid_cells(
    records: Sequence[Mapping[str, Any]],
    master: pd.DataFrame,
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> pd.DataFrame:
    """Build the complete saved-master-key audit grid with exactly one master merge.

    The saved ``master_results_table`` defines every valid ``(surrogate, recipe)`` cell.
    Cells with no successful source attacks are retained with zero count fields and null
    rates. This is the only helper that merges audit aggregates with master metadata.

    Args:
        records: Prediction-only audit records for successful source attacks.
        master: Saved master results table whose keys define the full grid.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        Complete master-keyed cell table with fixed CKA/DBS fields and nullable rates.
    """
    grid = _master_audit_grid(master)
    record_cells = aggregate_by_cell(
        records,
        benign_label=benign_label,
        injection_label=injection_label,
    )
    _validate_record_cell_keys(record_cells, grid)
    merged = grid.merge(
        record_cells,
        how="left",
        on=list(_KEY_COLUMNS),
        sort=False,
        validate="one_to_one",
    )
    completed = _complete_cell_counts(merged)
    return completed.loc[:, [*_CELL_COLUMNS, *_ASSOCIATION_FEATURES]]


def aggregate_records_on_complete_grid(
    records: Sequence[Mapping[str, Any]],
    complete_cells: pd.DataFrame,
    *,
    benign_label: int,
    injection_label: int = _INJECTION_LABEL,
) -> pd.DataFrame:
    """Aggregate records onto an existing complete master-key grid without another merge.

    This supports sensitivity cohorts: callers filter records first, then aggregate them over
    the same saved master grid used for the full cohort.

    Args:
        records: Filtered prediction-only audit records.
        complete_cells: Full cells from :func:`build_full_grid_cells`.
        benign_label: Class id meaning benign / not an injection.
        injection_label: Class id meaning injection.

    Returns:
        Complete cell grid containing the filtered cohort's counts and nullable rates.
    """
    grid = _master_audit_grid(complete_cells)
    record_cells = aggregate_by_cell(
        records,
        benign_label=benign_label,
        injection_label=injection_label,
    )
    _validate_record_cell_keys(record_cells, grid)
    grid_index = pd.MultiIndex.from_frame(grid.loc[:, list(_KEY_COLUMNS)])
    aligned = record_cells.set_index(list(_KEY_COLUMNS)).reindex(grid_index).reset_index()
    for feature in _ASSOCIATION_FEATURES:
        aligned[feature] = grid[feature].to_numpy()
    completed = _complete_cell_counts(aligned)
    return completed.loc[:, [*_CELL_COLUMNS, *_ASSOCIATION_FEATURES]]


def aggregate_audit_counts_from_frame(cells: pd.DataFrame) -> dict[str, int | float | None]:
    """Sum complete cell-level audit counts into one cohort summary.

    Args:
        cells: Cell aggregates with all audit count columns.

    Returns:
        Cohort-level counts and recomputed rates with null zero-denominator rates.
    """
    if cells.empty:
        return aggregate_audit_counts([], benign_label=0)
    source_successful = int(cells["source_successful"].sum())
    target_original_benign = int(cells["target_original_benign"].sum())
    target_original_injection = int(cells["target_original_injection"].sum())
    target_perturbed_benign = int(cells["target_perturbed_benign"].sum())
    true_target_flips = int(cells["true_target_flips"].sum())
    return {
        "source_successful": source_successful,
        "target_original_benign": target_original_benign,
        "target_original_injection": target_original_injection,
        "target_perturbed_benign": target_perturbed_benign,
        "true_target_flips": true_target_flips,
        "conditional_target_benign_rate": _safe_rate(
            target_perturbed_benign,
            source_successful,
        ),
        "true_target_flip_rate": _safe_rate(true_target_flips, target_original_injection),
    }


def _source_composition(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count successful-source records by canonical source identifier."""
    composition: dict[str, int] = {}
    for record in records:
        source = str(record["source"])
        composition[source] = composition.get(source, 0) + 1
    return composition


def _defined_rate_summary(values: Sequence[Any]) -> tuple[float | None, float | None, int]:
    """Return unweighted mean, maximum, and count of finite defined cell rates."""
    defined = [rate for value in values if (rate := _finite_optional_float(value)) is not None]
    if not defined:
        return None, None, 0
    return float(np.mean(defined)), float(np.max(defined)), len(defined)


def _fixed_surrogate_feature(values: Sequence[Any], feature: str, surrogate: str) -> float:
    """Return a fixed per-surrogate CKA/DBS value or reject inconsistent master metadata."""
    finite = [_finite_optional_float(value) for value in values]
    if any(value is None for value in finite):
        msg = f"surrogate {surrogate!r} has an undefined {feature!r} value"
        raise ValueError(msg)
    unique = {float(value) for value in finite if value is not None}
    if len(unique) != 1:
        msg = f"surrogate {surrogate!r} has recipe-varying {feature!r} values"
        raise ValueError(msg)
    return unique.pop()


def build_surrogate_macro_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a complete cell grid to one unweighted macro row per surrogate.

    Undefined cell rates are omitted from each outcome's macro mean and max. Consequently,
    a surrogate with no defined true-flip denominator remains ``None`` rather than being
    treated as having a zero true-flip rate.

    Args:
        cells: Complete master-keyed audit cell grid.

    Returns:
        One row per surrogate with CKA/DBS and outcome-specific macro summaries.
    """
    columns = [
        "surrogate",
        *_ASSOCIATION_FEATURES,
        *_COUNT_COLUMNS,
        *[
            field
            for outcome in _AUDIT_OUTCOMES
            for field in (
                f"{outcome}_macro_mean",
                f"{outcome}_macro_max",
                f"{outcome}_defined_cell_count",
            )
        ],
    ]
    rows: list[dict[str, Any]] = []
    for surrogate, group in cells.groupby("surrogate", sort=True):
        surrogate_name = str(surrogate)
        row: dict[str, Any] = {
            "surrogate": surrogate_name,
            **{
                feature: _fixed_surrogate_feature(
                    group[feature].tolist(),
                    feature,
                    surrogate_name,
                )
                for feature in _ASSOCIATION_FEATURES
            },
            **{column: int(group[column].sum()) for column in _COUNT_COLUMNS},
        }
        for outcome in _AUDIT_OUTCOMES:
            mean, maximum, count = _defined_rate_summary(group[outcome].tolist())
            row[f"{outcome}_macro_mean"] = mean
            row[f"{outcome}_macro_max"] = maximum
            row[f"{outcome}_defined_cell_count"] = count
        rows.append(row)
    nullable_columns = [
        f"{outcome}_{summary}"
        for outcome in _AUDIT_OUTCOMES
        for summary in ("macro_mean", "macro_max")
    ]
    return _normalise_optional_float_columns(
        pd.DataFrame.from_records(rows, columns=columns),
        nullable_columns,
    )


def _json_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values to strict JSON-compatible Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_optional_float(value)
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return DataFrame records containing only JSON-compatible scalar values."""
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _association_block(
    surrogate_macro: pd.DataFrame,
    *,
    outcome: str,
) -> dict[str, dict[str, Any]]:
    """Compute CKA/DBS associations from one macro row per surrogate."""
    outcome_column = f"{outcome}_macro_mean"
    result: dict[str, dict[str, Any]] = {}
    for feature in _ASSOCIATION_FEATURES:
        features: list[float] = []
        outcomes: list[float] = []
        for feature_value, outcome_value in zip(
            surrogate_macro[feature],
            surrogate_macro[outcome_column],
            strict=True,
        ):
            finite_feature = _finite_optional_float(feature_value)
            finite_outcome = _finite_optional_float(outcome_value)
            if finite_feature is None or finite_outcome is None:
                continue
            features.append(finite_feature)
            outcomes.append(finite_outcome)
        designed_pool = {
            "label": "designed surrogate pool",
            "unit": "one row per surrogate",
            "n_surrogates": len(outcomes),
        }
        if len(outcomes) < _MIN_ASSOCIATION_OBS:
            result[feature] = {
                "designed_pool": designed_pool,
                "exchangeability_null": {
                    "status": "not_estimable",
                    "rho": None,
                    "two_sided_p": None,
                    "n": len(outcomes),
                    "exact": False,
                    "p_value_label": "not estimable for the designed pool",
                },
            }
            continue
        try:
            stats = spearman_association(features, outcomes)
        except ValueError as error:
            result[feature] = {
                "designed_pool": designed_pool,
                "exchangeability_null": {
                    "status": "not_estimable",
                    "reason": str(error),
                    "rho": None,
                    "two_sided_p": None,
                    "n": len(outcomes),
                    "exact": False,
                    "p_value_label": "not estimable for the designed pool",
                },
            }
            continue
        exact = bool(stats["exact"])
        p_value_label = (
            "exact enumeration under the exchangeability null for the designed pool"
            if exact
            else "normal approximation under the exchangeability null for the designed pool"
        )
        result[feature] = {
            "designed_pool": designed_pool,
            "exchangeability_null": {
                "rho": float(stats["rho"]),
                "two_sided_p": float(stats["two_sided_p"]),
                "n": int(stats["n"]),
                "exact": exact,
                "p_value_label": p_value_label,
            },
        }
    return result


def _ablation_rng(seed: int, *, cohort_index: int, outcome_index: int) -> np.random.Generator:
    """Derive a deterministic ablation RNG for one cohort and outcome."""
    seed_sequence = np.random.SeedSequence([derive_seeds(seed).numpy, cohort_index, outcome_index])
    return np.random.default_rng(seed_sequence)


def _selection_ablation_block(
    surrogate_macro: pd.DataFrame,
    selection: Mapping[str, Sequence[str]],
    risk_params: Mapping[str, Any],
    seed: int,
    *,
    outcome: str,
    cohort_index: int,
    outcome_index: int,
) -> dict[str, Any]:
    """Run one M1/M2 ablation from defined surrogate-level macro outcomes."""
    mean_column = f"{outcome}_macro_mean"
    max_column = f"{outcome}_macro_max"
    mean_by_surrogate: dict[str, float] = {}
    max_by_surrogate: dict[str, float] = {}
    for row in _json_records(surrogate_macro):
        mean = _finite_optional_float(row[mean_column])
        maximum = _finite_optional_float(row[max_column])
        if mean is None or maximum is None:
            continue
        surrogate = str(row["surrogate"])
        mean_by_surrogate[surrogate] = mean
        max_by_surrogate[surrogate] = maximum
    m1 = [str(name) for name in selection["M1"] if str(name) in mean_by_surrogate]
    m2 = [str(name) for name in selection["M2"] if str(name) in mean_by_surrogate]
    designed_pool = {
        "label": "saved M1/M2 designed pool",
        "unit": "one row per surrogate",
        "n_m1": len(m1),
        "n_m2": len(m2),
    }
    if not m1 or not m2:
        return {
            "designed_pool": designed_pool,
            "one_sided_exchangeability_null": {
                "status": "not_estimable",
                "exact": False,
                "mean_p_value": None,
                "max_p_value": None,
                "p_value_label": "not estimable for the saved M1/M2 designed pool",
            },
        }
    stats = selection_ablation(
        mean_by_surrogate,
        max_by_surrogate,
        m1,
        m2,
        n_permutations=int(risk_params["ablation"]["n_permutations"]),
        rng=_ablation_rng(
            seed,
            cohort_index=cohort_index,
            outcome_index=outcome_index,
        ),
    )
    exact = bool(stats["exact"])
    p_value_label = (
        "exact enumeration under the exchangeability null for the designed M1/M2 pool"
        if exact
        else (
            "Monte Carlo label permutation under the exchangeability null for the designed "
            "M1/M2 pool"
        )
    )
    return {
        "designed_pool": designed_pool,
        "one_sided_exchangeability_null": {
            **stats,
            "p_value_label": p_value_label,
        },
    }


def _cohort_analysis_from_cells(
    cells: pd.DataFrame,
    source_composition: Mapping[str, int],
    selection: Mapping[str, Sequence[str]],
    risk_params: Mapping[str, Any],
    seed: int,
    *,
    cohort_index: int,
) -> dict[str, Any]:
    """Build count, macro, association, and ablation analysis from completed cells."""
    surrogate_macro = build_surrogate_macro_table(cells)
    return {
        **aggregate_audit_counts_from_frame(cells),
        "source_composition": {str(name): int(value) for name, value in source_composition.items()},
        "surrogate_macro": {
            "analysis_unit": "one row per surrogate",
            "aggregation": (
                "unweighted mean and maximum across each surrogate's defined recipe-level cells"
            ),
            "undefined_rate_handling": (
                "undefined cell rates are excluded; a surrogate-level outcome is null when "
                "none of its cells has that outcome denominator"
            ),
            "row_count": len(surrogate_macro),
            "rows": _json_records(surrogate_macro),
        },
        "cka_dbs_association": {
            outcome: _association_block(surrogate_macro, outcome=outcome)
            for outcome in _AUDIT_OUTCOMES
        },
        "selection_ablation": {
            outcome: _selection_ablation_block(
                surrogate_macro,
                selection,
                risk_params,
                seed,
                outcome=outcome,
                cohort_index=cohort_index,
                outcome_index=index,
            )
            for index, outcome in enumerate(_AUDIT_OUTCOMES)
        },
    }


def _cohort_analysis(
    records: Sequence[Mapping[str, Any]],
    cells: pd.DataFrame,
    selection: Mapping[str, Sequence[str]],
    risk_params: Mapping[str, Any],
    seed: int,
    *,
    cohort_index: int,
) -> dict[str, Any]:
    """Build count, macro, association, and ablation analysis for one record cohort."""
    return _cohort_analysis_from_cells(
        cells,
        _source_composition(records),
        selection,
        risk_params,
        seed,
        cohort_index=cohort_index,
    )


def attach_predictions(
    records: Sequence[Mapping[str, str]],
    original_predictions: Mapping[str, int],
    perturbed_predictions: Mapping[str, int],
) -> list[dict[str, int | str]]:
    """Attach target predictions and drop raw text from annotated records.

    Args:
        records: Annotated records with ``original`` and ``perturbed`` text keys.
        original_predictions: Target predictions keyed by original text.
        perturbed_predictions: Target predictions keyed by perturbed text.

    Returns:
        Prediction-only records suitable for aggregation.
    """
    enriched: list[dict[str, int | str]] = []
    for record in records:
        original = str(record["original"])
        perturbed = str(record["perturbed"])
        enriched.append(
            {
                "surrogate": str(record["surrogate"]),
                "recipe": str(record["recipe"]),
                "source": str(record["source"]),
                "original_pred": int(original_predictions[original]),
                "perturbed_pred": int(perturbed_predictions[perturbed]),
            }
        )
    return enriched


def baseline_counts_on_unique_originals(
    originals: Sequence[str],
    original_predictions: Mapping[str, int],
    *,
    benign_label: int,
) -> dict[str, int]:
    """Count target benign/injection predictions on unique original prompts.

    Args:
        originals: Unique original prompt texts.
        original_predictions: Target predictions keyed by original text.
        benign_label: Class id meaning benign / not an injection.

    Returns:
        Dict with ``target_original_benign`` and ``target_original_injection`` counts.
    """
    benign = sum(1 for text in originals if original_predictions[text] == benign_label)
    return {
        "target_original_benign": benign,
        "target_original_injection": len(originals) - benign,
    }


def flatten_successful_records(
    adversarial_examples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten successful attack records with surrogate/recipe identity attached.

    Args:
        adversarial_examples: Mapping from ``surrogate__recipe`` cell key to attack records.

    Returns:
        Successful records annotated with ``surrogate`` and ``recipe`` keys.
    """
    flattened: list[dict[str, Any]] = []
    for key, records in adversarial_examples.items():
        surrogate, recipe = key.split("__", 1)
        for record in records:
            if not record.get("success"):
                continue
            flattened.append(
                {
                    **record,
                    "surrogate": surrogate,
                    "recipe": recipe,
                }
            )
    return flattened


def build_summary_from_records(  # noqa: PLR0913
    records: Sequence[Mapping[str, Any]],
    cells: pd.DataFrame,
    selection: Mapping[str, Sequence[str]],
    risk_params: Mapping[str, Any],
    seed: int,
    *,
    excluded_source: str,
    benign_label: int,
    injection_label: int,
    unique_text_stats: Mapping[str, int],
    baseline_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build the audit summary using annotated prediction records for cohort splits.

    Args:
        records: Prediction-only annotated records.
        cells: Complete master-keyed cell aggregates from :func:`build_full_grid_cells`.
        selection: Saved M1/M2 membership.
        risk_params: Risk-stage parameters.
        seed: Root seed.
        excluded_source: Known target-training source excluded in sensitivity analysis.
        benign_label: Benign class id.
        injection_label: Injection class id.
        unique_text_stats: Unique text counts.
        baseline_counts: Target baseline counts on unique originals.

    Returns:
        JSON-serialisable audit summary without raw prompt text.
    """
    if excluded_source != KNOWN_TARGET_TRAINING_SOURCE:
        msg = (
            "the source-excluded audit must remove exactly "
            f"{KNOWN_TARGET_TRAINING_SOURCE!r}, got {excluded_source!r}"
        )
        raise ValueError(msg)
    sensitivity_records = [
        record for record in records if str(record["source"]) != KNOWN_TARGET_TRAINING_SOURCE
    ]
    sensitivity_cells = aggregate_records_on_complete_grid(
        sensitivity_records,
        cells,
        benign_label=benign_label,
        injection_label=injection_label,
    )
    m1 = [str(name) for name in selection["M1"]]
    m2 = [str(name) for name in selection["M2"]]
    full_analysis = _cohort_analysis(
        records,
        cells,
        selection,
        risk_params,
        seed,
        cohort_index=0,
    )
    sensitivity_analysis = _cohort_analysis(
        sensitivity_records,
        sensitivity_cells,
        selection,
        risk_params,
        seed,
        cohort_index=1,
    )
    return {
        "full_cohort": {
            **full_analysis,
            "unique_text_counts": {
                str(name): int(value) for name, value in unique_text_stats.items()
            },
            "target_baseline_on_unique_originals": {
                str(name): int(value) for name, value in baseline_counts.items()
            },
        },
        "known_source_excluded_sensitivity": {
            "analysis_label": "post-hoc sensitivity analysis excluding one known source",
            "exclusion_stage": "records filtered before per-cell aggregation",
            "excluded_source": KNOWN_TARGET_TRAINING_SOURCE,
            **sensitivity_analysis,
        },
        "membership": {
            "M1": m1,
            "M2": m2,
            "basis": "saved original mean CKA membership; audit changes outcomes only",
        },
    }


def build_raw_audit_context(
    *,
    excluded_source: str,
    unique_text_stats: Mapping[str, int],
    baseline_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build prediction-derived context needed to finalize an audit without raw text.

    Args:
        excluded_source: Known target-training source for the sensitivity analysis.
        unique_text_stats: Counts of unique originals, perturbations, and their union.
        baseline_counts: Target baseline counts on unique original prompts.

    Returns:
        JSON-serialisable context that carries no raw prompt text or target model state.
    """
    return {
        "excluded_source": str(excluded_source),
        "unique_text_counts": {str(name): int(value) for name, value in unique_text_stats.items()},
        "target_baseline_on_unique_originals": {
            str(name): int(value) for name, value in baseline_counts.items()
        },
    }


def _context_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a context mapping or raise a descriptive validation error."""
    if not isinstance(value, Mapping):
        msg = f"audit raw context field {field_name!r} must be a mapping"
        raise ValueError(msg)
    return value


def _context_counts(value: Any, field_name: str) -> dict[str, int]:
    """Convert a JSON context mapping of non-negative integral counts."""
    mapping = _context_mapping(value, field_name)
    counts: dict[str, int] = {}
    for name, raw_count in mapping.items():
        count = _finite_optional_float(raw_count)
        if count is None or count < 0.0 or not count.is_integer():
            msg = f"audit raw context field {field_name!r} has invalid count for {name!r}"
            raise ValueError(msg)
        counts[str(name)] = int(count)
    return counts


def _finalization_context(
    raw_context: Mapping[str, Any],
) -> tuple[str, dict[str, int], dict[str, int]]:
    """Extract context from either new raw context or the legacy saved audit summary."""
    if "full_cohort" in raw_context:
        full_cohort = _context_mapping(raw_context["full_cohort"], "full_cohort")
        sensitivity = _context_mapping(
            raw_context.get("known_source_excluded_sensitivity"),
            "known_source_excluded_sensitivity",
        )
        excluded_source = str(sensitivity.get("excluded_source"))
        unique_text_counts = _context_counts(
            full_cohort.get("unique_text_counts"),
            "full_cohort.unique_text_counts",
        )
        baseline_counts = _context_counts(
            full_cohort.get("target_baseline_on_unique_originals"),
            "full_cohort.target_baseline_on_unique_originals",
        )
    else:
        excluded_source = str(raw_context.get("excluded_source"))
        unique_text_counts = _context_counts(
            raw_context.get("unique_text_counts"),
            "unique_text_counts",
        )
        baseline_counts = _context_counts(
            raw_context.get("target_baseline_on_unique_originals"),
            "target_baseline_on_unique_originals",
        )
    if excluded_source != KNOWN_TARGET_TRAINING_SOURCE:
        msg = (
            "the source-excluded audit must remove exactly "
            f"{KNOWN_TARGET_TRAINING_SOURCE!r}, got {excluded_source!r}"
        )
        raise ValueError(msg)
    return excluded_source, unique_text_counts, baseline_counts


def _source_composition_from_aggregates(
    raw_sources: pd.DataFrame,
    *,
    excluded_source: str | None = None,
) -> dict[str, int]:
    """Sum source-success counts from source aggregates, optionally excluding one source."""
    source_aggregates = _validated_aggregate_frame(
        raw_sources,
        (*_KEY_COLUMNS, "source"),
    )
    if source_aggregates["source"].isna().any():
        msg = "audit source aggregates must define a source for every row"
        raise ValueError(msg)
    source_aggregates["source"] = source_aggregates["source"].astype(str)
    if excluded_source is not None:
        source_aggregates = source_aggregates.loc[source_aggregates["source"] != excluded_source]
    grouped = source_aggregates.groupby("source", sort=True)["source_successful"].sum()
    return {str(source): int(count) for source, count in grouped.items()}


def _source_excluded_cell_aggregates(
    raw_sources: pd.DataFrame,
    excluded_source: str,
) -> pd.DataFrame:
    """Sum non-excluded source aggregates to one row per audit cell."""
    source_aggregates = _validated_aggregate_frame(
        raw_sources,
        (*_KEY_COLUMNS, "source"),
    )
    if source_aggregates["source"].isna().any():
        msg = "audit source aggregates must define a source for every row"
        raise ValueError(msg)
    filtered = source_aggregates.loc[
        source_aggregates["source"].astype(str) != excluded_source,
        [*_KEY_COLUMNS, *_COUNT_COLUMNS],
    ]
    return (
        filtered.groupby(list(_KEY_COLUMNS), as_index=False, sort=True)[list(_COUNT_COLUMNS)]
        .sum()
        .reset_index(drop=True)
    )


def finalize_audit_from_aggregates(  # noqa: PLR0913
    raw_cells: pd.DataFrame,
    raw_sources: pd.DataFrame,
    raw_context: Mapping[str, Any],
    master_results_table: pd.DataFrame,
    cka_matrices: Mapping[str, Any],
    similarity_params: Mapping[str, Any],
    surrogate_selection: Mapping[str, Sequence[str]],
    risk_params: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild final audit artifacts from saved target-inference aggregates only.

    Args:
        raw_cells: Legacy-compatible target-inference aggregates per audit cell.
        raw_sources: Target-inference aggregates per audit cell and source.
        raw_context: New raw audit context or legacy final summary carrying non-text counts.
        master_results_table: Saved master table defining the complete audit cell grid.
        cka_matrices: Saved CKA matrices used to recompute corrected DBS.
        similarity_params: Similarity parameters containing ``dbs.box``.
        surrogate_selection: Saved M1/M2 membership.
        risk_params: Risk parameters for deterministic selection ablations.
        seed: Root reproducibility seed.

    Returns:
        Corrected complete audit cells and a complete JSON-serialisable audit summary.
    """
    excluded_source, unique_text_counts, baseline_counts = _finalization_context(raw_context)
    validate_source_aggregate_rollups(raw_cells, raw_sources)
    dbs_box = int(similarity_params["dbs"]["box"])
    cells = build_complete_grid_from_aggregates(
        raw_cells,
        master_results_table,
        cka_matrices,
        dbs_box=dbs_box,
    )
    sensitivity_cells = build_complete_grid_from_aggregates(
        _source_excluded_cell_aggregates(raw_sources, excluded_source),
        master_results_table,
        cka_matrices,
        dbs_box=dbs_box,
    )
    full_analysis = _cohort_analysis_from_cells(
        cells,
        _source_composition_from_aggregates(raw_sources),
        surrogate_selection,
        risk_params,
        seed,
        cohort_index=0,
    )
    sensitivity_analysis = _cohort_analysis_from_cells(
        sensitivity_cells,
        _source_composition_from_aggregates(raw_sources, excluded_source=excluded_source),
        surrogate_selection,
        risk_params,
        seed,
        cohort_index=1,
    )
    m1 = [str(name) for name in surrogate_selection["M1"]]
    m2 = [str(name) for name in surrogate_selection["M2"]]
    summary = {
        "full_cohort": {
            **full_analysis,
            "unique_text_counts": unique_text_counts,
            "target_baseline_on_unique_originals": baseline_counts,
        },
        "known_source_excluded_sensitivity": {
            "analysis_label": "post-hoc sensitivity analysis excluding one known source",
            "exclusion_stage": "records filtered before per-cell aggregation",
            "excluded_source": excluded_source,
            **sensitivity_analysis,
        },
        "membership": {
            "M1": m1,
            "M2": m2,
            "basis": "saved original mean CKA membership; audit changes outcomes only",
        },
    }
    return cells, summary
