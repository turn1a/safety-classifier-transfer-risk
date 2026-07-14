"""Tests for training-probe CKA sensitivity summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from transfer_risk.lib.similarity_sensitivity import (
    build_training_probe_sensitivity_summary,
    export_training_probe_similarity,
)


def _probe() -> pd.DataFrame:
    """Build a small in-memory balanced training probe."""
    return pd.DataFrame(
        [
            {
                "text": f"prompt-{index}",
                "label": index % 2,
                "source": f"source-{index % 3}",
                "split_marker": "train",
            }
            for index in range(6)
        ]
    )


def _original_similarity() -> pd.DataFrame:
    """Build an original similarity table whose legacy DBS must not be reused."""
    return pd.DataFrame(
        {
            "surrogate": [f"s{index}" for index in range(1, 7)],
            "mean_cka": [0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "dbs": [-99.0] * 6,
        }
    )


def _training_similarity() -> pd.DataFrame:
    """Build a training-probe similarity table with changed rank and membership."""
    return pd.DataFrame(
        {
            "surrogate": [f"s{index}" for index in range(1, 7)],
            "mean_cka": [0.87, 0.65, 0.82, 0.56, 0.43, 0.38],
            "dbs": [0.82, 0.61, 0.77, 0.51, 0.40, 0.31],
        }
    )


def _original_matrices() -> dict[str, list[list[float]]]:
    """Build saved original CKA matrices with distinct corrected DBS values."""
    return {f"s{index}": [[0.1 * index, 0.0], [0.0, 0.1 * index]] for index in range(1, 7)}


def _target_audit_summary() -> dict[str, object]:
    """Build a synthetic full-cohort true-target-flip summary."""
    outcomes = [0.60, 0.45, 0.55, 0.20, 0.30, 0.10]
    return {
        "full_cohort": {
            "surrogate_macro": {
                "rows": [
                    {
                        "surrogate": f"s{index}",
                        "true_target_flip_rate_macro_mean": outcome,
                        "true_target_flip_rate_macro_max": outcome,
                    }
                    for index, outcome in enumerate(outcomes, start=1)
                ]
            }
        }
    }


def test_training_probe_summary_compares_rank_membership_and_true_flip_sensitivity() -> None:
    summary = build_training_probe_sensitivity_summary(
        training_probe=_probe(),
        training_similarity=_training_similarity(),
        training_thresholds={"r1": 0.80, "r2": 0.40},
        training_selection={"M1": ["s1", "s3"], "M2": ["s4", "s6"]},
        original_similarity=_original_similarity(),
        original_selection={"M1": ["s1", "s2"], "M2": ["s5", "s6"]},
        original_cka_matrices=_original_matrices(),
        target_audit_summary=_target_audit_summary(),
        similarity_params={
            "pooling": "cls",
            "max_seq_len": 512,
            "cka": {"batch_size": 64},
            "dbs": {"box": 1},
            "thresholds": {"r1_quantile": 0.75, "r2_quantile": 0.25},
        },
        risk_params={"ablation": {"n_permutations": 10}},
        seed=17,
    )

    assert summary["probe_metadata"]["split"] == "train"
    assert summary["probe_metadata"]["label_counts"] == {"0": 3, "1": 3}
    assert (
        summary["rank_stability"]["original_vs_training_mean_cka"]["exchangeability_null"]["exact"]
        is True
    )
    assert (
        summary["rank_stability"]["original_vs_training_corrected_dbs"]["exchangeability_null"][
            "exact"
        ]
        is True
    )
    assert summary["membership_overlap"]["M1"]["intersection"] == ["s1"]
    assert summary["membership_overlap"]["M1"]["jaccard"] == pytest.approx(1 / 3)
    assert summary["membership_overlap"]["M2"]["intersection"] == ["s6"]
    true_flip = summary["true_target_flip_sensitivity"]
    assert (
        true_flip["training_probe_mean_cka_vs_full_cohort_macro_true_target_flip_rate"][
            "exchangeability_null"
        ]["exact"]
        is True
    )
    ablation = true_flip["training_probe_selection_ablation"]["one_sided_exchangeability_null"]
    assert ablation["exact"] is True
    assert ablation["m1_mean"] == pytest.approx(0.575)
    assert ablation["m2_mean"] == pytest.approx(0.15)
    assert "post-hoc" in summary["interpretation"]["analysis_label"]
    assert "no new attack data" in summary["interpretation"]["attack_data_note"]


def test_export_training_probe_similarity_limits_public_columns() -> None:
    table = _training_similarity().assign(internal_marker="not-public")

    exported = export_training_probe_similarity(table)

    assert list(exported.columns) == ["surrogate", "mean_cka", "dbs"]
    assert exported.to_dict("records") == _training_similarity().to_dict("records")


def _build_sensitivity_summary(
    *,
    training_probe: pd.DataFrame | None = None,
    training_similarity: pd.DataFrame | None = None,
    training_thresholds: dict[str, float] | None = None,
    training_selection: dict[str, Any] | None = None,
    original_similarity: pd.DataFrame | None = None,
    original_selection: dict[str, Any] | None = None,
    original_cka_matrices: dict[str, list[list[float]]] | None = None,
    target_audit_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid sensitivity summary with optional synthetic input substitutions."""
    return build_training_probe_sensitivity_summary(
        training_probe=_probe() if training_probe is None else training_probe,
        training_similarity=_training_similarity()
        if training_similarity is None
        else training_similarity,
        training_thresholds={"r1": 0.80, "r2": 0.40}
        if training_thresholds is None
        else training_thresholds,
        training_selection={"M1": ["s1", "s3"], "M2": ["s4", "s6"]}
        if training_selection is None
        else training_selection,
        original_similarity=_original_similarity()
        if original_similarity is None
        else original_similarity,
        original_selection={"M1": ["s1", "s2"], "M2": ["s5", "s6"]}
        if original_selection is None
        else original_selection,
        original_cka_matrices=_original_matrices()
        if original_cka_matrices is None
        else original_cka_matrices,
        target_audit_summary=_target_audit_summary()
        if target_audit_summary is None
        else target_audit_summary,
        similarity_params={
            "pooling": "cls",
            "max_seq_len": 512,
            "cka": {"batch_size": 64},
            "dbs": {"box": 1},
            "thresholds": {"r1_quantile": 0.75, "r2_quantile": 0.25},
        },
        risk_params={"ablation": {"n_permutations": 10}},
        seed=17,
    )


def test_export_training_probe_similarity_requires_all_public_columns() -> None:
    """An export cannot publish a partial similarity row schema."""
    similarity = _training_similarity().drop(columns="dbs")

    with pytest.raises(ValueError, match="missing required columns"):
        export_training_probe_similarity(similarity)


def test_sensitivity_summary_rejects_incompatible_similarity_pools() -> None:
    """Original and training CKA tables must identify the same configured surrogates."""
    training_similarity = _training_similarity()
    training_similarity.loc[5, "surrogate"] = "s7"

    with pytest.raises(ValueError, match="must use the same surrogate pool"):
        _build_sensitivity_summary(
            training_similarity=training_similarity,
            training_selection={"M1": ["s1", "s3"], "M2": ["s4", "s7"]},
        )


def test_sensitivity_summary_rejects_undefined_similarity_values() -> None:
    """A non-finite saved CKA value cannot enter a rank-stability comparison."""
    training_similarity = _training_similarity()
    training_similarity.loc[0, "mean_cka"] = float("nan")

    with pytest.raises(ValueError, match="must be a finite number"):
        _build_sensitivity_summary(training_similarity=training_similarity)


def test_sensitivity_summary_rejects_overlapping_selection_groups() -> None:
    """M1 and M2 must remain disjoint when rebuilding sensitivity evidence."""
    with pytest.raises(ValueError, match="M1 and M2 must not overlap"):
        _build_sensitivity_summary(training_selection={"M1": ["s1", "s3"], "M2": ["s3", "s6"]})


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"M2": ["s4", "s6"]}, r"missing 'M1'"),
        ({"M1": "s1", "M2": ["s4", "s6"]}, "must be a sequence"),
        ({"M1": ["s1", "s1"], "M2": ["s4", "s6"]}, "must contain unique"),
    ],
)
def test_sensitivity_summary_rejects_malformed_selection_values(
    selection: dict[str, Any],
    message: str,
) -> None:
    """Require named, sequence-shaped, unique membership in saved selections."""
    with pytest.raises(ValueError, match=message):
        _build_sensitivity_summary(training_selection=selection)


def test_sensitivity_summary_rejects_selection_members_outside_similarity_pool() -> None:
    """Saved selections cannot name surrogates absent from their similarity artifact."""
    with pytest.raises(ValueError, match="absent from its similarity table"):
        _build_sensitivity_summary(training_selection={"M1": ["s1", "missing"], "M2": ["s4", "s6"]})


def test_sensitivity_summary_requires_original_cka_matrix_for_every_surrogate() -> None:
    """Corrected DBS recomputation requires a saved matrix for each original surrogate."""
    matrices = _original_matrices()
    del matrices["s6"]

    with pytest.raises(ValueError, match="CKA matrices are missing surrogates"):
        _build_sensitivity_summary(original_cka_matrices=matrices)


def test_sensitivity_summary_rejects_malformed_target_audit_macro_rows() -> None:
    """Target-audit macro rows must be mappings before outcomes are reused."""
    target_audit_summary = {"full_cohort": {"surrogate_macro": {"rows": ["not a mapping"]}}}

    with pytest.raises(ValueError, match="must contain mappings"):
        _build_sensitivity_summary(target_audit_summary=target_audit_summary)


@pytest.mark.parametrize(
    "field",
    (
        "true_target_flip_rate_macro_mean",
        "true_target_flip_rate_macro_max",
    ),
)
def test_sensitivity_summary_rejects_boolean_optional_true_flip_rates(field: str) -> None:
    """Boolean values cannot stand in for nullable true-target-flip rate scalars."""
    target_audit_summary = _target_audit_summary()
    target_audit_summary["full_cohort"]["surrogate_macro"]["rows"][0][field] = True

    with pytest.raises(ValueError, match="must be a finite number"):
        _build_sensitivity_summary(target_audit_summary=target_audit_summary)


def test_sensitivity_summary_marks_no_defined_outcomes_not_estimable() -> None:
    """Undefined macro outcomes remain not estimable instead of becoming zero risk."""
    target_audit_summary = {
        "full_cohort": {
            "surrogate_macro": {
                "rows": [
                    {
                        "surrogate": f"s{index}",
                        "true_target_flip_rate_macro_mean": None,
                        "true_target_flip_rate_macro_max": None,
                    }
                    for index in range(1, 7)
                ]
            }
        }
    }

    summary = _build_sensitivity_summary(target_audit_summary=target_audit_summary)

    association = summary["true_target_flip_sensitivity"][
        "training_probe_mean_cka_vs_full_cohort_macro_true_target_flip_rate"
    ]["exchangeability_null"]
    ablation = summary["true_target_flip_sensitivity"]["training_probe_selection_ablation"][
        "one_sided_exchangeability_null"
    ]
    assert association["status"] == "not_estimable"
    assert association["n"] == 0
    assert ablation["status"] == "not_estimable"
    assert ablation["mean_p_value"] is None


def test_sensitivity_summary_requires_training_probe_label_and_source_metadata() -> None:
    """The public probe metadata records both canonical labels and source provenance."""
    probe = _probe().drop(columns="source")

    with pytest.raises(ValueError, match="training probe is missing required columns"):
        _build_sensitivity_summary(training_probe=probe)
