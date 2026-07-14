"""Tests for audit pipeline nodes with injected target predictions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from transfer_risk.lib import target_audit
from transfer_risk.pipelines.audit import nodes as audit_nodes


class _FakeModel:
    """Minimal stand-in for a torch module in node tests."""

    def to(self, _device: object) -> _FakeModel:
        """Return self for chaining."""
        return self

    def eval(self) -> _FakeModel:
        """Return self for chaining."""
        return self


def _fake_predict(_model: Any, _tokenizer: Any, texts: list[str], **_kwargs: Any) -> list[int]:
    """Deterministic fake target: benign when text ends with '-b', else injection."""
    return [0 if text.endswith("-b") else 1 for text in texts]


def _legacy_finalization_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, list[str]],
    dict[str, Any],
]:
    surrogates = [f"s{index}" for index in range(10)]
    recipes = [f"r{index}" for index in range(5)]
    master = pd.DataFrame(
        [
            {
                "surrogate": surrogate,
                "recipe": recipe,
                "mean_cka": 0.1 + 0.05 * index,
                "dbs": 0.9 - 0.01 * index,
            }
            for index, surrogate in enumerate(surrogates)
            for recipe in recipes
        ]
    )
    cka_matrices = {
        surrogate: np.array(
            [
                [0.1 + 0.01 * index, 0.2, 0.3],
                [0.4, 0.5, 0.6 + 0.01 * index],
            ]
        )
        for index, surrogate in enumerate(surrogates)
    }
    known_source = target_audit.KNOWN_TARGET_TRAINING_SOURCE
    records = [
        {
            "surrogate": "s0",
            "recipe": "r0",
            "source": "deepset/prompt-injections",
            "original_pred": 1,
            "perturbed_pred": 0,
        },
        {
            "surrogate": "s0",
            "recipe": "r0",
            "source": known_source,
            "original_pred": 1,
            "perturbed_pred": 0,
        },
        {
            "surrogate": "s0",
            "recipe": "r1",
            "source": "deepset/prompt-injections",
            "original_pred": 1,
            "perturbed_pred": 1,
        },
        {
            "surrogate": "s1",
            "recipe": "r0",
            "source": "deepset/prompt-injections",
            "original_pred": 0,
            "perturbed_pred": 0,
        },
        {
            "surrogate": "s1",
            "recipe": "r0",
            "source": known_source,
            "original_pred": 1,
            "perturbed_pred": 0,
        },
        {
            "surrogate": "s1",
            "recipe": "r1",
            "source": "deepset/prompt-injections",
            "original_pred": 1,
            "perturbed_pred": 0,
        },
    ]
    selection = {"M1": ["s0"], "M2": ["s1"]}
    risk_params = {"ablation": {"n_permutations": 10, "alpha": 0.05}}
    legacy_cells = target_audit.build_full_grid_cells(
        records,
        master,
        benign_label=0,
        injection_label=1,
    )
    legacy_summary = target_audit.build_summary_from_records(
        records,
        legacy_cells,
        selection,
        risk_params,
        seed=47,
        excluded_source=known_source,
        benign_label=0,
        injection_label=1,
        unique_text_stats={"unique_originals": 4, "unique_perturbations": 6, "unique_texts": 10},
        baseline_counts={"target_original_benign": 1, "target_original_injection": 3},
    )
    raw_sources = target_audit.aggregate_by_source(
        records,
        benign_label=0,
        injection_label=1,
    )
    return (
        legacy_cells,
        raw_sources,
        legacy_summary,
        master,
        cka_matrices,
        selection,
        risk_params,
    )


def test_finalize_target_audit_recomputes_dbs_without_changing_target_outcomes() -> None:
    """Rebuild the 50-cell audit from saved aggregates and corrected CKA matrices."""
    (
        legacy_cells,
        raw_sources,
        legacy_context,
        master,
        cka_matrices,
        selection,
        risk_params,
    ) = _legacy_finalization_inputs()

    cells, summary = audit_nodes.finalize_target_audit(
        raw_cells=legacy_cells,
        raw_sources=raw_sources,
        raw_context=legacy_context,
        master_results_table=master,
        cka_matrices=cka_matrices,
        similarity_params={"dbs": {"box": 0}},
        surrogate_selection=selection,
        risk_params=risk_params,
        seed=47,
    )

    count_rate_columns = [
        "surrogate",
        "recipe",
        "source_successful",
        "target_original_benign",
        "target_original_injection",
        "target_perturbed_benign",
        "true_target_flips",
        "conditional_target_benign_rate",
        "true_target_flip_rate",
    ]
    pd.testing.assert_frame_equal(
        cells.loc[:, count_rate_columns],
        legacy_cells.loc[:, count_rate_columns],
        check_dtype=False,
    )
    assert len(cells) == 50
    assert cells.loc[cells["surrogate"] == "s0", "dbs"].unique() == pytest.approx([0.3])
    assert (
        cells.loc[cells["surrogate"] == "s0", "dbs"].iloc[0]
        != legacy_cells.loc[legacy_cells["surrogate"] == "s0", "dbs"].iloc[0]
    )

    for cohort_name in ("full_cohort", "known_source_excluded_sensitivity"):
        expected = legacy_context[cohort_name]
        observed = summary[cohort_name]
        for field in (
            "source_successful",
            "target_original_benign",
            "target_original_injection",
            "target_perturbed_benign",
            "true_target_flips",
            "conditional_target_benign_rate",
            "true_target_flip_rate",
        ):
            assert observed[field] == expected[field]
        for outcome in ("conditional_target_benign_rate", "true_target_flip_rate"):
            associations = observed["cka_dbs_association"][outcome]
            assert set(associations) == {"mean_cka", "dbs"}
            assert associations["mean_cka"] == expected["cka_dbs_association"][outcome]["mean_cka"]
            assert associations["dbs"]["exchangeability_null"]["rho"] == pytest.approx(1.0)
            assert (
                associations["dbs"]["exchangeability_null"]["rho"]
                != expected["cka_dbs_association"][outcome]["dbs"]["exchangeability_null"]["rho"]
            )

    excluded = summary["known_source_excluded_sensitivity"]
    assert excluded["source_successful"] == 4
    assert excluded["true_target_flips"] == 2
    assert excluded["surrogate_macro"]["row_count"] == 10


def test_finalize_target_audit_rejects_unreconciled_source_aggregates() -> None:
    """Stop finalization before publishing when source rollups disagree with full cells."""
    (
        legacy_cells,
        raw_sources,
        legacy_context,
        master,
        cka_matrices,
        selection,
        risk_params,
    ) = _legacy_finalization_inputs()
    mismatched_sources = raw_sources.copy()
    mismatched_sources.loc[0, "source_successful"] += 1

    with pytest.raises(ValueError, match="source aggregate rollups"):
        audit_nodes.finalize_target_audit(
            raw_cells=legacy_cells,
            raw_sources=mismatched_sources,
            raw_context=legacy_context,
            master_results_table=master,
            cka_matrices=cka_matrices,
            similarity_params={"dbs": {"box": 0}},
            surrogate_selection=selection,
            risk_params=risk_params,
            seed=47,
        )


def test_run_target_audit_uses_injected_predictions_without_raw_text_outputs() -> None:
    """Build complete aggregates from injected predictions without exposing prompt text."""
    alpha = "audit-secret-alpha-b"
    alpha_perturbed = "audit-secret-alpha-pert-b"
    beta = "audit-secret-beta"
    beta_perturbed = "audit-secret-beta-pert-b"
    adversarial_examples = {
        "s1__r1": [
            {
                "original": alpha,
                "perturbed": alpha_perturbed,
                "success": True,
            },
            {
                "original": beta,
                "perturbed": beta_perturbed,
                "success": True,
            },
        ]
    }
    task_splits = {
        "test": pd.DataFrame(
            [
                {"text": alpha, "label": 1, "source": "deepset/prompt-injections"},
                {"text": beta, "label": 1, "source": "jackhhao/jailbreak-classification"},
            ]
        )
    }
    master = pd.DataFrame(
        [
            {"surrogate": "s1", "recipe": "r1", "mean_cka": 0.5, "dbs": 0.4},
            {"surrogate": "s1", "recipe": "r2", "mean_cka": 0.5, "dbs": 0.4},
        ]
    )
    cells, sources, summary = audit_nodes.run_target_audit(
        adversarial_examples=adversarial_examples,
        target={"model": _FakeModel(), "tokenizer": object()},
        task_splits=task_splits,
        master_results_table=master,
        transfer_params={"batch_size": 8, "max_seq_len": 32, "benign_label": 0},
        audit_params={
            "eval_set_size": 2,
            "max_prompt_chars": 3200,
            "excluded_training_source": "jackhhao/jailbreak-classification",
        },
        device_params={"policy": "cpu"},
        predict_fn=_fake_predict,
    )

    attacked_cell = cells.loc[cells["recipe"] == "r1"].iloc[0]
    zero_success_cell = cells.loc[cells["recipe"] == "r2"].iloc[0]
    assert len(cells) == 2
    assert attacked_cell["true_target_flips"] == 1
    assert attacked_cell["target_original_injection"] == 1
    assert zero_success_cell["source_successful"] == 0
    assert zero_success_cell["conditional_target_benign_rate"] is None
    assert zero_success_cell["true_target_flip_rate"] is None
    assert summary["excluded_source"] == "jackhhao/jailbreak-classification"
    assert summary["unique_text_counts"] == {
        "unique_originals": 2,
        "unique_perturbations": 2,
        "unique_texts": 4,
    }
    final_cells, final_summary = audit_nodes.finalize_target_audit(
        raw_cells=cells,
        raw_sources=sources,
        raw_context=summary,
        master_results_table=master,
        cka_matrices={"s1": np.array([[0.9, 0.1], [0.2, 0.7]])},
        similarity_params={"dbs": {"box": 0}},
        surrogate_selection={"M1": ["s1"], "M2": []},
        risk_params={"ablation": {"n_permutations": 10, "alpha": 0.05}},
        seed=3,
    )
    assert final_cells["dbs"].tolist() == pytest.approx([0.8, 0.8])
    assert final_summary["full_cohort"]["true_target_flips"] == 1
    assert final_summary["known_source_excluded_sensitivity"]["true_target_flips"] == 0
    for output in (
        cells.to_dict("records"),
        sources.to_dict("records"),
        summary,
        final_cells.to_dict("records"),
        final_summary,
    ):
        serialized = json.dumps(output, allow_nan=False)
        for raw_text in (alpha, alpha_perturbed, beta, beta_perturbed):
            assert raw_text not in serialized
