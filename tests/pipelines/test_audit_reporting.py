"""Tests for target-free public target-audit reporting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

mpl.use("Agg")

from transfer_risk.pipelines.audit_reporting import nodes as audit_reporting_nodes
from transfer_risk.pipelines.audit_reporting.pipeline import create_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RECIPES = ("bae", "bert-attack", "deepwordbug", "pwws", "textfooler")
_M1 = (
    "deberta-base-ft-seed",
    "deepset-deberta-injection",
    "deberta-small-pi-v2",
)
_M2 = (
    "llama-prompt-guard-22m",
    "deberta-base-pi-v1",
    "bilstm-attention",
)
_MACRO_RATES = {
    "bert-base-ft": 0.3500828384159088,
    "bilstm-attention": 0.09967613553320218,
    "deberta-base-ft-seed": 0.5567388639729065,
    "deberta-base-pi-v1": 0.29959568208855697,
    "deberta-small-pi-v2": 0.3280738802301993,
    "deepset-deberta-injection": 0.5352838732149078,
    "electra-small-ft": 0.4711285573595604,
    "llama-prompt-guard-22m": 0.0292927440552601,
    "llama-prompt-guard-86m": 0.1508360920125626,
    "roberta-base-ft": 0.4400717884528677,
}
_MEAN_CKA = {
    "bert-base-ft": 0.4450094194724594,
    "bilstm-attention": 0.25871896606466466,
    "deberta-base-ft-seed": 0.4754217532198447,
    "deberta-base-pi-v1": 0.329011060483958,
    "deberta-small-pi-v2": 0.451372545278504,
    "deepset-deberta-injection": 0.4542644298102016,
    "electra-small-ft": 0.4295596540530011,
    "llama-prompt-guard-22m": 0.4123087022398452,
    "llama-prompt-guard-86m": 0.4169064105029701,
    "roberta-base-ft": 0.4284723991738573,
}


def _assert_annotations_are_inside_axes(
    figure: mpl.figure.Figure,
    labels: set[str],
) -> None:
    """Assert that named point annotations are visible within the plotting axes."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    boxes = [text.get_window_extent(renderer) for text in axis.texts if text.get_text() in labels]
    assert len(boxes) == len(labels)
    for box in boxes:
        assert axis.bbox.contains(box.x0, box.y0)
        assert axis.bbox.contains(box.x1, box.y1)


def _synthetic_audit_cells() -> pd.DataFrame:
    """Build exactly 50 synthetic corrected audit cells, including private-text decoys."""
    rows: list[dict[str, Any]] = []
    for surrogate_index, (surrogate, macro_rate) in enumerate(_MACRO_RATES.items()):
        for recipe_index, recipe in enumerate(_RECIPES):
            true_flip_rate = macro_rate + ((recipe_index - 2) * 0.005)
            rows.append(
                {
                    "surrogate": surrogate,
                    "recipe": recipe,
                    "source_successful": 100,
                    "target_original_benign": 10,
                    "target_original_injection": 90,
                    "target_perturbed_benign": 40,
                    "true_target_flips": round(true_flip_rate * 90),
                    "conditional_target_benign_rate": 0.4,
                    "true_target_flip_rate": true_flip_rate,
                    "mean_cka": _MEAN_CKA[surrogate],
                    "dbs": 0.2 + (surrogate_index * 0.01),
                    "original": "private-original-prompt",
                    "perturbed": "private-perturbed-prompt",
                }
            )
    return pd.DataFrame.from_records(rows)


def _synthetic_audit_summary() -> dict[str, Any]:
    """Build a complete non-text audit summary with finalized true-flip statistics."""
    macro_rows = [
        {
            "surrogate": surrogate,
            "mean_cka": _MEAN_CKA[surrogate],
            "dbs": 0.2 + (index * 0.01),
            "true_target_flip_rate_macro_mean": rate,
        }
        for index, (surrogate, rate) in enumerate(_MACRO_RATES.items())
    ]
    true_flip_association = {
        "mean_cka": {
            "designed_pool": {
                "label": "designed surrogate pool",
                "unit": "one row per surrogate",
                "n_surrogates": 10,
            },
            "exchangeability_null": {
                "rho": 0.8303030303030303,
                "two_sided_p": 0.004710648148148148,
                "n": 10,
                "exact": True,
                "p_value_label": (
                    "exact enumeration under the exchangeability null for the designed pool"
                ),
            },
        }
    }
    true_flip_ablation = {
        "one_sided_exchangeability_null": {
            "m1_mean": 0.4733655391393379,
            "m2_mean": 0.14285485389233976,
            "mean_diff_pp": 33.051068524699815,
            "mean_p_value": 0.05,
            "exact": True,
            "n_m1": 3,
            "n_m2": 3,
            "p_value_label": (
                "exact enumeration under the exchangeability null for the designed M1/M2 pool"
            ),
        }
    }
    return {
        "full_cohort": {
            "source_successful": 4362,
            "target_original_benign": 353,
            "target_original_injection": 4009,
            "target_perturbed_benign": 1398,
            "true_target_flips": 1054,
            "true_target_flip_rate": 0.2629084559740584,
            "surrogate_macro": {
                "analysis_unit": "one row per surrogate",
                "row_count": 10,
                "rows": macro_rows,
            },
            "cka_dbs_association": {"true_target_flip_rate": true_flip_association},
            "selection_ablation": {"true_target_flip_rate": true_flip_ablation},
        },
        "known_source_excluded_sensitivity": {
            "analysis_label": "post-hoc sensitivity analysis excluding one known source",
            "excluded_source": "jackhhao/jailbreak-classification",
            "source_successful": 3384,
            "target_original_benign": 226,
            "target_original_injection": 3158,
            "target_perturbed_benign": 1203,
            "true_target_flips": 982,
            "true_target_flip_rate": 0.3109563014566181,
            "surrogate_macro": {
                "analysis_unit": "one row per surrogate",
                "row_count": 10,
                "rows": macro_rows,
            },
            "cka_dbs_association": {"true_target_flip_rate": true_flip_association},
            "selection_ablation": {"true_target_flip_rate": true_flip_ablation},
        },
        "membership": {
            "M1": list(_M1),
            "M2": list(_M2),
            "basis": "saved original mean CKA membership; audit changes outcomes only",
        },
    }


def _selection() -> dict[str, list[str]]:
    """Return the fixed synthetic M1/M2 membership."""
    return {"M1": list(_M1), "M2": list(_M2)}


def test_public_audit_cells_and_sources_select_only_safe_aggregate_columns() -> None:
    """Publish exactly 50 cells and source rows without private prompt text."""
    cells = _synthetic_audit_cells()
    sources = cells.assign(source="deepset/prompt-injections")

    public_cells = audit_reporting_nodes.build_public_target_audit_cells(cells)
    public_sources = audit_reporting_nodes.build_public_target_audit_sources(sources)

    assert len(public_cells) == 50
    assert list(public_cells.columns) == [
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
    ]
    assert list(public_sources.columns) == [
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
    ]
    serialized = public_cells.to_csv(index=False) + public_sources.to_csv(index=False)
    assert "private-original-prompt" not in serialized
    assert "private-perturbed-prompt" not in serialized


def test_public_audit_summary_is_strict_json_and_rejects_raw_text_fields() -> None:
    """Preserve finalized cohort analysis while rejecting accidental raw-text fields."""
    summary = _synthetic_audit_summary()

    public_summary = audit_reporting_nodes.build_public_target_audit_summary(summary)

    assert public_summary == summary
    assert json.dumps(public_summary, allow_nan=False)
    assert public_summary["full_cohort"]["surrogate_macro"]["row_count"] == 10
    assert (
        public_summary["known_source_excluded_sensitivity"]["analysis_label"]
        == "post-hoc sensitivity analysis excluding one known source"
    )
    assert (
        public_summary["full_cohort"]["cka_dbs_association"]["true_target_flip_rate"]["mean_cka"][
            "exchangeability_null"
        ]["p_value_label"]
        == "exact enumeration under the exchangeability null for the designed pool"
    )
    unsafe = deepcopy(summary)
    unsafe["full_cohort"]["original"] = "private-original-prompt"
    with pytest.raises(ValueError, match="raw-text"):
        audit_reporting_nodes.build_public_target_audit_summary(unsafe)


def test_true_flip_scatter_has_exact_annotation_points_labels_and_legend() -> None:
    """Render recipe and surrogate true-flip rates with exchangeability-qualified evidence."""
    figure = audit_reporting_nodes.plot_true_flip_scatter(
        _synthetic_audit_cells(), _synthetic_audit_summary()
    )
    axis = figure.axes[0]
    annotation_text = " ".join(text.get_text() for text in axis.texts)
    recipe_collections = [
        collection for collection in axis.collections if collection.get_label() in _RECIPES
    ]
    macro = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "surrogate macro true flip rate"
    )

    assert "Spearman rho = 0.83; exact-enumeration two-sided p = 0.0047; n = 10" in annotation_text
    assert "exchangeability null" in " ".join(text.get_text() for text in figure.texts).lower()
    assert "true target flip rate" in axis.get_ylabel().lower()
    assert {collection.get_label() for collection in recipe_collections} == set(_RECIPES)
    assert sum(len(collection.get_offsets()) for collection in recipe_collections) == 50
    assert len(macro.get_offsets()) == 10
    assert {
        "deberta-base-ft-seed",
        "deberta-base-pi-v1",
        "llama-prompt-guard-22m",
        "bilstm-attention",
    } <= {text.get_text() for text in axis.texts}
    _assert_annotations_are_inside_axes(
        figure,
        {
            "deberta-base-ft-seed",
            "deberta-base-pi-v1",
            "llama-prompt-guard-22m",
            "bilstm-attention",
        },
    )
    legend = axis.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {
        *_RECIPES,
        "surrogate macro true flip rate",
    }


def test_true_flip_scatter_keeps_legend_and_scope_note_visible_and_separate() -> None:
    """Keep the bottom legend and exchangeability scope note within distinct canvas regions."""
    figure = audit_reporting_nodes.plot_true_flip_scatter(
        _synthetic_audit_cells(), _synthetic_audit_summary()
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    legend = axis.get_legend()
    scope_note = next(
        text
        for text in figure.texts
        if text.get_text()
        == "Designed surrogate pool; exact p-value assumes an exchangeability null."
    )

    assert legend is not None
    legend_box = legend.get_window_extent(renderer)
    scope_note_box = scope_note.get_window_extent(renderer)
    assert not legend_box.overlaps(scope_note_box)
    for box in (legend_box, scope_note_box):
        assert figure.bbox.contains(box.x0, box.y0)
        assert figure.bbox.contains(box.x1, box.y1)


def test_true_flip_ablation_has_all_points_floor_annotation_and_group_encoding() -> None:
    """Render all six macro true-flip values with exact-floor context and visible groups."""
    figure = audit_reporting_nodes.plot_true_flip_ablation(_synthetic_audit_summary(), _selection())
    axis = figure.axes[0]
    annotation_text = " ".join(text.get_text() for text in axis.texts)
    m1, m2 = axis.collections[:2]

    assert "+33.1 pp" in annotation_text
    assert "exact one-sided p = .05" in annotation_text
    assert "1/20 floor; strict p < .05 rule not met" in annotation_text
    assert len(m1.get_offsets()) + len(m2.get_offsets()) == 6
    assert m1.get_offsets()[:, 1].tolist() == pytest.approx(
        [_MACRO_RATES[surrogate] for surrogate in _M1]
    )
    assert m2.get_offsets()[:, 1].tolist() == pytest.approx(
        [_MACRO_RATES[surrogate] for surrogate in _M2]
    )
    assert tuple(m1.get_facecolors()[0]) != tuple(m2.get_facecolors()[0])
    assert not np.array_equal(m1.get_paths()[0].vertices, m2.get_paths()[0].vertices)
    assert "true target flip rate" in axis.get_ylabel().lower()
    _assert_annotations_are_inside_axes(figure, set(_M1 + _M2))
    legend = axis.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"M1", "M2", "group mean"}
    assert figure.get_constrained_layout()


def test_audit_reporting_pipeline_is_target_free_and_uses_only_saved_aggregates() -> None:
    """Keep publication isolated from target inference, attacks, and normal reporting."""
    pipeline = create_pipeline()

    assert "target_model" not in pipeline.inputs()
    assert set(pipeline.inputs()) == {
        "target_audit_cells",
        "target_audit_raw_sources",
        "target_audit_summary",
        "surrogate_selection",
        "pub_master_results_table",
    }
    assert {node.name for node in pipeline.nodes} == {
        "build_public_target_audit_cells",
        "build_public_target_audit_sources",
        "build_public_target_audit_summary",
        "build_redacted_qualitative_audit",
        "plot_historical_conditional_scatter",
        "plot_true_flip_scatter",
        "plot_true_flip_ablation",
    }
    assert "transferred_examples" not in pipeline.inputs()
    assert "adversarial_examples" not in pipeline.inputs()


def test_audit_reporting_catalog_wraps_every_public_output_for_mlflow() -> None:
    """Persist every public audit table, summary, and figure through MLflow wrappers."""
    loader = OmegaConfigLoader(
        conf_source=str(_PROJECT_ROOT / "conf"),
        base_env="base",
        default_run_env="local",
    )
    catalog = loader["catalog"]
    public_datasets = {
        "pub_target_audit_cells": "pandas.CSVDataset",
        "pub_target_audit_sources": "pandas.CSVDataset",
        "pub_target_audit_summary": "json.JSONDataset",
        "pub_qualitative_examples": "json.JSONDataset",
        "fig_transfer_scatter": "matplotlib.MatplotlibDataset",
        "fig_true_flip_scatter": "matplotlib.MatplotlibDataset",
        "fig_true_flip_ablation": "matplotlib.MatplotlibDataset",
    }

    for name, dataset_type in public_datasets.items():
        entry = catalog[name]
        assert entry["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
        assert entry["dataset"]["type"] == dataset_type
        assert entry["dataset"]["filepath"].startswith("docs/")
        assert entry["artifact_path"] in {"tables", "results", "figures"}
