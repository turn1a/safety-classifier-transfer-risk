"""Focused tests for reporting pipeline nodes and figures."""

from __future__ import annotations

import inspect
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

mpl.use("Agg")

from transfer_risk.pipelines.reporting import nodes as reporting_nodes
from transfer_risk.pipelines.reporting.pipeline import create_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_ARTIFACTS = _PROJECT_ROOT / "docs" / "artifacts"


def _public_master() -> pd.DataFrame:
    """Load the checked-in public master table."""
    return pd.read_csv(_PUBLIC_ARTIFACTS / "master_results_table.csv")


def _public_metrics() -> dict[str, object]:
    """Load the checked-in public run metrics."""
    return json.loads((_PUBLIC_ARTIFACTS / "run_metrics.json").read_text())


def _public_selection() -> dict[str, list[str]]:
    """Derive M1/M2 membership from the checked-in public surrogate summary."""
    summary = pd.read_csv(_PUBLIC_ARTIFACTS / "surrogate_summary.csv")
    return {
        group: summary.loc[summary["similarity_group"] == group, "surrogate"].tolist()
        for group in ("M1", "M2")
    }


def _synthetic_master() -> pd.DataFrame:
    """Return a two-row in-memory master table for node behavior tests."""
    return pd.DataFrame(
        [
            {
                "surrogate": "alpha",
                "recipe": "bae",
                "n_successful": 10,
                "transfer_rate": 0.2,
                "mean_cka": 0.8,
                "dbs": 0.1,
                "target": "target/model",
            },
            {
                "surrogate": "beta",
                "recipe": "bae",
                "n_successful": 20,
                "transfer_rate": 0.25,
                "mean_cka": 0.3,
                "dbs": 0.9,
                "target": "target/model",
            },
        ]
    )


def _assert_annotations_do_not_overlap_or_clip(
    fig: mpl.figure.Figure,
    labels: set[str],
) -> None:
    """Assert selected annotation boxes are inside the canvas and pairwise disjoint."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        text.get_window_extent(renderer) for text in fig.axes[0].texts if text.get_text() in labels
    ]
    assert len(boxes) == len(labels)
    axes_box = fig.axes[0].bbox
    for box in boxes:
        assert axes_box.contains(box.x0, box.y0)
        assert axes_box.contains(box.x1, box.y1)
    assert all(not left.overlaps(right) for left, right in combinations(boxes, 2))


def test_recompute_master_dbs_changes_values_from_in_memory_matrices() -> None:
    original = _synthetic_master()
    matrices = {
        "alpha": np.array([[1.0, 0.2], [0.3, 0.8]]),
        "beta": np.array([[0.9, 0.1], [0.4, 0.7]]),
    }
    corrected = reporting_nodes.recompute_master_dbs(original, matrices, {"dbs": {"box": 0}})
    assert len(corrected) == 2
    assert "n_target_benign" in corrected.columns
    assert not corrected["dbs"].equals(original["dbs"])
    assert corrected["dbs"].tolist() == pytest.approx([0.9, 0.8])
    assert corrected["n_target_benign"].tolist() == [2, 5]


def test_export_public_master_table_has_expected_columns() -> None:
    master = _synthetic_master()
    master["n_target_benign"] = pd.Series([2, 5], dtype=int)
    table = reporting_nodes.build_public_master_results_table(master)
    assert len(table) == 2
    assert list(table.columns) == [
        "surrogate",
        "recipe",
        "n_successful",
        "n_target_benign",
        "transfer_rate",
        "mean_cka",
        "dbs",
        "target",
    ]
    assert table["n_target_benign"].dtype == int


def test_build_surrogate_summary_table_has_ten_rows() -> None:
    summary = reporting_nodes.build_surrogate_summary_table(_public_master(), _public_selection())
    assert len(summary) == 10
    assert set(summary["similarity_group"]) <= {"M1", "M2", "middle"}


def test_plot_transfer_scatter_uses_recipe_shapes_and_selected_annotations() -> None:
    master = _public_master()
    fig = reporting_nodes.plot_transfer_scatter(master)
    ax = fig.axes[0]
    annotation_text = " ".join(child.get_text() for child in ax.texts)
    assert "Spearman rho = 0.76" in annotation_text
    assert "exact two-sided p = 0.015" in annotation_text
    assert "n = 10" in annotation_text
    assert ax.get_title() == "Historical conditional target-benign rate vs mean CKA"
    assert ax.get_ylabel() == "Historical conditional target-benign rate"
    scope_note = " ".join(text.get_text() for text in fig.texts)
    assert "historical conditional target-benign rate" in scope_note.lower()
    assert "designed surrogate pool" in scope_note.lower()
    assert "exchangeability null" in scope_note.lower()
    assert "target flip" not in (annotation_text + scope_note).lower()
    recipes = set(master["recipe"])
    recipe_collections = [item for item in ax.collections if item.get_label() in recipes]
    assert {item.get_label() for item in recipe_collections} == recipes
    colors = {tuple(item.get_facecolors()[0]) for item in recipe_collections}
    markers = {tuple(item.get_paths()[0].vertices.round(4).ravel()) for item in recipe_collections}
    assert len(colors) == len(recipes)
    assert len(markers) == len(recipes)
    important = {
        "deberta-base-ft-seed",
        "deberta-base-pi-v1",
        "llama-prompt-guard-22m",
        "bilstm-attention",
    }
    surrogate_names = set(master["surrogate"])
    labels = {text.get_text() for text in ax.texts} & surrogate_names
    assert labels == important
    offsets = {text.get_position() for text in ax.texts if text.get_text() in important}
    assert len(offsets) == len(important)
    _assert_annotations_do_not_overlap_or_clip(fig, important)
    macro = next(
        item
        for item in ax.collections
        if item.get_label() == "surrogate macro historical conditional rate"
    )
    assert min(macro.get_sizes()) > max(item.get_sizes()[0] for item in recipe_collections)
    legend = ax.get_legend()
    assert legend is not None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    important_boxes = [
        text.get_window_extent(renderer) for text in ax.texts if text.get_text() in important
    ]
    assert all(not box.overlaps(legend_box) for box in important_boxes)
    scope = next(
        text for text in fig.texts if "Historical conditional target-benign rate" in text.get_text()
    )
    scope_box = scope.get_window_extent(renderer)
    assert fig.bbox.contains(scope_box.x0, scope_box.y0)
    assert fig.bbox.contains(scope_box.x1, scope_box.y1)
    assert not scope_box.overlaps(legend_box)


def test_plot_cka_ranked_applies_both_thresholds_only_to_mean_cka() -> None:
    master = _public_master()
    metrics = _public_metrics()
    thresholds = metrics["thresholds"]
    fig = reporting_nodes.plot_cka_ranked(master, _public_selection(), thresholds)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].get_yticklabels()) == 10
    cka_thresholds = sorted(float(line.get_xdata()[0]) for line in fig.axes[0].lines)
    assert cka_thresholds == pytest.approx(sorted(thresholds.values()))
    assert not fig.axes[1].lines
    assert {"r1", "r2"} <= set(fig.axes[0].get_title().split())
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"M1", "middle", "M2"}
    assert len({patch.get_hatch() for patch in fig.axes[0].patches}) == 3


def test_build_qualitative_examples_is_target_free_and_redacted() -> None:
    selected = reporting_nodes.build_qualitative_examples()
    assert len(selected) == 2
    labels = {item["label"] for item in selected}
    assert labels == {"semantic_preservation_uncertain", "meaning_changed"}
    assert all(
        set(item)
        == {"surrogate", "recipe", "n_words_changed", "label", "change_summary", "audit_note"}
        for item in selected
    )
    assert all("original" not in item and "perturbed" not in item for item in selected)


def test_build_public_run_metrics_node_matches_saved_headline_values() -> None:
    master = _public_master()
    frozen = _public_metrics()
    adversarial_examples = {
        "alpha__bae": [
            {"result_type": "SuccessfulAttackResult"},
            {"result_type": "FailedAttackResult"},
            {"result_type": "SkippedAttackResult"},
        ]
    }
    metrics = reporting_nodes.build_public_run_metrics_bundle(
        master,
        adversarial_examples,
        frozen["ablation"],
        frozen["thresholds"],
        {"eval_set_size": 191, "query_budget": 6000},
    )
    assert metrics["transfer"]["macro_cell_mean"] == pytest.approx(0.38216487917614733)
    assert metrics["associations"]["surrogate"]["mean_cka"]["n"] == 10
    assert metrics["attack_outcomes"]["attempted"] == 3


def test_reporting_catalog_wraps_public_outputs_for_mlflow() -> None:
    loader = OmegaConfigLoader(
        conf_source=str(_PROJECT_ROOT / "conf"),
        base_env="base",
        default_run_env="local",
    )
    catalog = loader["catalog"]
    public_datasets = {
        "pub_master_results_table": "pandas.CSVDataset",
        "pub_surrogate_summary": "pandas.CSVDataset",
        "pub_dataset_audit": "json.JSONDataset",
        "pub_model_validation_summary": "json.JSONDataset",
        "pub_run_metrics": "json.JSONDataset",
        "pub_results_manifest": "json.JSONDataset",
        "pub_qualitative_examples": "json.JSONDataset",
        "fig_cka_ranked": "matplotlib.MatplotlibDataset",
        "fig_transfer_scatter": "matplotlib.MatplotlibDataset",
    }
    for name, dataset_type in public_datasets.items():
        entry = catalog[name]
        assert entry["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
        assert entry["dataset"]["type"] == dataset_type
        assert entry["dataset"]["filepath"].startswith("docs/")
        assert entry["artifact_path"] in {"tables", "results", "figures"}
    assert "master_results_corrected" not in catalog
    assert "surrogate_metadata_bundle" not in catalog
    assert "fig_selection_ablation" not in catalog


def test_manifest_node_does_not_accept_corrected_master() -> None:
    parameters = inspect.signature(reporting_nodes.build_results_manifest_node).parameters
    assert "master_results_corrected" not in parameters


def test_reporting_wires_adversarial_examples_only_to_run_metrics() -> None:
    pipeline = create_pipeline()
    consumers = [item.name for item in pipeline.nodes if "adversarial_examples" in item.inputs]
    assert consumers == ["build_public_run_metrics"]
    manifest = next(item for item in pipeline.nodes if item.name == "build_results_manifest")
    assert "master_results_corrected" not in manifest.inputs
    qualitative = next(item for item in pipeline.nodes if item.name == "build_qualitative_examples")
    assert qualitative.inputs == []
    assert "transferred_examples" not in pipeline.inputs()
