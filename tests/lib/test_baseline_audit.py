"""Tests for canonical-label detector baseline counts and denominator handling."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from transfer_risk.lib.baseline_audit import summarize_detector_baseline


def test_pooled_and_source_metrics_preserve_canonical_label_denominators() -> None:
    summary = summarize_detector_baseline(
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        ["source-a", "source-a", "source-a", "source-a", "source-b", "source-b"],
        "archived-detector",
    )
    pooled = summary["pooled"]

    assert summary["target_id"] == "archived-detector"
    assert summary["label_values"] == {"canonical_negative": 0, "canonical_positive": 1}
    assert pooled["record_count"] == 6
    assert pooled["canonical_positive_count"] == 3
    assert pooled["canonical_negative_count"] == 3
    assert pooled["predicted_positive_count"] == 3
    assert pooled["predicted_negative_count"] == 3
    assert pooled["canonical_positive_predicted_positive_count"] == 2
    assert pooled["canonical_positive_predicted_negative_count"] == 1
    assert pooled["canonical_negative_predicted_positive_count"] == 1
    assert pooled["canonical_negative_predicted_negative_count"] == 2
    assert pooled["canonical_positive_recall"] == pytest.approx(2 / 3)
    assert pooled["canonical_negative_false_positive_rate"] == pytest.approx(1 / 3)
    assert pooled["canonical_positive_precision"] == pytest.approx(2 / 3)
    assert pooled["canonical_label_accuracy"] == pytest.approx(4 / 6)
    by_source = {row["source"]: row for row in summary["by_source"]}
    assert by_source["source-a"]["record_count"] == 4
    assert by_source["source-a"]["canonical_positive_recall"] == 0.5
    assert by_source["source-a"]["canonical_negative_false_positive_rate"] == 0.5
    assert by_source["source-a"]["canonical_positive_precision"] == 0.5
    assert by_source["source-a"]["canonical_label_accuracy"] == 0.5
    assert by_source["source-b"]["canonical_positive_recall"] == 1.0
    assert by_source["source-b"]["canonical_negative_false_positive_rate"] == 0.0
    assert by_source["source-b"]["canonical_label_accuracy"] == 1.0


def test_no_predicted_positive_has_null_precision() -> None:
    pooled = summarize_detector_baseline([1, 0], [0, 0], ["a", "a"], "detector")["pooled"]

    assert pooled["canonical_positive_recall"] == 0.0
    assert pooled["canonical_negative_false_positive_rate"] == 0.0
    assert pooled["canonical_positive_precision"] is None
    assert pooled["canonical_label_accuracy"] == 0.5


def test_single_canonical_class_has_null_absent_class_denominator() -> None:
    positive_only = summarize_detector_baseline([1, 1], [1, 0], ["a", "a"], "detector")["pooled"]
    negative_only = summarize_detector_baseline([0, 0], [1, 0], ["a", "a"], "detector")["pooled"]

    assert positive_only["canonical_positive_recall"] == 0.5
    assert positive_only["canonical_negative_false_positive_rate"] is None
    assert positive_only["canonical_positive_precision"] == 1.0
    assert negative_only["canonical_positive_recall"] is None
    assert negative_only["canonical_negative_false_positive_rate"] == 0.5
    assert negative_only["canonical_positive_precision"] == 0.0


def test_empty_input_has_zero_counts_and_no_defined_rates() -> None:
    summary = summarize_detector_baseline([], [], [], "detector")

    assert summary["pooled"]["record_count"] == 0
    assert summary["by_source"] == []
    for field in (
        "canonical_positive_recall",
        "canonical_negative_false_positive_rate",
        "canonical_positive_precision",
        "canonical_label_accuracy",
    ):
        assert summary["pooled"][field] is None


@pytest.mark.parametrize(
    ("expected", "predicted", "sources"),
    [([1], [], ["a"]), ([], [1], ["a"]), ([1], [1], []), ([1], [1], ["a", "b"])],
)
def test_lengths_must_match_before_aggregation(
    expected: list[int], predicted: list[int], sources: list[str]
) -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_detector_baseline(expected, predicted, sources, "detector")


@pytest.mark.parametrize("invalid", [True, False, 0.0, 1.0, "0", "1", None, -1, 2, np.bool_(1)])
@pytest.mark.parametrize("label_field", ["expected", "predicted"])
def test_binary_integer_labels_reject_coercion(invalid: Any, label_field: str) -> None:
    expected, predicted = [0], [0]
    if label_field == "expected":
        expected = [invalid]
    else:
        predicted = [invalid]
    with pytest.raises(ValueError, match="binary integers"):
        summarize_detector_baseline(expected, predicted, ["source-a"], "detector")


def test_numpy_integer_inputs_produce_json_serializable_builtin_values() -> None:
    summary = summarize_detector_baseline(
        [np.int64(1), np.int32(0)], [np.int32(0), np.int64(0)], ["a", "b"], "detector"
    )

    assert summary["pooled"]["record_count"] == 2
    assert json.loads(json.dumps(summary))["pooled"]["canonical_label_accuracy"] == 0.5


@pytest.mark.parametrize("source", [None, 0, "", " \t"])
def test_missing_source_identity_is_rejected(source: Any) -> None:
    with pytest.raises(ValueError, match="source"):
        summarize_detector_baseline([1], [1], [source], "detector")


def test_source_sequence_cannot_be_one_bare_string() -> None:
    with pytest.raises(ValueError, match="source"):
        summarize_detector_baseline([1], [1], "a", "detector")


@pytest.mark.parametrize("target_id", [None, 0, "", " \t"])
def test_target_identifier_must_be_present(target_id: Any) -> None:
    with pytest.raises(ValueError, match="target_id"):
        summarize_detector_baseline([1], [1], ["a"], target_id)


def test_source_names_are_preserved_and_source_counts_sum_to_the_pooled_counts() -> None:
    summary = summarize_detector_baseline(
        [1, 0, 1], [0, 1, 1], ["Source-A", "source-a", "Source-A"], "detector"
    )
    by_source = summary["by_source"]

    assert {row["source"] for row in by_source} == {"Source-A", "source-a"}
    for field in (
        "record_count",
        "canonical_positive_count",
        "canonical_negative_count",
        "predicted_positive_count",
        "predicted_negative_count",
        "canonical_positive_predicted_positive_count",
        "canonical_positive_predicted_negative_count",
        "canonical_negative_predicted_positive_count",
        "canonical_negative_predicted_negative_count",
    ):
        assert sum(row[field] for row in by_source) == summary["pooled"][field]
