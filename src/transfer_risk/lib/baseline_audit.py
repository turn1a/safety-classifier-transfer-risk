"""Pure detector-baseline summaries against saved canonical corpus labels.

Canonical positive combines injection and jailbreak material. Counts and metrics
describe agreement with those saved labels; they do not establish semantic ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _binary_labels(labels: Sequence[object]) -> list[int]:
    """Validate strict zero/one integer labels without truth-value or float coercion."""
    result: list[int] = []
    for value in labels:
        if isinstance(value, bool) or not isinstance(value, Integral) or value not in {0, 1}:
            msg = "expected and predicted labels must contain binary integers 0 or 1"
            raise ValueError(msg)
        result.append(int(value))
    return result


def _validate_sources(sources: Sequence[str]) -> None:
    """Require a sequence of nonempty source identifiers without changing their values."""
    if isinstance(sources, str):
        msg = "sources must be a sequence of source identifiers"
        raise ValueError(msg)
    if any(not isinstance(source, str) or not source.strip() for source in sources):
        msg = "each source must be a nonempty string identifier"
        raise ValueError(msg)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a fraction, leaving an absent denominator explicitly undefined."""
    return numerator / denominator if denominator else None


def _summarize_counts(
    expected: Sequence[int], predicted: Sequence[int]
) -> dict[str, int | float | None]:
    """Summarize validated binary labels for one finite saved-data group."""
    count = len(expected)
    canonical_positive = sum(expected)
    canonical_negative = count - canonical_positive
    predicted_positive = sum(predicted)
    predicted_negative = count - predicted_positive
    positive_predicted_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(expected, predicted, strict=True)
    )
    positive_predicted_negative = canonical_positive - positive_predicted_positive
    negative_predicted_positive = predicted_positive - positive_predicted_positive
    negative_predicted_negative = canonical_negative - negative_predicted_positive
    return {
        "record_count": count,
        "canonical_positive_count": canonical_positive,
        "canonical_negative_count": canonical_negative,
        "predicted_positive_count": predicted_positive,
        "predicted_negative_count": predicted_negative,
        "canonical_positive_predicted_positive_count": positive_predicted_positive,
        "canonical_positive_predicted_negative_count": positive_predicted_negative,
        "canonical_negative_predicted_positive_count": negative_predicted_positive,
        "canonical_negative_predicted_negative_count": negative_predicted_negative,
        "canonical_positive_recall": _rate(positive_predicted_positive, canonical_positive),
        "canonical_negative_false_positive_rate": _rate(
            negative_predicted_positive, canonical_negative
        ),
        "canonical_positive_precision": _rate(positive_predicted_positive, predicted_positive),
        "canonical_label_accuracy": _rate(
            positive_predicted_positive + negative_predicted_negative, count
        ),
    }


def summarize_detector_baseline(
    expected_labels: Sequence[int],
    predicted_labels: Sequence[int],
    sources: Sequence[str],
    target_id: str,
) -> dict[str, Any]:
    """Report pooled and source-specific detector agreement with canonical labels.

    Args:
        expected_labels: Saved integer corpus labels, zero for canonical negative and
            one for canonical positive. Positive labels mix injection and jailbreak examples.
        predicted_labels: Detector predictions using the same integer zero/one mapping.
        sources: Original source identifier for each example, preserved exactly.
        target_id: Nonempty identity of the evaluated detector.

    Returns:
        JSON-compatible pooled and source-specific confusion counts, canonical-positive
        recall and precision, canonical-negative false-positive rate, and canonical-label
        accuracy. Undefined denominators produce ``None``, including precision when no
        positive predictions exist. Empty input has zero counts and undefined metrics.

    Raises:
        ValueError: If lengths differ, a label is not an integer zero/one, a source is
            missing, or the detector identity is missing. Booleans and floats are rejected.
    """
    if not isinstance(target_id, str) or not target_id.strip():
        msg = "target_id must be a nonempty string"
        raise ValueError(msg)
    if not len(expected_labels) == len(predicted_labels) == len(sources):
        msg = "expected_labels, predicted_labels and sources must have the same length"
        raise ValueError(msg)
    _validate_sources(sources)
    expected, predicted = _binary_labels(expected_labels), _binary_labels(predicted_labels)
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for label, prediction, source in zip(expected, predicted, sources, strict=True):
        grouped[source].append((label, prediction))
    return {
        "schema_version": "baseline-audit-v1",
        "target_id": target_id,
        "label_values": {"canonical_negative": 0, "canonical_positive": 1},
        "label_scope": (
            "Canonical positive combines injection and jailbreak material. These metrics "
            "measure agreement with saved corpus labels; independent semantic adjudication "
            "has not been supplied by this summary."
        ),
        "pooled": _summarize_counts(expected, predicted),
        "by_source": [
            {
                "source": source,
                **_summarize_counts(
                    [labels[0] for labels in grouped[source]],
                    [labels[1] for labels in grouped[source]],
                ),
            }
            for source in sorted(grouped)
        ],
    }
