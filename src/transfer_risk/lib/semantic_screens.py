"""Summarize computational text-pair screens without inferring attack validity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

_NLI_LABELS = {"entailment", "contradiction", "neutral"}
_SCORES = ("embedding_cosine", "nli_min_entailment", "nli_max_contradiction")
_NLI_CLASSES = 3
_MATRIX_DIMENSIONS = 2


def cosine_pairs(left: npt.ArrayLike, right: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute aligned embedding cosines after validating finite nonzero vectors.

    Args:
        left: Two-dimensional original embedding matrix.
        right: Matching perturbed embedding matrix.

    Returns:
        One bounded cosine per pair.

    Raises:
        ValueError: If shapes differ, vectors are empty, nonfinite, or zero.
    """
    first, second = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if first.ndim != _MATRIX_DIMENSIONS or first.shape != second.shape or not first.size:
        msg = "embedding matrices must have the same nonempty two-dimensional shape"
        raise ValueError(msg)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        msg = "embedding vectors must be finite"
        raise ValueError(msg)
    norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    if np.any(norms <= 0):
        msg = "embedding cosine is undefined for zero vectors"
        raise ValueError(msg)
    return np.asarray(np.clip(np.sum(first * second, axis=1) / norms, -1.0, 1.0), dtype=np.float64)


def entailment_probabilities(
    logits: npt.ArrayLike, label_map: Mapping[int, str]
) -> dict[str, npt.NDArray[np.float64]]:
    """Map stable NLI softmax probabilities through an explicit three-class map.

    Args:
        logits: Finite pair-by-class model logits.
        label_map: Exact model class identifiers for entailment, contradiction and neutral.

    Returns:
        Probability vectors keyed by declared class names.

    Raises:
        ValueError: If logits or the class map are invalid.
    """
    if set(label_map) != set(range(_NLI_CLASSES)) or set(label_map.values()) != _NLI_LABELS:
        msg = "NLI needs explicit entailment, contradiction and neutral classes"
        raise ValueError(msg)
    values = np.asarray(logits, dtype=float)
    if (
        values.ndim != _MATRIX_DIMENSIONS
        or values.shape[1] != _NLI_CLASSES
        or not values.size
        or not np.isfinite(values).all()
    ):
        msg = "NLI logits must be a finite pair-by-three matrix"
        raise ValueError(msg)
    probabilities = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {label: probabilities[:, index] for index, label in label_map.items()}


def summarize_semantic_screens(
    scores: pd.DataFrame, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish sample score distributions and truncation counts without pair identifiers.

    Args:
        scores: Private scores with one row per sampled exact pair.
        provenance: Model revisions and inference settings, without corpus text.

    Returns:
        Aggregate diagnostic report with no validity threshold or inferred labels.

    Raises:
        ValueError: If pairs, scores, or coverage flags are invalid.
    """
    required = {"pair_id", "original_id", *_SCORES, "embedding_truncated", "nli_truncated"}
    if not required.issubset(scores.columns) or scores.empty:
        msg = "semantic screens require a nonempty complete score table"
        raise ValueError(msg)
    if scores["pair_id"].duplicated().any() or scores["pair_id"].isna().any():
        msg = "semantic screen pairs must be unique and nonmissing"
        raise ValueError(msg)
    distributions: dict[str, Any] = {}
    for name in _SCORES:
        values = scores[name].to_numpy(dtype=float)
        lower = -1 if name == "embedding_cosine" else 0
        if not np.isfinite(values).all() or np.any(values < lower) or np.any(values > 1):
            msg = f"{name} scores must be finite and within [{lower}, 1]"
            raise ValueError(msg)
        distributions[name] = {
            "minimum": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "maximum": float(values.max()),
        }
    for flag in ("embedding_truncated", "nli_truncated"):
        if not scores[flag].isin([True, False]).all():
            msg = "screen truncation flags must be boolean"
            raise ValueError(msg)
    return {
        "analysis_status": "post_hoc_computational_sample_screen",
        "sample_pairs": len(scores),
        "unique_originals": int(scores["original_id"].nunique()),
        "coverage": {
            "embedding_truncated_pairs": int(scores["embedding_truncated"].sum()),
            "nli_truncated_pairs": int(scores["nli_truncated"].sum()),
        },
        "scores": distributions,
        "provenance": dict(provenance),
        "semantic_validity_assessed": False,
        "injection_intent_assessed": False,
        "downstream_effectiveness_assessed": False,
        "human_reviews_used": False,
        "limitations": [
            "Distributions describe the stratified review sample, without population weighting.",
            "Embedding similarity and NLI probabilities have no validated intent threshold here.",
            "Truncated inputs can omit changed instructions or destinations.",
            "Scores are kept separate from the blinded human-review interface.",
        ],
    }
