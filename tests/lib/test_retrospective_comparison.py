import json
from pathlib import Path

import pandas as pd
import pytest

from transfer_risk.lib.retrospective_audit import reference_cells_digest
from transfer_risk.lib.retrospective_comparison import compare_saved_candidate_policies

_ROOT = Path(__file__).resolve().parents[2]


def _evidence() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    frames = {}
    reports = []
    for target, models in (("reference", ["a", "b", "c"]), ("other", ["a", "b"])):
        rows = []
        for index, model in enumerate(models):
            rows.append(
                {
                    "target": target,
                    "surrogate": model,
                    "recipe": "recipe",
                    "attempted": 10,
                    "source_successful": 5,
                    "source_failed": 4,
                    "source_skipped": 1,
                    "source_other": 0,
                    "target_original_positive": 8,
                    "target_original_benign": 2,
                    "target_original_positive_successful": 4,
                    "verified_flips_source_successful": index + 1,
                    "verified_flips_source_failed": 1,
                    "verified_flips_source_skipped": 0,
                    "verified_flips_source_other": 0,
                    "verified_flips_all_candidates": index + 2,
                    "originally_benign_candidates_benign": 2,
                }
            )
        frames[target] = pd.DataFrame(rows)
        reports.append(
            {
                "target": target,
                "included_surrogates": models,
                "excluded_surrogates": ["c"] if target == "other" else [],
                "unique_originals": 10,
                "unique_original_positive": 8,
                "unique_original_benign": 2,
                "aggregate_counts": frames[target].select_dtypes("number").sum().to_dict(),
                "inference_weights_sha256": "a" * 64,
                "tokenizer_vocab_sha256": "b" * 64,
                "loaded_hub_revision": None,
                "inference_dtype": "float32",
                "max_seq_len": 512,
                "device": "mps",
                "provenance_scope": "current_local_audit_only",
                "positive_label": 1,
                "benign_label": 0,
            }
        )
    similarity = pd.DataFrame({"surrogate": ["a", "b", "c"], "mean_cka": [0.1, 0.2, 0.3]})
    return (
        frames,
        similarity,
        {
            "reports": reports,
            "reference_check": {
                "counts_agree": True,
                "reference_cells_sha256": reference_cells_digest(frames["reference"]),
                "retained_surrogates": ["a", "b", "c"],
                "checked_cells": 3,
                "aggregate_counts": {
                    "attempted": 30,
                    "source_successful": 15,
                    "target_original_positive_successful": 12,
                    "verified_flips_source_successful": 6,
                },
            },
        },
    )


def _baseline(context: dict) -> dict:
    """Provide independently saved clean-input identities and canonical-positive counts."""
    reports = []
    for report in context["reports"]:
        reports.append(
            {
                "target_id": report["target"],
                "inference_weights_sha256": "a" * 64,
                "tokenizer_vocab_sha256": "b" * 64,
                "loaded_hub_revision": None,
                "inference_dtype": "float32",
                "max_seq_len": 512,
                "device": "mps",
                "provenance_scope": "current_local_audit_only",
                "label_values": {"canonical_positive": 1, "canonical_negative": 0},
                "pooled": {
                    "canonical_positive_count": 10,
                    "canonical_positive_predicted_positive_count": 8,
                    "canonical_positive_predicted_negative_count": 2,
                },
            }
        )
    return {"reports": reports}


def test_common_pool_uses_same_sources_and_counts_failed_candidates() -> None:
    frames, similarity, context = _evidence()
    report = compare_saved_candidate_policies(
        frames,
        similarity,
        context,
        attempts_per_cell=10,
        recipes=["recipe"],
        reference_target="reference",
        baseline_context=_baseline(context),
    )
    assert report["common_surrogates"] == ["a", "b"]
    assert report["common_pool"][0]["attempted"] == 20
    assert report["common_pool"][0]["source_gated_flips"] == 3
    assert report["common_pool"][0]["all_candidate_flips"] == 5
    assert report["common_pool"][0]["original_target_positive"] == 8
    assert report["reference_pool"]["association"]["rho"] == pytest.approx(1.0)
    assert report["independent_selection_validation"] is False
    assert "pair_id" not in json.dumps(report)


@pytest.mark.parametrize("change", ["duplicate", "missing_model", "budget", "counts", "target"])
def test_common_pool_rejects_incomplete_or_inconsistent_aggregates(change: str) -> None:
    frames, similarity, context = _evidence()
    frame = frames["reference"]
    if change == "duplicate":
        frames["reference"] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "missing_model":
        frames["reference"] = frame.iloc[:2]
    elif change == "budget":
        frame.loc[0, "attempted"] = 9
    elif change == "counts":
        frame.loc[0, "verified_flips_all_candidates"] = 9
    else:
        frame.loc[0, "target"] = "unexpected"
    with pytest.raises(ValueError):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=_baseline(context),
        )


@pytest.mark.parametrize(
    "change", ["stale_digest", "totals", "receipt_totals", "retained", "duplicate_context"]
)
def test_comparison_rejects_stale_or_mixed_context(change: str) -> None:
    """Internally plausible cells cannot reuse a receipt or context from other outputs."""
    frames, similarity, context = _evidence()
    if change == "stale_digest":
        frames["reference"].loc[0, "verified_flips_source_failed"] += 1
        frames["reference"].loc[0, "verified_flips_all_candidates"] += 1
    elif change == "totals":
        context["reports"][1]["aggregate_counts"]["source_failed"] += 1
    elif change == "receipt_totals":
        context["reference_check"]["aggregate_counts"]["source_successful"] += 1
    elif change == "retained":
        context["reference_check"]["retained_surrogates"] = ["a", "b"]
    else:
        context["reports"].append(context["reports"][0])
    with pytest.raises(ValueError):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=_baseline(context),
        )


def test_unchanged_committed_artifacts_pass_comparison() -> None:
    """The stricter boundary preserves the measured nine-transformer reference outcome."""
    root = _ROOT / "docs/artifacts"
    context = json.loads((root / "retrospective_audit_summary.json").read_text())
    frames = {
        report["target"]: pd.read_csv(root / f"retrospective_{report['target']}_cells.csv")
        for report in context["reports"]
    }
    report = compare_saved_candidate_policies(
        frames,
        pd.read_csv(root / "surrogate_summary.csv"),
        context,
        attempts_per_cell=191,
        recipes=["bae", "bert-attack", "deepwordbug", "pwws", "textfooler"],
        reference_target="protectai-v2",
        baseline_context=json.loads((root / "detector_baseline_summary.json").read_text()),
    )
    assert report["reference_pool"]["all_candidate_flips"] == 1403
    assert report["reference_pool"]["attempted"] == 8595


@pytest.mark.parametrize(
    "field",
    [
        "inference_weights_sha256",
        "tokenizer_vocab_sha256",
        "loaded_hub_revision",
        "inference_dtype",
        "max_seq_len",
        "device",
        "provenance_scope",
        "positive_label",
        "benign_label",
        "missing_target",
        "missing_field",
        "baseline_positive",
    ],
)
def test_comparison_requires_matching_independent_inference_context(field: str) -> None:
    """An unchanged aggregate table cannot conceal mixed checkpoint/settings provenance."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    if field == "missing_target":
        baseline["reports"].pop()
    elif field == "missing_field":
        del context["reports"][1]["inference_weights_sha256"]
    elif field == "baseline_positive":
        baseline["reports"][1]["pooled"]["canonical_positive_predicted_positive_count"] = 7
    else:
        context["reports"][1][field] = "changed"
    with pytest.raises(ValueError):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )


def test_constant_candidate_outcomes_retain_undefined_association_reason() -> None:
    """A constant candidate outcome has no correlation or p-value and retains its reason."""
    frames, similarity, context = _evidence()
    reference = frames["reference"]
    reference["verified_flips_source_successful"] = 1
    reference["verified_flips_all_candidates"] = 2
    context["reports"][0]["aggregate_counts"] = reference.select_dtypes("number").sum().to_dict()
    context["reference_check"]["reference_cells_sha256"] = reference_cells_digest(reference)
    context["reference_check"]["aggregate_counts"]["verified_flips_source_successful"] = 3
    report = compare_saved_candidate_policies(
        frames,
        similarity,
        context,
        attempts_per_cell=10,
        recipes=["recipe"],
        reference_target="reference",
        baseline_context=_baseline(context),
    )
    association = report["reference_pool"]["association"]
    assert association["rho"] is None
    assert association["two_sided_p"] is None
    assert "constant" in association["reason"]


@pytest.mark.parametrize("field", ["inference_weights_sha256", "tokenizer_vocab_sha256"])
@pytest.mark.parametrize("value", [None, "", " ", "a" * 63, "g" * 64, 123, [], True])
def test_matching_invalid_identity_evidence_cannot_certify_comparison(
    field: str, value: object
) -> None:
    """Equal missing or malformed fingerprints provide no checkpoint/tokenizer identity."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    context["reports"][0][field] = value
    baseline["reports"][0][field] = value
    with pytest.raises(ValueError, match=field):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )


@pytest.mark.parametrize("field", ["inference_dtype", "device", "provenance_scope"])
@pytest.mark.parametrize("value", [None, "", " ", 123, []])
def test_matching_invalid_inference_strings_cannot_certify_comparison(
    field: str, value: object
) -> None:
    """A shared empty or non-string inference setting cannot certify the saved context."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    context["reports"][0][field] = value
    baseline["reports"][0][field] = value
    with pytest.raises(ValueError, match=field):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )


@pytest.mark.parametrize("value", [None, 0, -1, True, "512", 512.0])
def test_matching_invalid_token_limit_cannot_certify_comparison(value: object) -> None:
    """The certified token window requires a positive integer limit."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    context["reports"][0]["max_seq_len"] = value
    baseline["reports"][0]["max_seq_len"] = value
    with pytest.raises(ValueError, match="max_seq_len"):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )


@pytest.mark.parametrize("value", ["", " ", 123, []])
def test_present_hub_revision_requires_nonempty_string(value: object) -> None:
    """Hub revisions may be absent, while a present revision must name an identity."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    context["reports"][0]["loaded_hub_revision"] = value
    baseline["reports"][0]["loaded_hub_revision"] = value
    with pytest.raises(ValueError, match="loaded_hub_revision"):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )


@pytest.mark.parametrize(
    "label,canonical",
    [("positive_label", "canonical_positive"), ("benign_label", "canonical_negative")],
)
@pytest.mark.parametrize("value", [None, "", True, 1.0, -1])
def test_matching_invalid_label_types_cannot_certify_comparison(
    label: str, canonical: str, value: object
) -> None:
    """Matching null, boolean or noninteger class indices are invalid label evidence."""
    frames, similarity, context = _evidence()
    baseline = _baseline(context)
    context["reports"][0][label] = value
    baseline["reports"][0]["label_values"][canonical] = value
    with pytest.raises(ValueError, match="label"):
        compare_saved_candidate_policies(
            frames,
            similarity,
            context,
            attempts_per_cell=10,
            recipes=["recipe"],
            reference_target="reference",
            baseline_context=baseline,
        )
