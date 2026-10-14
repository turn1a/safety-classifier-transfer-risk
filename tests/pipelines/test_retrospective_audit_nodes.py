"""Local-only candidate and baseline inference contracts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch

from transfer_risk.pipelines.retrospective_audit import nodes, pipeline


class _Tokenizer:
    """Small stable tokenizer vocabulary for fingerprint tests."""

    def get_vocab(self) -> dict[str, int]:
        """Return a fixed vocabulary."""
        return {"[PAD]": 0, "a": 1, "b": 2}


def _predict(_model: Any, _tokenizer: Any, texts: list[str], **_kwargs: Any) -> list[int]:
    return [int(text.endswith("positive")) for text in texts]


def test_local_audit_deduplicates_text_and_retains_failed_candidates() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "original_id": "o",
                "original": "original-positive",
                "perturbed": "benign",
            },
            {
                "pair_id": "p2",
                "original_id": "o",
                "original": "original-positive",
                "perturbed": "changed-positive",
            },
        ]
    )
    linkage = pd.DataFrame(
        [
            {
                "pair_id": pair,
                "original_id": "o",
                "source": "source",
                "surrogate": "s",
                "recipe": "r",
                "attempt_count": 1,
                "source_successful_count": successful,
                "source_failed_count": 1 - successful,
                "source_skipped_count": 0,
                "source_other_count": 0,
            }
            for pair, successful in [("p1", 0), ("p2", 1)]
        ]
    )
    calls: list[str] = []

    def record_predict(*args: Any, **kwargs: Any) -> list[int]:
        calls.extend(args[2])
        return _predict(*args, **kwargs)

    bundle = {"model": torch.nn.Linear(2, 2), "tokenizer": _Tokenizer()}
    params = {
        "max_seq_len": 512,
        "batch_size": 2,
        "targets": {
            "t": {
                "repo_id": "test/t",
                "positive_label": 1,
                "benign_label": 0,
                "positive_label_name": "injection",
                "excluded_surrogates": [],
            }
        },
    }
    predictions, cells, sources, context = nodes.run_retrospective_target(
        pairs,
        linkage,
        bundle,
        params,
        {"policy": "cpu"},
        target_id="t",
        predict_fn=record_predict,
    )
    assert sorted(calls) == ["benign", "changed-positive", "original-positive"]
    assert len(predictions) == 2
    assert cells["verified_flips_source_failed"].sum() == 1
    assert cells["verified_flips_source_successful"].sum() == 0
    assert sources["attempted"].sum() == 2
    assert len(context["inference_weights_sha256"]) == 64
    assert len(context["tokenizer_vocab_sha256"]) == 64
    assert context["provenance_scope"] == "current_local_audit_only"


def test_reference_check_requires_every_retained_cell_and_exact_counts() -> None:
    current = pd.DataFrame(
        [
            {
                "surrogate": "s",
                "recipe": "r",
                "source_successful": 3,
                "attempted": 191,
                "target_original_positive_successful": 2,
                "verified_flips_source_successful": 1,
            }
        ]
    )
    historical = pd.DataFrame(
        [
            {
                "surrogate": "s",
                "recipe": "r",
                "source_successful": 3,
                "target_original_injection": 2,
                "true_target_flips": 1,
            }
        ]
    )
    manifest = {"attack_attempts_per_cell": 191, "n_surrogates": 1, "n_recipes": 1}
    definition = {"excluded_surrogates": []}
    assert (
        nodes.verify_reference_counts(current, historical, manifest, definition)["counts_agree"]
        is True
    )
    historical.loc[0, "true_target_flips"] = 2
    with pytest.raises(ValueError, match="historical"):
        nodes.verify_reference_counts(current, historical, manifest, definition)


def test_reference_check_rejects_missing_whole_surrogate_and_short_attempt_budget() -> None:
    historical = pd.DataFrame(
        [
            {
                "surrogate": name,
                "recipe": "r",
                "source_successful": 3,
                "target_original_injection": 2,
                "true_target_flips": 1,
            }
            for name in ("s1", "s2")
        ]
    )
    current = historical.rename(
        columns={
            "target_original_injection": "target_original_positive_successful",
            "true_target_flips": "verified_flips_source_successful",
        }
    )
    current["attempted"] = 191
    manifest = {"attack_attempts_per_cell": 191, "n_surrogates": 2, "n_recipes": 1}
    definition = {"excluded_surrogates": []}
    with pytest.raises(ValueError, match="grid"):
        nodes.verify_reference_counts(current.iloc[:1], historical, manifest, definition)
    current.loc[0, "attempted"] = 3
    with pytest.raises(ValueError, match="budget"):
        nodes.verify_reference_counts(current, historical, manifest, definition)


def test_retrospective_summary_requires_matching_reference_check() -> None:
    context = {
        "target": "protectai-v2",
        "included_surrogates": ["s"],
        "aggregate_counts": {
            "attempted": 191,
            "source_successful": 3,
            "target_original_positive_successful": 2,
            "verified_flips_source_successful": 1,
        },
    }
    check = {
        "counts_agree": True,
        "retained_surrogates": ["s"],
        "aggregate_counts": context["aggregate_counts"],
    }
    result = nodes.assemble_retrospective_reports(check, context)
    assert result["reference_check"]["counts_agree"] is True
    check["counts_agree"] = False
    with pytest.raises(ValueError, match="reference"):
        nodes.assemble_retrospective_reports(check, context)


def test_snapshot_fingerprint_changes_when_weights_or_tokenizer_change() -> None:
    model = torch.nn.Linear(2, 2)
    tokenizer = _Tokenizer()
    first = nodes.fingerprint_loaded_classifier(model, tokenizer)
    with torch.no_grad():
        model.weight.add_(1)
    second = nodes.fingerprint_loaded_classifier(model, tokenizer)
    assert first["inference_weights_sha256"] != second["inference_weights_sha256"]
    assert first["tokenizer_vocab_sha256"] == second["tokenizer_vocab_sha256"]


def test_candidate_publication_rejects_failed_or_stale_reference_receipt() -> None:
    reference = pd.DataFrame(
        [
            {
                "surrogate": "s",
                "recipe": "r",
                "source_successful": 3,
                "attempted": 191,
                "target_original_positive_successful": 2,
                "verified_flips_source_successful": 1,
            }
        ]
    )
    historical = reference.rename(
        columns={
            "target_original_positive_successful": "target_original_injection",
            "verified_flips_source_successful": "true_target_flips",
        }
    )
    receipt = nodes.verify_reference_counts(
        reference,
        historical,
        {"attack_attempts_per_cell": 191, "n_surrogates": 1, "n_recipes": 1},
        {"excluded_surrogates": []},
    )
    cells, sources = (
        pd.DataFrame([{"target": "t", "attempted": 191}]),
        pd.DataFrame([{"target": "t", "attempted": 191}]),
    )
    published = nodes.publish_retrospective_tables(receipt, reference, cells, sources)
    assert published[0].equals(cells)
    assert published[1].equals(sources)
    with pytest.raises(ValueError, match="reference"):
        nodes.publish_retrospective_tables(
            {**receipt, "counts_agree": False}, reference, cells, sources
        )
    changed = reference.copy()
    changed.loc[0, "verified_flips_source_successful"] = 2
    with pytest.raises(ValueError, match="stale"):
        nodes.publish_retrospective_tables(receipt, changed, cells, sources)


def test_candidate_inference_has_no_public_outputs_before_reference_check(monkeypatch: Any) -> None:
    monkeypatch.setattr(pipeline, "retrospective_target_specs", lambda: ["protectai-v2", "other"])
    graph = pipeline.create_pipeline()
    for inference in graph.only_nodes(
        "audit_saved_candidates_protectai-v2", "audit_saved_candidates_other"
    ).nodes:
        assert all(not output.startswith("pub_") for output in inference.outputs)
    check = graph.only_nodes("verify_retrospective_reference_counts").nodes[0]
    assert "retrospective_cells__protectai-v2" in check.inputs
    for publication in graph.only_nodes(
        "publish_retrospective_protectai-v2", "publish_retrospective_other"
    ).nodes:
        assert "pub_retrospective_reference_check" in publication.inputs
        assert "retrospective_cells__protectai-v2" in publication.inputs
