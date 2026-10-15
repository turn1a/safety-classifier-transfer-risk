"""Exact-text provenance and conflict protection for staged review imports."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from transfer_risk.lib.semantic_audit import stable_pair_id
from transfer_risk.lib.semantic_flip_reviews import merge_flip_exports, progress_for_role
from transfer_risk.lib.semantic_review_tool import build_review_dataset, build_staged_review_dataset


def dataset():
    """Return a synthetic staged cohort and the older sorted cohort with matching text."""
    pairs = [
        {"pair_id": stable_pair_id(o, p), "original": o, "perturbed": p}
        for o, p in [("one", "two"), ("three", "four")]
    ]
    return build_staged_review_dataset(
        pairs, stage_one_pair_count=1, sample_trial_count=1
    ), build_review_dataset(pairs)


def rows(data, role="reviewer_1"):
    """Create synthetic explicit responses for a role and exact fingerprint."""
    return [
        {
            "pair_id": p["pair_id"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "reviewer_id": role,
            "independent_review": "yes",
            "semantic_preserved": "yes",
            "intent_preserved": "uncertain",
            "original_is_injection": "yes",
            "notes": "synthetic evidence",
            "review_status": "complete",
        }
        for p in data["pairs"]
    ]


def test_reuse_retains_exact_text_role_provenance_and_old_answers() -> None:

    new, old = dataset()
    merged = merge_flip_exports(
        new, [], [], pd.DataFrame(), reuse_dataset=old, reuse_first=rows(old)
    )
    assert len(merged) == 2
    assert set(merged.dataset_fingerprint) == {new["dataset_fingerprint"]}
    provenance = json.loads(merged.iloc[0].provenance)
    assert provenance[0]["source_dataset_fingerprint"] == old["dataset_fingerprint"]
    assert provenance[0]["reviewer_id"] == "reviewer_1"
    assert provenance[0]["exact_pair_fingerprint"]
    retained = merge_flip_exports(new, [], [], merged)
    pd.testing.assert_frame_equal(retained, merged)


def test_conflict_role_and_dataset_mismatch_refuse_import() -> None:

    new, old = dataset()
    saved = merge_flip_exports(new, rows(new), [], pd.DataFrame())
    changed = rows(new)
    changed[0]["intent_preserved"] = "yes"
    with pytest.raises(ValueError, match="conflict"):
        merge_flip_exports(new, changed, [], saved)
    with pytest.raises(ValueError, match="reviewer"):
        merge_flip_exports(new, rows(new, "reviewer_2"), [], pd.DataFrame())
    with pytest.raises(ValueError, match="fingerprint"):
        merge_flip_exports(new, rows(old), [], pd.DataFrame())
    tampered = saved.copy()
    tampered.loc[0, "provenance"] = "[]"
    with pytest.raises(ValueError, match="provenance"):
        merge_flip_exports(new, [], [], tampered)


def test_existing_import_cannot_hide_an_unknown_reviewer_role() -> None:
    new, _ = dataset()
    saved = merge_flip_exports(new, rows(new), [], pd.DataFrame())
    saved["reviewer_id"] = "outside"
    saved["provenance"] = saved["provenance"].map(
        lambda value: json.dumps(
            [{**entry, "reviewer_id": "outside"} for entry in json.loads(value)]
        )
    )
    with pytest.raises(ValueError, match="reviewer"):
        merge_flip_exports(new, [], [], saved)


def test_reused_answers_can_restore_only_their_own_role_into_the_new_browser() -> None:
    new, old = dataset()
    merged = merge_flip_exports(
        new, [], [], pd.DataFrame(), reuse_dataset=old, reuse_first=rows(old)
    )
    first = progress_for_role(new, merged, "reviewer_1")
    second = progress_for_role(new, merged, "reviewer_2")
    assert len(first["reviews"]) == 2
    assert second["reviews"] == {}
    assert first["dataset_fingerprint"] == new["dataset_fingerprint"]
    assert first["schema_version"] == "semantic-review-progress-v1"
    assert all(row["intent_preserved"] == "uncertain" for row in first["reviews"].values())
    with pytest.raises(ValueError, match="reviewer"):
        progress_for_role(new, merged, "outside")
