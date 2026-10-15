"""Private review-tool safety, role isolation, and export validation."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from transfer_risk.lib.semantic_audit import stable_pair_id
from transfer_risk.lib.semantic_review_tool import (
    REVIEW_CSV_COLUMNS,
    build_review_dataset,
    build_staged_review_dataset,
    combine_reviewer_exports,
    render_review_tool_html,
    review_tool_content_security_policy,
)

_TEMPLATE = Path(__file__).parents[2] / "src/transfer_risk/templates/semantic_review_tool.html"


class _HtmlParts(HTMLParser):
    """Collect actual parsed script text and tags from generated HTML."""

    def __init__(self) -> None:
        """Initialize parsed HTML collections."""
        super().__init__(convert_charrefs=False)
        self.scripts: dict[str, str] = {}
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.current = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect parsed tags and identify an inline script body."""
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script":
            self.current = str(attributes.get("id", ""))
            self.scripts[self.current] = ""

    def handle_data(self, data: str) -> None:
        """Collect script contents without interpreting them."""
        if self.current:
            self.scripts[self.current] += data

    def handle_endtag(self, tag: str) -> None:
        """Finish a script body."""
        if tag == "script":
            self.current = ""


def _candidates() -> list[dict[str, Any]]:
    """Create neutral synthetic pairs, with forbidden metadata to test blinding."""
    return [
        {
            "pair_id": stable_pair_id(original, perturbed),
            "original": original,
            "perturbed": perturbed,
            "surrogate": "hidden-model",
            "source": "hidden-source",
            "recipe": "hidden-recipe",
            "source_successful_count": 7,
            "score": 0.93,
        }
        for original, perturbed in [
            ("A neutral original.", "A neutral revision."),
            ("Another text.", "Another text."),
        ]
    ]


def _html(reviewer: str = "reviewer_1") -> str:
    """Render an actual source-controlled template using only synthetic text."""
    return render_review_tool_html(
        build_review_dataset(_candidates()), _TEMPLATE.read_text(), reviewer
    )


def _export(
    dataset: dict[str, Any], reviewer: str, *, complete: bool = True
) -> list[dict[str, str]]:
    """Make an explicit synthetic human export with no prompt text."""
    return [
        {
            "pair_id": pair["pair_id"],
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "reviewer_id": reviewer,
            "independent_review": "yes" if complete else "",
            "semantic_preserved": "yes" if complete else "",
            "intent_preserved": "yes" if complete else "",
            "original_is_injection": "yes" if complete else "",
            "notes": "",
            "review_status": "complete" if complete else "pending",
        }
        for pair in dataset["pairs"]
    ]


def _run_core(script: str) -> Any:
    """Execute the actual browser state/export core in an available JavaScript runtime."""
    runtime = shutil.which("node")
    if not runtime:
        pytest.skip("Node.js is unavailable for the browser-core behavior check")
    parsed = _HtmlParts()
    parsed.feed(_html())
    result = subprocess.run(  # noqa: S603 - executes controlled browser-core source and test fixtures
        [runtime, "--input-type=module"],
        input=parsed.scripts["review-core"] + "\n" + script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_tool_payload_is_whitelisted_and_role_bound_without_any_prefilled_labels() -> None:
    parsed = _HtmlParts()
    parsed.feed(_html())
    data = json.loads(parsed.scripts["review-data"])

    assert data["reviewer_id"] == "reviewer_1"
    assert data["pair_count"] == 2
    assert all(set(pair) == {"pair_id", "original", "perturbed"} for pair in data["pairs"])
    assert "hidden-model" not in _html()
    assert "hidden-source" not in _html()
    assert "hidden-recipe" not in _html()
    assert "source_successful_count" not in parsed.scripts["review-data"]
    assert not any("semantic_preserved" in pair for pair in data["pairs"])


def test_raw_script_breakout_html_and_separator_characters_roundtrip_as_exact_text() -> None:
    original = (
        '</script><img src="https://invalid.example/escape" onerror="alert(1)"> & \u2028\u2029'
    )
    perturbed = "__SEMANTIC_REVIEW_CSP__ __SEMANTIC_REVIEW_DATA__ <svg/onload=alert(2)>"
    dataset = build_review_dataset(
        [
            {
                "pair_id": stable_pair_id(original, perturbed),
                "original": original,
                "perturbed": perturbed,
            }
        ]
    )
    html = render_review_tool_html(dataset, _TEMPLATE.read_text(), "reviewer_1")
    parsed = _HtmlParts()
    parsed.feed(html)
    pair = json.loads(parsed.scripts["review-data"])["pairs"][0]

    assert pair["original"] == original
    assert pair["perturbed"] == perturbed
    assert set(parsed.scripts) == {"review-data", "review-core", "review-app"}
    assert not any(tag in {"img", "svg"} for tag, _ in parsed.tags)
    assert "\\u003c/script\\u003e" in parsed.scripts["review-data"]
    assert "\\u2028" in parsed.scripts["review-data"]
    assert "\\u2029" in parsed.scripts["review-data"]


def test_dataset_fingerprint_is_order_independent_but_changes_when_pair_text_changes() -> None:
    candidates = _candidates()
    original = build_review_dataset(candidates)
    reordered = build_review_dataset(candidates[::-1])
    changed = _candidates()
    changed[0]["perturbed"] = "A different revision."
    changed[0]["pair_id"] = stable_pair_id(changed[0]["original"], changed[0]["perturbed"])

    assert original["dataset_fingerprint"] == reordered["dataset_fingerprint"]
    assert original["dataset_fingerprint"] != build_review_dataset(changed)["dataset_fingerprint"]


def test_corrupt_or_duplicate_pair_identifiers_and_unknown_roles_are_rejected() -> None:
    candidates = _candidates()
    with pytest.raises(ValueError, match="duplicate"):
        build_review_dataset([candidates[0], candidates[0]])
    candidates[0]["pair_id"] = "corrupt"
    with pytest.raises(ValueError, match="pair_id"):
        build_review_dataset(candidates)
    with pytest.raises(ValueError, match="reviewer"):
        render_review_tool_html(build_review_dataset(_candidates()), _TEMPLATE.read_text(), "other")


def test_content_security_policy_allows_only_hash_bound_inline_assets_and_no_network() -> None:
    html = _html()
    policy = review_tool_content_security_policy(html)

    assert "default-src 'none'" in policy
    assert "connect-src 'none'" in policy
    assert "font-src 'none'" in policy
    assert "script-src 'sha256-" in policy
    assert "'unsafe-inline'" not in policy
    assert policy in html


def test_combination_checks_each_role_and_preserves_partial_rows_as_unresolved() -> None:
    dataset = build_review_dataset(_candidates())
    first, second = _export(dataset, "reviewer_1"), _export(dataset, "reviewer_2", complete=False)
    combined = combine_reviewer_exports(dataset, first, second)

    assert len(combined) == 2
    assert set(combined["reviewer_id"]) == {"reviewer_1"}
    assert set(combined.columns) == set(REVIEW_CSV_COLUMNS)
    assert not set(combined.columns) & {"original", "perturbed"}
    second[0]["semantic_preserved"] = "uncertain"
    second[0]["review_status"] = "in_progress"
    combined = combine_reviewer_exports(dataset, first, second)
    assert len(combined) == 3
    assert combined.iloc[-1]["independent_review"] == ""


@pytest.mark.parametrize(
    ("field", "bad", "match"),
    [
        ("dataset_fingerprint", "wrong", "fingerprint"),
        ("reviewer_id", "reviewer_2", "reviewer"),
        ("pair_id", "outside", "sample"),
        ("semantic_preserved", "probably", "label"),
        ("independent_review", "no", "independent"),
        ("review_status", "pending", "status"),
    ],
)
def test_invalid_export_is_rejected_before_combining(field: str, bad: str, match: str) -> None:
    dataset = build_review_dataset(_candidates())
    first, second = _export(dataset, "reviewer_1"), _export(dataset, "reviewer_2")
    first[0][field] = bad
    with pytest.raises(ValueError, match=match):
        combine_reviewer_exports(dataset, first, second)


def test_duplicate_export_rows_and_unexpected_text_columns_are_rejected() -> None:
    dataset = build_review_dataset(_candidates())
    first, second = _export(dataset, "reviewer_1"), _export(dataset, "reviewer_2")
    with pytest.raises(ValueError, match="duplicate"):
        combine_reviewer_exports(dataset, [*first, first[0]], second)
    first[0]["original"] = "unexpected private text"
    with pytest.raises(ValueError, match="columns"):
        combine_reviewer_exports(dataset, first, second)


def test_browser_core_starts_empty_and_never_assumes_independent_review() -> None:
    dataset = build_review_dataset(_candidates())
    result = _run_core(
        f"const data = {json.dumps(dataset)}; const C = globalThis.SemanticReviewCore; "
        "const state = C.freshState(data, 'reviewer_1'); "
        "console.log(JSON.stringify({row: C.rowFor(data.pairs[0].pair_id, state), "
        "key1: C.storageKey(data, 'reviewer_1'), key2: C.storageKey(data, 'reviewer_2')}));"
    )

    assert result["row"]["independent_review"] == ""
    assert result["row"]["semantic_preserved"] == ""
    assert result["key1"] != result["key2"]


def test_browser_progress_import_rejects_another_role_fingerprint_and_unknown_pair() -> None:
    dataset = build_review_dataset(_candidates())
    result = _run_core(
        f"const data = {json.dumps(dataset)}; const C = globalThis.SemanticReviewCore; "
        "const good = C.exportProgress(data, C.freshState(data, 'reviewer_1')); "
        "const outcomes = []; for (const variant of ["
        "{...good, reviewer_id:'reviewer_2'}, {...good, dataset_fingerprint:'wrong'}, "
        "{...good, reviews:{outside:{}}}]) { "
        "try { C.importProgress(data,'reviewer_1',variant); outcomes.push('accepted'); } "
        "catch (error) { outcomes.push('rejected'); }} console.log(JSON.stringify(outcomes));"
    )
    assert result == ["rejected", "rejected", "rejected"]


def test_browser_csv_quotes_notes_and_contains_no_sample_text_or_calibration() -> None:
    dataset = build_review_dataset(_candidates())
    result = _run_core(
        f"const data = {json.dumps(dataset)}; const C = globalThis.SemanticReviewCore; "
        "const state = C.freshState(data,'reviewer_1'); "
        "state.reviews[data.pairs[0].pair_id] = {...C.emptyRow(), "
        "notes:'A comma, a \"quote\"\\nnext line'}; "
        "console.log(JSON.stringify(C.exportCsv(data,state)));"
    )
    rows = list(csv.DictReader(StringIO(result)))

    assert len(rows) == 2
    assert rows[0]["notes"] == 'A comma, a "quote"\nnext line'
    assert rows[0]["independent_review"] == ""
    assert rows[0]["review_status"] == "in_progress"
    assert "A neutral original." not in result
    assert "calibration" not in result


def test_staged_dataset_preserves_explicit_queue_and_binds_stage_counts() -> None:

    candidates = _candidates()
    staged = build_staged_review_dataset(candidates, stage_one_pair_count=1, sample_trial_count=2)
    assert [pair["pair_id"] for pair in staged["pairs"]] == [row["pair_id"] for row in candidates]
    assert staged["schema_version"] == "semantic-review-staged-v1"
    assert staged["dataset_fingerprint"] != build_review_dataset(candidates)["dataset_fingerprint"]
    assert combine_reviewer_exports(staged, _export(staged, "reviewer_1"), []).shape[0] == 2
    assert "stage_one_pair_count" in render_review_tool_html(
        staged, _TEMPLATE.read_text(), "reviewer_1"
    )
    with pytest.raises(ValueError):
        build_staged_review_dataset(candidates, stage_one_pair_count=3, sample_trial_count=2)


def test_staged_exports_bind_full_cohort_identity_beyond_the_display_queue() -> None:
    first = build_staged_review_dataset(
        _candidates(), stage_one_pair_count=1, sample_trial_count=2, cohort_binding="population-one"
    )
    second = build_staged_review_dataset(
        _candidates(), stage_one_pair_count=1, sample_trial_count=2, cohort_binding="population-two"
    )
    assert first["dataset_fingerprint"] != second["dataset_fingerprint"]
    with pytest.raises(ValueError, match="fingerprint"):
        combine_reviewer_exports(second, _export(first, "reviewer_1"), [])


def test_staged_browser_json_restore_csv_export_and_conflict_protection() -> None:
    dataset = build_staged_review_dataset(
        _candidates(),
        stage_one_pair_count=1,
        sample_trial_count=2,
        cohort_binding="synthetic-trial-frame",
    )
    result = _run_core(
        f"const data={json.dumps(dataset)}; const C=globalThis.SemanticReviewCore; "
        "const state=C.freshState(data,'reviewer_1'); "
        "state.reviews[data.pairs[0].pair_id]={...C.emptyRow(),semantic_preserved:'yes',"
        "intent_preserved:'uncertain',original_is_injection:'yes',"
        "independent_review:'yes',notes:'synthetic only'}; "
        "const saved=JSON.parse(JSON.stringify(C.exportProgress(data,state))); "
        "const restored=C.importProgress(data,'reviewer_1',saved); "
        "const conflict=C.importProgress(data,'reviewer_1',C.exportProgress(data,state)); "
        "conflict.reviews[data.pairs[0].pair_id].semantic_preserved='no'; "
        "let rejected=false; try{C.mergeProgress(data,restored,conflict);}catch(_){rejected=true;} "
        "console.log(JSON.stringify({csv:C.exportCsv(data,restored),rejected,original:C.rowFor(data.pairs[0].pair_id,restored)}));"
    )
    rows = list(csv.DictReader(StringIO(result["csv"])))
    assert rows[0]["dataset_fingerprint"] == dataset["dataset_fingerprint"]
    assert rows[0]["review_status"] == "complete"
    assert rows[0]["intent_preserved"] == "uncertain"
    assert result["rejected"]
    assert result["original"]["semantic_preserved"] == "yes"
    assert "A neutral original." not in result["csv"]
