"""Pure private review-tool rendering and independently exported label validation.

The HTML contains exact text pairs and a fixed reviewer role. Human labels are
stored locally and imported separately. No model scores or outcomes enter the tool.
"""

from __future__ import annotations

import base64
import hashlib
import json
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

import pandas as pd

from transfer_risk.lib.semantic_audit import resolve_annotations, stable_pair_id

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REVIEWER_IDS = ("reviewer_1", "reviewer_2")
REVIEW_LABELS = ("semantic_preserved", "intent_preserved", "original_is_injection")
REVIEW_CSV_COLUMNS = (
    "pair_id",
    "dataset_fingerprint",
    "reviewer_id",
    "independent_review",
    *REVIEW_LABELS,
    "notes",
    "review_status",
)
_DATA_PLACEHOLDER = "__SEMANTIC_REVIEW_DATA__"
_CSP_PLACEHOLDER = "__SEMANTIC_REVIEW_CSP__"
_DATA_VERSION = "semantic-review-data-v1"


def _json(value: Any) -> str:
    """Encode deterministic JSON without escaping Unicode or allowing nonfinite values."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def build_review_dataset(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whitelist exact candidate texts and fingerprint the sorted private dataset.

    Args:
        candidates: Saved blinded sample rows. Additional metadata is discarded.

    Returns:
        Private dataset containing only exact pair IDs/texts and dataset-level metadata.

    Raises:
        ValueError: If text, pair IDs, duplicates, or an empty sample fail validation.
    """
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        pair_id, original, perturbed = (
            row.get(key) for key in ("pair_id", "original", "perturbed")
        )
        if (
            not isinstance(pair_id, str)
            or not isinstance(original, str)
            or not isinstance(perturbed, str)
        ):
            msg = "review candidates require string pair_id, original and perturbed fields"
            raise ValueError(msg)
        if pair_id != stable_pair_id(original, perturbed):
            msg = "review candidate pair_id does not match its exact saved text"
            raise ValueError(msg)
        if pair_id in seen:
            msg = "review candidates contain a duplicate pair_id"
            raise ValueError(msg)
        seen.add(pair_id)
        pairs.append({"pair_id": pair_id, "original": original, "perturbed": perturbed})
    if not pairs:
        msg = "the private review tool requires at least one selected pair"
        raise ValueError(msg)
    pairs.sort(key=lambda pair: pair["pair_id"])
    fingerprint = hashlib.sha256(_json([_DATA_VERSION, pairs]).encode("utf-8")).hexdigest()
    return {
        "schema_version": _DATA_VERSION,
        "dataset_fingerprint": fingerprint,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _checked_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a dataset through the whitelist and verify its stored fingerprint."""
    rebuilt = build_review_dataset(dataset["pairs"])
    if any(
        dataset.get(key) != rebuilt[key]
        for key in ("schema_version", "dataset_fingerprint", "pair_count")
    ):
        msg = "review dataset fingerprint or metadata does not match its exact pairs"
        raise ValueError(msg)
    return rebuilt


class _InlineAssets(HTMLParser):
    """Collect executable inline scripts and styles for content-security-policy hashes."""

    def __init__(self) -> None:
        """Initialize parser state without modifying asset text."""
        super().__init__(convert_charrefs=False)
        self.assets: dict[str, list[str]] = {"script": [], "style": []}
        self.current = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Begin an inline executable script or stylesheet."""
        if tag in self.assets and dict(attrs).get("type") != "application/json":
            self.current = tag
            self.assets[tag].append("")

    def handle_data(self, data: str) -> None:
        """Preserve exact asset text for browser-compatible SHA-256 hashes."""
        if self.current:
            self.assets[self.current][-1] += data

    def handle_endtag(self, tag: str) -> None:
        """End an asset body."""
        if tag == self.current:
            self.current = ""


def review_tool_content_security_policy(html: str) -> str:
    """Allow only the document's hashed inline code and styles, with no network access.

    Args:
        html: Generated private review document.

    Returns:
        Policy suitable for a meta element and an HTTP response header. The server
        additionally denies framing in its response headers.
    """
    parsed = _InlineAssets()
    parsed.feed(html)
    directives = ["default-src 'none'"]
    for kind in ("script", "style"):
        hashes = [
            "'sha256-"
            + base64.b64encode(hashlib.sha256(asset.encode("utf-8")).digest()).decode("ascii")
            + "'"
            for asset in parsed.assets[kind]
        ]
        directives.append(kind + "-src " + (" ".join(hashes) if hashes else "'none'"))
    directives.extend(
        [
            "img-src data:",
            "font-src 'none'",
            "connect-src 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        ]
    )
    return "; ".join(directives)


def render_review_tool_html(
    dataset: Mapping[str, Any],
    template: str,
    reviewer_id: str,
) -> str:
    """Embed whitelisted private pairs in a fixed-role offline HTML document safely.

    Args:
        dataset: Exact candidate dataset created by :func:`build_review_dataset`.
        template: Source-controlled static HTML/CSS/JavaScript template.
        reviewer_id: Fixed role, ``reviewer_1`` or ``reviewer_2``.

    Returns:
        Private self-contained HTML. JSON uses explicit escapes for HTML delimiters
        and JavaScript line separators; the UI renders raw strings with ``textContent``.

    Raises:
        ValueError: If the role, dataset fingerprint, or template placeholders are invalid.
    """
    if reviewer_id not in REVIEWER_IDS:
        msg = "the review tool must be bound to reviewer_1 or reviewer_2"
        raise ValueError(msg)
    if template.count(_DATA_PLACEHOLDER) != 1 or template.count(_CSP_PLACEHOLDER) != 1:
        msg = "review HTML template must contain each data and CSP placeholder exactly once"
        raise ValueError(msg)
    payload = _json({**_checked_dataset(dataset), "reviewer_id": reviewer_id})
    for character, escape in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(character, escape)
    preview = template.replace(_DATA_PLACEHOLDER, payload)
    policy = review_tool_content_security_policy(preview)
    return template.replace(_CSP_PLACEHOLDER, policy).replace(_DATA_PLACEHOLDER, payload)


def _string(row: Mapping[str, Any], key: str) -> str:
    """Read an explicit export string without coercing machine labels or identities."""
    value = row.get(key)
    if not isinstance(value, str):
        msg = "review export fields must be explicit strings"
        raise ValueError(msg)
    return value


def _review_status(row: Mapping[str, str]) -> str:
    """Derive completion only from explicit labels and independent-review attestation."""
    if all(row[label] for label in REVIEW_LABELS) and row["independent_review"] == "yes":
        return "complete"
    if any(row[label] for label in REVIEW_LABELS) or row["independent_review"] or row["notes"]:
        return "in_progress"
    return "pending"


def _validated_export(
    dataset: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    reviewer: str,
) -> list[dict[str, str]]:
    """Validate one role's possibly partial export without inventing absent responses."""
    pair_ids = {pair["pair_id"] for pair in dataset["pairs"]}
    seen: set[str] = set()
    validated: list[dict[str, str]] = []
    for raw in rows:
        if set(raw) != set(REVIEW_CSV_COLUMNS):
            msg = "review export columns must match the tool CSV schema and exclude prompt text"
            raise ValueError(msg)
        row = {key: _string(raw, key) for key in REVIEW_CSV_COLUMNS}
        if row["dataset_fingerprint"] != dataset["dataset_fingerprint"]:
            msg = "review export dataset fingerprint does not match the current saved sample"
            raise ValueError(msg)
        if row["reviewer_id"] != reviewer:
            msg = "review export reviewer identity does not match its assigned role"
            raise ValueError(msg)
        if row["pair_id"] not in pair_ids:
            msg = "review export pair_id is outside the selected sample"
            raise ValueError(msg)
        if row["pair_id"] in seen:
            msg = "review export contains a duplicate pair_id for one reviewer"
            raise ValueError(msg)
        seen.add(row["pair_id"])
        if any(row[label] not in {"", "yes", "no", "uncertain"} for label in REVIEW_LABELS):
            msg = "review export labels must be yes, no, uncertain, or empty while pending"
            raise ValueError(msg)
        if row["independent_review"] not in {"", "yes"}:
            msg = "independent_review must remain pending or contain an explicit yes attestation"
            raise ValueError(msg)
        if row["review_status"] != _review_status(row):
            msg = "review export status contradicts its labels or independence attestation"
            raise ValueError(msg)
        if row["review_status"] != "pending":
            validated.append(row)
    return validated


def combine_reviewer_exports(
    dataset: Mapping[str, Any],
    reviewer_1_rows: Sequence[Mapping[str, Any]],
    reviewer_2_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Combine two private role-bound exports only after validating both completely.

    Args:
        dataset: Current exact saved probability sample.
        reviewer_1_rows: User's CSV export, possibly partial.
        reviewer_2_rows: Partner's separate CSV export, possibly partial.

    Returns:
        Private annotation dataframe with no prompt columns. Untouched pending rows
        are omitted so export alone cannot increase reviewed coverage. Missing labels,
        missing attestations, uncertainty and disagreement remain unresolved downstream.

    Raises:
        ValueError: If either export has a wrong fingerprint/role, unknown or duplicate
            pair, unrecognized columns, invalid labels, or inconsistent completion status.
    """
    checked = _checked_dataset(dataset)
    rows = [
        *_validated_export(checked, reviewer_1_rows, REVIEWER_IDS[0]),
        *_validated_export(checked, reviewer_2_rows, REVIEWER_IDS[1]),
    ]
    annotations = pd.DataFrame(rows, columns=list(REVIEW_CSV_COLUMNS))
    resolve_annotations(annotations, [pair["pair_id"] for pair in checked["pairs"]], REVIEWER_IDS)
    return annotations
