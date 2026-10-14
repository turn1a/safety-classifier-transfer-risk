"""Loopback serving and catalog-only export-combination boundaries."""

from __future__ import annotations

from http.client import HTTPConnection
from threading import Thread
from typing import Any

import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset

from transfer_risk.lib.semantic_audit import stable_pair_id
from transfer_risk.lib.semantic_review_tool import REVIEW_CSV_COLUMNS, build_review_dataset
from transfer_risk.scripts.semantic_review_server import (
    build_review_server,
    combine_catalog_exports,
    load_review_html,
)


def _catalog() -> DataCatalog:
    """Build an in-memory catalog with a valid private sample and explicit exports."""
    candidates = pd.DataFrame(
        [
            {
                "pair_id": stable_pair_id("Neutral original.", "Neutral revision."),
                "original": "Neutral original.",
                "perturbed": "Neutral revision.",
            }
        ]
    )
    dataset = build_review_dataset(candidates.to_dict("records"))
    exports = {
        role: pd.DataFrame(
            [
                {
                    "pair_id": candidates.iloc[0]["pair_id"],
                    "dataset_fingerprint": dataset["dataset_fingerprint"],
                    "reviewer_id": role,
                    "independent_review": "yes",
                    "semantic_preserved": "yes",
                    "intent_preserved": "uncertain",
                    "original_is_injection": "uncertain",
                    "notes": "Insufficient context.",
                    "review_status": "complete",
                }
            ],
            columns=list(REVIEW_CSV_COLUMNS),
        )
        for role in ("reviewer_1", "reviewer_2")
    }
    return DataCatalog(
        {
            "semantic_audit_annotation_candidates": MemoryDataset(candidates),
            "semantic_audit_reviewer_1_export": MemoryDataset(exports["reviewer_1"]),
            "semantic_audit_reviewer_2_export": MemoryDataset(exports["reviewer_2"]),
            "semantic_audit_annotations": MemoryDataset(pd.DataFrame()),
            "semantic_audit_review_tool_reviewer_1": MemoryDataset(
                "<!doctype html><title>Role one</title>"
            ),
            "semantic_audit_review_tool_reviewer_2": MemoryDataset(
                "<!doctype html><title>Role two</title>"
            ),
        }
    )


def test_loopback_server_serves_only_one_document_and_rejects_uploads_and_wrong_hosts() -> None:
    server = build_review_server("<!doctype html><title>Private synthetic review</title>", port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        port = server.server_address[1]
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"Private synthetic review" in response.read()
        assert response.getheader("Cache-Control").startswith("no-store")
        assert "connect-src 'none'" in response.getheader("Content-Security-Policy")
        assert response.getheader("X-Frame-Options") == "DENY"
        connection.close()
        for method, path, expected in [
            ("GET", "/private.csv", 404),
            ("GET", "/../annotations.csv", 404),
            ("GET", "/?reviewer=reviewer_2", 404),
            ("POST", "/", 405),
            ("PUT", "/", 405),
        ]:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request(method, path, body="must not be stored" if method != "GET" else None)
            response = connection.getresponse()
            assert response.status == expected
            assert b"Private synthetic review" not in response.read()
            connection.close()
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", "/", headers={"Host": "attacker.invalid"})
        response = connection.getresponse()
        assert response.status == 403
        assert b"Private synthetic review" not in response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_html_selection_is_fixed_to_the_assigned_catalog_role() -> None:
    catalog = _catalog()

    assert "Role one" in load_review_html(catalog, "reviewer_1")
    assert "Role two" in load_review_html(catalog, "reviewer_2")
    with pytest.raises(ValueError, match="reviewer"):
        load_review_html(catalog, "../../annotations")


def test_valid_catalog_exports_are_saved_without_prompt_text_or_false_consensus() -> None:
    catalog = _catalog()
    result = combine_catalog_exports(catalog, {"reviewer_ids": ["reviewer_1", "reviewer_2"]})
    annotations = catalog.load("semantic_audit_annotations")

    assert result["submitted_rows"] == 2
    assert result["complete_rows"] == 2
    assert result["selected_pairs"] == 1
    assert set(annotations["intent_preserved"]) == {"uncertain"}
    assert not set(annotations.columns) & {"original", "perturbed"}


def test_bad_export_cannot_replace_existing_catalog_annotations() -> None:
    catalog = _catalog()
    previous = pd.DataFrame({"keep": ["existing evidence"]})
    catalog.save("semantic_audit_annotations", previous)
    bad = catalog.load("semantic_audit_reviewer_2_export")
    bad.loc[0, "dataset_fingerprint"] = "wrong"
    catalog.save("semantic_audit_reviewer_2_export", bad)

    with pytest.raises(ValueError, match="fingerprint"):
        combine_catalog_exports(catalog, {"reviewer_ids": ["reviewer_1", "reviewer_2"]})
    pd.testing.assert_frame_equal(catalog.load("semantic_audit_annotations"), previous)


@pytest.mark.parametrize("identities", [[], ["one", "two"], ["reviewer_1", "reviewer_1"]])
def test_combination_requires_the_two_confirmed_reviewer_roles(identities: list[str]) -> None:
    with pytest.raises(ValueError, match="reviewer_ids"):
        combine_catalog_exports(_catalog(), {"reviewer_ids": identities})


@pytest.mark.parametrize("port", [-1, 65536, True])
def test_server_rejects_invalid_port_values(port: Any) -> None:
    with pytest.raises(ValueError, match="port"):
        build_review_server("<!doctype html>", port=port)
