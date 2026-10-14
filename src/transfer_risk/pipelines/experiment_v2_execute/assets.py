"""Gate catalog materialization of pinned offline recipe assets before core registration."""

from __future__ import annotations

from typing import Any

from transfer_risk.lib.experiment_v2.protocol import content_digest
from transfer_risk.lib.experiment_v2.schemas import Protocol
from transfer_risk.pipelines.experiment_v2_execute.nodes import (
    _check_context,
    _check_structure,
    _fail,
)


def gate_assets_preparation(
    parameters: dict[str, Any],
    context: dict[str, Any],
    root_seed: int,
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate asset copying without requiring yet-unrealized core evidence or model hashes.

    Args:
        parameters: Intended v2 run, private root, seed, and source commit.
        context: Actual isolated committed source state.
        root_seed: Shared project root seed.
        structure: Intended committed axes bound by the asset-preparation factory.

    Returns:
        Fresh preparation receipt; the catalog independently validates every asset pin.
    """
    protocol = Protocol.model_validate(parameters)
    _check_context(context, protocol.execution_ref)
    _check_structure(protocol, structure)
    if (
        root_seed != protocol.root_seed
        or protocol.private_root != f"data/experiment_v2/{protocol.run_id}"
    ):
        _fail("asset preparation must use the intended isolated run and project seed")
    return {
        "protocol_sha256": content_digest(protocol),
        "run_id": protocol.run_id,
        "execution_ref": protocol.execution_ref,
        "private_root": protocol.private_root,
    }


def materialize_asset_bundle(
    receipt: dict[str, Any], source_bundle: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Pass pinned staging bytes to their catalog-owned immutable cache boundary.

    Args:
        receipt: Fresh preparation gate.
        source_bundle: Non-text operator input with source_root and frozen manifest.
        parameters: Intended private v2 run.

    Returns:
        The bundle validated, copied, and hashed by ExperimentV2AttackAssetsDataset.save.
    """
    protocol = Protocol.model_validate(parameters)
    if (
        receipt.get("protocol_sha256") != content_digest(protocol)
        or source_bundle.get("manifest", {}).get("run_id") != protocol.run_id
    ):
        _fail("auxiliary preparation source has a stale or different run identity")
    return source_bundle
