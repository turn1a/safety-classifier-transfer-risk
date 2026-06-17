"""Nodes for the models pipeline (SPEC.md §5, §7)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from transfer_risk.devices import resolve_device
from transfer_risk.lib.seeds import derive_seeds
from transfer_risk.pipelines.models.registry import (
    assert_hf_auth,
    requires_gated_auth,
    validate_surrogate_specs,
)
from transfer_risk.pipelines.models.training import finetune_transformer, train_bilstm

logger = logging.getLogger(__name__)
_MODELS_DIR = Path("data/06_models")


def build_surrogate_registry(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the surrogate pool; verify HF auth if gated models are used.

    Args:
        params: The ``models`` parameter block (``target``, ``surrogates``).

    Returns:
        A registry dict with the ``target`` id and the validated ``surrogates`` specs.
    """
    specs = list(params["surrogates"])
    validate_surrogate_specs(specs)
    if requires_gated_auth(specs):
        logger.info("HuggingFace authenticated as %s (gated models enabled)", assert_hf_auth())
    return {"target": params["target"], "surrogates": specs}


def prepare_surrogates(
    splits: dict[str, pd.DataFrame],
    registry: dict[str, Any],
    models_params: dict[str, Any],
    device_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Record pre-fine-tuned models, fine-tune the rest, and train the BiLSTM outlier.

    Args:
        splits: ``train`` / ``val`` / ``test`` DataFrames.
        registry: Output of :func:`build_surrogate_registry`.
        models_params: The ``models`` block (``finetune`` and ``bilstm`` sub-configs).
        device_params: The ``device`` block (``policy``).
        seed: The project root seed.

    Returns:
        A manifest mapping each surrogate name to ``{kind, source, ...metadata}``.
    """
    device = resolve_device(device_params["policy"])
    torch_seed = derive_seeds(seed).torch
    train_df, val_df = splits["train"], splits["val"]
    finetune_cfg, bilstm_cfg = models_params["finetune"], models_params["bilstm"]
    manifest: dict[str, Any] = {}
    for spec in registry["surrogates"]:
        name, kind = spec["name"], spec["kind"]
        if kind == "pretrained":
            manifest[name] = {
                "kind": "pretrained",
                "source": spec["id"],
                "gated": bool(spec.get("gated", False)),
            }
        elif kind == "finetune":
            out_dir = _MODELS_DIR / name
            meta = finetune_transformer(
                spec["id"], train_df, val_df, out_dir, finetune_cfg, device, torch_seed
            )
            manifest[name] = {
                "kind": "finetune",
                "source": str(out_dir),
                "backbone": spec["id"],
                **meta,
            }
        else:  # bilstm
            out_dir = _MODELS_DIR / name
            meta = train_bilstm(
                train_df, val_df, out_dir, params=bilstm_cfg, device=device, seed=torch_seed
            )
            manifest[name] = {"kind": "bilstm", "source": str(out_dir), **meta}
        logger.info("Prepared surrogate %s (%s)", name, kind)
    return manifest
