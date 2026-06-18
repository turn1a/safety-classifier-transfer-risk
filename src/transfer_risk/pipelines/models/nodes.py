"""Nodes for the models pipeline (SPEC.md §5, §7).

Each surrogate is prepared by its own node so the work is catalog-driven and resumable: a
``pretrained`` model is materialised from its ``hub.{name}`` source into a ``surrogate.{name}``
checkpoint unchanged, a ``finetune`` backbone is fine-tuned, and the BiLSTM is trained from
scratch. Every transformer surrogate is then exported to ONNX in-process. No model is loaded
from the Hub or saved to disk inside a node — the ``hub.{name}`` / ``surrogate.{name}`` /
``onnx.{name}`` catalog datasets own all model I/O.
"""

from __future__ import annotations

import logging
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


def build_surrogate_registry(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the surrogate pool; verify HF auth if gated models are used.

    Args:
        params: the ``models`` parameter block (``target``, ``surrogates``).

    Returns:
        A registry dict with the ``target`` id and the validated ``surrogates`` specs.
    """
    specs = list(params["surrogates"])
    validate_surrogate_specs(specs)
    if requires_gated_auth(specs):
        logger.info("HuggingFace authenticated as %s (gated models enabled)", assert_hf_auth())
    return {"target": params["target"], "surrogates": specs}


def _device_and_seed(device_params: dict[str, Any], seed: int) -> tuple[Any, int]:
    """Resolve the training device and the derived torch seed."""
    return resolve_device(device_params["policy"]), derive_seeds(seed).torch


def materialize_surrogate(hub_bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pass a pretrained Hub model through to its checkpoint unchanged; return bundle + metadata.

    Args:
        hub_bundle: the ``{"kind", "model", "tokenizer"}`` bundle loaded from ``hub__{name}``.

    Returns:
        ``(bundle, meta)``: the same bundle (Kedro saves it via ``surrogate.{name}``) and a
        provenance fragment.
    """
    num_params = int(sum(p.numel() for p in hub_bundle["model"].parameters()))
    return hub_bundle, {"kind": "pretrained", "num_params": num_params}


def train_surrogate(
    backbone: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    models_params: dict[str, Any],
    device_params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fine-tune a backbone on the canonical task; return its checkpoint bundle and metadata.

    Args:
        backbone: the ``{"model", "tokenizer"}`` loaded from the surrogate's ``hub.{name}``.
        splits: ``train`` / ``val`` / ``test`` DataFrames.
        models_params: the ``models`` block (``finetune`` sub-config).
        device_params: the ``device`` block (``policy``).
        seed: the project root seed.

    Returns:
        ``(bundle, meta)`` for the ``surrogate.{name}`` output and its provenance fragment.
    """
    device, torch_seed = _device_and_seed(device_params, seed)
    return finetune_transformer(
        backbone, splits["train"], splits["val"], models_params["finetune"], device, torch_seed
    )


def train_bilstm_surrogate(
    splits: dict[str, pd.DataFrame],
    models_params: dict[str, Any],
    device_params: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the from-scratch BiLSTM surrogate; return its checkpoint bundle and metadata."""
    device, torch_seed = _device_and_seed(device_params, seed)
    return train_bilstm(
        splits["train"],
        splits["val"],
        params=models_params["bilstm"],
        device=device,
        seed=torch_seed,
    )


def export_onnx(surrogate_path: str, models_params: dict[str, Any]) -> dict[str, Any]:
    """Export a transformer surrogate to an ONNX graph (+ tokenizer) for the fast victim path.

    Args:
        surrogate_path: the local checkpoint directory from ``surrogate.{name}``.
        models_params: the ``models`` block (the ``finetune.max_seq_len`` trace length).

    Returns:
        ``{"onnx_bytes", "tokenizer"}`` for the ``onnx.{name}`` output.
    """
    import torch  # noqa: PLC0415

    from transfer_risk.modeling import export_to_onnx, load_transformer  # noqa: PLC0415

    model, tokenizer = load_transformer(surrogate_path, torch.device("cpu"))
    max_seq_len = int(models_params["finetune"]["max_seq_len"])
    onnx_bytes = export_to_onnx(model, tokenizer, max_seq_len=max_seq_len)
    return {"onnx_bytes": onnx_bytes, "tokenizer": tokenizer}
