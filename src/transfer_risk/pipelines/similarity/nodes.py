"""Nodes for the similarity pipeline (SPEC.md §3.1-§3.3).

The pure maths is delegated to :mod:`transfer_risk.lib` (CKA, DBS, thresholds); per-layer
representation extraction lives in :mod:`transfer_risk.modeling`. The target's representations
are computed once (it arrives as a loaded ``target_model`` bundle); each surrogate's CKA is its
own node (input its ``surrogate.{name}`` checkpoint path), and a reduce step assembles them.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from transfer_risk.devices import resolve_device
from transfer_risk.lib.cka import cka_matrix
from transfer_risk.lib.dbs import diagonal_box_similarity
from transfer_risk.lib.seeds import derive_seeds
from transfer_risk.lib.thresholds import calibrate
from transfer_risk.modeling import layer_representations, representations_from_loaded

logger = logging.getLogger(__name__)


def build_probe_set(canonical: pd.DataFrame, params: dict[str, Any], seed: int) -> pd.DataFrame:
    """Sample a fixed, label-balanced probe set used for every model's CKA."""
    n_probe = int(params["n_probe"])
    rng_seed = derive_seeds(seed).numpy
    half = n_probe // 2
    positives = canonical[canonical["label"] == 1]
    negatives = canonical[canonical["label"] == 0]
    probe = pd.concat(
        [
            positives.sample(min(half, len(positives)), random_state=rng_seed),
            negatives.sample(min(n_probe - half, len(negatives)), random_state=rng_seed),
        ]
    )
    return probe.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)


def compute_target_reps(
    target: dict[str, Any],
    probe_set: pd.DataFrame,
    sim_params: dict[str, Any],
    device_params: dict[str, Any],
) -> list[Any]:
    """Compute the frozen target's per-layer probe representations once (reused by every CKA)."""
    device = resolve_device(device_params["policy"])
    return representations_from_loaded(
        target["model"],
        target["tokenizer"],
        probe_set["text"].tolist(),
        pooling=sim_params["pooling"],
        max_seq_len=int(sim_params.get("max_seq_len", 256)),
        batch_size=sim_params["cka"]["batch_size"],
        device=device,
    )


def compute_cka(
    surrogate_path: str,
    target_reps: list[Any],
    probe_set: pd.DataFrame,
    sim_params: dict[str, Any],
    device_params: dict[str, Any],
    *,
    kind: str,
) -> Any:
    """Build the target-vs-surrogate layer CKA matrix for one surrogate.

    Args:
        surrogate_path: the surrogate's local checkpoint directory (from ``surrogate.{name}``).
        target_reps: the precomputed target representations.
        probe_set: the shared probe texts.
        sim_params: the ``similarity`` block.
        device_params: the ``device`` block.
        kind: the surrogate kind (``pretrained`` / ``finetune`` / ``bilstm``), bound at build time.

    Returns:
        The layer-pair CKA matrix.
    """
    device = resolve_device(device_params["policy"])
    spec = {"kind": kind, "source": surrogate_path}
    surrogate_reps = layer_representations(
        spec,
        probe_set["text"].tolist(),
        pooling=sim_params["pooling"],
        max_seq_len=int(sim_params.get("max_seq_len", 256)),
        batch_size=sim_params["cka"]["batch_size"],
        device=device,
    )
    matrix = cka_matrix(target_reps, surrogate_reps)
    logger.info("CKA matrix shape %s", matrix.shape)
    return matrix


def reduce_cka(*matrices: Any, names: list[str]) -> dict[str, Any]:
    """Assemble the per-surrogate CKA matrices into a ``{surrogate -> matrix}`` mapping."""
    return dict(zip(names, matrices, strict=True))


def reduce_similarity(cka_matrices: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    """Reduce each CKA matrix to (mean CKA, DBS) scalars per surrogate."""
    box = int(params["dbs"]["box"])
    rows = [
        {
            "surrogate": name,
            "mean_cka": float(matrix.mean()),
            "dbs": diagonal_box_similarity(matrix, box),
        }
        for name, matrix in cka_matrices.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_cka", ascending=False).reset_index(drop=True)


def calibrate_thresholds(
    similarity_table: pd.DataFrame, params: dict[str, Any]
) -> dict[str, float]:
    """Calibrate r1/r2 from the observed mean-CKA distribution (quartiles)."""
    quantiles = params["thresholds"]
    thresholds = calibrate(
        similarity_table["mean_cka"].tolist(),
        quantiles["r1_quantile"],
        quantiles["r2_quantile"],
    )
    return {"r1": thresholds.r1, "r2": thresholds.r2}


def select_surrogates(
    similarity_table: pd.DataFrame, thresholds: dict[str, float]
) -> dict[str, Any]:
    """Split surrogates into high-similarity (M1) and low-similarity (M2) pools."""
    r1, r2 = thresholds["r1"], thresholds["r2"]
    high = similarity_table.loc[similarity_table["mean_cka"] >= r1, "surrogate"].tolist()
    low = similarity_table.loc[similarity_table["mean_cka"] <= r2, "surrogate"].tolist()
    logger.info("Selected M1=%s M2=%s (r1=%.4f r2=%.4f)", high, low, r1, r2)
    return {"M1": high, "M2": low, "r1": r1, "r2": r2, "signal": "mean_cka"}
