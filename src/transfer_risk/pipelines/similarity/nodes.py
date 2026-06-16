"""Nodes for the similarity pipeline (SPEC.md §3.1-§3.3).

The pure maths is delegated to :mod:`transfer_risk.lib` (CKA, DBS, thresholds);
per-layer representation extraction lives in :mod:`transfer_risk.modeling`.
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
from transfer_risk.modeling import layer_representations

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


def compute_cka_matrices(
    probe_set: pd.DataFrame,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    sim_params: dict[str, Any],
    device_params: dict[str, Any],
) -> dict[str, Any]:
    """Build the target-vs-surrogate layer CKA matrix for each surrogate."""
    device = resolve_device(device_params["policy"])
    texts = probe_set["text"].tolist()
    pooling = sim_params["pooling"]
    batch_size = sim_params["cka"]["batch_size"]
    max_seq_len = int(sim_params.get("max_seq_len", 256))
    target_spec = {"kind": "pretrained", "source": registry["target"]}
    target_reps = layer_representations(
        target_spec,
        texts,
        pooling=pooling,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )
    matrices: dict[str, Any] = {}
    for name, entry in manifest.items():
        surrogate_reps = layer_representations(
            entry,
            texts,
            pooling=pooling,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
        )
        matrices[name] = cka_matrix(target_reps, surrogate_reps)
        logger.info("CKA matrix %s: shape %s", name, matrices[name].shape)
    return matrices


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
