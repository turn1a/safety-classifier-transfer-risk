"""Nodes for the risk pipeline (SPEC.md §3.1 steps 6-7, §11)."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from transfer_risk.lib.ablation import selection_ablation
from transfer_risk.lib.association import spearman_association
from transfer_risk.lib.public_bundle import apply_corrected_dbs
from transfer_risk.lib.seeds import derive_seeds

logger = logging.getLogger(__name__)
_FEATURES = ["mean_cka", "dbs"]
_MIN_CV = 5
_MIN_CORR = 3
_SUCCESS_EFFECT_PP = 5.0


def recompute_risk_master_dbs(
    master_results_table: pd.DataFrame,
    cka_matrices: dict[str, Any],
    params_similarity: dict[str, Any],
) -> pd.DataFrame:
    """Recompute DBS for the risk stage from saved CKA matrices.

    Args:
        master_results_table: Original saved transfer-stage master table.
        cka_matrices: Saved target-vs-surrogate CKA matrices.
        params_similarity: Similarity-stage parameters supplying ``dbs.box``.

    Returns:
        A separate corrected master table for regression, ablation, and run metrics.
    """
    box = int(params_similarity["dbs"]["box"])
    corrected = apply_corrected_dbs(master_results_table, cka_matrices, box=box)
    logger.info("Prepared corrected DBS risk input for %d rows (box=%d)", len(corrected), box)
    return corrected


def fit_regressors(master: pd.DataFrame, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit shallow tree regressors and report similarity-vs-transfer associations.

    With the small sample (n ~ surrogates x recipes) the trees are read for feature
    importance as descriptive context, not primary inferential evidence. The primary
    association evidence is Spearman(similarity, transfer) computed after aggregating
    transfer rates by surrogate. Recipe-level associations are reported separately as a
    sensitivity view. Cross-validated R2 is reported only when n is large enough.

    Rows whose similarity features are non-finite are dropped first: a surrogate that
    failed to yield valid representations (so CKA/DBS could not be computed) cannot inform
    the regression, and scikit-learn rejects NaN inputs outright.
    """
    finite = np.isfinite(master[_FEATURES].to_numpy()).all(axis=1)
    if not bool(finite.all()):
        dropped = sorted(master.loc[~finite, "surrogate"].unique())
        logger.warning(
            "Dropping %d row(s) with non-finite %s before regression (surrogates: %s)",
            int((~finite).sum()),
            _FEATURES,
            dropped,
        )
        master = master.loc[finite]
    features = master[_FEATURES].to_numpy()
    target = master["transfer_rate"].to_numpy()
    n = len(master)
    tree = DecisionTreeRegressor(max_depth=params["decision_tree"]["max_depth"], random_state=seed)
    forest = RandomForestRegressor(
        n_estimators=params["random_forest"]["n_estimators"],
        max_depth=params["random_forest"]["max_depth"],
        random_state=seed,
    )
    tree.fit(features, target)
    forest.fit(features, target)
    result: dict[str, Any] = {
        "n_samples": n,
        "feature_names": _FEATURES,
        "decision_tree_importances": tree.feature_importances_.tolist(),
        "random_forest_importances": forest.feature_importances_.tolist(),
        "decision_tree_cv_r2": None,
        "random_forest_cv_r2": None,
        "surrogate_association": {},
        "recipe_association": {},
        "spearman": {},
        "models": {"decision_tree": tree, "random_forest": forest},
    }
    if n >= _MIN_CV:
        result["decision_tree_cv_r2"] = float(cross_val_score(tree, features, target, cv=5).mean())
        result["random_forest_cv_r2"] = float(
            cross_val_score(forest, features, target, cv=5).mean()
        )
    surrogate_means = _aggregate_surrogate_transfer(master)
    result["surrogate_association"] = _association_by_feature(surrogate_means)
    result["recipe_association"] = _association_by_feature(master)
    # Compatibility alias: keep the historical key while pointing it at the new primary result.
    result["spearman"] = result["surrogate_association"]
    logger.info(
        (
            "Regression fit on %d rows (%d surrogate aggregates); RF importances %s; "
            "primary surrogate association keys %s"
        ),
        n,
        len(surrogate_means),
        result["random_forest_importances"],
        sorted(result["surrogate_association"]),
    )
    return result


def run_ablation(
    master: pd.DataFrame,
    selection: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Compare high-CKA (M1) vs low-CKA (M2) surrogates on transfer rate.

    The pure :func:`selection_ablation` runs a one-sided permutation test on the difference in
    group-mean transfer rate between M1 (similarity >= r1) and M2 (similarity <= r2), on both
    each surrogate's mean-across-recipes and max-across-recipes transfer. With small groups the
    permutation p-value has a coarse floor (three vs three gives a minimum of 1/20 = 0.05), so
    the exact p-value is reported, not thresholded blindly. ``effect_size_pp`` and
    ``empirical_p_value`` alias the mean-transfer contrast for the metrics node and the figure.
    """
    transfer_mean = {
        str(name): float(value)
        for name, value in master.groupby("surrogate")["transfer_rate"].mean().items()
    }
    transfer_max = {
        str(name): float(value)
        for name, value in master.groupby("surrogate")["transfer_rate"].max().items()
    }
    m1 = [name for name in selection["M1"] if name in transfer_mean]
    m2 = [name for name in selection["M2"] if name in transfer_mean]
    if not m1 or not m2:
        logger.warning("Ablation skipped: M1=%s M2=%s (need >=1 attacked surrogate each)", m1, m2)
        return {
            "m1": m1,
            "m2": m2,
            "effect_size_pp": 0.0,
            "empirical_p_value": 1.0,
            "success_criterion_met": False,
            "note": "empty M1 or M2 after intersecting with attacked surrogates",
        }
    n_permutations = int(params["ablation"]["n_permutations"])
    rng = np.random.default_rng(derive_seeds(seed).numpy)
    stats = selection_ablation(
        transfer_mean, transfer_max, m1, m2, n_permutations=n_permutations, rng=rng
    )
    met = bool(
        stats["mean_diff_pp"] >= _SUCCESS_EFFECT_PP
        and stats["mean_p_value"] < params["ablation"]["alpha"]
    )
    result: dict[str, Any] = {
        "m1": m1,
        "m2": m2,
        **stats,
        "effect_size_pp": stats["mean_diff_pp"],
        "empirical_p_value": stats["mean_p_value"],
        "n_permutations": n_permutations,
        "success_criterion_met": met,
    }
    logger.info(
        "Ablation M1 vs M2: mean %.3f/%.3f (%+.1fpp p=%.3f), max %.3f/%.3f (%+.1fpp p=%.3f)",
        stats["m1_mean"],
        stats["m2_mean"],
        stats["mean_diff_pp"],
        stats["mean_p_value"],
        stats["m1_max_mean"],
        stats["m2_max_mean"],
        stats["max_diff_pp"],
        stats["max_p_value"],
    )
    return result


def track_run_metrics(
    master: pd.DataFrame,
    ablation: dict[str, Any],
    regressors: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Log the headline run scalars to MLflow and return them as a flat dict.

    Gathers the cross-surrogate aggregates — mean/max transfer rate, surrogate-level
    primary similarity-vs-transfer associations, recipe-level sensitivity associations,
    random-forest importances (descriptive only), the ablation effect/p-value, and the
    calibrated thresholds — and logs them as metrics on the active MLflow run (opened
    by kedro-mlflow). The same dict is returned so it is also persisted locally
    (``run_metrics``) for inspection.

    Args:
        master: master results table (one row per surrogate x recipe).
        ablation: the ablation result dict.
        regressors: the regression result dict (feature names, importances, associations).
        thresholds: the calibrated ``{"r1", "r2"}`` thresholds.

    Returns:
        Flat ``{metric_name: value}`` dict of the logged run metrics.
    """
    transfer = master["transfer_rate"]
    metrics: dict[str, float] = {
        "transfer_rate_mean": float(transfer.mean()),
        "transfer_rate_max": float(transfer.max()),
        "n_observations": float(len(master)),
        "threshold_r1": float(thresholds["r1"]),
        "threshold_r2": float(thresholds["r2"]),
        "ablation_effect_pp": float(ablation["effect_size_pp"]),
        "ablation_p_value": float(ablation["empirical_p_value"]),
        "ablation_max_effect_pp": float(ablation.get("max_diff_pp", 0.0)),
        "ablation_max_p_value": float(ablation.get("max_p_value", 1.0)),
        "m1_mean_transfer": float(ablation.get("m1_mean", 0.0)),
        "m2_mean_transfer": float(ablation.get("m2_mean", 0.0)),
    }
    for feature, importance in zip(
        regressors["feature_names"], regressors["random_forest_importances"], strict=False
    ):
        metrics[f"rf_importance_{feature}"] = float(importance)
    primary_association = regressors.get("surrogate_association", regressors.get("spearman", {}))
    recipe_association = regressors.get("recipe_association", {})
    for feature, stats in primary_association.items():
        rho, p_value, n_obs, exact = _association_scalars(stats)
        metrics[f"surrogate_spearman_{feature}_rho"] = rho
        metrics[f"surrogate_spearman_{feature}_p"] = p_value
        metrics[f"surrogate_spearman_{feature}_n"] = n_obs
        metrics[f"surrogate_spearman_{feature}_exact"] = exact
        # Backward-compatible metric names now map to the primary surrogate-level result.
        metrics[f"spearman_{feature}_rho"] = rho
        metrics[f"spearman_{feature}_p"] = p_value
    for feature, stats in recipe_association.items():
        rho, p_value, n_obs, exact = _association_scalars(stats)
        metrics[f"recipe_spearman_{feature}_rho"] = rho
        metrics[f"recipe_spearman_{feature}_p"] = p_value
        metrics[f"recipe_spearman_{feature}_n"] = n_obs
        metrics[f"recipe_spearman_{feature}_exact"] = exact
    _log_mlflow_metrics(metrics)
    logger.info("Logged %d run metrics to MLflow", len(metrics))
    return metrics


def _aggregate_surrogate_transfer(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recipe-level rows to one transfer/similarity row per surrogate.

    Args:
        master: Master results table (one row per surrogate x recipe).

    Returns:
        DataFrame with one row per surrogate and columns ``mean_cka``, ``dbs``,
        and mean ``transfer_rate``.
    """
    return (
        master.groupby("surrogate", as_index=False)
        .agg(
            mean_cka=("mean_cka", "mean"),
            dbs=("dbs", "mean"),
            transfer_rate=("transfer_rate", "mean"),
        )
        .reset_index(drop=True)
    )


def _association_by_feature(table: pd.DataFrame) -> dict[str, dict[str, float | int | bool]]:
    """Compute per-feature Spearman association against ``transfer_rate``.

    Args:
        table: Input table with feature columns and ``transfer_rate``.

    Returns:
        Mapping ``feature -> {rho, two_sided_p, n, exact}``. Empty when the table
        has fewer than ``_MIN_CORR`` rows.
    """
    if len(table) < _MIN_CORR:
        return {}
    association: dict[str, dict[str, float | int | bool]] = {}
    for feature in _FEATURES:
        association[feature] = spearman_association(
            table[feature].tolist(),
            table["transfer_rate"].tolist(),
        )
    return association


def _association_scalars(stats: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract metric-ready scalars from association stats.

    Args:
        stats: Association stats dict.

    Returns:
        Tuple ``(rho, p_value, n_obs, exact_flag)`` as floats.
    """
    rho = float(stats["rho"])
    p_value = float(stats["two_sided_p"]) if "two_sided_p" in stats else float(stats["p"])
    n_obs = float(stats.get("n", 0))
    exact = float(bool(stats.get("exact", False)))
    return rho, p_value, n_obs, exact


def _log_mlflow_metrics(metrics: dict[str, float]) -> None:
    """Log the finite metrics to the active MLflow run, if kedro-mlflow has opened one."""
    import mlflow  # noqa: PLC0415  # optional tracking glue; imported only when used

    if mlflow.active_run() is None:
        return
    mlflow.log_metrics({key: value for key, value in metrics.items() if math.isfinite(value)})
