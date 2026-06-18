"""Nodes for the risk pipeline (SPEC.md §3.1 steps 6-7, §11)."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from transfer_risk.lib.ablation import selection_ablation
from transfer_risk.lib.seeds import derive_seeds

logger = logging.getLogger(__name__)
_FEATURES = ["mean_cka", "dbs"]
_MIN_CV = 5
_MIN_CORR = 3
_SUCCESS_EFFECT_PP = 5.0


def fit_regressors(master: pd.DataFrame, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fit shallow DecisionTree + RandomForest regressors and report correlations.

    With the small sample (n ~ surrogates x recipes) the trees are read for feature
    importance, not prediction; Spearman(similarity, transfer) is the primary
    correlation evidence. Cross-validated R2 is reported only when n is large enough.

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
        "spearman": {},
        "models": {"decision_tree": tree, "random_forest": forest},
    }
    if n >= _MIN_CV:
        result["decision_tree_cv_r2"] = float(cross_val_score(tree, features, target, cv=5).mean())
        result["random_forest_cv_r2"] = float(
            cross_val_score(forest, features, target, cv=5).mean()
        )
    if n >= _MIN_CORR:
        for feature in _FEATURES:
            rho, p_value = spearmanr(master[feature], master["transfer_rate"])
            result["spearman"][feature] = {"rho": float(rho), "p": float(p_value)}
    logger.info(
        "Regression fit on %d rows; RF importances %s", n, result["random_forest_importances"]
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

    Gathers the cross-surrogate aggregates — mean/max transfer rate, the
    similarity-vs-transfer Spearman correlations, random-forest feature importances, the
    ablation effect/p-value, and the calibrated thresholds — and logs them as metrics on
    the active MLflow run (opened by kedro-mlflow). The same dict is returned so it is
    also persisted locally (``run_metrics``) for inspection.

    Args:
        master: master results table (one row per surrogate x recipe).
        ablation: the ablation result dict.
        regressors: the regression result dict (feature names, importances, spearman).
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
    for feature, stats in regressors.get("spearman", {}).items():
        metrics[f"spearman_{feature}_rho"] = float(stats["rho"])
        metrics[f"spearman_{feature}_p"] = float(stats["p"])
    _log_mlflow_metrics(metrics)
    logger.info("Logged %d run metrics to MLflow", len(metrics))
    return metrics


def _log_mlflow_metrics(metrics: dict[str, float]) -> None:
    """Log the finite metrics to the active MLflow run, if kedro-mlflow has opened one."""
    import mlflow  # noqa: PLC0415  # optional tracking glue; imported only when used

    if mlflow.active_run() is None:
        return
    mlflow.log_metrics({key: value for key, value in metrics.items() if math.isfinite(value)})
