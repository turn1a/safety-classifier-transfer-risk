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

from transfer_risk.lib.ablation import ablation_statistics
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
    """
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
    """Compare CKA-guided selection vs random selection on max transfer rate.

    Guided selection is M1 + M2. The pure :func:`ablation_statistics` draws ``n_bootstrap``
    random subsets of the same size from the full pool; the empirical p-value is the
    fraction of random subsets whose max transfer rate matches or beats the guided one.
    """
    per_surrogate = master.groupby("surrogate")["transfer_rate"].max()
    per_surrogate_max = {str(name): float(value) for name, value in per_surrogate.items()}
    pool = list(per_surrogate_max)
    guided = [name for name in [*selection["M1"], *selection["M2"]] if name in per_surrogate_max]
    if not guided:
        return {"effect_size_pp": 0.0, "empirical_p_value": 1.0, "note": "no guided surrogates"}
    n_bootstrap = int(params["ablation"]["n_bootstrap"])
    rng = np.random.default_rng(derive_seeds(seed).numpy)
    stats = ablation_statistics(per_surrogate_max, guided, pool, n_bootstrap, rng)
    met = bool(
        stats["effect_size_pp"] >= _SUCCESS_EFFECT_PP
        and stats["empirical_p_value"] < params["ablation"]["alpha"]
    )
    result: dict[str, Any] = {
        "guided_surrogates": guided,
        **stats,
        "n_bootstrap": n_bootstrap,
        "success_criterion_met": met,
    }
    logger.info(
        "Ablation: guided=%.3f random=%.3f effect=%.1fpp p=%.3f",
        stats["guided_max_transfer"],
        stats["random_max_mean"],
        stats["effect_size_pp"],
        stats["empirical_p_value"],
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
