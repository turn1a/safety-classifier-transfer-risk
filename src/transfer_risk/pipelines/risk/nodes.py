"""Nodes for the risk pipeline (SPEC.md §3.1 steps 6-7, §11)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from transfer_risk.lib.seeds import derive_seeds

if TYPE_CHECKING:
    import pandas as pd

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


def _random_subset(pool: list[str], size: int, rng: Any) -> list[str]:
    if size >= len(pool):
        return pool
    return [str(item) for item in rng.choice(pool, size=size, replace=False)]


def run_ablation(
    master: pd.DataFrame,
    selection: dict[str, Any],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Compare CKA-guided selection vs random selection on max transfer rate.

    Guided selection is M1 + M2. For each of ``n_bootstrap`` reseeds a random subset of
    the same size is drawn from the surrogate pool; the empirical p-value is the
    fraction of random subsets whose max transfer rate matches or beats the guided one.
    """
    per_surrogate = master.groupby("surrogate")["transfer_rate"].max()
    pool = [str(name) for name in per_surrogate.index]
    guided = [name for name in [*selection["M1"], *selection["M2"]] if name in pool]
    if not guided:
        return {"effect_size_pp": 0.0, "empirical_p_value": 1.0, "note": "no guided surrogates"}
    guided_max = float(per_surrogate.loc[guided].max())
    size = len(guided)
    n_bootstrap = int(params["ablation"]["n_bootstrap"])
    rng = np.random.default_rng(derive_seeds(seed).numpy)
    random_maxes = np.array(
        [
            float(per_surrogate.loc[_random_subset(pool, size, rng)].max())
            for _ in range(n_bootstrap)
        ]
    )
    effect_pp = (guided_max - float(random_maxes.mean())) * 100.0
    p_value = float((random_maxes >= guided_max).mean())
    met = bool(effect_pp >= _SUCCESS_EFFECT_PP and p_value < params["ablation"]["alpha"])
    result = {
        "guided_surrogates": guided,
        "guided_max_transfer": guided_max,
        "random_max_mean": float(random_maxes.mean()),
        "random_max_std": float(random_maxes.std()),
        "effect_size_pp": effect_pp,
        "empirical_p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "success_criterion_met": met,
    }
    logger.info(
        "Ablation: guided=%.3f random=%.3f effect=%.1fpp p=%.3f",
        guided_max,
        result["random_max_mean"],
        effect_pp,
        p_value,
    )
    return result
