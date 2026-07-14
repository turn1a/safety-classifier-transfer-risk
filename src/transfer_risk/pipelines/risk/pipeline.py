"""Risk pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.risk.nodes import (
    fit_regressors,
    recompute_risk_master_dbs,
    run_ablation,
    track_run_metrics,
)


def create_pipeline() -> Pipeline:
    """Assemble the risk pipeline."""
    return Pipeline(
        [
            node(
                recompute_risk_master_dbs,
                inputs=["master_results_table", "cka_matrices", "params:similarity"],
                outputs="risk_master_results_corrected",
                name="recompute_risk_master_dbs",
            ),
            node(
                fit_regressors,
                inputs=["risk_master_results_corrected", "params:risk", "params:seed"],
                outputs="regressors",
                name="fit_regressors",
            ),
            node(
                run_ablation,
                inputs=[
                    "risk_master_results_corrected",
                    "surrogate_selection",
                    "params:risk",
                    "params:seed",
                ],
                outputs="ablation_results",
                name="run_ablation",
            ),
            node(
                track_run_metrics,
                inputs=[
                    "risk_master_results_corrected",
                    "ablation_results",
                    "regressors",
                    "thresholds",
                ],
                outputs="run_metrics",
                name="track_run_metrics",
            ),
        ]
    )
