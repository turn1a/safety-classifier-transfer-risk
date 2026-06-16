"""Risk pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.risk.nodes import (
    fit_regressors,
    run_ablation,
    track_run_metrics,
)


def create_pipeline() -> Pipeline:
    """Assemble the risk pipeline."""
    return Pipeline(
        [
            node(
                fit_regressors,
                inputs=["master_results_table", "params:risk", "params:seed"],
                outputs="regressors",
                name="fit_regressors",
            ),
            node(
                run_ablation,
                inputs=[
                    "master_results_table",
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
                    "master_results_table",
                    "ablation_results",
                    "regressors",
                    "thresholds",
                ],
                outputs="run_metrics",
                name="track_run_metrics",
            ),
        ]
    )
