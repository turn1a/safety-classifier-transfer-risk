"""Reporting pipeline assembly."""

from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.reporting.nodes import (
    plot_cka_heatmap,
    plot_regression_ablation,
    plot_transfer_scatter,
)


def create_pipeline() -> Pipeline:
    """Assemble the reporting pipeline."""
    return Pipeline(
        [
            node(
                plot_cka_heatmap,
                inputs="similarity_table",
                outputs="fig_cka_heatmap",
                name="plot_cka_heatmap",
            ),
            node(
                plot_transfer_scatter,
                inputs="master_results_table",
                outputs="fig_transfer_scatter",
                name="plot_transfer_scatter",
            ),
            node(
                plot_regression_ablation,
                inputs=["regressors", "ablation_results"],
                outputs="fig_regression_ablation",
                name="plot_regression_ablation",
            ),
        ]
    )
