"""Smoke stage: a single runnable node proving catalog + MLflow wiring.

Unlike the domain pipelines (whose nodes are stubs in this scaffold), this node
is implemented so ``kedro run --pipeline smoke`` executes end to end.
"""

from transfer_risk.pipelines.smoke.pipeline import create_pipeline

__all__ = ["create_pipeline"]
