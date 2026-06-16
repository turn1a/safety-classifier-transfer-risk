"""Risk stage: fit transfer-rate regressors and run the CKA-guided-vs-random ablation."""

from transfer_risk.pipelines.risk.pipeline import create_pipeline

__all__ = ["create_pipeline"]
