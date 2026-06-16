"""Data stage: build the canonical prompt-injection dataset and deterministic splits."""

from transfer_risk.pipelines.data.pipeline import create_pipeline

__all__ = ["create_pipeline"]
