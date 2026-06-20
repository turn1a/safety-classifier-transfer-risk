"""The attacks pipeline serves DeBERTa victims from torch and the standard models from ONNX.

DeBERTa-v2/v3's disentangled-attention ONNX graph triggers a MatMul dimension mismatch in the
cloud box's aarch64 onnxruntime at every sequence length (HF transformers #18237), so those
surrogates carry ``victim: torch`` and attack from their torch checkpoint instead. These build-time
tests guard the routing and the config requirement (a new DeBERTa surrogate without the flag is the
exact mistake that aborted a full cloud sweep). They build the pipeline only, so they stay in the
fast suite (no textattack/torch import).
"""

from __future__ import annotations

from transfer_risk.pipelines._dynamic import surrogate_specs
from transfer_risk.pipelines.attacks.pipeline import create_pipeline


def _victim_dataset_kind(pipeline: object, surrogate: str) -> str:
    """Return ``onnx`` or ``surrogate`` for the victim dataset wired into a surrogate's attacks."""
    for node in pipeline.nodes:  # type: ignore[attr-defined]
        if not node.name.startswith("attack_"):
            continue
        for dataset in node.inputs:
            if dataset.endswith(f"__{surrogate}") and dataset.startswith(("onnx__", "surrogate__")):
                return dataset.split("__", 1)[0]
    return ""


def test_attacks_victim_routing_follows_flag() -> None:
    """The victim is torch for the BiLSTM and victim: torch surrogates, ONNX for the rest."""
    pipeline = create_pipeline()
    routings = {_victim_dataset_kind(pipeline, spec["name"]) for spec in surrogate_specs()}
    assert routings == {"onnx", "surrogate"}  # both paths are exercised by the pool
    for spec in surrogate_specs():
        torch_victim = spec["kind"] == "bilstm" or spec.get("victim") == "torch"
        expected = "surrogate" if torch_victim else "onnx"
        assert _victim_dataset_kind(pipeline, spec["name"]) == expected, spec["name"]


def test_all_deberta_surrogates_are_flagged_torch() -> None:
    """Every DeBERTa surrogate must carry victim: torch (ONNX fails on aarch64)."""
    deberta = [spec for spec in surrogate_specs() if "deberta" in spec["name"]]
    assert deberta  # the pool has DeBERTa surrogates to protect
    for spec in deberta:
        assert spec.get("victim") == "torch", (
            f"{spec['name']} is a DeBERTa surrogate and must be `victim: torch`: the box's aarch64 "
            "onnxruntime cannot run its disentangled-attention ONNX graph"
        )
