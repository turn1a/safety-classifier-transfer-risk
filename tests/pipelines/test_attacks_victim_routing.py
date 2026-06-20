"""Every attack victim is served from its torch checkpoint by default.

The cloud box's aarch64 onnxruntime fails every transformer's fused ONNX attention with a MatMul
dimension mismatch — DeBERTa-v2 disentangled attention and the standard BERT/RoBERTa/ELECTRA graphs
alike — so the sweep serves every surrogate from its torch checkpoint. A transformer may opt back
into its faster ONNX graph with ``victim: onnx`` only where that is verified; the base config opts
none in. These build-time tests guard the routing (no textattack/torch import, so they stay fast).
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


def test_every_victim_defaults_to_the_torch_checkpoint() -> None:
    """No surrogate wires an ONNX victim — the aarch64 box can't run transformer ONNX."""
    pipeline = create_pipeline()
    routing = {s["name"]: _victim_dataset_kind(pipeline, s["name"]) for s in surrogate_specs()}
    onnx_victims = sorted(name for name, kind in routing.items() if kind == "onnx")
    assert not onnx_victims, f"surrogates wiring an ONNX victim (broken on aarch64): {onnx_victims}"
    assert set(routing.values()) == {"surrogate"}


def test_base_config_opts_no_surrogate_into_onnx() -> None:
    """The base config opts no surrogate into ONNX (victim: onnx is for verified platforms only)."""
    opted_in = [spec["name"] for spec in surrogate_specs() if spec.get("victim") == "onnx"]
    assert not opted_in, f"victim: onnx is not safe on the aarch64 run box: {opted_in}"
