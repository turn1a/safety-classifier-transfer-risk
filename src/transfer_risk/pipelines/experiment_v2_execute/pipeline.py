"""Assemble opt-in v2 stages from the dedicated prospective configuration."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node

from transfer_risk.pipelines.experiment_v2_execute import assets, nodes, target_preparation

if TYPE_CHECKING:
    from collections.abc import Callable


def _structure(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Read v2 graph structure without inheriting the historical base source pool.

    Args:
        parameters: Explicit test structure, or the dedicated v2 environment.

    Returns:
        Prospective model, replica, recipe, and target-role structure.
    """
    if parameters is not None:
        return parameters
    loader = OmegaConfigLoader(
        conf_source=str(Path(__file__).resolve().parents[4] / "conf"),
        env="experiment_v2",
        base_env="base",
        default_run_env="experiment_v2",
    )
    return dict(loader["parameters"]["experiment_v2"])


def _model_datasets(parameters: dict[str, Any]) -> list[str]:
    """List every saved model/replica in stable protocol order.

    Args:
        parameters: Prospective protocol structure.

    Returns:
        Exact private checkpoint dataset names.
    """
    return [
        f"v2_model__{model['id']}__r{replicate}"
        for model in parameters["models"]
        for replicate in parameters["replicates"]
    ]


def empty_backbone(receipt: dict[str, Any]) -> dict[str, Any]:
    """Supply the from-scratch BiLSTM input only after the fresh model gate.

    Args:
        receipt: Fresh validated model-stage receipt.

    Returns:
        No pretrained backbone for a from-scratch model.
    """
    if not receipt.get("protocol_sha256"):
        message = "BiLSTM setup requires a fresh validated model gate"
        raise ValueError(message)
    return {}


def create_models_pipeline(parameters: dict[str, Any] | None = None) -> Pipeline:
    """Build one catalog-owned train/materialize node per model and replica.

    Args:
        parameters: Optional explicit structure used by synthetic tests.

    Returns:
        A model-stage graph requiring a clean isolated registered source checkout.
    """
    structure = _structure(parameters)
    assembled = [
        node(
            partial(nodes.gate_models, structure=nodes.execution_structure(structure)),
            inputs=[
                "params:experiment_v2",
                "v2_corpus_evidence",
                "v2_registration_evidence",
                "v2_execution_context",
                "params:seed",
                "params:experiment_v2_execution",
                "v2_attack_asset_manifest",
            ],
            outputs="v2_models_receipt",
            name="v2_gate_models",
            tags=["experiment_v2", "v2_models"],
        )
    ]
    if any(model["kind"] == "bilstm" for model in structure["models"]):
        assembled.append(
            node(
                empty_backbone,
                inputs="v2_models_receipt",
                outputs="v2_empty_backbone",
                name="v2_empty_bilstm_backbone",
                tags=["experiment_v2", "v2_models"],
            )
        )
    collect_inputs = ["v2_models_receipt", "params:experiment_v2", "v2_corpus_evidence"]
    for model in structure["models"]:
        for replicate in structure["replicates"]:
            identifier = model["id"]
            checkpoint = f"v2_model__{identifier}__r{replicate}"
            metadata = f"v2_model_meta__{identifier}__r{replicate}"
            backbone = "v2_empty_backbone" if model["kind"] == "bilstm" else f"v2_hub__{identifier}"
            assembled.append(
                node(
                    partial(
                        nodes.prepare_model, cell={"model_id": identifier, "replicate": replicate}
                    ),
                    inputs=[
                        "v2_models_receipt",
                        "params:experiment_v2",
                        "v2_corpus_evidence",
                        "v2_private_train",
                        "v2_private_validation",
                        backbone,
                        "params:experiment_v2_execution",
                    ],
                    outputs=[checkpoint, metadata],
                    name=f"v2_prepare_{identifier}_r{replicate}",
                    tags=["experiment_v2", "v2_models", identifier],
                )
            )
            collect_inputs.extend([checkpoint, metadata])
    assembled.append(
        node(
            nodes.collect_models,
            inputs=collect_inputs,
            outputs="v2_models_manifest",
            name="v2_collect_models",
            tags=["experiment_v2", "v2_models"],
        )
    )
    return Pipeline(assembled)


def _create_phase_pipeline(phase: str, parameters: dict[str, Any] | None) -> Pipeline:
    """Build a complete phase with separate generation and target audit boundaries.

    Args:
        phase: Calibration or evaluation.
        parameters: Optional synthetic structure.

    Returns:
        Fresh phase gate, complete baselines, resumable attempts, target audits, and records.
    """
    structure = _structure(parameters)
    current, receipt = f"v2_{phase}_current_models", f"v2_{phase}_receipt"
    assembled = [
        node(
            nodes.refresh_models_manifest,
            inputs=["v2_models_manifest", *_model_datasets(structure)],
            outputs=current,
            name=f"v2_{phase}_refresh_models",
            tags=["experiment_v2", f"v2_{phase}"],
        )
    ]
    gate_inputs = [
        "params:experiment_v2",
        "v2_corpus_evidence",
        "v2_registration_evidence",
        "v2_execution_context",
        current,
    ]
    gate_function: Callable[..., dict[str, Any]]
    if phase == "calibration":
        gate_inputs.append("params:experiment_v2_execution")
        gate_function = nodes.gate_calibration
    else:
        gate_inputs.extend(
            [
                "v2_calibration_records",
                "v2_selection_lock",
                "v2_selection_registration",
                "params:experiment_v2_execution",
            ]
        )
        gate_function = nodes.gate_evaluation
    gate_inputs.append("v2_attack_asset_manifest")
    assembled.append(
        node(
            partial(gate_function, structure=nodes.execution_structure(structure)),
            inputs=gate_inputs,
            outputs=receipt,
            name=f"v2_gate_{phase}",
            tags=["experiment_v2", f"v2_{phase}"],
        )
    )
    assembled.append(
        node(
            nodes.score_phase_originals,
            inputs=[
                receipt,
                "params:experiment_v2",
                "v2_corpus_evidence",
                current,
                f"v2_private_{phase}",
                "params:experiment_v2_execution",
            ],
            outputs=f"v2_{phase}_clean_predictions",
            name=f"v2_{phase}_clean_predictions",
            tags=["experiment_v2", f"v2_{phase}"],
        )
    )
    if phase == "calibration":
        assembled.append(
            node(
                nodes.compute_probe_similarities,
                inputs=[
                    receipt,
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    current,
                    "v2_private_probe",
                    "params:experiment_v2_execution",
                ],
                outputs=f"v2_{phase}_similarities",
                name="v2_calibration_probe_cka",
                tags=["experiment_v2", "v2_calibration"],
            )
        )
    else:
        assembled.append(
            node(
                nodes.no_similarities,
                inputs=receipt,
                outputs=f"v2_{phase}_similarities",
                name="v2_evaluation_no_cka",
                tags=["experiment_v2", "v2_evaluation"],
            )
        )
    audited_datasets = []
    sources = sorted(
        {source for role in structure["target_roles"] for source in role["candidate_ids"]}
    )
    for source in sources:
        for replicate in structure["replicates"]:
            for recipe in structure["recipes"]:
                suffix = f"{source}__r{replicate}__{recipe}"
                private = f"v2_{phase}_attempts__{suffix}"
                audited = f"v2_{phase}_audited__{suffix}"
                cell = {
                    "phase": phase,
                    "source_id": source,
                    "replicate": replicate,
                    "recipe": recipe,
                }
                assembled.append(
                    node(
                        partial(nodes.run_attack_shard, cell=cell),
                        inputs=[
                            receipt,
                            "params:experiment_v2",
                            "v2_corpus_evidence",
                            current,
                            f"v2_private_{phase}",
                            "params:experiment_v2_execution",
                        ],
                        outputs=private,
                        name=f"v2_{phase}_attack_{suffix}",
                        tags=["experiment_v2", f"v2_{phase}", source, recipe],
                    )
                )
                assembled.append(
                    node(
                        nodes.audit_attack_shard,
                        inputs=[
                            receipt,
                            "params:experiment_v2",
                            current,
                            private,
                            "params:experiment_v2_execution",
                        ],
                        outputs=audited,
                        name=f"v2_{phase}_audit_{suffix}",
                        tags=["experiment_v2", f"v2_{phase}", source, recipe],
                    )
                )
                audited_datasets.append(audited)
    assembled.append(
        node(
            nodes.collect_phase_records,
            inputs=[
                receipt,
                "params:experiment_v2",
                "v2_corpus_evidence",
                "v2_registration_evidence",
                current,
                f"v2_{phase}_clean_predictions",
                f"v2_{phase}_similarities",
                *audited_datasets,
            ],
            outputs=f"v2_{phase}_raw_records",
            name=f"v2_collect_{phase}_records",
            tags=["experiment_v2", f"v2_{phase}"],
        )
    )
    return Pipeline(assembled)


def create_calibration_pipeline(parameters: dict[str, Any] | None = None) -> Pipeline:
    """Assemble calibration execution and independent-probe CKA without human votes.

    Args:
        parameters: Optional explicit synthetic structure.

    Returns:
        Complete raw calibration evidence from newly realized private checkpoints.
    """
    return _create_phase_pipeline("calibration", parameters)


def create_evaluation_pipeline(parameters: dict[str, Any] | None = None) -> Pipeline:
    """Assemble held-out execution gated by a registered frozen selector.

    Args:
        parameters: Optional explicit synthetic structure.

    Returns:
        Complete raw evaluation evidence with no probe extraction or selector recomputation.
    """
    return _create_phase_pipeline("evaluation", parameters)


def create_annotation_pipeline(phase: str) -> Pipeline:
    """Assemble a target-free human-evidence merge without heavy execution dependencies.

    Args:
        phase: Calibration or evaluation.

    Returns:
        Reviewed phase records ready for pure selection or analysis.

    Raises:
        ValueError: If the requested phase is unknown.
    """
    if phase not in {"calibration", "evaluation"}:
        message = "v2 annotation phase must be calibration or evaluation"
        raise ValueError(message)
    return Pipeline(
        [
            node(
                nodes.merge_annotations,
                inputs=[
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    "v2_registration_evidence",
                    f"v2_{phase}_raw_records",
                    f"v2_{phase}_annotations",
                ],
                outputs=f"v2_{phase}_records",
                name=f"v2_merge_{phase}_human_annotations",
                tags=["experiment_v2", "v2_annotations"],
            )
        ]
    )


def create_target_prepare_pipeline(parameters: dict[str, Any] | None = None) -> Pipeline:
    """Assemble independent-target preparation before core protocol registration.

    Args:
        parameters: Optional explicit graph axes for synthetic tests.

    Returns:
        A separately registered target-training stage exporting exact prepared hashes.
    """
    structure = nodes.execution_structure(_structure(parameters))
    return Pipeline(
        [
            node(
                partial(target_preparation.gate_target_preparation, structure=structure),
                inputs=[
                    "params:experiment_v2_target_preparation",
                    "v2_target_preparation_evidence",
                    "params:experiment_v2",
                    "v2_corpus_evidence",
                    "v2_execution_context",
                    "params:seed",
                ],
                outputs="v2_target_preparation_receipt",
                name="v2_gate_target_preparation",
                tags=["experiment_v2", "v2_target_preparation"],
            ),
            node(
                target_preparation.prepare_target,
                inputs=[
                    "v2_target_preparation_receipt",
                    "params:experiment_v2_target_preparation",
                    "v2_target_preparation_evidence",
                    "v2_target_private_train",
                    "v2_target_private_validation",
                    "v2_target_backbone",
                    "params:experiment_v2_execution",
                ],
                outputs=["v2_prepared_target", "v2_prepared_target_metadata"],
                name="v2_prepare_independent_target",
                tags=["experiment_v2", "v2_target_preparation"],
            ),
            node(
                target_preparation.export_target_preparation,
                inputs=[
                    "v2_target_preparation_receipt",
                    "v2_prepared_target",
                    "v2_prepared_target_metadata",
                ],
                outputs="v2_target_preparation_record",
                name="v2_export_target_preparation",
                tags=["experiment_v2", "v2_target_preparation"],
            ),
        ]
    )


def create_assets_prepare_pipeline(parameters: dict[str, Any] | None = None) -> Pipeline:
    """Materialize separately pinned operator asset inputs without training or inference.

    Args:
        parameters: Optional explicit graph axes for synthetic tests.

    Returns:
        Fresh isolated-source gate and an immutable verified offline asset cache.
    """
    structure = nodes.execution_structure(_structure(parameters))
    return Pipeline(
        [
            node(
                partial(assets.gate_assets_preparation, structure=structure),
                inputs=["params:experiment_v2", "v2_execution_context", "params:seed"],
                outputs="v2_attack_assets_preparation_receipt",
                name="v2_gate_attack_asset_preparation",
                tags=["experiment_v2", "v2_assets_preparation"],
            ),
            node(
                assets.materialize_asset_bundle,
                inputs=[
                    "v2_attack_assets_preparation_receipt",
                    "v2_attack_asset_source",
                    "params:experiment_v2",
                ],
                outputs="v2_attack_asset_manifest",
                name="v2_materialize_offline_attack_assets",
                tags=["experiment_v2", "v2_assets_preparation"],
            ),
        ]
    )


def create_pipeline() -> Pipeline:
    """Expose a nonempty opt-in alias for Kedro discovery without executing any stage.

    Returns:
        The separately gated v2 model stage, excluded from the historical default chain.
    """
    return create_models_pipeline()
