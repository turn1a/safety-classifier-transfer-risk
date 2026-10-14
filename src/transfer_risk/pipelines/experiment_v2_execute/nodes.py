"""Catalog-bound future v2 execution with complete private and non-text accounting."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

from transfer_risk.datasets.experiment_v2_attack_assets_dataset import (
    AttackAssetManifest,
    _validate_manifest,
)
from transfer_risk.lib.experiment_v2.analysis import freeze_selection
from transfer_risk.lib.experiment_v2.protocol import (
    component_seed,
    content_digest,
    is_digest,
    parse_timestamp,
    partition_digest,
    require_valid_preflight,
    validate_preflight,
)
from transfer_risk.lib.experiment_v2.records import validate_records
from transfer_risk.lib.experiment_v2.schemas import (
    AttackAttempt,
    CandidatePrediction,
    CleanPrediction,
    CorpusEvidence,
    ExperimentRecords,
    ModelIdentity,
    ModelRealization,
    Protocol,
    RegistrationEvidence,
)
from transfer_risk.pipelines.experiment_v2_execute import runtime

PredictFn = Callable[[ModelIdentity, dict[str, str], list[str], dict[str, Any]], list[str]]
AttackFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], int, int], dict[str, Any]]

_EXECUTION_BINDINGS = (
    "protocol_sha256",
    "corpus_sha256",
    "execution_ref",
    "execution_spec_sha256",
    "registration_evidence_sha256",
    "registration_registered_at",
    "attack_assets_sha256",
    "phase",
    "selection_lock_sha256",
    "selection_registry_uri",
    "selection_registry_evidence_sha256",
    "selection_registered_at",
)


def _fail(message: str) -> NoReturn:
    """Raise an execution-contract error without exposing private input text.

    Args:
        message: Violated invariant.

    Raises:
        ValueError: Always.
    """
    raise ValueError(message)


def execution_spec_digest(protocol: Protocol, settings: dict[str, Any]) -> str:
    """Bind every inference and attack setting to the registered execution specification.

    Args:
        protocol: Frozen task, windows, recipes, budgets, seed, and execution ref.
        settings: Runtime batch sizes, representation pooling, and device policy.

    Returns:
        Canonical non-text execution-specification SHA256.
    """
    return content_digest(
        {
            "recipes": protocol.recipes,
            "source_query_budget": protocol.source_query_budget,
            "query_batch_size": settings["query_batch_size"],
            "inference_batch_size": settings["inference_batch_size"],
            "pooling": settings["pooling"],
            "device_policy": settings["device_policy"],
            "model_windows": {model.id: model.max_tokens for model in protocol.models},
            "root_seed": protocol.root_seed,
            "execution_ref": protocol.execution_ref,
            "source_feedback": "source_only",
            "input_text_policy": "exact_registered_utf8",
            "attack_assets_sha256": protocol.attack_assets_sha256,
            "strict_offline_assets": True,
        }
    )


def input_policy_digest(protocol: Protocol) -> str:
    """Bind exact UTF-8 content and tokenizer truncation without altering originals.

    Args:
        protocol: Registered per-model token windows.

    Returns:
        Canonical non-text input-policy SHA256.
    """
    return content_digest(
        {
            "text_encoding": "utf-8",
            "content_hash": "sha256",
            "normalization": "none",
            "character_clipping": None,
            "tokenizer_truncation": "registered_max_tokens",
            "model_windows": {model.id: model.max_tokens for model in protocol.models},
            "grouping": "separate_registered_evidence",
        }
    )


def _check_context(context: dict[str, Any], execution_ref: str | None) -> None:
    """Reject mutable, primary, or wrong-branch source before heavy catalog inputs load.

    Args:
        context: Fresh source and hardware facts supplied by the read-only catalog.
        execution_ref: Registered complete source commit.

    Raises:
        ValueError: If isolation, source cleanliness, or the execution commit differs.
    """
    if context.get("branch") != "codex/methodology-recompute":
        _fail("v2 execution requires the codex/methodology-recompute branch")
    if context.get("isolated_worktree") is not True:
        _fail("v2 execution requires an isolated worktree")
    if context.get("source_clean") is not True:
        _fail("v2 execution requires committed clean source, including untracked code")
    if context.get("execution_ref") != execution_ref or not execution_ref:
        _fail("actual execution commit differs from registered source")
    if not context.get("hardware") or not context.get("project_root"):
        _fail("actual source and hardware provenance is absent")


def execution_structure(parameters: dict[str, Any]) -> dict[str, Any]:
    """Capture only the committed axes used to construct the future execution graph.

    Args:
        parameters: Full protocol or an already reduced structural descriptor.

    Returns:
        Model id/kind order, replicas, recipes, and target/source-role membership.
    """
    return {
        "models": [{"id": model["id"], "kind": model["kind"]} for model in parameters["models"]],
        "replicates": list(parameters["replicates"]),
        "recipes": list(parameters["recipes"]),
        "target_roles": [
            {
                "id": role["id"],
                "model_id": role["model_id"],
                "candidate_ids": list(role["candidate_ids"]),
            }
            for role in parameters["target_roles"]
        ],
    }


def _check_structure(protocol: Protocol, structure: dict[str, Any] | None) -> None:
    """Reject runtime axes that differ from the graph before any text or ML load.

    Args:
        protocol: Fully resolved runtime protocol, including overrideable evidence pins.
        structure: Bound committed graph axes, supplied by every execution factory.
    """
    if structure is not None and execution_structure(protocol.model_dump()) != execution_structure(
        structure
    ):
        _fail("registered model, replica, recipe or target/source axes differ from the built graph")


def _check_settings(protocol: Protocol, settings: dict[str, Any] | None) -> str:
    """Reject runtime overrides that change registered budgets or training recipes.

    Args:
        protocol: Registered model recipes and execution specification.
        settings: Supplied runtime settings.

    Returns:
        The exact registered runtime-specification digest.

    Raises:
        ValueError: If settings are absent, unregistered, or incompatible with models.
    """
    if not settings:
        _fail("registered v2 execution settings are absent")
    if (
        settings.get("pooling") != "cls"
        or not isinstance(settings.get("inference_batch_size"), int)
        or settings["inference_batch_size"] <= 0
        or not isinstance(settings.get("query_batch_size"), int)
        or settings["query_batch_size"] <= 0
        or settings.get("device_policy") not in {"cpu", "mps_or_cpu"}
    ):
        _fail("v2 execution settings contain an invalid window, batch, pooling or device")
    digest = execution_spec_digest(protocol, settings)
    if digest != protocol.attack_spec_sha256:
        _fail("actual execution settings differ from the registered attack specification")
    if input_policy_digest(protocol) != protocol.input_policy_sha256:
        _fail("actual text handling differs from the registered input policy")
    for model in protocol.models:
        if model.kind != "frozen":
            recipe = settings.get("training", {}).get(model.id)
            if not recipe or content_digest(recipe) != model.training_recipe_sha256:
                _fail("actual model training recipe differs from its registered hash")
            if recipe.get("max_seq_len") != model.max_tokens:
                _fail("training token window differs from the registered model window")
            if model.kind == "bilstm" and recipe.get("padding_policy") != "masked":
                _fail("prospective BiLSTM training requires masked padding")
    return digest


def _check_asset_descriptor(
    protocol: Protocol, context: dict[str, Any], descriptor: dict[str, Any] | None
) -> dict[str, Any]:
    """Bind the catalog-verified cache to the registered manifest and private run.

    Args:
        protocol: Registered exact auxiliary-manifest identity.
        context: Actual isolated checkout path.
        descriptor: Dataset-verified manifest and realized local asset paths.

    Returns:
        The unchanged isolated auxiliary descriptor.
    """
    if not descriptor:
        _fail("registered verified offline attack assets are absent")
    manifest = AttackAssetManifest.model_validate(descriptor["manifest"])
    _validate_manifest(manifest, protocol.run_id)
    digest = content_digest(manifest)
    if descriptor.get("manifest_sha256") != digest or protocol.attack_assets_sha256 != digest:
        _fail("actual auxiliary asset manifest differs from its registered identity")
    root = Path(context["project_root"]) / protocol.private_root / "attack_assets" / "cache"
    if descriptor.get("cache_root") != str(root):
        _fail("auxiliary asset cache is outside the isolated catalog-owned run")
    expected_paths = {name: str(root / role.path) for name, role in manifest.roles.items()}
    expected_paths.update({name: str(root / path) for name, path in manifest.cache_paths.items()})
    if descriptor.get("paths") != expected_paths:
        _fail("auxiliary role/cache paths differ from their registered local manifest")
    return descriptor


def gate_models(
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    context: dict[str, Any],
    root_seed: int,
    execution_settings: dict[str, Any] | None = None,
    attack_assets: dict[str, Any] | None = None,
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require registered, isolated, clean source before loading text or model inputs.

    Args:
        parameters: Frozen core protocol.
        corpus_evidence: Grouped, reviewed original metadata.
        registration_evidence: External protocol/corpus registration attestation.
        context: Fresh actual checkout and host facts.
        root_seed: Shared project seed.
        execution_settings: Runtime settings whose hashes were registered.
        attack_assets: Verified catalog-owned offline auxiliary manifest/cache paths.
        structure: Committed graph axes bound by the factory; runtime pins remain overrideable.

    Returns:
        A fresh receipt binding all dependent heavy nodes to validated evidence.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_context(context, protocol.execution_ref)
    _check_structure(protocol, structure)
    if root_seed != protocol.root_seed:
        _fail("project seed differs from the registered v2 seed")
    registration = RegistrationEvidence.model_validate(registration_evidence)
    receipt = require_valid_preflight(validate_preflight(protocol, corpus, registration))
    started_at = datetime.now(UTC)
    if parse_timestamp(registration.registered_at) >= started_at:
        _fail("core registration must precede model and phase execution")
    execution_digest = _check_settings(protocol, execution_settings)
    assets = _check_asset_descriptor(protocol, context, attack_assets)
    return {
        **receipt,
        "execution_spec_sha256": execution_digest,
        "execution_ref": protocol.execution_ref,
        "registration_evidence_sha256": content_digest(registration),
        "registration_registered_at": registration.registered_at,
        "attack_assets_sha256": assets["manifest_sha256"],
        "_attack_assets": assets,
        "project_root": context["project_root"],
        "hardware": context["hardware"],
        "started_at": started_at.isoformat(),
    }


def _check_receipt(
    receipt: dict[str, Any],
    protocol: Protocol,
    corpus: CorpusEvidence | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    """Reject stale protocol, corpus, or settings in dependent execution nodes.

    Args:
        receipt: Fresh gate result.
        protocol: This node's parsed protocol.
        corpus: This node's corpus evidence when present.
        settings: This node's runtime settings when present.

    Raises:
        ValueError: If registered evidence or settings changed after the gate.
    """
    if receipt.get("protocol_sha256") != content_digest(protocol):
        _fail("v2 execution receipt has a stale protocol")
    if receipt.get("attack_assets_sha256") != protocol.attack_assets_sha256:
        _fail("v2 execution receipt has stale auxiliary asset identity")
    if corpus is not None and receipt.get("corpus_sha256") != content_digest(corpus):
        _fail("v2 execution receipt has a stale corpus")
    if settings is not None and receipt.get("execution_spec_sha256") != execution_spec_digest(
        protocol, settings
    ):
        _fail("v2 execution receipt has stale runtime settings")


def _check_artifact_execution(header: dict[str, Any] | None, receipt: dict[str, Any]) -> None:
    """Reject saved rows from another protocol, registration, runtime, phase, or selector.

    Args:
        header: Immutable originating execution metadata.
        receipt: Fresh current phase or model gate.

    Raises:
        ValueError: If evidence identity or saved temporal ordering is invalid.
    """
    if not isinstance(header, dict) or any(
        header.get(key) != receipt.get(key) for key in _EXECUTION_BINDINGS
    ):
        _fail("saved artifact has a stale execution or selection identity")
    if parse_timestamp(header.get("completed_at")) < parse_timestamp(header.get("started_at")):
        _fail("saved execution artifact completes before it started")
    if parse_timestamp(header.get("started_at")) <= parse_timestamp(
        header.get("registration_registered_at")
    ):
        _fail("saved execution artifact must start after initial registration")
    if header.get("selection_registered_at") and parse_timestamp(header.get("started_at")) <= (
        parse_timestamp(header["selection_registered_at"])
    ):
        _fail("saved held-out execution precedes selection registration")
    if not header.get("hardware"):
        _fail("saved execution artifact lacks actual hardware provenance")


def execution_header(
    receipt: dict[str, Any], inherited: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Stamp private artifacts while retaining earlier origins across resumed audit nodes.

    Args:
        receipt: Fresh phase/model execution gate.
        inherited: Original private generation header when an audit resumes later.

    Returns:
        Immutable execution identity, original start, current completion, and hardware.
    """
    start = receipt["started_at"]
    hardware = {receipt["hardware"]}
    if inherited is not None:
        _check_artifact_execution(inherited, receipt)
        start = min((start, inherited["started_at"]), key=parse_timestamp)
        hardware.add(inherited["hardware"])
    return {
        **{key: receipt.get(key) for key in _EXECUTION_BINDINGS},
        "started_at": start,
        "completed_at": datetime.now(UTC).isoformat(),
        "hardware": "; ".join(sorted(hardware)),
    }


def phase_frame(
    private_texts: list[dict[str, Any]], corpus: CorpusEvidence, partition: str
) -> pd.DataFrame:
    """Join exact private originals to one sealed registered partition.

    Args:
        private_texts: Private catalog rows containing only id and text.
        corpus: Registered non-text membership and content hashes.
        partition: One requested partition, with no sampling or random splitting.

    Returns:
        The original registered order with text, label, id, and group_id columns.

    Raises:
        ValueError: If coverage, exact content hashes, or private row fields differ.
    """
    records = [row for row in corpus.records if row.partition == partition]
    indexed = {row["id"]: row for row in private_texts}
    if len(indexed) != len(private_texts) or set(indexed) != {row.id for row in records}:
        _fail("private original partition coverage differs from registered membership")
    result = []
    for row in records:
        private = indexed[row.id]
        if set(private) != {"id", "text"} or not isinstance(private["text"], str):
            _fail("private original row contains invalid fields")
        if hashlib.sha256(private["text"].encode()).hexdigest() != row.content_sha256:
            _fail("private original content differs from the registered content hash")
        result.append(
            {"id": row.id, "text": private["text"], "label": row.label, "group_id": row.group_id}
        )
    return pd.DataFrame(result)


def _model_key(model_id: str, replicate: int) -> str:
    """Return a stable private model/replicate key.

    Args:
        model_id: Registered detector id.
        replicate: Registered seed replica.

    Returns:
        The matching artifact-map key.
    """
    return f"{model_id}:r{replicate}"


def _check_manifest(
    protocol: Protocol, corpus: CorpusEvidence, manifest: dict[str, Any], root: str
) -> None:
    """Check realized artifact coverage and isolation before phase inference.

    Args:
        protocol: Registered model/replicate grid.
        corpus: Registered training evidence.
        manifest: Saved realized identities and exact private descriptors.
        root: Actual isolated checkout root from the context catalog.

    Raises:
        ValueError: If realization coverage, provenance, identity, or isolation differs.
    """
    if manifest.get("protocol_sha256") != content_digest(protocol) or manifest.get(
        "corpus_sha256"
    ) != content_digest(corpus):
        _fail("model manifest differs from registered protocol or corpus")
    expected = {
        _model_key(model.id, index) for model in protocol.models for index in protocol.replicates
    }
    realizations = [ModelRealization.model_validate(row) for row in manifest["realizations"]]
    keys = {_model_key(row.model_id, row.replicate) for row in realizations}
    if (
        keys != expected
        or len(realizations) != len(expected)
        or set(manifest["artifacts"]) != expected
    ):
        _fail("model manifest realization coverage is incomplete or duplicated")
    by_model = {model.id: model for model in protocol.models}
    for realization in realizations:
        model = by_model[realization.model_id]
        descriptor = manifest["artifacts"][_model_key(model.id, realization.replicate)]
        if (
            descriptor["checkpoint_sha256"] != realization.checkpoint_sha256
            or descriptor["tokenizer_sha256"] != realization.tokenizer_sha256
            or not is_digest(realization.checkpoint_sha256)
            or not is_digest(realization.tokenizer_sha256)
            or realization.execution_ref != protocol.execution_ref
            or not Path(descriptor["path"]).is_relative_to(Path(root) / protocol.private_root)
        ):
            _fail("model artifact identity or isolated path differs from realization evidence")
        expected_seed = (
            None
            if model.kind == "frozen"
            else component_seed(protocol.root_seed, realization.replicate, f"model:{model.id}")
        )
        expected_partition = None if model.kind == "frozen" else partition_digest(corpus, "train")
        if (realization.training_seed, realization.train_partition_sha256) != (
            expected_seed,
            expected_partition,
        ):
            _fail("model realization has stale training membership or seed provenance")
    for role in protocol.target_roles:
        identities = {
            manifest["artifacts"][_model_key(role.model_id, index)]["checkpoint_sha256"]
            for index in protocol.replicates
        }
        if len(identities) != 1:
            _fail("frozen target identity changes across replicas")
        for source in role.candidate_ids:
            if any(
                manifest["artifacts"][_model_key(source, index)]["checkpoint_sha256"] in identities
                for index in protocol.replicates
            ):
                _fail("realized target weights are excluded from their own source pool")
    for preparation in corpus.target_preparations:
        role = next(role for role in protocol.target_roles if role.id == preparation.target_role)
        if any(
            (
                manifest["artifacts"][_model_key(role.model_id, index)]["checkpoint_sha256"],
                manifest["artifacts"][_model_key(role.model_id, index)]["tokenizer_sha256"],
            )
            != (preparation.checkpoint_sha256, preparation.tokenizer_sha256)
            for index in protocol.replicates
        ):
            _fail("prepared target fingerprint differs from its preparation evidence")


def refresh_models_manifest(
    manifest: dict[str, Any], *descriptors: dict[str, str]
) -> dict[str, Any]:
    """Rehash saved checkpoints through the catalog before every heavy phase.

    Args:
        manifest: Saved model realizations and private artifact descriptors.
        descriptors: Fresh descriptors from every model/replica dataset in stable order.

    Returns:
        The unchanged model manifest after checking every artifact identity.

    Raises:
        ValueError: If any artifact path or content changed after training/materialization.
    """
    current = {descriptor["path"]: descriptor for descriptor in descriptors}
    if len(descriptors) != len(manifest["artifacts"]) or any(
        current.get(saved["path"]) != saved for saved in manifest["artifacts"].values()
    ):
        _fail("saved model/tokenizer content changed before phase execution")
    return manifest


def gate_calibration(
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    context: dict[str, Any],
    models_manifest: dict[str, Any],
    execution_settings: dict[str, Any],
    attack_assets: dict[str, Any] | None = None,
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate calibration and independent-probe inference before private text is opened.

    Args:
        parameters: Registered protocol.
        corpus_evidence: Registered grouped corpus metadata.
        registration_evidence: External original registration.
        context: Fresh actual code and host facts.
        models_manifest: Freshly checked realized models.
        execution_settings: Registered runtime parameters.
        attack_assets: Fresh verified isolated offline asset descriptor.
        structure: Committed execution axes bound at graph construction.

    Returns:
        A calibration-only execution receipt.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    receipt = gate_models(
        parameters,
        corpus_evidence,
        registration_evidence,
        context,
        protocol.root_seed,
        execution_settings,
        attack_assets,
        structure=structure,
    )
    _check_manifest(protocol, corpus, models_manifest, context["project_root"])
    _check_artifact_execution(models_manifest.get("_execution"), receipt)
    return {**receipt, "phase": "calibration"}


def gate_evaluation(  # noqa: PLR0913
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    context: dict[str, Any],
    models_manifest: dict[str, Any],
    calibration_records: dict[str, Any],
    lock: dict[str, Any],
    selection_registration: dict[str, Any],
    execution_settings: dict[str, Any] | None = None,
    attack_assets: dict[str, Any] | None = None,
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a reproduced, externally registered selector before held-out text is read.

    Args:
        parameters: Registered protocol.
        corpus_evidence: Registered original group membership.
        registration_evidence: Original external registration.
        context: Fresh actual checkout and host facts.
        models_manifest: Freshly checked exact model realizations.
        calibration_records: Completed calibration and human annotation evidence.
        lock: Frozen source choices from that calibration evidence.
        selection_registration: External attestation binding the exact frozen lock.
        execution_settings: Registered runtime settings.
        attack_assets: Fresh verified isolated offline asset descriptor.
        structure: Committed execution axes bound at graph construction.

    Returns:
        A held-out execution receipt including selection-registration provenance.
    """
    if not lock or not calibration_records or not selection_registration:
        _fail(
            "evaluation requires completed calibration and an externally registered selection lock"
        )
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    registration = RegistrationEvidence.model_validate(registration_evidence)
    receipt = gate_models(
        parameters,
        corpus_evidence,
        registration_evidence,
        context,
        protocol.root_seed,
        execution_settings,
        attack_assets,
        structure=structure,
    )
    _check_manifest(protocol, corpus, models_manifest, context["project_root"])
    _check_artifact_execution(models_manifest.get("_execution"), receipt)
    calibration = ExperimentRecords.model_validate(calibration_records)
    expected = freeze_selection(
        protocol, corpus, registration, calibration, protocol.root_seed, str(lock.get("frozen_at"))
    )
    if expected != lock or selection_registration.get("lock_sha256") != lock["lock_sha256"]:
        _fail("evaluation selection lock differs from completed calibration or its registration")
    if (
        not str(selection_registration.get("registry_uri", "")).startswith("https://")
        or not is_digest(selection_registration.get("evidence_sha256"))
        or parse_timestamp(selection_registration.get("registered_at"))
        < parse_timestamp(lock["frozen_at"])
        or parse_timestamp(selection_registration.get("registered_at"))
        >= parse_timestamp(receipt["started_at"])
    ):
        _fail("selection external registration must precede held-out evaluation")
    if calibration.realizations != [
        ModelRealization.model_validate(row) for row in models_manifest["realizations"]
    ]:
        _fail("model realization identities changed after calibration")
    return {
        **receipt,
        "phase": "evaluation",
        "selection_lock_sha256": lock["lock_sha256"],
        "selection_registry_uri": selection_registration["registry_uri"],
        "selection_registry_evidence_sha256": selection_registration["evidence_sha256"],
        "selection_registered_at": selection_registration["registered_at"],
    }


def prepare_model(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    train_texts: list[dict[str, Any]],
    validation_texts: list[dict[str, Any]],
    backbone: dict[str, Any],
    execution_settings: dict[str, Any],
    *,
    cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train or materialize one model replica using exact registered group partitions.

    Args:
        receipt: Fresh model-stage gate receipt.
        parameters: Registered protocol.
        corpus_evidence: Registered train/validation membership and content identities.
        train_texts: Private training originals.
        validation_texts: Private validation originals.
        backbone: Pinned upstream bundle or private prepared-target descriptor.
        execution_settings: Registered fixed architecture-specific training recipes.
        cell: Model id and replica bound by pipeline assembly.

    Returns:
        Catalog-owned checkpoint bundle and non-text training/materialization metadata.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus, execution_settings)
    model = next(model for model in protocol.models if model.id == cell["model_id"])
    replicate = int(cell["replicate"])
    seed = (
        None
        if model.kind == "frozen"
        else component_seed(protocol.root_seed, replicate, f"model:{model.id}")
    )
    metadata = {
        "model_id": model.id,
        "replicate": replicate,
        "training_seed": seed,
        "train_partition_sha256": None
        if model.kind == "frozen"
        else partition_digest(corpus, "train"),
        "execution_ref": protocol.execution_ref,
    }
    if model.kind == "frozen" and "path" in backbone:
        if (backbone["checkpoint_sha256"], backbone["tokenizer_sha256"]) != (
            model.revision,
            model.tokenizer_revision,
        ):
            _fail("independently prepared frozen target differs from registered fingerprints")
        return {"copy_from": backbone}, {**metadata, "_execution": execution_header(receipt)}
    if model.kind != "bilstm":
        expected = {
            "repository": model.repository,
            "revision": model.revision,
            "tokenizer_repository": model.tokenizer_repository,
            "tokenizer_revision": model.tokenizer_revision,
        }
        if backbone.get("input_identity") != expected:
            _fail("backbone input identity differs from registered immutable model pins")
    if model.kind == "frozen":
        return backbone, {**metadata, "_execution": execution_header(receipt)}
    train = phase_frame(train_texts, corpus, "train")
    validation = phase_frame(validation_texts, corpus, "validation")
    config = execution_settings["training"][model.id]
    if seed is None:
        _fail("trained model has no derived training seed")
    bundle, training_metadata = runtime.train_model(
        model,
        deepcopy(backbone),
        train,
        validation,
        config,
        execution_settings["device_policy"],
        seed,
    )
    return bundle, {
        **metadata,
        "training_metadata": training_metadata,
        "_execution": execution_header(receipt),
    }


def collect_models(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    *artifacts_and_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Bind saved model hashes to seed and train-partition provenance.

    Args:
        receipt: Fresh model-stage gate.
        parameters: Registered protocol.
        corpus_evidence: Registered corpus metadata.
        artifacts_and_metadata: Alternating saved descriptors and training metadata.

    Returns:
        Complete private model realization manifest.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus)
    artifacts, realizations = {}, []
    execution = execution_header(receipt)
    for index in range(0, len(artifacts_and_metadata), 2):
        descriptor, metadata = artifacts_and_metadata[index : index + 2]
        _check_artifact_execution(metadata.get("_execution"), receipt)
        execution["started_at"] = min(
            (execution["started_at"], metadata["_execution"]["started_at"]), key=parse_timestamp
        )
        model_id, replicate = metadata["model_id"], metadata["replicate"]
        artifacts[_model_key(model_id, replicate)] = descriptor
        fields = set(ModelRealization.model_fields) - {"checkpoint_sha256", "tokenizer_sha256"}
        realizations.append(
            ModelRealization.model_validate(
                {
                    **{key: metadata[key] for key in fields},
                    "checkpoint_sha256": descriptor["checkpoint_sha256"],
                    "tokenizer_sha256": descriptor["tokenizer_sha256"],
                }
            ).model_dump()
        )
    manifest = {
        "protocol_sha256": content_digest(protocol),
        "corpus_sha256": content_digest(corpus),
        "artifacts": artifacts,
        "realizations": realizations,
        "_execution": {
            **execution,
            "hardware": "; ".join(
                sorted(
                    {
                        receipt["hardware"],
                        *{item["_execution"]["hardware"] for item in artifacts_and_metadata[1::2]},
                    }
                )
            ),
        },
    }
    _check_manifest(protocol, corpus, manifest, receipt["project_root"])
    return manifest


def score_phase_originals(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    manifest: dict[str, Any],
    private_texts: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    predict_fn: PredictFn = runtime.predict_model,
) -> list[dict[str, Any]]:
    """Score every phase original under every model replica, including benign controls.

    Args:
        receipt: Calibration/evaluation gate.
        parameters: Registered protocol.
        corpus_evidence: Registered original metadata.
        manifest: Realized model fingerprints.
        private_texts: Only this phase's private originals.
        settings: Registered inference settings.
        predict_fn: Inference adapter; tests replace the expensive operation.

    Returns:
        The complete non-text clean prediction grid.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus, settings)
    frame = phase_frame(private_texts, corpus, receipt["phase"])
    result = []
    runtime_settings = {**settings, "_attack_assets": receipt.get("_attack_assets")}
    for model in protocol.models:
        for replicate in protocol.replicates:
            descriptor = manifest["artifacts"][_model_key(model.id, replicate)]
            labels = predict_fn(model, descriptor, frame["text"].tolist(), runtime_settings)
            for original, raw in zip(frame["id"], labels, strict=True):
                result.append(
                    {
                        **CleanPrediction(
                            model_id=model.id,
                            replicate=replicate,
                            original_id=original,
                            raw_label=raw,
                            checkpoint_sha256=descriptor["checkpoint_sha256"],
                            tokenizer_sha256=descriptor["tokenizer_sha256"],
                        ).model_dump(),
                        "_execution": execution_header(receipt),
                    }
                )
    return result


def compute_probe_similarities(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    manifest: dict[str, Any],
    private_probe: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute globally centered full-probe CKA only during calibration.

    Args:
        receipt: Calibration-only gate.
        parameters: Registered protocol.
        corpus_evidence: Independent probe metadata.
        manifest: Realized private model descriptors.
        private_probe: Exact separately grouped probe originals.
        settings: Registered pooling, windows, and batch settings.

    Returns:
        Every role/source/replicate CKA identity record.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus, settings)
    if receipt.get("phase") != "calibration":
        _fail("evaluation cannot recompute representation selection")
    probe = phase_frame(private_probe, corpus, "probe")["text"].tolist()
    return [
        {**row, "_execution": execution_header(receipt)}
        for row in runtime.compute_similarities(protocol, corpus, manifest, probe, settings)
    ]


def run_attack_shard(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    manifest: dict[str, Any],
    private_texts: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    cell: dict[str, Any],
    attack_fn: AttackFn = runtime.attack_original,
) -> list[dict[str, Any]]:
    """Retain every fixed-budget positive-original attempt in one source/recipe replica.

    Args:
        receipt: Fresh phase gate.
        parameters: Registered source pool, recipes, query caps, and seeds.
        corpus_evidence: Registered phase originals.
        manifest: Realized source/target artifact fingerprints.
        private_texts: Only this phase's originals.
        settings: Registered source attack settings.
        cell: Phase, source, recipe, and replica bound by pipeline assembly.
        attack_fn: Expensive attack adapter replaced by synthetic fixtures in tests.

    Returns:
        Private candidate texts plus strict non-text attempts, including failures and errors.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus, settings)
    if receipt.get("phase") != cell["phase"]:
        _fail("attempt shard phase differs from its execution gate")
    source, replicate, recipe = cell["source_id"], cell["replicate"], cell["recipe"]
    candidates = {source for role in protocol.target_roles for source in role.candidate_ids}
    if (
        source not in candidates
        or recipe not in protocol.recipes
        or replicate not in protocol.replicates
    ):
        _fail("attempt shard contains an unregistered source, recipe or replica")
    model = next(model for model in protocol.models if model.id == source)
    descriptor = manifest["artifacts"][_model_key(source, replicate)]
    seed = component_seed(protocol.root_seed, replicate, f"attack:{source}:{recipe}")
    frame = phase_frame(private_texts, corpus, cell["phase"])
    result = []
    model_input = {"identity": model.model_dump(), "artifact": descriptor}
    runtime_settings = {**settings, "_attack_assets": receipt.get("_attack_assets")}
    if attack_fn is runtime.attack_original:
        try:
            model_input["runtime_wrapper"] = runtime.make_source_wrapper(
                model, descriptor, runtime_settings
            )
        except Exception as error:
            model_input["setup_error"] = type(error).__name__
    for row in frame[frame["label"] == 1].to_dict("records"):
        original: dict[str, Any] = {
            "id": row["id"],
            "text": row["text"],
            "label": row["label"],
            "group_id": row["group_id"],
        }
        returned = attack_fn(
            model_input,
            original,
            {**runtime_settings, "recipe": recipe},
            seed,
            protocol.source_query_budget,
        )
        text = returned["candidate_text"]
        attempt = AttackAttempt(
            id=f"{cell['phase']}:{replicate}:{source}:{original['id']}:{recipe}",
            source_id=source,
            replicate=replicate,
            original_id=original["id"],
            recipe=recipe,
            status=returned["status"],
            source_queries=returned["source_queries"],
            target_search_queries=0,
            attack_seed=seed,
            candidate_sha256=None if text is None else hashlib.sha256(text.encode()).hexdigest(),
            source_raw_perturbed=returned["source_raw_perturbed"],
            source_checkpoint_sha256=descriptor["checkpoint_sha256"],
            source_tokenizer_sha256=descriptor["tokenizer_sha256"],
            target_predictions=[],
            mechanical_screen="unassessable",
            mechanical_evidence_sha256=None,
        )
        minimum_queries = 0 if attempt.status == "error" and text is None else 1
        if not minimum_queries <= attempt.source_queries <= protocol.source_query_budget:
            _fail("attack adapter did not preserve an actual bounded source query count")
        result.append(
            {
                "attempt": attempt.model_dump(),
                "candidate_text": text,
                "error_type": returned.get("error_type"),
                "_execution": execution_header(receipt),
            }
        )
    return result


def audit_attack_shard(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    manifest: dict[str, Any],
    private_attempts: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    predict_fn: PredictFn = runtime.predict_model,
) -> list[dict[str, Any]]:
    """Score all returned candidates after generation, including source failures.

    Args:
        receipt: Fresh phase gate.
        parameters: Registered frozen targets and actual label maps.
        manifest: Exact realized target artifact descriptors.
        private_attempts: Source generation outputs with private candidate texts.
        settings: Registered target inference settings.
        predict_fn: Expensive inference adapter replaced by synthetic fixtures in tests.

    Returns:
        Non-text complete attempts with target audit predictions.
    """
    protocol = Protocol.model_validate(parameters)
    _check_receipt(receipt, protocol, settings=settings)
    for row in private_attempts:
        _check_artifact_execution(row.get("_execution"), receipt)
    attempts = [AttackAttempt.model_validate(row["attempt"]) for row in private_attempts]
    candidates = [
        (index, row["candidate_text"])
        for index, row in enumerate(private_attempts)
        if row["candidate_text"] is not None
    ]
    runtime_settings = {**settings, "_attack_assets": receipt.get("_attack_assets")}
    for replicate in protocol.replicates:
        subset = [
            (index, text) for index, text in candidates if attempts[index].replicate == replicate
        ]
        if not subset:
            continue
        texts = [text for _index, text in subset]
        for target_id in sorted({role.model_id for role in protocol.target_roles}):
            target = next(model for model in protocol.models if model.id == target_id)
            descriptor = manifest["artifacts"][_model_key(target_id, replicate)]
            labels = predict_fn(target, descriptor, texts, runtime_settings)
            for (index, text), raw in zip(subset, labels, strict=True):
                if hashlib.sha256(text.encode()).hexdigest() != attempts[index].candidate_sha256:
                    _fail("private candidate content changed before target audit")
                attempts[index].target_predictions.append(
                    CandidatePrediction(
                        model_id=target_id,
                        raw_label=raw,
                        checkpoint_sha256=descriptor["checkpoint_sha256"],
                        tokenizer_sha256=descriptor["tokenizer_sha256"],
                    )
                )
    return [
        {**attempt.model_dump(), "_execution": execution_header(receipt, row["_execution"])}
        for attempt, row in zip(attempts, private_attempts, strict=True)
    ]


def collect_phase_records(
    receipt: dict[str, Any],
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    manifest: dict[str, Any],
    clean_predictions: list[dict[str, Any]],
    similarities: list[dict[str, Any]],
    *audited_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the complete realized, baseline, CKA, attempt, and target-audit grid.

    Args:
        receipt: Fresh phase gate and selection provenance.
        parameters: Registered protocol.
        corpus_evidence: Registered original group membership.
        registration_evidence: Original external registration.
        manifest: Exact realized model evidence.
        clean_predictions: Every model/replica/original baseline.
        similarities: Calibration-only independent probe records, empty in evaluation.
        audited_shards: Every planned source/recipe/replica shard after target auditing.

    Returns:
        Non-text raw ExperimentRecords with no fabricated human or application evidence.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    _check_receipt(receipt, protocol, corpus)
    rows = [*clean_predictions, *similarities, *(row for shard in audited_shards for row in shard)]
    for row in rows:
        _check_artifact_execution(row.get("_execution"), receipt)
    earliest_start = min((row["_execution"]["started_at"] for row in rows), key=parse_timestamp)
    hardware = "; ".join(sorted({row["_execution"]["hardware"] for row in rows}))
    stripped_clean = [
        {key: value for key, value in row.items() if key != "_execution"}
        for row in clean_predictions
    ]
    stripped_similarities = [
        {key: value for key, value in row.items() if key != "_execution"} for row in similarities
    ]
    stripped_attempts = [
        {key: value for key, value in row.items() if key != "_execution"}
        for shard in audited_shards
        for row in shard
    ]
    records = ExperimentRecords.model_validate(
        {
            "phase": receipt["phase"],
            "protocol_sha256": content_digest(protocol),
            "corpus_sha256": content_digest(corpus),
            "execution_ref": protocol.execution_ref,
            "started_at": earliest_start,
            "completed_at": datetime.now(UTC).isoformat(),
            "hardware": hardware,
            "realizations": manifest["realizations"],
            "clean_predictions": stripped_clean,
            "similarities": stripped_similarities,
            "attempts": stripped_attempts,
            "annotations": [],
            "application_records": [],
            **{
                key: receipt[key]
                for key in (
                    "selection_lock_sha256",
                    "selection_registry_uri",
                    "selection_registry_evidence_sha256",
                    "selection_registered_at",
                )
                if key in receipt
            },
        }
    )
    validate_records(
        protocol,
        corpus,
        RegistrationEvidence.model_validate(registration_evidence),
        records,
        protocol.root_seed,
    )
    return records.model_dump(mode="json")


def merge_annotations(
    parameters: dict[str, Any],
    corpus_evidence: dict[str, Any],
    registration_evidence: dict[str, Any],
    raw_records: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach supplied human judgments without repeating any model or attack operation.

    Args:
        parameters: Registered protocol and annotation specification.
        corpus_evidence: Registered original identities.
        registration_evidence: Original external registration.
        raw_records: Immutable heavy-stage non-text output.
        annotations: Human-origin blinded semantic, intent, and task judgments.

    Returns:
        Validated records for target-free calibration selection or held-out analysis.
    """
    protocol = Protocol.model_validate(parameters)
    corpus = CorpusEvidence.model_validate(corpus_evidence)
    records = ExperimentRecords.model_validate({**raw_records, "annotations": annotations})
    validate_records(
        protocol,
        corpus,
        RegistrationEvidence.model_validate(registration_evidence),
        records,
        protocol.root_seed,
    )
    return records.model_dump(mode="json")


def no_similarities(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Supply an empty similarity boundary for held-out execution.

    Args:
        receipt: Fresh held-out gate.

    Returns:
        No representation scores, preserving the frozen selector.
    """
    if receipt.get("phase") != "evaluation":
        _fail("empty similarity output requires held-out evaluation")
    return []
