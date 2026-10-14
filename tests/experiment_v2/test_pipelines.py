"""V2 runs explicitly, records blockers, and leaves the historical default untouched."""

from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from transfer_risk.lib.experiment_v2.protocol import build_execution_manifest, validate_preflight
from transfer_risk.lib.experiment_v2.schemas import (
    CorpusEvidence,
    Protocol,
    RegistrationEvidence,
)
from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.experiment_v2_analysis.pipeline import (
    create_pipeline as analysis_pipeline,
)
from transfer_risk.pipelines.experiment_v2_preflight.pipeline import (
    create_pipeline as preflight_pipeline,
)

from .test_analysis import complete_fixture


def test_v2_opt_in_pipelines_never_enter_historical_composites():
    pipelines = register_pipelines()
    v2_names = {
        "experiment_v2_preflight",
        "experiment_v2_selection",
        "experiment_v2_analysis",
        "experiment_v2_execute",
        "experiment_v2_target_prepare",
        "experiment_v2_assets_prepare",
        "experiment_v2_models",
        "experiment_v2_calibration_execute",
        "experiment_v2_evaluation_execute",
        "experiment_v2_calibration_annotations",
        "experiment_v2_evaluation_annotations",
    }
    assert v2_names <= set(pipelines)
    for name in ("__default__", "stage", "downstream", "audit", "paper_reporting"):
        assert not any(node.name.startswith("v2_") for node in pipelines[name].nodes)
    for name in v2_names:
        inputs = pipelines[name].inputs()
        assert all(item.startswith(("v2_", "params:")) for item in inputs)
        assert "target_model" not in inputs
        assert "task_splits" not in inputs
        assert "canonical_dataset" not in inputs


def test_checked_in_template_persists_blockers_then_fails_without_claiming_ready(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    loader = OmegaConfigLoader(conf_source=str(root / "conf"), base_env="base", env="experiment_v2")
    protocol = loader["parameters"]["experiment_v2"]

    report = validate_preflight(
        Protocol.model_validate(protocol),
        CorpusEvidence(grouping_method=None, grouping_evidence_sha256=None, records=[]),
        RegistrationEvidence(),
    )
    assert report["status"] == "blocked"
    assert report["rerun_ready"] is False
    monkeypatch.chdir(tmp_path)
    datasets = {
        "params:experiment_v2": MemoryDataset(protocol),
        "params:seed": MemoryDataset(20260616),
        "v2_corpus_evidence": MemoryDataset(
            {"grouping_method": None, "grouping_evidence_sha256": None, "records": []}
        ),
        "v2_registration_evidence": MemoryDataset({}),
        "v2_preflight_report": MemoryDataset(),
        "v2_execution_manifest": MemoryDataset(),
        "v2_preflight_receipt": MemoryDataset(),
    }
    catalog = DataCatalog(datasets)
    with pytest.raises(ValueError, match="preflight blocked"):
        SequentialRunner().run(preflight_pipeline(), catalog)
    persisted = catalog.load("v2_preflight_report")
    assert persisted["status"] == "blocked"
    assert persisted["human_original_review_complete"] is False
    assert not catalog.exists("v2_preflight_receipt")


def test_target_free_analysis_runs_through_real_kedro_nodes_with_synthetic_records():
    protocol, corpus, registration, calibration, evaluation, lock = complete_fixture()
    datasets = {
        "params:experiment_v2": MemoryDataset(protocol.model_dump()),
        "params:seed": MemoryDataset(20260616),
        "v2_corpus_evidence": MemoryDataset(corpus.model_dump()),
        "v2_registration_evidence": MemoryDataset(registration.model_dump()),
        "v2_calibration_records": MemoryDataset(calibration.model_dump()),
        "v2_evaluation_records": MemoryDataset(evaluation.model_dump()),
        "v2_selection_lock": MemoryDataset(lock),
        "v2_preflight_report": MemoryDataset(),
        "v2_execution_manifest": MemoryDataset(),
        "v2_preflight_receipt": MemoryDataset(),
        "v2_public_summary": MemoryDataset(),
    }
    catalog = DataCatalog(datasets)
    SequentialRunner().run(analysis_pipeline(), catalog)
    report = catalog.load("v2_public_summary")
    assert report["selection"]["primary"]["cka_yield"] == 1.0
    assert report["outcomes"]["primary"]["attempted"] == 96
    assert report["analysis_only"] is True


def test_environment_catalog_has_guarded_run_specific_mlflow_outputs():
    root = Path(__file__).resolve().parents[2]
    loader = OmegaConfigLoader(conf_source=str(root / "conf"), base_env="base", env="experiment_v2")
    catalog = loader["catalog"]
    for name in (
        "v2_preflight_report",
        "v2_execution_manifest",
        "v2_preflight_receipt",
        "v2_selection_lock",
    ):
        assert catalog[name]["type"].endswith("ExperimentV2JSONDataset")
        assert catalog[name]["filepath"].startswith("data/experiment_v2/prospective-v2/")
    public = catalog["v2_public_summary"]
    assert public["type"] == "kedro_mlflow.io.artifacts.MlflowArtifactDataset"
    assert public["dataset"]["type"].endswith("ExperimentV2JSONDataset")
    assert public["dataset"]["filepath"].startswith("docs/artifacts/experiment_v2/prospective-v2/")


def test_execution_graphs_resolve_every_boundary_without_loading_models_or_text():
    root = Path(__file__).resolve().parents[2]
    loader = OmegaConfigLoader(conf_source=str(root / "conf"), base_env="base", env="experiment_v2")
    configuration = {
        name: config for name, config in loader["catalog"].items() if name.startswith("v2_")
    }
    catalog = DataCatalog.from_config(configuration)
    pipelines = register_pipelines()
    names = (
        "experiment_v2_target_prepare",
        "experiment_v2_assets_prepare",
        "experiment_v2_models",
        "experiment_v2_calibration_execute",
        "experiment_v2_evaluation_execute",
        "experiment_v2_calibration_annotations",
        "experiment_v2_evaluation_annotations",
    )
    for name in names:
        for boundary in pipelines[name].datasets():
            if not boundary.startswith("params:"):
                assert catalog.get(boundary) is not None, (name, boundary)
    for name in (
        "v2_models_receipt",
        "v2_attack_assets_preparation_receipt",
        "v2_target_preparation_receipt",
        "v2_calibration_receipt",
        "v2_evaluation_receipt",
        "v2_calibration_current_models",
        "v2_evaluation_current_models",
    ):
        assert isinstance(catalog.get(name), MemoryDataset)


def test_execution_manifest_paths_match_catalog_files_and_shard_factories():
    root = Path(__file__).resolve().parents[2]
    loader = OmegaConfigLoader(conf_source=str(root / "conf"), base_env="base", env="experiment_v2")
    protocol = Protocol.model_validate(loader["parameters"]["experiment_v2"])
    corpus = CorpusEvidence(grouping_method=None, grouping_evidence_sha256=None, records=[])
    report = validate_preflight(protocol, corpus, RegistrationEvidence())
    manifest = build_execution_manifest(protocol, corpus, protocol.root_seed, report)
    paths = manifest["private_artifacts"]
    configuration = loader["catalog"]
    expected = {
        "train": "v2_private_train",
        "validation": "v2_private_validation",
        "probe": "v2_private_probe",
        "calibration": "v2_private_calibration",
        "evaluation": "v2_private_evaluation",
        "target_train": "v2_target_private_train",
        "target_validation": "v2_target_private_validation",
        "prepared_target": "v2_prepared_target",
        "target_preparation_record": "v2_target_preparation_record",
        "checkpoints": "v2_model__{model_id}__r{replicate}",
        "model_metadata": "v2_model_meta__{model_id}__r{replicate}",
        "models_manifest": "v2_models_manifest",
        "probe_similarities": "v2_calibration_similarities",
        "attempt_shards": "v2_{phase}_attempts__{source_id}__r{replicate}__{recipe}",
        "audited_shards": "v2_{phase}_audited__{source_id}__r{replicate}__{recipe}",
        "selection_lock": "v2_selection_lock",
    }
    for field, name in expected.items():
        assert paths[field] == configuration[name]["filepath"]
    for phase in ("calibration", "evaluation"):
        for field, name in (
            ("clean_predictions", f"v2_{phase}_clean_predictions"),
            ("raw_phase_records", f"v2_{phase}_raw_records"),
            ("reviewed_phase_records", f"v2_{phase}_records"),
        ):
            assert paths[field].format(phase=phase) == configuration[name]["filepath"]
    assert (
        paths["attack_asset_manifest"]
        == configuration["v2_attack_asset_manifest"]["manifest_filepath"]
    )
    assert paths["attack_asset_cache"] == configuration["v2_attack_asset_manifest"]["cache_root"]
    assert manifest["runtime_gate"]["execution_ref"] is None
    assert manifest["rerun_ready"] is False
