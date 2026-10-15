"""Run-specific v2 catalogs bind real Kedro paths without changing isolation."""

import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks.manager import _create_hook_manager
from kedro.io import DatasetError
from kedro.runner import SequentialRunner

import transfer_risk.datasets.experiment_v2_json_dataset as module
from transfer_risk.datasets.experiment_v2_immutable_json_dataset import (
    ExperimentV2ImmutableJSONDataset,
)
from transfer_risk.datasets.experiment_v2_json_dataset import ExperimentV2JSONDataset
from transfer_risk.pipeline_registry import register_pipelines
from transfer_risk.pipelines.experiment_v2_preflight.pipeline import create_pipeline


@pytest.mark.parametrize(
    "filepath",
    [
        "data/06_models/summary.json",
        "data/experiment_v2/shared/summary.json",
        "data/experiment_v2/safe-run/../old/summary.json",
        "/absolute/summary.json",
        "s3://historical/summary.json",
    ],
)
def test_historical_shared_or_escaping_catalog_path_is_rejected(filepath):
    with pytest.raises(DatasetError, match="isolated"):
        ExperimentV2JSONDataset(filepath=filepath, run_id="safe-run")


@pytest.mark.parametrize("visibility,prefix", [("private", "data"), ("public", "docs/artifacts")])
def test_isolated_json_roundtrips_without_touching_historical_paths(
    tmp_path, monkeypatch, visibility, prefix
):
    monkeypatch.chdir(tmp_path)
    dataset = ExperimentV2JSONDataset(
        filepath=f"{prefix}/experiment_v2/safe-run/summary.json",
        run_id="safe-run",
        visibility=visibility,
    )
    dataset.save({"aggregate": 3})
    assert dataset.load() == {"aggregate": 3}
    assert not Path("data/07_model_output").exists()


def test_output_symlink_cannot_redirect_a_valid_lexical_path_to_historical_data(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    historical = tmp_path / "data/07_model_output"
    historical.mkdir(parents=True)
    link = tmp_path / "data/experiment_v2/safe-run"
    link.parent.mkdir(parents=True)
    dataset = ExperimentV2JSONDataset(
        filepath="data/experiment_v2/safe-run/summary.json", run_id="safe-run"
    )
    link.symlink_to(historical, target_is_directory=True)
    with pytest.raises(DatasetError, match="symlink"):
        dataset.save({"aggregate": 3})
    assert list(historical.iterdir()) == []


@pytest.mark.parametrize("visibility,prefix", [("private", "data"), ("public", "docs/artifacts")])
def test_kedro_absolute_json_paths_roundtrip_within_the_current_project_run(
    tmp_path, monkeypatch, visibility, prefix
):
    monkeypatch.chdir(tmp_path)
    path = Path.cwd() / prefix / "experiment_v2/safe-run/report.json"
    dataset = ExperimentV2JSONDataset(filepath=str(path), run_id="safe-run", visibility=visibility)
    dataset.save({"aggregate": 3})
    assert dataset.load() == {"aggregate": 3}
    assert path.exists()


@pytest.mark.parametrize("location", ["outside", "sibling", "other-run", "traversal"])
def test_absolute_json_paths_cannot_select_a_different_project_or_run(
    tmp_path, monkeypatch, location
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    if location == "outside":
        path = tmp_path / "outside/data/experiment_v2/safe-run/report.json"
    elif location == "sibling":
        path = tmp_path / "project-copy/data/experiment_v2/safe-run/report.json"
    elif location == "other-run":
        path = project / "data/experiment_v2/safe-run-copy/report.json"
    else:
        path = project / "data/experiment_v2/safe-run/../other-run/report.json"
    with pytest.raises(DatasetError, match="isolated"):
        ExperimentV2JSONDataset(filepath=str(path), run_id="safe-run")


def test_json_writer_remains_bound_to_its_project_after_a_cwd_change(tmp_path, monkeypatch):
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(project)
    dataset = ExperimentV2JSONDataset(
        filepath="data/experiment_v2/safe-run/report.json", run_id="safe-run"
    )
    monkeypatch.chdir(elsewhere)
    dataset.save({"aggregate": 3})
    assert (project / "data/experiment_v2/safe-run/report.json").is_file()
    assert not (elsewhere / "data").exists()


def test_initial_absolute_symlink_path_cannot_redirect_to_another_project(tmp_path, monkeypatch):
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(project)
    link = project / "data/experiment_v2/safe-run"
    link.parent.mkdir(parents=True)
    link.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(DatasetError, match="symlink"):
        ExperimentV2JSONDataset(filepath=str(link / "report.json"), run_id="safe-run")


def test_relative_json_path_from_a_project_subdirectory_uses_the_project_root(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    notebooks = project / "notebooks"
    notebooks.mkdir(parents=True)
    (project / "pyproject.toml").write_text('[tool.kedro]\npackage_name = "transfer_risk"\n')
    monkeypatch.chdir(notebooks)
    dataset = ExperimentV2JSONDataset(
        filepath="data/experiment_v2/safe-run/report.json", run_id="safe-run"
    )
    dataset.save({"aggregate": 3})
    assert (project / "data/experiment_v2/safe-run/report.json").is_file()
    assert not (notebooks / "data").exists()


def test_real_kedro_context_materializes_v2_and_persists_template_blockers(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[tool.kedro]\npackage_name = "transfer_risk"\n')
    shutil.copytree(
        repository / "conf/experiment_v2/templates", project / "conf/experiment_v2/templates"
    )
    monkeypatch.chdir(project)
    loader = OmegaConfigLoader(
        conf_source=str(repository / "conf"), base_env="base", env="experiment_v2"
    )
    context = KedroContext(
        package_name="transfer_risk",
        project_path=project,
        config_loader=loader,
        env="experiment_v2",
        hook_manager=_create_hook_manager(),
    )
    context._pipelines_to_validate = ["experiment_v2_preflight"]
    catalog = context.catalog
    # MLflow's actual catalog hook materializes every explicit entry before running nodes.
    assert catalog.items()
    pipelines = register_pipelines()
    for name, pipeline in pipelines.items():
        if name.startswith("experiment_v2"):
            for boundary in pipeline.datasets():
                if not boundary.startswith("params:"):
                    assert catalog.get(boundary) is not None
    with pytest.raises(ValueError, match="preflight blocked"):
        SequentialRunner().run(create_pipeline(), catalog)
    report = catalog.load("v2_preflight_report")
    assert report["status"] == "blocked" and report["rerun_ready"] is False
    assert (project / "data/experiment_v2/prospective-v2/analysis/preflight.json").is_file()
    assert (
        project / "data/experiment_v2/prospective-v2/analysis/execution_manifest.json"
    ).is_file()
    assert not catalog.exists("v2_preflight_receipt")
    assert not (project / "data/experiment_v2/prospective-v2/models").exists()


@pytest.mark.parametrize("redirect", [False, True])
def test_real_catalog_public_wrapper_logs_only_its_checked_absolute_path(
    tmp_path, monkeypatch, redirect
):
    repository = Path(__file__).resolve().parents[2]
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    (project / "pyproject.toml").write_text('[tool.kedro]\npackage_name = "transfer_risk"\n')
    monkeypatch.chdir(project)
    context = KedroContext(
        package_name="transfer_risk",
        project_path=project,
        config_loader=OmegaConfigLoader(
            conf_source=str(repository / "conf"), base_env="base", env="experiment_v2"
        ),
        env="experiment_v2",
        hook_manager=_create_hook_manager(),
    )
    context._pipelines_to_validate = ["experiment_v2_preflight"]
    catalog = context.catalog
    catalog.get("v2_public_summary")
    log_artifact = Mock()
    monkeypatch.setattr("mlflow.log_artifact", log_artifact)
    run = project / "docs/artifacts/experiment_v2/prospective-v2"
    if redirect:
        run.parent.mkdir(parents=True)
        run.symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.chdir(elsewhere)
    if redirect:
        with pytest.raises(DatasetError, match="symlink"):
            catalog.save("v2_public_summary", {"synthetic_aggregate": 1})
        log_artifact.assert_not_called()
        assert list(elsewhere.iterdir()) == []
    else:
        catalog.save("v2_public_summary", {"synthetic_aggregate": 1})
        assert catalog.load("v2_public_summary") == {"synthetic_aggregate": 1}
        assert (run / "summary.json").is_file()
        assert not (elsewhere / "docs").exists()
        log_artifact.assert_called_once_with(str(run / "summary.json"), "experiment_v2")


@pytest.mark.parametrize("immutable", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), object()])
def test_failed_serialization_preserves_existing_json_and_leaves_no_new_artifact(
    tmp_path, monkeypatch, immutable, bad
):

    monkeypatch.chdir(tmp_path)
    cls = ExperimentV2ImmutableJSONDataset if immutable else ExperimentV2JSONDataset
    path = tmp_path / "data/experiment_v2/safe-run/evidence.json"
    dataset = cls(filepath=str(path), run_id="safe-run", save_args={"allow_nan": True})
    with pytest.raises((DatasetError, ValueError, TypeError)):
        dataset.save({"partial": [1, bad]})
    assert not path.exists()
    dataset.save({"complete": 1})
    if not immutable:
        with pytest.raises((DatasetError, ValueError, TypeError)):
            dataset.save({"partial": [1, bad]})
    assert dataset.load() == {"complete": 1}


def test_immutable_publication_does_not_clobber_a_concurrent_writer(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data/experiment_v2/safe-run/evidence.json"
    dataset = ExperimentV2ImmutableJSONDataset(filepath=str(path), run_id="safe-run")
    monkeypatch.setattr(dataset, "exists", lambda: False)
    dataset.save({"first": 1})
    with pytest.raises(DatasetError, match="immutable"):
        dataset.save({"second": 2})
    assert dataset.load() == {"first": 1}


@pytest.mark.parametrize("immutable", [False, True])
def test_interrupted_atomic_json_publication_preserves_destination(
    tmp_path, monkeypatch, immutable
):

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data/experiment_v2/safe-run/evidence.json"
    cls = ExperimentV2ImmutableJSONDataset if immutable else ExperimentV2JSONDataset
    dataset = cls(filepath=str(path), run_id="safe-run")
    if not immutable:
        dataset.save({"previous": 1})

    def interrupted(*_args):
        msg = "synthetic interrupted publication"
        raise OSError(msg)

    monkeypatch.setattr(module.os, "link" if immutable else "replace", interrupted)
    with pytest.raises((DatasetError, OSError), match="interrupted publication"):
        dataset.save({"new": 2})
    if immutable:
        assert not path.exists()
    else:
        assert dataset.load() == {"previous": 1}
    assert list(path.parent.iterdir()) == ([] if immutable else [path])
