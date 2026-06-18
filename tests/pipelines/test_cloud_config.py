"""The cloud environment redirects only the boundary datasets to S3, via globals.

These guard the catalog-owns-S3 mechanism (the scoped ``tr.bucket`` resolver + per-env
``globals.yml``) without touching real S3: a ``--env cloud`` catalog must resolve the boundary
artifacts (splits, surrogate checkpoints, ONNX graphs, adversarial partitions) to ``s3://`` and
leave everything else (similarity, transfer, the assembled table) on local paths; the base env
must stay fully local and need no ``TR_BUCKET``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kedro.config import OmegaConfigLoader

from transfer_risk.settings import CONFIG_LOADER_ARGS

_CONF = str(Path(__file__).resolve().parents[2] / "conf")
_BUCKET = "tr-test-bucket"


def _catalog(env: str | None) -> dict[str, Any]:
    loader = OmegaConfigLoader(
        conf_source=_CONF,
        env=env,
        base_env="base",
        default_run_env="local",
        **CONFIG_LOADER_ARGS,
    )
    return loader["catalog"]


@pytest.fixture
def cloud_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("TR_BUCKET", _BUCKET)
    monkeypatch.setenv("TR_REGION", "eu-central-1")
    return _catalog("cloud")


def test_boundary_datasets_resolve_to_s3(cloud_catalog: dict[str, Any]) -> None:
    prefix = f"s3://{_BUCKET}/"
    assert cloud_catalog["task_splits"]["filepath"].startswith(prefix)
    assert cloud_catalog["surrogate__{name}"]["filepath"].startswith(prefix)
    assert cloud_catalog["onnx__{name}"]["filepath"].startswith(prefix)
    assert cloud_catalog["adversarial__{cellkey}"]["filepath"].startswith(prefix)
    assert cloud_catalog["adversarial_shard__{shardkey}"]["filepath"].startswith(prefix)


def test_non_boundary_datasets_stay_local(cloud_catalog: dict[str, Any]) -> None:
    # The local downstream (transfer/risk/reporting) writes these even under --env cloud.
    assert cloud_catalog["similarity_table"]["filepath"].startswith("data/")
    assert cloud_catalog["transfer_results"]["filepath"].startswith("data/")
    assert cloud_catalog["adversarial_examples"]["filepath"].startswith("data/")


def test_base_env_is_fully_local() -> None:
    # No TR_BUCKET set: the base env must resolve without it and stay local.
    catalog = _catalog(None)
    assert catalog["surrogate__{name}"]["filepath"].startswith("data/")
    assert catalog["task_splits"]["filepath"].startswith("data/")
    assert catalog["adversarial__{cellkey}"]["filepath"].startswith("data/")
