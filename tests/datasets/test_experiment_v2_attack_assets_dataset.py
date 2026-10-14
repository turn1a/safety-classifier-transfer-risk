"""Verify catalog-owned offline recipe assets with synthetic file bytes only."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
from kedro.io import DatasetError

from transfer_risk.lib.experiment_v2.protocol import content_digest


def asset_manifest(run_id):
    paths = [
        "masked-lm/config.json",
        "masked-lm/model.safetensors",
        "masked-lm/tokenizer.json",
        "semantic-encoder/modules.json",
        "semantic-encoder/0_Transformer/config.json",
        "semantic-encoder/0_Transformer/model.safetensors",
        "semantic-encoder/0_Transformer/tokenizer.json",
        "textattack/post_install_check_3",
        *[
            f"textattack/word_embeddings/paragramcf/{name}"
            for name in (
                "paragram.npy",
                "wordlist.pickle",
                "mse_dist.p",
                "cos_sim.p",
                "nn.npy",
            )
        ],
        "nltk/corpora/wordnet.zip",
        "nltk/corpora/omw-1.4.zip",
        "nltk/corpora/stopwords/english",
        *[
            f"nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{name}.json"
            for name in ("weights", "tagdict", "classes")
        ],
        "nltk/taggers/universal_tagset/en-ptb.map",
        "nltk/tokenizers/punkt_tab/english/abbrev_types.txt",
    ]
    contents = {
        path: b"" if path.endswith("post_install_check_3") else path.encode() for path in paths
    }
    return {
        "schema_version": 2,
        "run_id": run_id,
        "cache_paths": {"textattack": "textattack", "nltk": "nltk", "huggingface": "huggingface"},
        "roles": {
            role: {"repository": f"synthetic/{role}", "revision": "a" * 40, "path": path}
            for role, path in (
                ("masked_lm", "masked-lm"),
                ("semantic_encoder", "semantic-encoder"),
                ("counterfitted_embeddings", "textattack/word_embeddings/paragramcf"),
                ("nltk", "nltk"),
            )
        },
        "files": [
            {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(contents.items())
        ],
    }, contents


def _dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_attack_assets_dataset")
    return module.ExperimentV2AttackAssetsDataset(
        manifest_filepath="data/experiment_v2/test-run/attack_assets/manifest.json",
        cache_root="data/experiment_v2/test-run/attack_assets/cache",
        run_id="test-run",
    )


def _staging(tmp_path, manifest, contents):
    root = tmp_path / "data/experiment_v2/test-run/attack_assets/staging"
    for path, data in contents.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {"source_root": str(root), "manifest": manifest}


def test_asset_boundary_materializes_and_rechecks_every_file_byte(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    manifest, contents = asset_manifest("test-run")
    dataset.save(_staging(tmp_path, manifest, contents))
    descriptor = dataset.load()
    assert descriptor["manifest_sha256"] == content_digest(manifest)
    assert descriptor["paths"]["masked_lm"].endswith("attack_assets/cache/masked-lm")
    (Path(descriptor["cache_root"]) / "masked-lm/model.safetensors").write_bytes(b"changed bytes")
    with pytest.raises(DatasetError, match=r"hash|bytes|identity"):
        dataset.load()


def test_auxiliary_assets_reject_unpinned_sources_before_copying(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    manifest, contents = asset_manifest("test-run")
    manifest["roles"]["masked_lm"]["revision"] = None
    with pytest.raises(DatasetError, match=r"pin|revision|immutable"):
        dataset.save(_staging(tmp_path, manifest, contents))
    assert not (tmp_path / "data/experiment_v2/test-run/attack_assets/cache").exists()


def test_auxiliary_assets_reject_missing_and_unlisted_bytes(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    manifest, contents = asset_manifest("test-run")
    bundle = _staging(tmp_path, manifest, contents)
    source = Path(bundle["source_root"])
    (source / "unexpected-model.bin").write_bytes(b"unregistered asset")
    with pytest.raises(DatasetError, match=r"inventory|unlisted|coverage"):
        dataset.save(bundle)
    (source / "unexpected-model.bin").unlink()
    (source / "nltk/corpora/wordnet.zip").unlink()
    with pytest.raises(DatasetError, match=r"inventory|missing|coverage"):
        dataset.save(bundle)


def test_auxiliary_cache_cannot_escape_its_private_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_attack_assets_dataset")
    with pytest.raises(DatasetError, match="isolated"):
        module.ExperimentV2AttackAssetsDataset(
            manifest_filepath="data/experiment_v2/test-run/attack_assets/manifest.json",
            cache_root="data/legacy/textattack",
            run_id="test-run",
        )


def test_auxiliary_nlp_inventory_requires_every_actual_pos_tagger_file(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    manifest, contents = asset_manifest("test-run")
    missing = (
        "nltk/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json"
    )
    manifest["files"] = [row for row in manifest["files"] if row["path"] != missing]
    contents.pop(missing)
    with pytest.raises(DatasetError, match=r"POS|resource|missing"):
        dataset.save(_staging(tmp_path, manifest, contents))


def test_absolute_auxiliary_paths_and_staging_keep_the_captured_project_after_cwd_change(
    tmp_path, monkeypatch
):
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(project)
    root = Path.cwd() / "data/experiment_v2/test-run/attack_assets"
    module = importlib.import_module("transfer_risk.datasets.experiment_v2_attack_assets_dataset")
    dataset = module.ExperimentV2AttackAssetsDataset(
        manifest_filepath=str(root / "manifest.json"),
        cache_root=str(root / "cache"),
        run_id="test-run",
    )
    manifest, contents = asset_manifest("test-run")
    bundle = _staging(project, manifest, contents)
    monkeypatch.chdir(elsewhere)
    dataset.save(bundle)
    descriptor = dataset.load()
    assert descriptor["cache_root"] == str(root / "cache")
    assert descriptor["manifest_sha256"] == content_digest(manifest)
    assert not (elsewhere / "data").exists()


@pytest.mark.parametrize("source_kind", ["sibling-project", "traversal", "symlink"])
def test_auxiliary_staging_cannot_escape_the_captured_run(tmp_path, monkeypatch, source_kind):
    project = tmp_path / "project"
    project.mkdir()
    dataset = _dataset(project, monkeypatch)
    manifest, contents = asset_manifest("test-run")
    bundle = _staging(project, manifest, contents)
    if source_kind == "sibling-project":
        bundle = _staging(tmp_path / "sibling", manifest, contents)
    elif source_kind == "traversal":
        bundle["source_root"] = f"{bundle['source_root']}/../staging"
    else:
        source = Path(bundle["source_root"])
        moved = tmp_path / "saved-assets"
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(DatasetError, match=r"isolated|symlink"):
        dataset.save(bundle)
    assert not (project / "data/experiment_v2/test-run/attack_assets/cache").exists()
