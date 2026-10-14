"""Materialize and verify immutable local TextAttack auxiliary assets through the catalog."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, NoReturn

from kedro.io import AbstractDataset, DatasetError

from transfer_risk.datasets.experiment_v2_immutable_json_dataset import (
    ExperimentV2ImmutableJSONDataset,
)
from transfer_risk.datasets.experiment_v2_json_dataset import ExperimentV2JSONDataset
from transfer_risk.datasets.experiment_v2_paths import guarded_run_path
from transfer_risk.lib.experiment_v2.protocol import content_digest, is_digest
from transfer_risk.lib.experiment_v2.schemas import EvidenceRecord

_ROLES = {"masked_lm", "semantic_encoder", "counterfitted_embeddings", "nltk"}
_CACHE_NAMES = {"textattack", "nltk", "huggingface"}


class AuxiliaryRole(EvidenceRecord):
    """Pin an auxiliary model, archive, or dataset and its local realized directory."""

    repository: str | None
    revision: str | None
    path: str


class AuxiliaryFile(EvidenceRecord):
    """Pin exact file bytes under the dedicated realized auxiliary cache root."""

    path: str
    sha256: str | None


class AttackAssetManifest(EvidenceRecord):
    """Bind every model, tokenizer, embedding, and NLP resource needed by the recipes."""

    schema_version: Literal[2]
    run_id: str
    cache_paths: dict[str, str]
    roles: dict[str, AuxiliaryRole]
    files: list[AuxiliaryFile]


def _asset_fail(message: str) -> NoReturn:
    """Raise a private asset-boundary error without printing file content.

    Args:
        message: Violated immutable asset invariant.

    Raises:
        DatasetError: Always.
    """
    raise DatasetError(message)


def _relative_asset_path(value: str) -> Path:
    """Reject absolute and traversing paths in a frozen auxiliary manifest.

    Args:
        value: Cache-root-relative asset filepath or role directory.

    Returns:
        The validated relative path.
    """
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or value in {".", ""}:
        _asset_fail("auxiliary asset paths must stay within their isolated cache root")
    return path


def _validate_manifest(manifest: AttackAssetManifest, run_id: str) -> None:
    """Require complete immutable pins and a nonempty exact file inventory.

    Args:
        manifest: Parsed auxiliary asset inventory.
        run_id: Declared private run identity.
    """
    if (
        manifest.run_id != run_id
        or set(manifest.roles) != _ROLES
        or set(manifest.cache_paths) != _CACHE_NAMES
    ):
        _asset_fail("auxiliary manifest requires the exact run, model, embedding and NLP roles")
    for role in manifest.roles.values():
        if not role.repository or not is_digest(role.revision, lengths=(40, 64)):
            _asset_fail("auxiliary sources require immutable repository and revision pins")
        _relative_asset_path(role.path)
    for cache in manifest.cache_paths.values():
        _relative_asset_path(cache)
    if (
        manifest.roles["nltk"].path != manifest.cache_paths["nltk"]
        or manifest.roles["counterfitted_embeddings"].path
        != (f"{manifest.cache_paths['textattack']}/word_embeddings/paragramcf")
        or len(set(manifest.cache_paths.values())) != len(_CACHE_NAMES)
    ):
        _asset_fail("auxiliary role/cache paths must match the actual offline recipe loaders")
    if not manifest.files or len({item.path for item in manifest.files}) != len(manifest.files):
        _asset_fail("auxiliary asset file inventory is absent or duplicated")
    for item in manifest.files:
        _relative_asset_path(item.path)
        if not is_digest(item.sha256):
            _asset_fail("every auxiliary file requires its actual immutable byte hash")
    _require_recipe_files(manifest)


def _require_recipe_files(manifest: AttackAssetManifest) -> None:
    """Require the files used by the registered MLM, encoder, embedding, and NLP loaders.

    Args:
        manifest: Complete pinned auxiliary file inventory.
    """
    paths = {item.path for item in manifest.files}
    textattack = manifest.cache_paths["textattack"]
    required = {f"{textattack}/post_install_check_3"}
    embeddings = manifest.roles["counterfitted_embeddings"].path
    required |= {
        f"{embeddings}/{name}"
        for name in (
            "paragram.npy",
            "wordlist.pickle",
            "mse_dist.p",
            "cos_sim.p",
            "nn.npy",
        )
    }
    required.add(f"{manifest.roles['masked_lm'].path}/config.json")
    required.add(f"{manifest.roles['semantic_encoder'].path}/modules.json")
    if not required <= paths:
        _asset_fail("auxiliary manifest is missing recipe model, embedding or offline marker files")
    nltk = manifest.cache_paths["nltk"]
    required_nlp = {
        f"{nltk}/corpora/stopwords/english",
        f"{nltk}/taggers/universal_tagset/en-ptb.map",
        *{
            f"{nltk}/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.{name}.json"
            for name in ("weights", "tagdict", "classes")
        },
    }
    if not required_nlp <= paths:
        _asset_fail("auxiliary manifest is missing English POS, tagset or stopword resource files")
    resource_alternatives = [
        (f"{nltk}/corpora/wordnet.zip", f"{nltk}/corpora/wordnet/"),
        (f"{nltk}/corpora/omw-1.4.zip", f"{nltk}/corpora/omw-1.4/"),
    ]
    if any(
        not any(
            any(path == prefix or path.startswith(prefix) for path in paths)
            for prefix in alternatives
        )
        for alternatives in resource_alternatives
    ):
        _asset_fail("auxiliary manifest is missing WordNet, stopword or POS/tagset resources")
    for role_name in ("masked_lm", "semantic_encoder"):
        prefix = f"{manifest.roles[role_name].path}/"
        files = [path for path in paths if path.startswith(prefix)]
        if not any(path.endswith((".safetensors", ".bin")) for path in files) or not any(
            Path(path).name.startswith("tokenizer")
            or Path(path).name in {"vocab.txt", "vocab.json", "spiece.model"}
            for path in files
        ):
            _asset_fail("auxiliary neural roles require weights and tokenizer/vocabulary bytes")


def verify_asset_cache(root: Path, manifest: AttackAssetManifest) -> None:
    """Verify complete realized file coverage and hashes without importing NLP/ML libraries.

    Only runtime lock files are excluded from the inventory. Unlisted content,
    symlinks, absent files, and changed bytes fail before any recipe search.

    Args:
        root: Catalog-owned realized or staging cache root.
        manifest: Exact frozen auxiliary inventory.
    """
    _validate_manifest(manifest, manifest.run_id)
    if not root.is_dir() or root.is_symlink():
        _asset_fail("auxiliary cache root is absent or redirected")
    files = list(root.rglob("*"))
    if any(item.is_symlink() for item in files):
        _asset_fail("auxiliary cache cannot contain a symlink")
    actual = {
        item.relative_to(root).as_posix(): item
        for item in files
        if item.is_file() and not item.name.endswith(".lock") and ".locks" not in item.parts
    }
    expected = {item.path: item.sha256 for item in manifest.files}
    if set(actual) != set(expected):
        _asset_fail("auxiliary cache file inventory coverage differs from its frozen manifest")
    for name, path in actual.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[name]:
            _asset_fail("auxiliary cache file bytes differ from their registered hash identity")
    marker = root / manifest.cache_paths["textattack"] / "post_install_check_3"
    if marker.read_bytes() != b"":
        _asset_fail("offline TextAttack installation marker must be the reviewed empty file")


class ExperimentV2AttackAssetsDataset(AbstractDataset[dict[str, Any], dict[str, Any]]):
    """Own a verified immutable offline cache and its separately stored non-text manifest."""

    def __init__(
        self,
        *,
        manifest_filepath: str,
        cache_root: str,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Bind manifest and cache directories to one isolated local v2 run.

        Args:
            manifest_filepath: Relative or absolute private manifest path outside the cache.
            cache_root: Relative or absolute realized cache directory in this project's run.
            run_id: Dedicated prospective run identity.
            metadata: Optional Kedro metadata.
        """
        self._manifest = ExperimentV2ImmutableJSONDataset(filepath=manifest_filepath, run_id=run_id)
        self._boundary = ExperimentV2JSONDataset(
            filepath=f"{cache_root}/_isolation.json", run_id=run_id
        )
        self._project_root = self._boundary._project_root
        self._root = self._boundary._configured_path.parent
        self._run_root = self._project_root / "data" / "experiment_v2" / run_id
        self._run_id = run_id
        if self._manifest._configured_path.is_relative_to(self._root):
            _asset_fail("auxiliary manifest must be stored outside its hashed cache directory")
        self.metadata = metadata

    def load(self) -> dict[str, Any]:
        """Recheck every pinned file byte and return only local verified role/cache paths.

        Returns:
            Manifest, its digest, private cache root, and realized local asset paths.
        """
        self._boundary._assert_no_symlink()
        manifest = AttackAssetManifest.model_validate(self._manifest.load())
        _validate_manifest(manifest, self._run_id)
        verify_asset_cache(self._root, manifest)
        paths = {name: str(self._root / role.path) for name, role in manifest.roles.items()}
        paths.update({name: str(self._root / path) for name, path in manifest.cache_paths.items()})
        return {
            "manifest_sha256": content_digest(manifest),
            "manifest": manifest.model_dump(),
            "cache_root": str(self._root),
            "paths": paths,
        }

    def save(self, data: dict[str, Any]) -> None:
        """Copy operator-supplied pinned staging bytes through the catalog, without downloads.

        Args:
            data: Source_root and manifest of separately prepared pinned asset inputs.

        Raises:
            DatasetError: If pins, bytes, isolation, or immutable output constraints fail.
        """
        self._boundary._assert_no_symlink()
        manifest = AttackAssetManifest.model_validate(data["manifest"])
        _validate_manifest(manifest, self._run_id)
        source_value = data.get("source_root")
        if not isinstance(source_value, str):
            _asset_fail("pinned auxiliary staging source is absent")
        source = guarded_run_path(
            source_value, project_root=self._project_root, run_id=self._run_id
        )
        if not source.is_relative_to(self._run_root) or source == self._root:
            _asset_fail(
                "auxiliary staging requires a separate directory in the isolated private run"
            )
        verify_asset_cache(source, manifest)
        if self._root.exists() or self._manifest.exists():
            _asset_fail("realized auxiliary cache and manifest are immutable for this run")
        self._root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".v2-assets-", dir=self._root.parent) as temporary:
            staging = Path(temporary) / "cache"
            for item in manifest.files:
                destination = staging / item.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / item.path, destination)
            for directory in manifest.cache_paths.values():
                (staging / directory).mkdir(parents=True, exist_ok=True)
            verify_asset_cache(staging, manifest)
            staging.replace(self._root)
        try:
            self._manifest.save(manifest.model_dump(mode="json"))
        except Exception:
            shutil.rmtree(self._root)
            raise

    def exists(self) -> bool:
        """Check whether the immutable manifest/cache boundary was materialized.

        Returns:
            Whether both manifest and local realized directory exist.
        """
        self._boundary._assert_no_symlink()
        return self._manifest.exists() and self._root.is_dir()

    def _describe(self) -> dict[str, str]:
        """Describe the private offline auxiliary boundary without exposing contents.

        Returns:
            Run identity and private cache root.
        """
        return {"run_id": self._run_id, "cache_root": str(self._root)}
