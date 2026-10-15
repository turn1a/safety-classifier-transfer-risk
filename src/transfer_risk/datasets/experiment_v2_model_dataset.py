"""Persist immutable private v2 checkpoints and fingerprint their exact saved artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from typing import TYPE_CHECKING, Any

from kedro.io import AbstractDataset, DatasetError

from transfer_risk.datasets.experiment_v2_json_dataset import ExperimentV2JSONDataset
from transfer_risk.datasets.experiment_v2_paths import guarded_run_path
from transfer_risk.datasets.surrogate_model_dataset import SurrogateModelDataset

if TYPE_CHECKING:
    from pathlib import Path


def _canonical_bytes(value: Any) -> bytes:
    """Encode fingerprint manifests with fixed JSON formatting.

    Args:
        value: JSON-compatible manifest or tokenizer configuration.

    Returns:
        UTF-8 JSON with sorted keys, no whitespace, and unescaped Unicode.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _file_digest(path: Path) -> str:
    """Hash file bytes without loading a checkpoint into memory.

    Args:
        path: One private artifact file.

    Returns:
        Its lowercase SHA256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _weight_artifact(path: Path) -> bool:
    """Recognize saved transformer and BiLSTM weights and inference configuration.

    Args:
        path: Relative checkpoint artifact path.

    Returns:
        Whether it belongs in the checkpoint fingerprint.
    """
    return (
        path.name in {"config.json", "model.pt"}
        or path.suffix == ".safetensors"
        or (path.name.startswith("pytorch_model") and path.suffix == ".bin")
        or path.name.endswith((".safetensors.index.json", ".bin.index.json"))
    )


def checkpoint_descriptor(path: Path) -> dict[str, str]:
    """Fingerprint a local checkpoint and every saved tokenizer or vocabulary artifact.

    Checkpoint files are config.json, model.pt, *.safetensors, pytorch_model*.bin,
    and their *.index.json manifests. Every other saved file belongs to the
    tokenizer manifest. A BiLSTM adds a canonical virtual tokenizer artifact
    containing its vocab and max_seq_len. Each sorted manifest contains objects
    with path and sha256 keys; its canonical JSON bytes are hashed with SHA256.

    Args:
        path: Isolated private saved model directory.

    Returns:
        Absolute local path and separate checkpoint/tokenizer fingerprints.

    Raises:
        DatasetError: If files are absent, redirected, or lack model/tokenizer artifacts.
    """
    if not path.is_dir():
        msg = "v2 checkpoint directory is absent"
        raise DatasetError(msg)
    files = sorted(path.rglob("*"))
    if any(item.is_symlink() for item in files):
        msg = "v2 checkpoint cannot contain a symlink"
        raise DatasetError(msg)
    weights: list[dict[str, str]] = []
    tokenizer: list[dict[str, str]] = []
    for item in files:
        if item.is_file():
            entry = {"path": item.relative_to(path).as_posix(), "sha256": _file_digest(item)}
            (weights if _weight_artifact(item) else tokenizer).append(entry)
    config_path = path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if "vocab" in config:
            token_bytes = _canonical_bytes(
                {"vocab": config["vocab"], "max_seq_len": config["max_seq_len"]}
            )
            tokenizer.append(
                {
                    "path": "$bilstm_tokenizer.json",
                    "sha256": hashlib.sha256(token_bytes).hexdigest(),
                }
            )
    if not weights or not tokenizer or not config_path.is_file():
        msg = "v2 checkpoint requires saved weights, configuration, and tokenizer artifacts"
        raise DatasetError(msg)
    return {
        "path": str(path.absolute()),
        "checkpoint_sha256": hashlib.sha256(_canonical_bytes(weights)).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            _canonical_bytes(sorted(tokenizer, key=lambda entry: entry["path"]))
        ).hexdigest(),
    }


class ExperimentV2ModelDataset(AbstractDataset[dict[str, Any], dict[str, Any]]):
    """Publish verified immutable payload and metadata bundles within one private run."""

    def __init__(
        self, *, filepath: str, run_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Bind the bundle and staging directory to one isolated catalog path.

        Args:
            filepath: Private bundle directory.
            run_id: Dedicated run identity.
            metadata: Optional Kedro metadata.
        """
        self._boundary = ExperimentV2JSONDataset(
            filepath=f"{filepath}/_isolation.json", run_id=run_id
        )
        self._project_root = self._boundary._project_root
        self._path = self._boundary._configured_path.parent
        self._staging = self._path.with_name(f".{self._path.name}.staging")
        self._completion = ExperimentV2JSONDataset(
            filepath=str(self._staging / "complete.json"), run_id=run_id
        )
        self._run_root = self._project_root / "data" / "experiment_v2" / run_id
        self.metadata = metadata

    def _verify(self, root: Path) -> dict[str, Any]:
        """Verify every payload byte and transactional metadata before exposing a bundle.

        Args:
            root: Final or interrupted staging directory.

        Returns:
            Payload descriptor and its authenticated metadata projection.

        Raises:
            DatasetError: If completion, payload, or metadata is missing or corrupt.
        """
        guarded_run_path(root, project_root=self._project_root, run_id=self._run_root.name)
        if any(item.is_symlink() for item in root.rglob("*")):
            msg = "v2 bundle cannot contain a symlink"
            raise DatasetError(msg)
        try:
            marker = json.loads((root / "complete.json").read_text())
            metadata = json.loads((root / "metadata.json").read_text())
            descriptor = checkpoint_descriptor(root / "payload")
            expected = {
                "checkpoint_sha256": descriptor["checkpoint_sha256"],
                "tokenizer_sha256": descriptor["tokenizer_sha256"],
                "metadata_sha256": hashlib.sha256(_canonical_bytes(metadata)).hexdigest(),
            }
            if marker != expected or {item.name for item in root.iterdir()} != {
                "payload",
                "metadata.json",
                "complete.json",
            }:
                msg = "v2 bundle identity is corrupt"
                raise DatasetError(msg)
        except (OSError, ValueError, TypeError) as error:
            msg = "v2 bundle is incomplete or corrupt"
            raise DatasetError(msg) from error
        return {**descriptor, "_metadata": metadata}

    def _recover(self) -> None:
        """Publish a verified completed staging directory left by an interrupted save.

        Raises:
            DatasetError: If staging is incomplete, corrupt, or redirected.
        """
        self._boundary._assert_no_symlink()
        guarded_run_path(self._staging, project_root=self._project_root, run_id=self._run_root.name)
        if self._staging.exists():
            self._verify(self._staging)
            if self._path.exists():
                msg = "v2 final and staging bundles both exist; inspect the run"
                raise DatasetError(msg)
            self._staging.rename(self._path)

    def load(self) -> dict[str, Any]:
        """Load a completed bundle, recovering verified staged publication if necessary.

        Returns:
            Exact saved payload identities and its non-text metadata.
        """
        self._recover()
        return self._verify(self._path)

    def save(self, data: dict[str, Any]) -> None:
        """Stage payload and metadata, seal their identities, then publish one bundle.

        Args:
            data: Existing model bundle or same-run copy descriptor, with _metadata.

        Raises:
            DatasetError: If immutable output exists or a copied payload changed.
        """
        if self.exists():
            msg = "v2 checkpoint outputs are immutable; choose a fresh run"
            raise DatasetError(msg)
        if not isinstance(data.get("_metadata"), dict) or not data["_metadata"]:
            message = "v2 checkpoint bundle requires nonempty training or preparation metadata"
            raise DatasetError(message)
        metadata_bytes = _canonical_bytes(data["_metadata"])
        if "copy_from" in data:
            descriptor = data["copy_from"]
            source = guarded_run_path(
                descriptor["path"], project_root=self._project_root, run_id=self._run_root.name
            )
            actual = checkpoint_descriptor(source)
            if any(actual[key] != descriptor[key] for key in actual):
                msg = "prepared checkpoint identity changed before copying"
                raise DatasetError(msg)
        self._staging.mkdir(parents=True, exist_ok=False)
        payload = self._staging / "payload"
        if "copy_from" in data:
            shutil.copytree(source, payload)
        else:
            SurrogateModelDataset(filepath=str(payload), metadata=self.metadata).save(data)
        descriptor = checkpoint_descriptor(payload)
        (self._staging / "metadata.json").write_bytes(metadata_bytes)
        marker = {key: descriptor[key] for key in ("checkpoint_sha256", "tokenizer_sha256")}
        marker["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
        # A single atomic marker authorizes recovery; payload hashing excludes this directory.
        self._completion._publish(marker, immutable=True)
        self._recover()

    def exists(self) -> bool:
        """Check verified completion, refusing partial directories as recoverable outputs.

        Returns:
            Whether a complete immutable bundle is present.
        """
        self._recover()
        if not self._path.exists():
            return False
        self._verify(self._path)
        return True

    def _describe(self) -> dict[str, str]:
        """Describe the isolated private bundle boundary.

        Returns:
            Local bundle path and exact run identity.
        """
        return {"filepath": str(self._path), "run_id": self._run_root.name}
