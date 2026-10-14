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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


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


class ExperimentV2ModelDataset(AbstractDataset[dict[str, Any], dict[str, str]]):
    """Persist models under one local v2 run and return verified private descriptors."""

    def __init__(
        self, *, filepath: str, run_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Bind a checkpoint to the exact private run root.

        Args:
            filepath: Relative or absolute checkpoint directory in this project's private run.
            run_id: Dedicated run identity.
            metadata: Optional Kedro metadata.
        """
        self._boundary = ExperimentV2JSONDataset(
            filepath=f"{filepath}/_isolation.json", run_id=run_id
        )
        self._project_root = self._boundary._project_root
        self._path = self._boundary._configured_path.parent
        self._run_root = self._project_root / "data" / "experiment_v2" / run_id
        self._dataset = SurrogateModelDataset(filepath=str(self._path), metadata=metadata)
        self.metadata = metadata

    def load(self) -> dict[str, str]:
        """Return fresh fingerprints after rechecking private path isolation.

        Returns:
            A descriptor of the exact saved checkpoint and tokenizer.
        """
        self._boundary._assert_no_symlink()
        return checkpoint_descriptor(self._path)

    def save(self, data: dict[str, Any]) -> None:
        """Save a new checkpoint or copy an independently prepared frozen target.

        Args:
            data: Normal model bundle, or copy_from with a private saved descriptor.

        Raises:
            DatasetError: If a checkpoint exists or a copied input escapes this run.
        """
        self._boundary._assert_no_symlink()
        if self.exists():
            msg = "v2 checkpoint outputs are immutable; choose a fresh run or resume missing nodes"
            raise DatasetError(msg)
        if "copy_from" not in data:
            self._dataset.save(data)
            return
        descriptor = data["copy_from"]
        source = guarded_run_path(
            descriptor["path"], project_root=self._project_root, run_id=self._run_root.name
        )
        if not source.is_relative_to(self._run_root):
            msg = "copied v2 checkpoint must belong to the same isolated private run"
            raise DatasetError(msg)
        if checkpoint_descriptor(source) != descriptor:
            msg = "prepared checkpoint identity changed before copying"
            raise DatasetError(msg)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, self._path)

    def exists(self) -> bool:
        """Check whether immutable private checkpoint files already exist.

        Returns:
            Whether the checkpoint directory contains any artifacts.
        """
        self._boundary._assert_no_symlink()
        return self._path.is_dir() and any(self._path.iterdir())

    def _describe(self) -> dict[str, str]:
        """Describe the isolated private checkpoint boundary.

        Returns:
            Its local filepath and exact run identity.
        """
        return {"filepath": str(self._path), "run_id": self._run_root.name}
