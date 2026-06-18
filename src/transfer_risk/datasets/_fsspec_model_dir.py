"""Shared base for datasets that persist a model *directory* over fsspec.

torch / transformers / onnxruntime read and write local directories, not ``s3://`` URIs, so
these datasets follow the ``kedro_datasets`` ``TensorFlowModelDataset`` pattern: ``save``
serialises into a temporary directory then ``fs.put``s it to the (possibly remote) target,
and ``load`` materialises a remote directory into a stable local cache (race-safe for
``ParallelRunner``) and returns the *local path*, which each consumer reloads with its own
options. Per-format serialisation is provided by subclasses via :meth:`_write_dir`.
"""

from __future__ import annotations

import shutil
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import fsspec
from kedro.io.core import (
    AbstractVersionedDataset,
    DatasetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

from transfer_risk.lib.remote import model_cache_path

_LOCAL_PROTOCOLS = ("file", "local")


class FsspecModelDirDataset(AbstractVersionedDataset[Any, str]):
    """Persist a model directory to any fsspec filesystem; load it back as a local path.

    Subclasses implement :meth:`_write_dir` to serialise their bundle into a local directory;
    everything else (the fsspec transfer, the local-vs-remote handling, the race-safe cache)
    is shared. ``load`` returns a local directory path rather than a loaded object because each
    consumer reloads with its own options (hidden states for CKA, inference/ONNX for attacks)
    and because attack workers load the victim in a separate process from a path.
    """

    def __init__(
        self,
        *,
        filepath: str,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        version: Version | None = None,
    ) -> None:
        """Wire the fsspec filesystem for ``filepath`` (local or remote).

        Args:
            filepath: directory path, optionally prefixed with a protocol (e.g. ``s3://``).
            credentials: passed to ``fsspec.filesystem``; usually unset so the botocore default
                chain / EC2 instance profile is used.
            fs_args: extra ``fsspec.filesystem`` arguments.
            cache_dir: local root for materialised copies of a remote directory; defaults to a
                shared per-host temp directory so ``ParallelRunner`` workers reuse one copy.
            metadata: arbitrary metadata, ignored by Kedro.
            version: Kedro versioning (unused; accepted for catalog compatibility).
        """
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)
        self._protocol = protocol
        self._fs = fsspec.filesystem(protocol, **_credentials, **_fs_args)
        self._cache_dir = cache_dir
        self.metadata = metadata
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _write_dir(self, data: Any, dest: Path) -> None:
        """Serialise ``data`` into the local directory ``dest`` (implemented by subclasses).

        Args:
            data: the bundle to persist.
            dest: an existing local temporary directory to write into.
        """
        raise NotImplementedError

    def save(self, data: Any) -> None:
        """Serialise ``data`` to a temp directory and ``fs.put`` it to the target path."""
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with tempfile.TemporaryDirectory(prefix="tr_model_save_") as tmp:
            self._write_dir(data, Path(tmp))
            if self._fs.exists(save_path):
                self._fs.rm(save_path, recursive=True)
            self._fs.put(f"{tmp.rstrip('/')}/", f"{save_path.rstrip('/')}/", recursive=True)
        self._invalidate_cache()

    def load(self) -> str:
        """Return a local directory path: the target itself if local, else a cached copy."""
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        if self._protocol in _LOCAL_PROTOCOLS:
            return load_path
        return self._materialize(load_path)

    def _materialize(self, load_path: str) -> str:
        """Download a remote directory to a stable local cache once (race-safe); return its path.

        Args:
            load_path: the remote directory (e.g. ``s3://bucket/key``).

        Returns:
            The local cache directory path.

        Raises:
            OSError: if the download fails and no concurrent worker produced the cache.
        """
        cache_root = self._cache_dir or str(Path(tempfile.gettempdir()) / "transfer_risk_models")
        final = Path(model_cache_path(load_path, cache_root))
        if final.is_dir() and any(final.iterdir()):
            return str(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.parent / f".tmp-{uuid4().hex}-{final.name}"
        self._fs.get(f"{load_path.rstrip('/')}/", f"{staging}/", recursive=True)
        try:
            staging.replace(final)
        except OSError:
            # A concurrent worker materialised it first; reuse theirs and drop our staging copy.
            shutil.rmtree(staging, ignore_errors=True)
            if not (final.is_dir() and any(final.iterdir())):
                raise
        return str(final)

    def _exists(self) -> bool:
        """Whether the target directory exists and is non-empty."""
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False
        return bool(self._fs.exists(load_path)) and bool(self._fs.ls(load_path))

    def _describe(self) -> dict[str, Any]:
        """Return a printable description of this dataset."""
        return {"filepath": self._filepath, "protocol": self._protocol}

    def _release(self) -> None:
        """Invalidate the filesystem cache on release."""
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate the underlying filesystem cache for this path."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)
