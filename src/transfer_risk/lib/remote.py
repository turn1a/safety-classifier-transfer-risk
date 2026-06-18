"""Pure helpers for locating local copies of remote model directories.

torch / transformers / onnxruntime read local directories, not ``s3://`` URIs, so the custom
model-directory datasets download a remote checkpoint into a local cache before handing the
path to those libraries. The path arithmetic for that cache lives here (pure, no I/O) so it
can be unit-tested; the datasets perform the actual fsspec transfer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


def model_cache_path(remote_path: str, cache_root: str) -> str:
    """Return the deterministic local cache directory for a remote model directory.

    The name keeps the remote basename (for readability) and appends a short digest of the
    full remote path, so two remotes that share a basename do not collide in the cache.

    Args:
        remote_path: the remote model directory (e.g. ``s3://bucket/data/06_models/foo``).
        cache_root: the local root under which materialised copies live.

    Returns:
        The local cache directory path, as a string.
    """
    stripped = remote_path.rstrip("/")
    name = PurePosixPath(stripped).name or "model"
    digest = hashlib.blake2b(stripped.encode(), digest_size=4).hexdigest()
    return str(Path(cache_root) / f"{name}-{digest}")
