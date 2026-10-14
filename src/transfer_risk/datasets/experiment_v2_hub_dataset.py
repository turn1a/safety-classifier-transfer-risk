"""Load only immutable v2 backbone and tokenizer revisions through the catalog."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError

from transfer_risk.datasets.hub_model_dataset import HuggingFaceHubModelDataset


def load_pinned_tokenizer(repository: str, revision: str, load_args: dict[str, Any]) -> Any:
    """Load an independently pinned tokenizer at this catalog boundary.

    Args:
        repository: Tokenizer repository identity.
        revision: Complete immutable upstream commit.
        load_args: Local cache and offline settings.

    Returns:
        The tokenizer from that exact revision.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(repository, revision=revision, **load_args)


def tokenizer_commit(
    tokenizer: Any, repository: str, revision: str, load_args: dict[str, Any]
) -> str | None:
    """Resolve tokenizer provenance from actual loader state and cached snapshot paths.

    Transformers can consume _commit_hash while constructing a tokenizer. Resolved
    vocabulary/tokenizer paths retain the immutable Hub snapshot commit. When neither
    field is present, the pinned tokenizer configuration supplies the loader's commit.

    Args:
        tokenizer: The actually loaded tokenizer.
        repository: Pinned tokenizer repository.
        revision: Required immutable tokenizer commit.
        load_args: Shared offline and cache settings.

    Returns:
        One consistent observed commit, or no verifiable identity.
    """
    observed = set()
    direct = tokenizer.init_kwargs.get("_commit_hash")
    if isinstance(direct, str):
        observed.add(direct)
    for value in tokenizer.init_kwargs.values():
        if not isinstance(value, str):
            continue
        parts = Path(value).parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots" and re.fullmatch("[0-9a-f]{40}", parts[index + 1]):
                observed.add(parts[index + 1])
    if observed:
        return next(iter(observed)) if len(observed) == 1 else None
    from transformers.models.auto.tokenization_auto import get_tokenizer_config  # noqa: PLC0415

    config = get_tokenizer_config(repository, revision=revision, **load_args)
    commit = config.get("_commit_hash")
    return commit if isinstance(commit, str) else None


class ExperimentV2HubModelDataset(AbstractDataset[Any, dict[str, Any]]):
    """Return a backbone with independently checked immutable input identities."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None,
        tokenizer_repo_id: str | None = None,
        tokenizer_revision: str | None = None,
        load_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store prospective input pins without loading or downloading them.

        Args:
            repo_id: Backbone or frozen historical detector repository.
            revision: Required complete immutable upstream model commit.
            tokenizer_repo_id: Separate tokenizer repository, or the model repository.
            tokenizer_revision: Tokenizer commit, or the model commit.
            load_args: Model loading settings and offline/cache options.
            metadata: Optional Kedro metadata.
        """
        self._repo_id = repo_id
        self._revision = revision
        self._tokenizer_repo = tokenizer_repo_id or repo_id
        self._tokenizer_revision = tokenizer_revision or revision
        self._load_args = dict(load_args or {})
        self.metadata = metadata

    def load(self) -> dict[str, Any]:
        """Load pinned inputs and verify the commits reported by their actual loaders.

        Returns:
            Existing transformer bundle plus unmodified immutable input identities.

        Raises:
            DatasetError: If immutable pins are missing or actual commits differ.
        """
        pins = [self._revision, self._tokenizer_revision]
        if any(not isinstance(pin, str) or not re.fullmatch("[0-9a-f]{40}", pin) for pin in pins):
            msg = "v2 upstream model and tokenizer require complete immutable commit revisions"
            raise DatasetError(msg)
        identity = {
            "repository": self._repo_id,
            "revision": self._revision,
            "tokenizer_repository": self._tokenizer_repo,
            "tokenizer_revision": self._tokenizer_revision,
        }
        dataset = HuggingFaceHubModelDataset(
            repo_id=self._repo_id, load_args={**self._load_args, "revision": self._revision}
        )
        bundle = dataset.load()
        shared = {
            key: value
            for key, value in self._load_args.items()
            if key in {"local_files_only", "cache_dir", "subfolder", "force_download"}
        }
        if (self._tokenizer_repo, self._tokenizer_revision) != (self._repo_id, self._revision):
            bundle["tokenizer"] = load_pinned_tokenizer(
                self._tokenizer_repo, str(self._tokenizer_revision), shared
            )
        observed_model = getattr(bundle["model"].config, "_commit_hash", None)
        observed_tokenizer = tokenizer_commit(
            bundle["tokenizer"], self._tokenizer_repo, str(self._tokenizer_revision), shared
        )
        if observed_model != self._revision or observed_tokenizer != self._tokenizer_revision:
            msg = "loaded v2 model/tokenizer identity differs from immutable input pins"
            raise DatasetError(msg)
        return {**bundle, "input_identity": identity}

    def save(self, data: Any) -> None:  # noqa: ARG002
        """Reject writes to an upstream input model.

        Args:
            data: Unsupported write value.

        Raises:
            DatasetError: Always, because this catalog dataset is read-only.
        """
        msg = "v2 upstream model input is read-only"
        raise DatasetError(msg)

    def _describe(self) -> dict[str, Any]:
        """Describe immutable input pins without accessing their repositories.

        Returns:
            Model and tokenizer repository identities and prospective commits.
        """
        return {
            "repo_id": self._repo_id,
            "revision": self._revision,
            "tokenizer_repo_id": self._tokenizer_repo,
            "tokenizer_revision": self._tokenizer_revision,
        }
