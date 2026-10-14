"""Bind existing TextAttack recipes to catalog-verified local assets and disable downloads."""

from __future__ import annotations

import os
import sys
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, NoReturn

from transfer_risk.datasets.experiment_v2_attack_assets_dataset import (
    AttackAssetManifest,
    verify_asset_cache,
)
from transfer_risk.lib.experiment_v2.protocol import content_digest

OFFLINE_ENVIRONMENT = (
    "TA_CACHE_DIR",
    "TA_DEVICE",
    "TA_SENTENCE_ENCODER",
    "NLTK_DATA",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_ASSETS_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "TFHUB_CACHE_DIR",
)
_STATE: dict[str, Any] | None = None
_NLTK_RESOURCES = {
    "wordnet": "corpora/wordnet/",
    "omw-1.4": "corpora/omw-1.4/",
    "stopwords": "corpora/stopwords/",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger/",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
    "universal_tagset": "taggers/universal_tagset/",
    "punkt": "tokenizers/punkt/",
    "punkt_tab": "tokenizers/punkt_tab/",
}


class OfflineAssetError(RuntimeError):
    """Signal an absent, changed, or unregistered asset without permitting a download."""


def _no_download(*_args: Any, **_kwargs: Any) -> NoReturn:
    """Reject network-backed asset acquisition in a registered attack process.

    Args:
        *_args: Existing library download positional arguments.
        **_kwargs: Existing library download keyword arguments.

    Raises:
        OfflineAssetError: Always.
    """
    message = "registered TextAttack execution is offline; network asset downloads are forbidden"
    raise OfflineAssetError(message)


def _environment(paths: dict[str, str], device: str) -> dict[str, str]:
    """Resolve all asset/cache environment variables to the isolated local run.

    Args:
        paths: Exact dataset-owned auxiliary role and cache paths.
        device: Actual source/auxiliary device resolved by the existing device helper.

    Returns:
        The complete strict-offline environment contract.
    """
    return {
        "TA_CACHE_DIR": paths["textattack"],
        "TA_DEVICE": device,
        "TA_SENTENCE_ENCODER": "sentence-transformers",
        "NLTK_DATA": paths["nltk"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": paths["huggingface"],
        "HF_HUB_CACHE": f"{paths['huggingface']}/hub",
        "HF_ASSETS_CACHE": f"{paths['huggingface']}/assets",
        "PYTORCH_TRANSFORMERS_CACHE": paths["huggingface"],
        "XDG_CACHE_HOME": paths["textattack"],
        "TFHUB_CACHE_DIR": paths["textattack"],
    }


def _restrict_nltk(nltk: Any, root: str) -> None:
    """Use only registered NLP files, including cache-only PWWS download requests.

    Args:
        nltk: Existing NLTK package, imported after setting NLTK_DATA.
        root: Catalog-verified local NLP resource directory.
    """
    nltk.data.path[:] = [root]
    nltk.data._resource_cache.clear()
    mapping = getattr(getattr(nltk, "tag", None), "mapping", None)
    if mapping is not None:
        mapping._MAPPINGS.clear()
    for owner, name in (
        (getattr(nltk, "tag", None), "_get_tagger"),
        (getattr(nltk, "tokenize", None), "_get_punkt_tokenizer"),
    ):
        function = getattr(owner, name, None)
        if function is not None and hasattr(function, "cache_clear"):
            function.cache_clear()
    for name in ("wordnet", "omw", "omw14", "stopwords"):
        corpus = vars(nltk.corpus).get(name)
        unload = vars(corpus).get("_unload") if corpus is not None else None
        if callable(unload):
            unload()

    def cached_resource(info_or_id: str | list[str], *args: Any, **kwargs: Any) -> bool:
        """Accept only an already-present registered NLTK resource.

        Args:
            info_or_id: Existing recipe's package id or ids.
            *args: Unused legacy downloader positional arguments.
            **kwargs: Existing recipe's downloader keyword arguments.

        Returns:
            True after checking the registered local resource.

        Raises:
            OfflineAssetError: If the request would need additional NLP bytes.
        """
        if args or kwargs.get("download_dir") not in {None, root}:
            message = "NLTK offline resource requests cannot change the registered cache"
            raise OfflineAssetError(message)
        names = [info_or_id] if isinstance(info_or_id, str) else info_or_id
        if not isinstance(names, list) or any(name not in _NLTK_RESOURCES for name in names):
            message = "requested NLP resource is absent from the registered offline cache"
            raise OfflineAssetError(message)
        for name in names:
            try:
                nltk.data.find(_NLTK_RESOURCES[name], paths=[root])
            except LookupError as error:
                message = "requested NLP resource is absent from the registered offline cache"
                raise OfflineAssetError(message) from error
        return True

    nltk.download = cached_resource


def _restrict_textattack(paths: dict[str, str], manifest: AttackAssetManifest) -> None:
    """Keep existing algorithms and thresholds while binding every auxiliary load.

    Args:
        paths: Verified local cache and model directories.
        manifest: Registered byte inventory used to restrict cache requests.
    """
    install: Any = import_module("textattack.shared.utils.install")
    utils: Any = import_module("textattack.shared.utils")
    install.TEXTATTACK_CACHE_DIR = paths["textattack"]
    listed = {row.path for row in manifest.files}
    textattack_prefix = manifest.cache_paths["textattack"]

    def cached_path(name: str, skip_if_cached: bool = True) -> str:  # noqa: FBT001, FBT002
        """Return a registered existing TextAttack cache object without downloading.

        Args:
            name: Existing recipe's cache-relative path.
            skip_if_cached: Existing cache policy; forced acquisition is rejected.

        Returns:
            The already verified isolated asset path.

        Raises:
            OfflineAssetError: If the request is unregistered or absent.
        """
        relative = Path(name)
        inventory = f"{textattack_prefix}/{relative.as_posix()}"
        path = Path(paths["textattack"]) / relative
        if (
            not skip_if_cached
            or relative.is_absolute()
            or ".." in relative.parts
            or not name
            or not path.exists()
            or path.is_symlink()
            or not any(item == inventory or item.startswith(f"{inventory}/") for item in listed)
        ):
            message = "requested TextAttack asset is absent from its registered offline cache"
            raise OfflineAssetError(message)
        return str(path)

    def cached_url(_url: str, save_path: str, skip_if_cached: bool = True) -> str:  # noqa: FBT001, FBT002
        """Accept an existing cache path without using its download URL.

        Args:
            _url: Existing library URL, which is never requested.
            save_path: Registered cache-relative path.
            skip_if_cached: Existing cache policy.

        Returns:
            The verified existing path.
        """
        return cached_path(save_path, skip_if_cached)

    for module in (install, utils):
        module.download_from_s3 = cached_path
        module.download_from_url = cached_url
        module.http_get = _no_download
    utils.GLOBAL_OBJECTS.pop("textattack_counterfitted_GLOVE_embedding", None)
    bae: Any = import_module("textattack.attack_recipes.bae_garg_2019")
    bae.WordSwapMaskedLM = partial(bae.WordSwapMaskedLM, masked_language_model=paths["masked_lm"])
    encoders: Any = import_module("textattack.constraints.semantics.sentence_encoders")
    encoders.SentenceTransformerEncoder = partial(
        encoders.SentenceTransformerEncoder, model_name=paths["semantic_encoder"]
    )


def configure_offline_assets(descriptor: dict[str, Any] | None, device: str) -> None:
    """Verify assets before importing TextAttack and configure its exact local loaders.

    Each source shard rechecks all bytes. A process can use only one auxiliary
    identity and actual device, so library-global caches cannot cross runs.

    Args:
        descriptor: Catalog-owned verified manifest and realized paths.
        device: Actual device string supplied by the existing device resolver.

    Raises:
        OfflineAssetError: If the cache is unconfigured, stale, or already contaminated.
    """
    global _STATE  # noqa: PLW0603
    if not descriptor:
        message = "registered offline auxiliary assets must be configured before TextAttack"
        raise OfflineAssetError(message)
    manifest = AttackAssetManifest.model_validate(descriptor["manifest"])
    if content_digest(manifest) != descriptor.get("manifest_sha256"):
        message = "offline auxiliary manifest has a stale byte identity"
        raise OfflineAssetError(message)
    root = Path(descriptor["cache_root"])
    expected = {name: str(root / role.path) for name, role in manifest.roles.items()}
    expected.update({name: str(root / path) for name, path in manifest.cache_paths.items()})
    if descriptor.get("paths") != expected:
        message = "offline auxiliary paths differ from the registered manifest"
        raise OfflineAssetError(message)
    verify_asset_cache(root, manifest)
    environment = _environment(expected, device)
    identity = {
        "manifest_sha256": descriptor["manifest_sha256"],
        "cache_root": str(root),
        "environment": environment,
    }
    if _STATE is not None and identity != _STATE:
        message = "a registered offline process cannot switch auxiliary manifests or devices"
        raise OfflineAssetError(message)
    if _STATE is None and "textattack" in sys.modules:
        message = "TextAttack was imported before the registered offline cache was configured"
        raise OfflineAssetError(message)
    os.environ.update(environment)
    if _STATE is not None:
        return
    _restrict_nltk(import_module("nltk"), expected["nltk"])
    hub: Any = import_module("huggingface_hub.constants")
    hub.HF_HUB_OFFLINE = True
    hub.HF_HOME = expected["huggingface"]
    hub.HF_HUB_CACHE = environment["HF_HUB_CACHE"]
    hub.HF_ASSETS_CACHE = environment["HF_ASSETS_CACHE"]
    _restrict_textattack(expected, manifest)
    # TextAttack's install module overwrites HF_HOME; restore the dedicated cache.
    os.environ.update(environment)
    _STATE = identity


def require_offline_assets() -> None:
    """Reject unconfigured or subsequently altered download/cache settings.

    Raises:
        OfflineAssetError: If the actual strict-offline environment differs from its binding.
    """
    if _STATE is None or any(
        os.environ.get(key) != value for key, value in _STATE["environment"].items()
    ):
        message = "TextAttack recipes require a configured registered offline asset boundary"
        raise OfflineAssetError(message)
