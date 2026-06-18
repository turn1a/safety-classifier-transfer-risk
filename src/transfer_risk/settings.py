"""Kedro project settings for transfer_risk.

kedro-mlflow registers its tracking hooks automatically via plugin entry points (config
in ``conf/base/mlflow.yml``), so no explicit ``HOOKS`` entry is required. Console + file
logging is configured in ``conf/logging.yml`` (auto-discovered by Kedro).

``CONFIG_LOADER_ARGS`` registers two *scoped* OmegaConf resolvers, ``tr.bucket`` and
``tr.region``, so ``conf/cloud/globals.yml`` can build ``s3://`` roots from the box's
environment (``TR_BUCKET`` / ``TR_REGION``). Kedro 1.4 enables the built-in ``oc.env``
resolver only for credentials, and the docs discourage re-enabling it globally; two
single-purpose resolvers expose exactly the cloud bucket and region and nothing else. See
https://docs.kedro.org/en/stable/configure/advanced_configuration/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from transfer_risk.hooks import CloudpickleDatasetHook

# Register a cloudpickle reducer for ForkingPickler-unfriendly datasets so the attack sweep can
# run under ParallelRunner with the (unused-by-it) kedro-mlflow datasets present in the catalog.
HOOKS = (CloudpickleDatasetHook(),)

# Load .env before any node imports torch, so the Apple-Silicon / tokenizer flags
# (PYTORCH_ENABLE_MPS_FALLBACK, TOKENIZERS_PARALLELISM, ...) and HF_TOKEN take effect.
# Kedro imports settings during project bootstrap, ahead of pipeline execution. Existing
# shell variables win (override=False), so an exported value is never clobbered.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _require_env(name: str) -> str:
    """Return the environment variable ``name``, raising if it is unset or empty.

    Args:
        name: the environment variable to read.

    Returns:
        The variable's value.

    Raises:
        KeyError: if the variable is unset or empty (the cloud config cannot resolve).
    """
    value = os.environ.get(name)
    if not value:
        msg = f"environment variable {name!r} is required to resolve the cloud config but is unset"
        raise KeyError(msg)
    return value


def _resolve_bucket(*_args: object) -> str:
    """Resolve ``${tr.bucket:}`` to the cloud S3 bucket from ``TR_BUCKET`` (used in globals)."""
    return _require_env("TR_BUCKET")


def _resolve_region(*_args: object) -> str:
    """Resolve ``${tr.region:}`` to the AWS region from ``TR_REGION`` (used in globals)."""
    return _require_env("TR_REGION")


CONFIG_LOADER_ARGS = {
    # Kedro's default CONFIG_LOADER_ARGS carries these; since we override the dict to add
    # resolvers, we must restate them, or base_env/default_run_env fall back to "" and the loader
    # globs conf/** across every environment (raising on duplicate keys between base and thin).
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {
        "tr.bucket": _resolve_bucket,
        "tr.region": _resolve_region,
    },
}
