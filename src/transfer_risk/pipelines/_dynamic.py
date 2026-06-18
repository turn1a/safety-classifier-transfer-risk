"""Build-time configuration for the dynamically generated pipelines.

The models, similarity, and attacks pipelines create one node per surrogate (and per attack
shard), so ``create_pipeline`` needs the surrogate specs, recipes, and shard size while the
pipeline is being built. Kedro does not inject ``params:`` at that point, so this module reads
them directly with ``OmegaConfigLoader``. The structure environment comes from ``KEDRO_ENV``
(the only env that changes the *structure* is ``thin``; ``cloud`` changes only paths, so it
shares the base structure), and the result is cached so the three pipelines read config once.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from kedro.config import OmegaConfigLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONF_SOURCE = str(_PROJECT_ROOT / "conf")


@lru_cache(maxsize=1)
def _structure() -> dict[str, Any]:
    """Load the surrogate specs and attack parameters for the build-time structure env.

    Returns:
        ``{"surrogates": [...], "attacks": {...}}`` for the ``KEDRO_ENV`` environment (base
        otherwise).
    """
    env = os.environ.get("KEDRO_ENV") or "local"
    loader = OmegaConfigLoader(
        conf_source=_CONF_SOURCE, env=env, base_env="base", default_run_env="local"
    )
    params = loader["parameters"]
    return {"surrogates": list(params["models"]["surrogates"]), "attacks": dict(params["attacks"])}


def surrogate_specs() -> list[dict[str, Any]]:
    """Return the surrogate specs (``{name, kind, gated?}``) for the build-time structure env."""
    return [dict(spec) for spec in _structure()["surrogates"]]


def attack_params() -> dict[str, Any]:
    """Return the attack parameters (recipes, eval_set_size, shard_size, ...) for the build env."""
    return dict(_structure()["attacks"])
