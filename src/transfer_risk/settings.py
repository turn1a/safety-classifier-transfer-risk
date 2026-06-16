"""Kedro project settings for transfer_risk.

kedro-mlflow registers its tracking hooks automatically via plugin entry points (config
in ``conf/base/mlflow.yml``), so no explicit ``HOOKS`` entry is required. Console + file
logging is configured in ``conf/logging.yml`` (auto-discovered by Kedro). See the Kedro
docs to customise the config loader, session store, or hooks as the project grows:
https://docs.kedro.org/en/stable/configure/configuration_basics/
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env before any node imports torch, so the Apple-Silicon / tokenizer flags
# (PYTORCH_ENABLE_MPS_FALLBACK, TOKENIZERS_PARALLELISM, ...) and HF_TOKEN take effect.
# Kedro imports settings during project bootstrap, ahead of pipeline execution. Existing
# shell variables win (override=False), so an exported value is never clobbered.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
