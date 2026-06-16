"""Download the offline NLP assets the attack recipes need.

Run via ``just setup-data`` (``python -m transfer_risk.scripts.fetch_assets``). Idempotent:
re-running re-checks and skips what is already cached. PWWS needs WordNet; the default
nltk POS path needs the perceptron tagger; TextFooler needs counter-fitted word
embeddings; the default semantic-similarity constraint uses a sentence-transformers model.
This replaces the silent download that used to live in the deleted subenv script.
"""

from __future__ import annotations

import logging

import nltk
from sentence_transformers import SentenceTransformer
from textattack.transformations import WordSwapEmbedding

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fetch_assets")

_NLTK_RESOURCES = (
    "wordnet",
    "omw-1.4",
    "stopwords",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "universal_tagset",
)
_SENTENCE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def fetch_nltk() -> None:
    """Download the NLTK corpora and taggers the attack recipes use."""
    for resource in _NLTK_RESOURCES:
        logger.info("nltk: %s", resource)
        nltk.download(resource, quiet=True)


def fetch_sentence_encoder() -> None:
    """Pre-download the default sentence-transformers semantic-similarity model."""
    logger.info("sentence-transformers: %s", _SENTENCE_ENCODER)
    SentenceTransformer(_SENTENCE_ENCODER)


def fetch_textattack_embeddings() -> None:
    """Pre-download TextAttack's counter-fitted word embeddings (used by TextFooler)."""
    logger.info("textattack: counter-fitted word embeddings")
    WordSwapEmbedding()


def main() -> None:
    """Fetch every offline asset the attack pipeline needs."""
    fetch_nltk()
    fetch_sentence_encoder()
    fetch_textattack_embeddings()
    logger.info("All attack assets are present.")


if __name__ == "__main__":
    main()
