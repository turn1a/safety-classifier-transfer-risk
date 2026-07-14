"""Pure deterministic sampling for training-only CKA sensitivity probes.

The helpers in this module operate only on an already-loaded training split. They preserve
every input column, including source provenance, while selecting a fixed canonical-label balance.
"""

from __future__ import annotations

import pandas as pd

from transfer_risk.lib.seeds import derive_seeds

_REQUIRED_COLUMNS = frozenset({"text", "label", "source"})


def sample_balanced_training_probe(
    train: pd.DataFrame,
    *,
    n_per_label: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic, balanced CKA probe from an existing training split.

    Args:
        train: Saved training split with canonical ``text``, ``label``, and ``source`` columns.
        n_per_label: Number of rows sampled for each canonical label (one and zero).
        seed: Root reproducibility seed.

    Returns:
        A shuffled copy containing exactly ``n_per_label`` rows for each canonical label and
        all original columns.

    Raises:
        ValueError: If required columns are absent, ``n_per_label`` is not positive, or either
            canonical class has too few training rows.
    """
    if n_per_label <= 0:
        msg = "n_per_label must be positive"
        raise ValueError(msg)
    missing = sorted(_REQUIRED_COLUMNS.difference(train.columns))
    if missing:
        msg = f"training split is missing required columns: {missing}"
        raise ValueError(msg)

    positives = train.loc[train["label"] == 1]
    negatives = train.loc[train["label"] == 0]
    if len(positives) < n_per_label or len(negatives) < n_per_label:
        msg = (
            "training split must contain at least "
            f"{n_per_label} rows for each canonical label; found "
            f"label 1={len(positives)}, label 0={len(negatives)}"
        )
        raise ValueError(msg)

    sampling_seed = derive_seeds(seed).numpy
    sampled = pd.concat(
        [
            positives.sample(n=n_per_label, random_state=sampling_seed),
            negatives.sample(n=n_per_label, random_state=sampling_seed),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=sampling_seed).reset_index(drop=True)
