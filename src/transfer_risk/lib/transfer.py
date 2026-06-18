"""Transfer-rate computation (SPEC.md §3.1 step 5).

Pure helper for the transfer stage. The frozen target classifies a surrogate's successful
adversarial examples; the transfer rate is the fraction it *also* labels benign, i.e. the
fraction of attacks that carry over from surrogate to target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def transfer_rate(target_predictions: Sequence[int], benign_label: int = 0) -> float:
    """Fraction of adversarial examples the frozen target also predicts as benign.

    Args:
        target_predictions: the target's predicted labels on the surrogate's successful
            adversarial examples (each example flipped injection -> benign on the surrogate).
        benign_label: the class id that means "not an injection"; a target prediction equal
            to it means the attack transferred.

    Returns:
        ``transferred / total`` in ``[0, 1]``; ``0.0`` for an empty input.
    """
    if not target_predictions:
        return 0.0
    transferred = sum(1 for prediction in target_predictions if prediction == benign_label)
    return transferred / len(target_predictions)
