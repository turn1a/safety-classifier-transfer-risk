"""Transfer-rate computation (SPEC.md §3.1 step 5).

Pure helper for the transfer stage. The frozen target classifies a surrogate's successful
adversarial examples on perturbed text only; the returned rate is the conditional fraction
the target labels benign given a successful source attack — not a true target flip, because
original-prompt target predictions were not stored in v1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def transfer_rate(target_predictions: Sequence[int], benign_label: int = 0) -> float:
    """Conditional target-benign rate on perturbed prompts after successful source attacks.

    Args:
        target_predictions: the target's predicted labels on the surrogate's successful
            adversarial examples (each example flipped injection -> benign on the surrogate).
        benign_label: the class id that means "not an injection"; a target prediction equal
            to it counts toward the conditional benign rate.

    Returns:
        ``benign_on_perturbed / total`` in ``[0, 1]``; ``0.0`` for an empty input. This is
        not a true target flip rate because original-prompt target predictions are ignored.
    """
    if not target_predictions:
        return 0.0
    transferred = sum(1 for prediction in target_predictions if prediction == benign_label)
    return transferred / len(target_predictions)
