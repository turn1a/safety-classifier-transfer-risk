"""Tests for transfer_risk.lib.transfer: the transfer-rate definition (SPEC.md §3.1 step 5)."""

from transfer_risk.lib.transfer import transfer_rate


def test_empty_predictions_is_zero() -> None:
    assert transfer_rate([]) == 0.0


def test_all_predicted_benign_is_one() -> None:
    assert transfer_rate([0, 0, 0]) == 1.0


def test_none_predicted_benign_is_zero() -> None:
    assert transfer_rate([1, 1, 1]) == 0.0


def test_half_predicted_benign_is_one_half() -> None:
    assert transfer_rate([0, 1, 0, 1]) == 0.5


def test_respects_a_non_default_benign_label() -> None:
    assert transfer_rate([2, 2, 1], benign_label=2) == 2 / 3
