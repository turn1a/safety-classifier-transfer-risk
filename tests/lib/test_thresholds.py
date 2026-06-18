"""Threshold calibration ordering (SPEC.md §3.1 step 3, §12)."""

from __future__ import annotations

import pytest

from transfer_risk.lib.thresholds import calibrate


def test_thresholds_are_ordered_in_unit_interval() -> None:
    thresholds = calibrate([0.1, 0.3, 0.5, 0.7, 0.9])
    assert 0.0 < thresholds.r2 < thresholds.r1 < 1.0


def test_empty_distribution_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate([])


def test_degenerate_distribution_raises() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        calibrate([0.5, 0.5, 0.5, 0.5])
