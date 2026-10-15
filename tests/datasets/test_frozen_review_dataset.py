"""Immutable private cohort storage refuses accidental preparation replacement."""

import pytest

from transfer_risk.datasets.frozen_review_dataset import FrozenReviewDataset


def test_frozen_review_save_is_idempotent_but_refuses_different_cohort(tmp_path) -> None:

    dataset = FrozenReviewDataset(filepath=str(tmp_path / "cohort.json"))
    dataset.save({"cohort": "one"})
    assert dataset.load() == {"cohort": "one"}
    dataset.save({"cohort": "one"})
    before = (tmp_path / "cohort.json").read_bytes()
    with pytest.raises(Exception, match="frozen"):
        dataset.save({"cohort": "two"})
    assert (tmp_path / "cohort.json").read_bytes() == before
