import pytest
from kedro.io.core import DatasetError

from transfer_risk.datasets.semantic_model_dataset import PinnedSemanticModelDataset


@pytest.mark.parametrize("revision", ["main", "v1", "", "abc123"])
def test_semantic_models_require_immutable_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        PinnedSemanticModelDataset(repo_id="test/model", revision=revision, role="encoder")


def test_semantic_models_reject_arbitrary_model_classes() -> None:
    with pytest.raises(ValueError, match="role"):
        PinnedSemanticModelDataset(repo_id="test/model", revision="a" * 40, role="remote")


def test_semantic_model_dataset_is_read_only() -> None:
    dataset = PinnedSemanticModelDataset(repo_id="test/model", revision="a" * 40, role="nli")
    with pytest.raises(DatasetError, match="read-only"):
        dataset.save({})
    assert dataset._describe()["revision"] == "a" * 40
