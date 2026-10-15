"""Optional finalized public evidence retains a distinct absent state."""

import pytest
from kedro.io import DataCatalog, DatasetError


def test_catalog_optional_review_lifecycle_refuses_malformed_content(tmp_path) -> None:
    """Catalog loads absent evidence as None and rejects corrupt or nonobject JSON."""
    path = tmp_path / "summary.json"
    catalog = DataCatalog.from_config(
        {
            "pub_review": {
                "type": "transfer_risk.datasets.optional_review_dataset.OptionalReviewDataset",
                "filepath": str(path),
            }
        }
    )
    assert catalog.load("pub_review") is None
    catalog.save("pub_review", {"review_status": "complete"})
    assert catalog.load("pub_review") == {"review_status": "complete"}
    for content in ("{broken", "null", "[]"):
        path.write_text(content)
        with pytest.raises(DatasetError):
            catalog.load("pub_review")
