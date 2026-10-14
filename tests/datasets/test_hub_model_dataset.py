"""Hub model and tokenizer loads use the same pinned revision without network access."""

import sys
from types import SimpleNamespace

import pytest

from transfer_risk.datasets.hub_model_dataset import HuggingFaceHubModelDataset


@pytest.mark.parametrize(
    "load_args,expected_shared",
    [
        (
            {"revision": "a" * 40, "local_files_only": True, "num_labels": 2},
            {"revision": "a" * 40, "local_files_only": True},
        ),
        ({}, {}),
    ],
)
def test_model_and_tokenizer_share_revision_and_offline_options(
    monkeypatch, load_args, expected_shared
):
    calls = []
    model_bundle = object()
    tokenizer_bundle = object()

    def load_model(repo, **kwargs):
        calls.append(("model", repo, kwargs))
        return model_bundle

    def load_tokenizer(repo, **kwargs):
        calls.append(("tokenizer", repo, kwargs))
        if kwargs != {"token": None, **expected_shared}:
            msg = "tokenizer did not receive identical immutable load options"
            raise AssertionError(msg)
        return tokenizer_bundle

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        ),
    )
    dataset = HuggingFaceHubModelDataset(repo_id="synthetic/detector", load_args=load_args)
    bundle = dataset.load()
    assert bundle["model"] is model_bundle
    assert bundle["tokenizer"] is tokenizer_bundle
    assert calls[0][1] == calls[1][1] == "synthetic/detector"
    assert all(calls[0][2].get(key) == value for key, value in expected_shared.items())
    assert "num_labels" not in calls[1][2]
