"""Round-trip tests for SurrogateModelDataset (BiLSTM path; the HF path needs a download).

The check is behavioural: after save -> load the reconstructed model must produce the same
logits on the same input (weights and config round-trip exactly), not merely "load runs".
``load`` now returns the local directory path, so the test reconstructs the model from it —
the same way the modeling layer does — and exercises both a local path and an ``s3://``-like
remote (``memory://``) that must be materialised to a local cache.
"""

import json
from pathlib import Path

import torch

from transfer_risk.datasets.surrogate_model_dataset import SurrogateModelDataset
from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier

_CONFIG = {
    "vocab": {"<pad>": 0, "<unk>": 1, "ignore": 2, "rules": 3},
    "embed_dim": 8,
    "hidden_dim": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "max_seq_len": 16,
}


def _reconstruct(path: str) -> BiLSTMClassifier:
    config = json.loads((Path(path) / "config.json").read_text())
    model = BiLSTMClassifier(
        len(config["vocab"]),
        config["embed_dim"],
        config["hidden_dim"],
        config["num_layers"],
        config["dropout"],
    )
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu"))
    return model.eval()


def test_bilstm_local_round_trip(tmp_path: Path) -> None:
    model = BiLSTMClassifier(len(_CONFIG["vocab"]), 8, 4, 1, 0.0).eval()
    dataset = SurrogateModelDataset(filepath=str(tmp_path / "bilstm-attention"))
    dataset.save({"kind": "bilstm", "model": model, "config": _CONFIG})
    assert dataset.exists()

    loaded_path = dataset.load()
    ids = torch.tensor([[2, 3, 1]])
    with torch.no_grad():
        expected = model(ids)
        actual = _reconstruct(loaded_path)(ids)
    assert torch.allclose(expected, actual)


def test_bilstm_remote_round_trip_materializes(tmp_path: Path) -> None:
    model = BiLSTMClassifier(len(_CONFIG["vocab"]), 8, 4, 1, 0.0).eval()
    dataset = SurrogateModelDataset(
        filepath="memory:///models/bilstm-remote", cache_dir=str(tmp_path / "cache")
    )
    dataset.save({"kind": "bilstm", "model": model, "config": _CONFIG})
    assert dataset.exists()

    loaded_path = dataset.load()
    assert str(tmp_path / "cache") in loaded_path  # materialised locally, not an s3:// URI
    ids = torch.tensor([[2, 3, 1]])
    with torch.no_grad():
        expected = model(ids)
        actual = _reconstruct(loaded_path)(ids)
    assert torch.allclose(expected, actual)


def test_missing_checkpoint_does_not_exist(tmp_path: Path) -> None:
    assert not SurrogateModelDataset(filepath=str(tmp_path / "absent")).exists()
