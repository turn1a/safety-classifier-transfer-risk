"""Round-trip tests for SurrogateModelDataset (BiLSTM path; the HF path needs a download).

The check is behavioural: after save -> load the reconstructed model must produce the same
logits on the same input (weights and config round-trip exactly), not merely "load runs".
"""

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


def test_bilstm_save_load_round_trip(tmp_path: Path) -> None:
    model = BiLSTMClassifier(len(_CONFIG["vocab"]), 8, 4, 1, 0.0).eval()
    dataset = SurrogateModelDataset(str(tmp_path / "bilstm-attention"))
    dataset.save({"kind": "bilstm", "model": model, "config": _CONFIG})
    assert dataset.exists()

    loaded = dataset.load()
    assert loaded["kind"] == "bilstm"
    assert loaded["config"] == _CONFIG
    ids = torch.tensor([[2, 3, 1]])
    with torch.no_grad():
        expected = model(ids)
        actual = loaded["model"](ids)
    assert torch.allclose(expected, actual)


def test_missing_checkpoint_does_not_exist(tmp_path: Path) -> None:
    assert not SurrogateModelDataset(str(tmp_path / "absent")).exists()
