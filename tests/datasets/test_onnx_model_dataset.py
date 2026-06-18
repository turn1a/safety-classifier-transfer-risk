"""Round-trip tests for OnnxModelDataset using a stub tokenizer (no network).

The dataset's job is directory persistence: ``model.onnx`` bytes plus the tokenizer files must
survive save -> load on both a local path and an ``s3://``-like remote (``memory://``). A stub
tokenizer with a ``save_pretrained`` keeps the test offline.
"""

from pathlib import Path

from transfer_risk.datasets.onnx_model_dataset import OnnxModelDataset

_GRAPH = b"onnx-graph-bytes-\x00\x01\x02"


class _StubTokenizer:
    """Minimal stand-in that writes a marker file, mimicking ``save_pretrained``."""

    def save_pretrained(self, dest: str) -> None:
        """Write a tokenizer marker file into ``dest``."""
        (Path(dest) / "tokenizer.json").write_text('{"stub": true}')


def test_onnx_local_round_trip(tmp_path: Path) -> None:
    dataset = OnnxModelDataset(filepath=str(tmp_path / "surrogate"))
    dataset.save({"onnx_bytes": _GRAPH, "tokenizer": _StubTokenizer()})
    assert dataset.exists()

    path = Path(dataset.load())
    assert (path / "model.onnx").read_bytes() == _GRAPH
    assert (path / "tokenizer.json").exists()


def test_onnx_remote_round_trip_materializes(tmp_path: Path) -> None:
    dataset = OnnxModelDataset(
        filepath="memory:///onnx/surrogate-remote", cache_dir=str(tmp_path / "cache")
    )
    dataset.save({"onnx_bytes": _GRAPH, "tokenizer": _StubTokenizer()})
    assert dataset.exists()

    path = Path(dataset.load())
    assert str(tmp_path / "cache") in str(path)
    assert (path / "model.onnx").read_bytes() == _GRAPH
    assert (path / "tokenizer.json").exists()
