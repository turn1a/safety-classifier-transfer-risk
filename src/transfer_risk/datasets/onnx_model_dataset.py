"""A Kedro dataset for a surrogate's exported ONNX graph (plus its tokenizer).

The attack sweep serves transformer victims from an ONNX Runtime session, which is faster on
CPU than the torch checkpoint. The graph is produced in-pipeline by the ONNX export node
(``torch.onnx.export`` to an in-memory buffer) and persisted here as the ``onnx.{name}``
factory, so it travels to and from S3 through the catalog like every other artifact. ``load``
returns the local directory path (``model.onnx`` plus the tokenizer files), which
:class:`~transfer_risk.pipelines.attacks.runner.ONNXModelWrapper` opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transfer_risk.datasets._fsspec_model_dir import FsspecModelDirDataset

if TYPE_CHECKING:
    from pathlib import Path

_GRAPH = "model.onnx"


class OnnxModelDataset(FsspecModelDirDataset):
    """Persist an exported ONNX graph and its tokenizer to a directory.

    The saved value is ``{"onnx_bytes": <bytes>, "tokenizer": <tokenizer>}``: the serialised
    ONNX graph (so the export node performs no filesystem I/O) and the matching tokenizer.
    ``load`` returns the local directory path containing ``model.onnx`` and the tokenizer files.
    """

    def _write_dir(self, data: Any, dest: Path) -> None:
        """Write ``model.onnx`` from the serialised graph and save the tokenizer alongside it."""
        (dest / _GRAPH).write_bytes(data["onnx_bytes"])
        data["tokenizer"].save_pretrained(dest)
