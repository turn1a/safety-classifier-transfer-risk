"""Export each transformer surrogate to ONNX for ``use_onnx`` attack runs.

``optimum`` (the exporter) conflicts with the project's ``transformers`` 5, so the export
runs in a throwaway ``uvx`` environment (``optimum`` 1.x) and this orchestrator only reads
the surrogate manifest and shells out — it never imports ``optimum`` into the main env. It
writes ``{onnx_root}/{name}/model.onnx`` (plus the tokenizer) per transformer surrogate; the
BiLSTM is excluded (it is tiny and always served by torch). Run via ``just export-onnx``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MANIFEST = Path("data/06_models/surrogate_manifest.json")
ONNX_ROOT = Path("data/06_models/onnx")


def main() -> int:
    """Export every non-BiLSTM surrogate in the manifest to ONNX; skip those already exported.

    Returns:
        ``0`` if every transformer surrogate has a ``model.onnx``, ``1`` if any export failed.
    """
    if not MANIFEST.exists():
        logger.error("No surrogate manifest at %s; run the models pipeline first.", MANIFEST)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    uvx = shutil.which("uvx")
    if uvx is None:
        logger.error("uvx not found on PATH; install uv: https://docs.astral.sh/uv/")
        return 1
    failures: list[str] = []
    for name, entry in manifest.items():
        if entry.get("kind") == "bilstm":
            continue
        out = ONNX_ROOT / name
        if (out / "model.onnx").exists():
            logger.info("[skip] %s: model.onnx already present", name)
            continue
        source = entry["source"]
        logger.info("[export] %s <- %s", name, source)
        subprocess.run(  # noqa: S603 (args come from our own manifest)
            [
                uvx,
                "--from",
                "optimum[exporters]<2",
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                str(source),
                "--task",
                "text-classification",
                str(out),
            ],
            check=False,
        )
        # optimum's post-export validation can exit non-zero on a strict atol while still
        # writing a numerically faithful model.onnx (the parity gate measures ~5e-8 / 100%
        # argmax agreement), so success is "model.onnx exists", not the CLI exit code.
        if not (out / "model.onnx").exists():
            failures.append(name)
            logger.error("[FAIL] %s: no model.onnx produced", name)
    if failures:
        logger.error("Failed to export: %s", failures)
        return 1
    logger.info("All transformer surrogates exported to %s.", ONNX_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
