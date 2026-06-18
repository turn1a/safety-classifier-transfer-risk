"""Load trained classifiers and extract per-layer representations.

Shared glue (imports torch + transformers) used by the similarity stage (and later
the attack/transfer stages). Transformers expose every layer via
``output_hidden_states``; the BiLSTM exposes its two pooled layers via
``representations``. A model "spec" is a manifest entry (or the target) with ``kind``
in {pretrained, finetune, bilstm} and a ``source`` (a HuggingFace id or a local dir).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transfer_risk.pipelines.models.bilstm import BiLSTMClassifier, encode, pad_batch

if TYPE_CHECKING:
    from collections.abc import Mapping

FloatArray = npt.NDArray[np.float64]


def layer_representations(
    spec: Mapping[str, Any],
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Return one pooled ``(n_texts, d)`` matrix per layer for the model in ``spec``."""
    if spec["kind"] == "bilstm":
        return _bilstm_representations(spec["source"], texts, batch_size=batch_size, device=device)
    return _transformer_representations(
        spec["source"],
        texts,
        pooling=pooling,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )


def _pool(layer: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool a ``(batch, seq, hidden)`` layer to ``(batch, hidden)``."""
    if pooling == "cls":
        return layer[:, 0, :]
    expanded = mask.unsqueeze(-1).to(layer.dtype)
    summed = (layer * expanded).sum(dim=1)
    return summed / expanded.sum(dim=1).clamp(min=1.0)


def representations_from_loaded(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Per-layer pooled reps from an already-loaded transformer bundle (e.g. ``target_model``).

    The model must have been loaded with ``output_hidden_states=True``. Used for the frozen
    target, which arrives as a loaded catalog bundle rather than a checkpoint path.
    """
    return _transformer_reps(
        model.to(device).eval(),
        tokenizer,
        texts,
        pooling=pooling,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )


def _transformer_representations(
    source: str,
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Return one pooled ``(n_texts, d)`` matrix per transformer layer (embeddings skipped)."""
    tokenizer = AutoTokenizer.from_pretrained(source)
    # fp32 regardless of checkpoint dtype: some backbones (deberta-v3) ship fp16 weights, and
    # fp16 ops are slow/unimplemented on CPU. fp32 also keeps activations comparable across the
    # pool for CKA.
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            source, output_hidden_states=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    return _transformer_reps(
        model,
        tokenizer,
        texts,
        pooling=pooling,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
    )


def _transformer_reps(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    pooling: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Compute per-layer pooled reps from a loaded transformer (the shared inner loop)."""
    per_layer: list[list[FloatArray]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden_states = model(**encoded).hidden_states
            mask = encoded["attention_mask"]
            # Skip hidden_states[0] (the static embedding layer): under CLS pooling its
            # [CLS] vector is identical for every input, so its CKA would be undefined.
            for index, layer in enumerate(hidden_states[1:]):
                pooled: FloatArray = _pool(layer, mask, pooling).cpu().numpy().astype(np.float64)
                if index >= len(per_layer):
                    per_layer.append([])
                per_layer[index].append(pooled)
    return [np.concatenate(chunks, axis=0).astype(np.float64) for chunks in per_layer]


def _bilstm_representations(
    source: str,
    texts: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> list[FloatArray]:
    """Return the BiLSTM's two pooled layer matrices (mean embedding, mean LSTM output)."""
    config = json.loads((Path(source) / "config.json").read_text())
    vocab = config["vocab"]
    model = BiLSTMClassifier(
        len(vocab),
        config["embed_dim"],
        config["hidden_dim"],
        config["num_layers"],
        config["dropout"],
    )
    model.load_state_dict(torch.load(Path(source) / "model.pt", map_location=device))
    model.to(device).eval()
    max_len = config["max_seq_len"]
    per_layer: list[list[FloatArray]] = [[], []]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids = pad_batch(
                [encode(text, vocab, max_len) for text in texts[start : start + batch_size]]
            ).to(device)
            for index, rep in enumerate(model.representations(ids)):
                per_layer[index].append(rep.cpu().numpy().astype(np.float64))
    return [np.concatenate(chunks, axis=0).astype(np.float64) for chunks in per_layer]


def load_transformer(source: str, device: torch.device) -> tuple[Any, Any]:
    """Load a transformer classifier + tokenizer for inference (no hidden states)."""
    tokenizer = AutoTokenizer.from_pretrained(source)
    # fp32: fp16 checkpoints (deberta-v3) are slow/unsupported for CPU inference, and the
    # attack sweep runs surrogates on CPU.
    model = (
        AutoModelForSequenceClassification.from_pretrained(source, dtype=torch.float32)
        .to(device)
        .eval()
    )
    return model, tokenizer


def predict(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    """Predict class labels for a list of texts with a loaded transformer classifier."""
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
    return predictions


class _LogitsOnly(torch.nn.Module):
    """Wrap a classifier so its forward returns the logits tensor, not the HF output object.

    ``torch.onnx.export`` traces tensor outputs cleanly; the HuggingFace ``ModelOutput`` object
    is awkward to trace, so this thin wrapper exposes ``input_ids`` / ``attention_mask`` /
    ``token_type_ids`` and returns ``.logits`` directly.
    """

    def __init__(self, model: Any) -> None:
        """Store the wrapped classifier."""
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: Any, attention_mask: Any = None, token_type_ids: Any = None
    ) -> Any:
        """Return the classifier's logits for the given encoded inputs."""
        kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs).logits


def export_to_onnx(model: Any, tokenizer: Any, *, max_seq_len: int = 256, opset: int = 17) -> bytes:
    """Export a transformer classifier to an ONNX graph in-process and return its bytes.

    Uses ``torch.onnx.export`` (no ``optimum``, so it runs in the main environment) with
    batch- and sequence-dynamic axes, so ONNX Runtime serves variable-length batches. The graph
    is written to an in-memory buffer (no filesystem I/O), and the caller persists it through the
    ``onnx.{name}`` catalog dataset. The result is validated against the torch checkpoint by the
    parity gate before the faster ONNX victim path is trusted.

    Args:
        model: the trained classifier to export.
        tokenizer: its tokenizer (used to build a representative sample input).
        max_seq_len: sequence length of the sample input traced during export.
        opset: ONNX opset version.

    Returns:
        The serialised ONNX graph.
    """
    import io  # noqa: PLC0415 (stdlib, kept local to the only function that needs it)

    wrapper = _LogitsOnly(model.to("cpu").eval())
    sample = tokenizer(
        "prompt injection export probe",
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
    )
    candidate_inputs = ("input_ids", "attention_mask", "token_type_ids")
    input_names = [name for name in candidate_inputs if name in sample]
    args = tuple(sample[name] for name in input_names)
    dynamic_axes: dict[str, dict[int, str]] = {
        name: {0: "batch", 1: "sequence"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            buffer,  # type: ignore[arg-type]  # torch stub narrows f to str/PathLike; BytesIO works
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    return buffer.getvalue()
