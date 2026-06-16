"""Surrogate registry: resolve and validate the configured model pool.

Pure validation lives here; the HF-auth precheck for gated models is the only piece
that touches the network. Adding a surrogate is a single config entry (SPEC.md §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from huggingface_hub import whoami

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

VALID_KINDS = frozenset({"pretrained", "finetune", "bilstm"})


def validate_surrogate_specs(specs: Sequence[Mapping[str, Any]]) -> None:
    """Validate surrogate specs: unique names, known kinds, and ids where required.

    Raises:
        ValueError: On a duplicate name, an unknown kind, or a transformer kind
            (``pretrained`` / ``finetune``) missing its ``id``.
    """
    names = [spec.get("name") for spec in specs]
    if len(names) != len(set(names)):
        msg = f"surrogate names must be unique, got {names}"
        raise ValueError(msg)
    for spec in specs:
        kind = spec.get("kind")
        if kind not in VALID_KINDS:
            msg = f"{spec.get('name')!r}: unknown kind {kind!r} (expected {sorted(VALID_KINDS)})"
            raise ValueError(msg)
        if kind in {"pretrained", "finetune"} and not spec.get("id"):
            msg = f"{spec.get('name')!r}: kind {kind!r} requires an 'id'"
            raise ValueError(msg)


def requires_gated_auth(specs: Sequence[Mapping[str, Any]]) -> bool:
    """Return True if any spec is a gated model that needs HF authentication."""
    return any(spec.get("gated") for spec in specs)


def assert_hf_auth() -> str:
    """Ensure a usable HuggingFace token is present; return the account name.

    Raises:
        RuntimeError: With setup guidance if no valid token is configured.
    """
    try:
        return str(whoami().get("name", "unknown"))
    except Exception as exc:  # any failure here means no usable token
        msg = (
            "Gated models are configured but no usable HuggingFace token was found. "
            "Accept the licences on the model pages, then run `hf auth login` "
            "(or export HF_TOKEN)."
        )
        raise RuntimeError(msg) from exc
