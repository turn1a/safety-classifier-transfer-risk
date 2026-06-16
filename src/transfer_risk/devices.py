"""Torch device selection (glue — imports torch).

Shared by the model, similarity, attack, and transfer stages so every node honours
the same ``device.policy`` parameter. ``mps_or_cpu`` prefers Apple MPS and falls back
to CPU when MPS is unavailable.
"""

from __future__ import annotations

import torch


def resolve_device(policy: str) -> torch.device:
    """Resolve a device policy string to a concrete ``torch.device``.

    Args:
        policy: ``"mps_or_cpu"`` (prefer MPS, else CPU) or ``"cpu"`` (force CPU).

    Returns:
        The selected ``torch.device``.
    """
    if policy != "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
