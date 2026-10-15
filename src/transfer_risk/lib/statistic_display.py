"""Format defined and undefined manuscript statistics without changing numeric precision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def format_number(value: float | None, spec: str, *, reason: str) -> str:
    """Render a numeric value or state why the statistic is undefined.

    Args:
        value: Defined statistic or an explicit null.
        spec: Existing numeric format specification.
        reason: Explanation used only for a null value.

    Returns:
        Formatted number or an explicit undefined label and reason.
    """
    return f"{value:{spec}}" if value is not None else f"Undefined ({reason})"


def format_percent(value: float | None, *, reason: str, suffix: str = "") -> str:
    """Render a one-decimal percentage or its missing-denominator explanation."""
    if value is None:
        return format_number(None, ".1f", reason=reason)
    return f"{100 * value:.1f}{suffix}"


def association_cells(stats: Mapping[str, Any] | None) -> tuple[str, str]:
    """Render correlation and p-value cells with a saved or missing-evidence reason."""
    values = stats or {}
    reason = values.get("reason") or "association unavailable"
    return (
        format_number(values.get("rho"), ".3f", reason=reason),
        format_number(values.get("two_sided_p"), ".2g", reason=reason),
    )
