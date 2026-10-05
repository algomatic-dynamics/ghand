"""Action filtering helpers."""

from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to an inclusive range."""

    return min(max(value, lower), upper)
