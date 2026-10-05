"""Joint and actuator limit models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimit:
    """Inclusive position limit for a joint in radians."""

    lower: float
    upper: float
