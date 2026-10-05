"""Conversions between LEAP Hand API joint vectors and ghand joint names.

LEAP V1 KAIZEN is compatible with the LEAP Hand v1, so the LEAP Hand API drives
it unchanged. These helpers translate between the LEAP vector representation and
ghand's named joints. The LEAP Hand API orders its 16 values by Dynamixel ID:
index 0..3, middle 4..7, ring 8..11, thumb 12..15. ``LEAPsim`` values are
radians centred on the servo mid-point, while ``LEAPhand`` values add pi so
that the mid-point reads pi. ghand measures radians from the same mid-point
with the packaged joint direction and orders joints ring, middle, index,
thumb. Only the right hand is supported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

from ghand.hand.calibration import DEFAULT_JOINT_DIRECTIONS, GHAND_RIGHT_MOTOR_IDS
from ghand.hand.joints import RIGHT_HAND_SIDE, make_hand_joint_names


LEAP_VECTOR_LENGTH: Final[int] = 16
LEAP_HAND_OFFSET_RAD: Final[float] = math.pi


def leap_sim_vector_to_joint_radians(vector: Sequence[float]) -> dict[str, float]:
    """Map a ``LEAPsim`` vector (zero-centred radians) to ghand joint radians."""

    values = _validate_vector(vector)
    return {
        joint_name: direction * values[dxl_id]
        for joint_name, dxl_id, direction in zip(
            make_hand_joint_names(RIGHT_HAND_SIDE),
            GHAND_RIGHT_MOTOR_IDS,
            DEFAULT_JOINT_DIRECTIONS,
            strict=True,
        )
    }


def joint_radians_to_leap_sim_vector(joints: Mapping[str, float]) -> tuple[float, ...]:
    """Map ghand joint radians to a ``LEAPsim`` vector; missing joints read zero."""

    vector = [0.0] * LEAP_VECTOR_LENGTH
    for joint_name, dxl_id, direction in zip(
        make_hand_joint_names(RIGHT_HAND_SIDE),
        GHAND_RIGHT_MOTOR_IDS,
        DEFAULT_JOINT_DIRECTIONS,
        strict=True,
    ):
        vector[dxl_id] = direction * _finite(joints.get(joint_name, 0.0), joint_name)
    return tuple(vector)


def leap_hand_vector_to_joint_radians(vector: Sequence[float]) -> dict[str, float]:
    """Map a ``LEAPhand`` vector (pi-centred radians) to ghand joint radians."""

    values = _validate_vector(vector)
    return leap_sim_vector_to_joint_radians(
        [value - LEAP_HAND_OFFSET_RAD for value in values]
    )


def joint_radians_to_leap_hand_vector(joints: Mapping[str, float]) -> tuple[float, ...]:
    """Map ghand joint radians to a ``LEAPhand`` vector; missing joints read pi."""

    return tuple(
        value + LEAP_HAND_OFFSET_RAD for value in joint_radians_to_leap_sim_vector(joints)
    )


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(_finite(value, f"LEAP vector[{index}]") for index, value in enumerate(vector))
    if len(values) != LEAP_VECTOR_LENGTH:
        raise ValueError(
            f"LEAP joint vectors must contain {LEAP_VECTOR_LENGTH} values, got {len(values)}."
        )
    return values


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result
