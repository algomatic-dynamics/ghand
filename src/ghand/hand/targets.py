"""Timestamped joint-target contracts keyed by ghand joint name."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from ghand.hand.joints import make_hand_joint_names


NORMALIZED_TARGET_LOWER: Final[float] = -1.0
NORMALIZED_TARGET_UPPER: Final[float] = 1.0
CURRENT_LIMIT_LOWER_MA: Final[float] = 0.0
VALID_HAND_JOINT_NAMES: Final[frozenset[str]] = frozenset(
    (*make_hand_joint_names("left"), *make_hand_joint_names("right"))
)


@dataclass(frozen=True)
class JointTarget:
    """Timestamped normalized hand target keyed by ghand joint name."""

    timestamp_ns: int
    joints: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")

        normalized: dict[str, float] = {}
        for joint_name, value in self.joints.items():
            if joint_name not in VALID_HAND_JOINT_NAMES:
                raise ValueError(f"Unknown hand joint target: {joint_name!r}.")
            normalized[joint_name] = _validate_normalized_target(value, joint_name)
        object.__setattr__(self, "joints", normalized)

    @classmethod
    def empty(cls, *, timestamp_ns: int) -> JointTarget:
        """Return a no-op target update."""

        return cls(timestamp_ns=timestamp_ns, joints={})

    @classmethod
    def neutral(cls, *, timestamp_ns: int, side: str = "right") -> JointTarget:
        """Return an explicit all-zero target for one hand side."""

        return cls(
            timestamp_ns=timestamp_ns,
            joints={joint_name: 0.0 for joint_name in make_hand_joint_names(side)},
        )

    def to_vector(self, joint_names: Sequence[str]) -> tuple[float, ...]:
        """Return target values in a requested joint order, defaulting missing values to zero."""

        return tuple(float(self.joints.get(joint_name, 0.0)) for joint_name in joint_names)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class RadianJointTarget:
    """Timestamped hand target in joint-angle radians keyed by ghand joint name."""

    timestamp_ns: int
    joints: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")

        angles: dict[str, float] = {}
        for joint_name, value in self.joints.items():
            if joint_name not in VALID_HAND_JOINT_NAMES:
                raise ValueError(f"Unknown hand joint target: {joint_name!r}.")
            angles[joint_name] = _validate_radian_target(value, joint_name)
        object.__setattr__(self, "joints", angles)

    @classmethod
    def empty(cls, *, timestamp_ns: int) -> RadianJointTarget:
        """Return a no-op target update."""

        return cls(timestamp_ns=timestamp_ns, joints={})

    @classmethod
    def neutral(cls, *, timestamp_ns: int, side: str = "right") -> RadianJointTarget:
        """Return an explicit all-zero radian target for one hand side."""

        return cls(
            timestamp_ns=timestamp_ns,
            joints={joint_name: 0.0 for joint_name in make_hand_joint_names(side)},
        )

    def to_vector(self, joint_names: Sequence[str]) -> tuple[float, ...]:
        """Return target radians in a requested joint order, defaulting missing values to zero."""

        return tuple(float(self.joints.get(joint_name, 0.0)) for joint_name in joint_names)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class CurrentLimitTarget:
    """Timestamped current-limit target in milliamps keyed by ghand joint name."""

    timestamp_ns: int
    currents_ma: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")

        currents: dict[str, float] = {}
        for joint_name, value in self.currents_ma.items():
            if joint_name not in VALID_HAND_JOINT_NAMES:
                raise ValueError(f"Unknown hand joint current limit: {joint_name!r}.")
            currents[joint_name] = _validate_current_limit_ma(value, joint_name)
        object.__setattr__(self, "currents_ma", currents)

    @classmethod
    def empty(cls, *, timestamp_ns: int) -> CurrentLimitTarget:
        """Return a no-op current-limit target update."""

        return cls(timestamp_ns=timestamp_ns, currents_ma={})

    @classmethod
    def constant(
        cls,
        *,
        timestamp_ns: int,
        current_limit_ma: float,
        side: str = "right",
    ) -> CurrentLimitTarget:
        """Return a constant current limit for all joints on one hand side."""

        current = _validate_current_limit_ma(current_limit_ma, "current_limit_ma")
        return cls(
            timestamp_ns=timestamp_ns,
            currents_ma={joint_name: current for joint_name in make_hand_joint_names(side)},
        )

    def to_vector(self, joint_names: Sequence[str]) -> tuple[float, ...]:
        """Return current limits in a requested joint order, defaulting missing values to zero."""

        return tuple(float(self.currents_ma.get(joint_name, 0.0)) for joint_name in joint_names)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class PositionCurrentAction:
    """Combined hand action with joint-position radians and current limits."""

    position: RadianJointTarget
    current_limit: CurrentLimitTarget

    def __post_init__(self) -> None:
        if self.position.timestamp_ns != self.current_limit.timestamp_ns:
            raise ValueError("position and current_limit timestamps must match.")

    @property
    def timestamp_ns(self) -> int:
        """Return the shared action timestamp."""

        return self.position.timestamp_ns

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _validate_normalized_target(value: float, joint_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{joint_name} target must be finite.")
    if not NORMALIZED_TARGET_LOWER <= normalized <= NORMALIZED_TARGET_UPPER:
        raise ValueError(
            f"{joint_name} target must be in normalized range "
            f"{NORMALIZED_TARGET_LOWER}..{NORMALIZED_TARGET_UPPER}."
        )
    return normalized


def _validate_radian_target(value: float, joint_name: str) -> float:
    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError(f"{joint_name} target angle must be finite.")
    return angle


def _validate_current_limit_ma(value: float, joint_name: str) -> float:
    current = float(value)
    if not math.isfinite(current):
        raise ValueError(f"{joint_name} current limit must be finite.")
    if current < CURRENT_LIMIT_LOWER_MA:
        raise ValueError(
            f"{joint_name} current limit must be >= {CURRENT_LIMIT_LOWER_MA} mA."
        )
    return current
