"""URDF joint limit helpers shared by Rerun and the operator UI."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final


DEFAULT_JOINT_LOWER_RAD: Final[float] = -math.pi
DEFAULT_JOINT_UPPER_RAD: Final[float] = math.pi


@dataclass(frozen=True)
class UrdfJointLimit:
    """Position limit for one URDF joint, in radians."""

    joint_name: str
    lower: float = DEFAULT_JOINT_LOWER_RAD
    upper: float = DEFAULT_JOINT_UPPER_RAD

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("URDF joint limits must be finite.")
        if self.lower >= self.upper:
            raise ValueError("URDF joint lower limit must be less than upper limit.")

    @property
    def center(self) -> float:
        """Return the midpoint of the joint range."""

        return (self.lower + self.upper) / 2.0

    @property
    def half_range(self) -> float:
        """Return half of the joint range."""

        return (self.upper - self.lower) / 2.0

    def target_to_angle(self, value: float, *, scale: float = 1.0) -> float:
        """Map a normalized target in [-1, +1] into this URDF joint range."""

        clamped = _clamp(float(value), -1.0, 1.0)
        return (self.center + clamped * self.half_range) * scale

    def angle_to_target(self, angle: float) -> float:
        """Map a URDF joint angle back into normalized target space."""

        return _clamp((float(angle) - self.center) / self.half_range, -1.0, 1.0)

    def degree_range(self) -> tuple[float, float]:
        """Return the joint range in degrees."""

        return (math.degrees(self.lower), math.degrees(self.upper))


@dataclass(frozen=True)
class UrdfPoseModel:
    """URDF-backed pose conversion model for display and visualization."""

    limits: Mapping[str, UrdfJointLimit]

    @classmethod
    def empty(cls) -> UrdfPoseModel:
        """Return a model that assigns default limits to all requested joints."""

        return cls(limits={})

    @classmethod
    def from_urdf_file(cls, path: Path | str) -> UrdfPoseModel:
        """Load joint limits from a URDF file."""

        return cls(limits=load_urdf_joint_limits(path))

    def joint_limit(self, joint_name: str) -> UrdfJointLimit:
        """Return the configured limit or a default -pi..+pi limit."""

        return self.limits.get(joint_name, UrdfJointLimit(joint_name=joint_name))

    def target_to_angle(
        self,
        *,
        dxl_id: int,
        value: float,
        joint_angle_scale: float = 1.0,
    ) -> float:
        """Convert a normalized target into a URDF joint angle."""

        return self.joint_limit(_joint_name_for_dxl_id(dxl_id)).target_to_angle(
            value,
            scale=joint_angle_scale,
        )

    def angle_to_target(self, *, dxl_id: int, angle: float) -> float:
        """Convert a URDF joint angle into normalized target space."""

        return self.joint_limit(_joint_name_for_dxl_id(dxl_id)).angle_to_target(angle)

    def target_to_degrees(self, *, dxl_id: int, value: float) -> float:
        """Convert a normalized target into display degrees."""

        return math.degrees(self.target_to_angle(dxl_id=dxl_id, value=value))

    def degrees_to_target(self, *, dxl_id: int, degrees: float) -> float:
        """Convert display degrees into normalized target space."""

        return self.angle_to_target(dxl_id=dxl_id, angle=math.radians(degrees))

    def degree_range(self, *, dxl_id: int) -> tuple[float, float]:
        """Return the UI slider range for a Dynamixel-backed URDF joint."""

        lower, upper = self.joint_limit(_joint_name_for_dxl_id(dxl_id)).degree_range()
        return (min(lower, upper), max(lower, upper))


def load_urdf_joint_limits(path: Path | str) -> dict[str, UrdfJointLimit]:
    """Load non-fixed URDF joint limits, defaulting missing limits to -pi..+pi."""

    root = ET.parse(path).getroot()
    limits: dict[str, UrdfJointLimit] = {}
    for joint in _children_named(root, "joint"):
        name = joint.get("name")
        joint_type = joint.get("type", "")
        if not name or joint_type == "fixed":
            continue

        limit = _first_child_named(joint, "limit")
        if limit is None:
            limits[name] = UrdfJointLimit(joint_name=name)
            continue

        lower = _finite_float(limit.get("lower"))
        upper = _finite_float(limit.get("upper"))
        if lower is None or upper is None or lower >= upper:
            limits[name] = UrdfJointLimit(joint_name=name)
        else:
            limits[name] = UrdfJointLimit(joint_name=name, lower=lower, upper=upper)
    return limits


def _joint_name_for_dxl_id(dxl_id: int) -> str:
    return f"id{int(dxl_id)}"


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _children_named(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _tag_name(child.tag) == name]


def _first_child_named(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _tag_name(child.tag) == name:
            return child
    return None


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
