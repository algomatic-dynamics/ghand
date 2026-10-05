"""Joint vocabulary of the LEAP V1 KAIZEN hand: sides, joint order, and channel names."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal


HandSide = Literal["left", "right"]
LEFT_HAND_SIDE: Final[HandSide] = "left"
RIGHT_HAND_SIDE: Final[HandSide] = "right"
DEFAULT_HAND_SIDES: Final[tuple[HandSide, ...]] = (RIGHT_HAND_SIDE,)
BIMANUAL_HAND_SIDES: Final[tuple[HandSide, ...]] = (LEFT_HAND_SIDE, RIGHT_HAND_SIDE)
HAND_SIDE_PREFIXES: Final[dict[HandSide, str]] = {
    LEFT_HAND_SIDE: "left_hand",
    RIGHT_HAND_SIDE: "right_hand",
}

JOINT_POSITION_KEY: Final[str] = "pos"
JOINT_PRESENT_CURRENT_KEY: Final[str] = "present_current_ma"
JOINT_CURRENT_LIMIT_KEY: Final[str] = "current_limit_ma"
JOINT_CURRENT_RESIDUAL_KEY: Final[str] = "current_residual_ma"
JOINT_CONTACT_KEY: Final[str] = "contact"

# A 16-DoF, four-finger joint-space layout matching the packaged LEAP V1 KAIZEN URDF.
# The order intentionally follows the URDF/Dynamixel ID hierarchy:
# ring=id0..3, middle=id4..7, index=id8..11, thumb=id12..15.
# Values are joint positions in radians.
GHAND_HAND_JOINTS: Final[tuple[str, ...]] = (
    "ring_mcp_side",
    "ring_mcp_flex",
    "ring_pip_flex",
    "ring_dip_flex",
    "middle_mcp_side",
    "middle_mcp_flex",
    "middle_pip_flex",
    "middle_dip_flex",
    "index_mcp_side",
    "index_mcp_flex",
    "index_pip_flex",
    "index_dip_flex",
    "thumb_cmc_side",
    "thumb_cmc_flex",
    "thumb_mcp_flex",
    "thumb_ip_flex",
)
GHAND_HAND_JOINT_POSITION_NAMES: Final[tuple[str, ...]] = tuple(
    f"{joint}.{JOINT_POSITION_KEY}" for joint in GHAND_HAND_JOINTS
)



@dataclass(frozen=True)
class JointComponent:
    """Ordered joint-position channels for one robot component."""

    prefix: str
    joints: tuple[str, ...]
    position_key: str = JOINT_POSITION_KEY

    def __post_init__(self) -> None:
        joints = tuple(self.joints)
        validate_name_part(self.prefix, "component prefix")
        validate_name_part(self.position_key, "position key")
        if not joints:
            raise ValueError("JointComponent must contain at least one joint.")
        for joint in joints:
            validate_name_part(joint, "joint name")
        if len(set(joints)) != len(joints):
            raise ValueError(f"Duplicate joints in component {self.prefix!r}.")
        object.__setattr__(self, "joints", joints)

    def position_names(self) -> tuple[str, ...]:
        """Return LeRobot channel names for this component."""

        return tuple(f"{self.prefix}.{joint}.{self.position_key}" for joint in self.joints)



def validate_name_part(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty.")
    if "/" in value:
        raise ValueError(f"{label} must not contain '/': {value!r}.")



def validate_hand_side(side: HandSide | str) -> HandSide:
    normalized = side.lower()
    if normalized not in HAND_SIDE_PREFIXES:
        valid = ", ".join(sorted(HAND_SIDE_PREFIXES))
        raise ValueError(f"Unsupported hand side {side!r}. Expected one of: {valid}.")
    return normalized  # type: ignore[return-value]


def make_hand_joint_component(
    side: HandSide | str,
    *,
    position_key: str = JOINT_POSITION_KEY,
) -> JointComponent:
    """Return the LEAP V1 KAIZEN hand component for a side."""

    normalized_side = validate_hand_side(side)
    return JointComponent(
        prefix=HAND_SIDE_PREFIXES[normalized_side],
        joints=GHAND_HAND_JOINTS,
        position_key=position_key,
    )


def compose_joint_position_names(components: Sequence[JointComponent]) -> tuple[str, ...]:
    """Concatenate joint-position channels from multiple robot components."""

    if not components:
        raise ValueError("At least one joint component must be provided.")

    names: list[str] = []
    seen: set[str] = set()
    for component in components:
        for name in component.position_names():
            if name in seen:
                raise ValueError(f"Duplicate joint position channel: {name!r}.")
            seen.add(name)
            names.append(name)
    return tuple(names)



def make_hand_joint_names(side: HandSide | str) -> tuple[str, ...]:
    """Return LeRobot channel names for one LEAP V1 KAIZEN hand component."""

    return make_hand_joint_component(side).position_names()


def make_hand_joint_current_names(side: HandSide | str) -> tuple[str, ...]:
    """Return present-current observation channel names for one LEAP V1 KAIZEN hand."""

    return make_hand_joint_component(
        side,
        position_key=JOINT_PRESENT_CURRENT_KEY,
    ).position_names()


def make_hand_joint_current_limit_names(side: HandSide | str) -> tuple[str, ...]:
    """Return current-limit action channel names for one LEAP V1 KAIZEN hand."""

    return make_hand_joint_component(
        side,
        position_key=JOINT_CURRENT_LIMIT_KEY,
    ).position_names()
