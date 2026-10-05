"""Joint-to-motor calibration of the LEAP V1 KAIZEN hand and the fixed packaged mapping."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

from ghand.hand.joints import make_hand_joint_names
from ghand.hand.model import default_right_hand_urdf_path
from ghand.hand.urdf import load_urdf_joint_limits
from ghand.hardware import (
    POSITION_CONTROL_MAX_TICK,
    POSITION_CONTROL_MIN_TICK,
    POSITION_TICKS_PER_REVOLUTION,
)
from ghand.safety.filters import clamp


# Standard assembly: the servo mid-point (180 degrees) is joint angle zero.
DEFAULT_ZERO_TICK: Final[int] = POSITION_TICKS_PER_REVOLUTION // 2
DEFAULT_TICKS_PER_RADIAN: Final[float] = POSITION_TICKS_PER_REVOLUTION / (2.0 * math.pi)
DEFAULT_URDF_DXL_IDS: Final[tuple[int, ...]] = tuple(range(16))
DEFAULT_JOINT_DIRECTIONS: Final[tuple[int, ...]] = (
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
)
DEFAULT_JOINT_TICKS_PER_RADIAN: Final[tuple[float, ...]] = (
    DEFAULT_TICKS_PER_RADIAN,
) * 16
DEFAULT_JOINT_LOWER_RADIANS: Final[tuple[float, ...]] = (-math.pi,) * 16
DEFAULT_JOINT_UPPER_RADIANS: Final[tuple[float, ...]] = (math.pi,) * 16
GHAND_RIGHT_MOTOR_IDS: Final[tuple[int, ...]] = (
    8,
    9,
    10,
    11,
    4,
    5,
    6,
    7,
    0,
    1,
    2,
    3,
    12,
    13,
    14,
    15,
)


@dataclass(frozen=True)
class JointCalibration:
    """Radian-based joint-to-motor mapping usable before full calibration exists."""

    joint_name: str
    dxl_id: int
    urdf_dxl_id: int | None = None
    zero_tick: int = DEFAULT_ZERO_TICK
    ticks_per_radian: float = DEFAULT_TICKS_PER_RADIAN
    direction: int = 1
    lower_rad: float = -math.pi
    upper_rad: float = math.pi

    def __post_init__(self) -> None:
        if self.ticks_per_radian <= 0:
            raise ValueError("ticks_per_radian must be positive.")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1.")
        if not math.isfinite(self.lower_rad) or not math.isfinite(self.upper_rad):
            raise ValueError("joint angle limits must be finite.")
        if self.lower_rad >= self.upper_rad:
            raise ValueError("lower_rad must be less than upper_rad.")
        if not self.lower_rad <= 0.0 <= self.upper_rad:
            raise ValueError("joint angle limits must include zero radians.")
        if self.urdf_dxl_id is not None and self.urdf_dxl_id < 0:
            raise ValueError("urdf_dxl_id must be non-negative.")

    @property
    def pose_dxl_id(self) -> int:
        """Return the URDF/model joint ID used for display and visualization."""

        return self.dxl_id if self.urdf_dxl_id is None else self.urdf_dxl_id

    def clamp_angle_rad(self, value_rad: float) -> float:
        """Clamp a joint angle in radians to the configured range."""

        return clamp(float(value_rad), self.lower_rad, self.upper_rad)

    def angle_rad_to_tick(self, value_rad: float) -> int:
        """Convert a joint angle in radians to a conservative position tick."""

        clamped = self.clamp_angle_rad(value_rad)
        tick = self.zero_tick + self.direction * clamped * self.ticks_per_radian
        return int(
            round(clamp(tick, POSITION_CONTROL_MIN_TICK, POSITION_CONTROL_MAX_TICK))
        )

    def tick_to_angle_rad(self, tick: int) -> float:
        """Convert a position tick back into joint angle radians."""

        value = self.direction * (int(tick) - self.zero_tick) / self.ticks_per_radian
        return self.clamp_angle_rad(value)

    def normalized_to_angle_rad(self, value: float) -> float:
        """Map a normalized retargeting value in [-1, +1] into joint radians."""

        normalized = clamp(float(value), -1.0, 1.0)
        if normalized >= 0:
            return self.clamp_angle_rad(normalized * self.upper_rad)
        return self.clamp_angle_rad((-normalized) * self.lower_rad)

    def clamp_target(self, value: float) -> float:
        """Compatibility alias for clamping a radian target."""

        return self.clamp_angle_rad(value)

    def target_to_tick(self, value: float) -> int:
        """Compatibility alias for converting a radian target to a tick."""

        return self.angle_rad_to_tick(value)

    def tick_to_target(self, tick: int) -> float:
        """Compatibility alias for converting a tick to radians."""

        return self.tick_to_angle_rad(tick)


def make_default_joint_calibrations(
    *,
    side: str = "right",
    motor_ids: Sequence[int] = tuple(range(16)),
    urdf_dxl_ids: Sequence[int] = DEFAULT_URDF_DXL_IDS,
    directions: Sequence[int] = DEFAULT_JOINT_DIRECTIONS,
    zero_ticks: Sequence[int] | None = None,
    ticks_per_radian: Sequence[float] = DEFAULT_JOINT_TICKS_PER_RADIAN,
    lower_radians: Sequence[float] = DEFAULT_JOINT_LOWER_RADIANS,
    upper_radians: Sequence[float] = DEFAULT_JOINT_UPPER_RADIANS,
) -> tuple[JointCalibration, ...]:
    """Create a temporary joint-to-motor mapping before calibration is available."""

    joint_names = make_hand_joint_names(side)
    resolved_zero_ticks = tuple(
        zero_ticks or (DEFAULT_ZERO_TICK,) * len(joint_names)
    )
    if len(motor_ids) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} motor IDs, got {len(motor_ids)}."
        )
    if len(urdf_dxl_ids) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} URDF joint IDs, got {len(urdf_dxl_ids)}."
        )
    if len(directions) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} motor directions, got {len(directions)}."
        )
    if len(resolved_zero_ticks) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} zero ticks, got {len(resolved_zero_ticks)}."
        )
    if len(ticks_per_radian) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} ticks-per-radian values, got {len(ticks_per_radian)}."
        )
    if len(lower_radians) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} lower radian limits, got {len(lower_radians)}."
        )
    if len(upper_radians) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} upper radian limits, got {len(upper_radians)}."
        )
    return tuple(
        JointCalibration(
            joint_name=joint_name,
            dxl_id=dxl_id,
            urdf_dxl_id=urdf_dxl_id,
            zero_tick=zero_tick,
            direction=direction,
            ticks_per_radian=ticks_per_rad,
            lower_rad=lower_rad,
            upper_rad=upper_rad,
        )
        for joint_name, dxl_id, urdf_dxl_id, direction, zero_tick, ticks_per_rad, lower_rad, upper_rad in zip(
            joint_names,
            motor_ids,
            urdf_dxl_ids,
            directions,
            resolved_zero_ticks,
            ticks_per_radian,
            lower_radians,
            upper_radians,
            strict=True,
        )
    )


REAL_HAND_JOINT_CALIBRATION_SCHEMA_VERSION: Final[str] = (
    "ghand-packaged-real-hand-joint-mapping-v1"
)
# Standard LEAP Hand servos use 180 degrees as simulator q=0.
REAL_HAND_ZERO_TICK: Final[int] = DEFAULT_ZERO_TICK
REAL_HAND_ZERO_SOURCE: Final[str] = "leap_hand_standard_180_degree"


def make_real_hand_joint_calibrations() -> tuple[JointCalibration, ...]:
    """Build the fixed standard-assembly motor mapping with packaged URDF limits."""

    motor_ids = GHAND_RIGHT_MOTOR_IDS
    base = make_default_joint_calibrations(
        side="right",
        motor_ids=motor_ids,
        zero_ticks=(REAL_HAND_ZERO_TICK,) * len(motor_ids),
    )
    limits = load_urdf_joint_limits(default_right_hand_urdf_path())
    calibrations: list[JointCalibration] = []
    for item in base:
        joint_id = f"id{item.pose_dxl_id}"
        limit = limits.get(joint_id)
        if limit is None:
            raise ValueError(f"Packaged URDF is missing the {joint_id} joint limit.")
        calibration = replace(
            item,
            lower_rad=limit.lower,
            upper_rad=limit.upper,
        )
        endpoint_ticks = tuple(
            calibration.zero_tick
            + calibration.direction * angle_rad * calibration.ticks_per_radian
            for angle_rad in (calibration.lower_rad, calibration.upper_rad)
        )
        if not all(
            POSITION_CONTROL_MIN_TICK <= tick <= POSITION_CONTROL_MAX_TICK
            for tick in endpoint_ticks
        ):
            raise ValueError(
                f"The fixed zero tick {calibration.zero_tick} for "
                f"{calibration.joint_name} cannot represent its complete URDF range."
            )
        calibrations.append(calibration)
    return tuple(calibrations)


def real_hand_joint_calibration_provenance() -> dict[str, str]:
    """Return a deterministic identity for the packaged motor-to-model mapping."""

    urdf_sha256 = _sha256(default_right_hand_urdf_path())
    identity = {
        "schema_version": REAL_HAND_JOINT_CALIBRATION_SCHEMA_VERSION,
        "source_urdf_sha256": urdf_sha256,
        "reference_pose": "packaged_urdf_zero",
        "zero_source": REAL_HAND_ZERO_SOURCE,
        "calibrations": [
            asdict(calibration) for calibration in make_real_hand_joint_calibrations()
        ],
    }
    return {
        "schema_version": REAL_HAND_JOINT_CALIBRATION_SCHEMA_VERSION,
        "sha256": _sha256_json(identity),
        "source_urdf_sha256": urdf_sha256,
        "reference_pose": "packaged_urdf_zero",
        "zero_source": REAL_HAND_ZERO_SOURCE,
    }


def validate_real_hand_homing_offsets(values: Mapping[int, int]) -> dict[int, int]:
    """Require the standard zero Homing Offset for every configured motor."""

    expected_ids = set(GHAND_RIGHT_MOTOR_IDS)
    if set(values) != expected_ids:
        raise ValueError(
            "Homing Offset readback did not return the configured 16 Dynamixel IDs."
        )
    normalized: dict[int, int] = {}
    for dxl_id in GHAND_RIGHT_MOTOR_IDS:
        normalized[dxl_id] = _required_int(
            values[dxl_id], f"Homing Offset for ID {dxl_id}"
        )
    nonzero = {dxl_id: offset for dxl_id, offset in normalized.items() if offset != 0}
    if nonzero:
        details = ", ".join(
            f"ID {dxl_id}={offset}" for dxl_id, offset in nonzero.items()
        )
        raise ValueError(
            "Homing Offset(20) must be 0 for the packaged 2048-tick q=0 mapping; "
            f"found {details}. Restore the standard LEAP Hand motor configuration."
        )
    return normalized


def validate_real_hand_joint_calibrations(
    raw_calibrations: object,
) -> tuple[JointCalibration, ...]:
    """Validate embedded calibrations against the fixed packaged mapping."""

    if not isinstance(raw_calibrations, list) or len(raw_calibrations) != 16:
        raise ValueError("Real-hand mapping must contain exactly 16 joints.")
    parsed: list[JointCalibration] = []
    for index, raw in enumerate(raw_calibrations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid joint mapping entry {index}.")
        try:
            parsed.append(
                JointCalibration(
                    joint_name=str(raw["joint_name"]),
                    dxl_id=_required_int(
                        raw["dxl_id"], f"calibrations[{index}].dxl_id"
                    ),
                    urdf_dxl_id=_required_int(
                        raw["urdf_dxl_id"],
                        f"calibrations[{index}].urdf_dxl_id",
                    ),
                    zero_tick=_required_int(
                        raw["zero_tick"], f"calibrations[{index}].zero_tick"
                    ),
                    ticks_per_radian=_required_float(
                        raw["ticks_per_radian"],
                        f"calibrations[{index}].ticks_per_radian",
                    ),
                    direction=_required_int(
                        raw["direction"], f"calibrations[{index}].direction"
                    ),
                    lower_rad=_required_float(
                        raw["lower_rad"], f"calibrations[{index}].lower_rad"
                    ),
                    upper_rad=_required_float(
                        raw["upper_rad"], f"calibrations[{index}].upper_rad"
                    ),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Joint mapping entry {index} is missing {exc.args[0]}."
            ) from exc

    expected = make_real_hand_joint_calibrations()
    for actual, required in zip(parsed, expected, strict=True):
        if (
            actual.joint_name != required.joint_name
            or actual.dxl_id != required.dxl_id
            or actual.pose_dxl_id != required.pose_dxl_id
            or actual.zero_tick != required.zero_tick
            or actual.direction != required.direction
            or not math.isclose(
                actual.ticks_per_radian, required.ticks_per_radian, abs_tol=1e-12
            )
            or not math.isclose(actual.lower_rad, required.lower_rad, abs_tol=1e-12)
            or not math.isclose(actual.upper_rad, required.upper_rad, abs_tol=1e-12)
        ):
            raise ValueError(
                "Real-hand mapping does not match the current packaged LEAP V1 KAIZEN model "
                f"at entry {actual.joint_name!r}."
            )
    return tuple(parsed)


def validate_joint_calibration_provenance(raw: object) -> dict[str, str]:
    """Validate the packaged mapping identity in a measurement manifest."""

    if not isinstance(raw, Mapping):
        raise ValueError("Measurement has no packaged real-hand mapping provenance.")
    expected = real_hand_joint_calibration_provenance()
    for field, value in expected.items():
        if raw.get(field) != value:
            raise ValueError(
                "Measurement uses a different packaged real-hand mapping "
                f"({field} mismatch)."
            )
    return expected


def raw_joint_angle_rad(calibration: JointCalibration, tick: int) -> float:
    """Convert an encoder tick without hiding an out-of-range result by clamping."""

    return (
        calibration.direction
        * (int(tick) - calibration.zero_tick)
        / calibration.ticks_per_radian
    )


def joint_limit_tolerance_rad(calibration: JointCalibration) -> float:
    """Allow one encoder tick for rounding at a configured limit."""

    return 1.0 / calibration.ticks_per_radian + 1e-12


def _required_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _required_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
