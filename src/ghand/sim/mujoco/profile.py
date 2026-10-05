"""Versioned MuJoCo physics and control contract for LEAP V1 KAIZEN."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Final

from ghand.hand import (
    JointCalibration,
    load_urdf_joint_limits,
    make_default_joint_calibrations,
)


MUJOCO_PHYSICS_PROFILE_SCHEMA_VERSION: Final[str] = "v4"
MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION: Final[str] = "ghand-mujoco-validated-profile-v4"
_LEGACY_APPROVED_PROFILE_SCHEMA_VERSION: Final[str] = "ghand-mujoco-validated-profile-v3"
VALIDATED_MUJOCO_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "actuator_kp",
    "joint_damping",
)
_MUJOCO_PROFILE_VECTOR_LENGTHS: Final[dict[str, int]] = {
    "palm_collision_pos_m": 3,
    "palm_collision_size_m": 3,
    "palm_friction": 3,
    "finger_friction": 3,
    "floor_friction": 3,
    "cube_friction": 3,
    "turn_handle_center_pos_m": 3,
    "turn_handle_friction": 3,
    "joint_limit_solref": 2,
    "joint_limit_solimp": 5,
    "hand_self_collision_solref": 2,
    "hand_self_collision_solimp": 5,
}
HAND_SELF_COLLISION_DISABLED: Final[str] = "disabled"
HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1: Final[str] = "safe_cross_link_v1"
HAND_SELF_COLLISION_MODES: Final[frozenset[str]] = frozenset(
    {
        HAND_SELF_COLLISION_DISABLED,
        HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1,
    }
)


class _MujocoJointCalibration(JointCalibration):
    """Keep teleop normalization independent from URDF safety limits."""

    def normalized_to_angle_rad(self, value: float) -> float:
        """Use the canonical +/-pi target scale, then apply the URDF clamp."""

        normalized = min(max(float(value), -1.0), 1.0)
        return self.clamp_angle_rad(normalized * math.pi)


@dataclass(frozen=True)
class MujocoPhysicsProfile:
    """Physical parameters shared by MJCF export and runtime control paths.

    Joint position limits deliberately do not live in this dataclass. They are
    loaded from the source URDF so the generated MJCF and runtime action clamp
    cannot drift from the robot model.
    """

    name: str = "ghand-default"
    schema_version: str = MUJOCO_PHYSICS_PROFILE_SCHEMA_VERSION
    physics_timestep_s: float = 0.002
    control_hz: float = 10.0
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    solver_impratio: float = 1.0
    target_velocity_limit_rad_s: float = 20.0
    joint_damping: float = 0.025
    joint_armature: float = 0.001
    joint_frictionloss: float = 0.0
    joint_limit_solref: tuple[float, float] = (0.02, 1.0)
    joint_limit_solimp: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )
    actuator_kp: float = 6.0
    actuator_velocity_feedback_gain: float = 0.0
    actuator_force_limit: float = 0.8
    finger_collision_radius_m: float = 0.006
    finger_collision_half_length_m: float = 0.018
    palm_collision_pos_m: tuple[float, float, float] = (0.02, 0.008, -0.04)
    palm_collision_size_m: tuple[float, float, float] = (0.055, 0.06, 0.012)
    palm_friction: tuple[float, float, float] = (1.1, 0.02, 0.001)
    finger_friction: tuple[float, float, float] = (1.25, 0.03, 0.001)
    floor_z_m: float = -0.08
    floor_friction: tuple[float, float, float] = (1.2, 0.02, 0.001)
    cube_half_size_m: float = 0.036
    cube_mass_kg: float = 0.1
    cube_friction: tuple[float, float, float] = (1.4, 0.04, 0.002)
    turn_handle_center_pos_m: tuple[float, float, float] = (0.08, 0.02, 0.10)
    turn_handle_bar_half_length_m: float = 0.12
    turn_handle_bar_radius_m: float = 0.014
    turn_handle_knob_radius_m: float = 0.018
    turn_handle_hub_radius_m: float = 0.018
    turn_handle_hub_half_length_m: float = 0.012
    turn_handle_mass_kg: float = 0.12
    turn_handle_damping: float = 0.08
    turn_handle_armature: float = 0.002
    turn_handle_frictionloss: float = 0.003
    turn_handle_friction: tuple[float, float, float] = (1.5, 0.05, 0.002)
    hand_collision_contype: int = 1
    hand_collision_conaffinity: int = 0
    hand_self_collision_mode: str = HAND_SELF_COLLISION_SAFE_CROSS_LINK_V1
    hand_self_collision_solref: tuple[float, float] = (0.005, 1.0)
    hand_self_collision_solimp: tuple[float, float, float, float, float] = (
        0.99,
        0.999,
        0.0005,
        0.5,
        2.0,
    )
    hand_self_collision_margin_m: float = 0.0015
    cube_collision_contype: int = 2
    cube_collision_conaffinity: int = 5
    floor_collision_contype: int = 4
    floor_collision_conaffinity: int = 3
    turn_handle_collision_contype: int = 8
    turn_handle_collision_conaffinity: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MuJoCo physics profile name must not be empty.")
        if not self.schema_version.strip():
            raise ValueError("MuJoCo physics profile schema_version must not be empty.")
        if not self.integrator.strip() or not self.cone.strip():
            raise ValueError("MuJoCo integrator and cone must not be empty.")
        if self.hand_self_collision_mode not in HAND_SELF_COLLISION_MODES:
            modes = ", ".join(sorted(HAND_SELF_COLLISION_MODES))
            raise ValueError(f"hand_self_collision_mode must be one of: {modes}.")

        for field_name in (
            "physics_timestep_s",
            "control_hz",
            "solver_impratio",
            "target_velocity_limit_rad_s",
            "actuator_kp",
            "actuator_force_limit",
            "finger_collision_radius_m",
            "finger_collision_half_length_m",
            "cube_half_size_m",
            "cube_mass_kg",
            "turn_handle_bar_half_length_m",
            "turn_handle_bar_radius_m",
            "turn_handle_knob_radius_m",
            "turn_handle_hub_radius_m",
            "turn_handle_hub_half_length_m",
            "turn_handle_mass_kg",
        ):
            _validate_positive(getattr(self, field_name), field_name)
        steps_per_control = 1.0 / (self.control_hz * self.physics_timestep_s)
        if not math.isclose(
            steps_per_control,
            round(steps_per_control),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "control_hz must produce an integer number of physics timesteps."
            )
        for field_name in (
            "joint_damping",
            "joint_armature",
            "joint_frictionloss",
            "actuator_velocity_feedback_gain",
            "hand_self_collision_margin_m",
            "turn_handle_damping",
            "turn_handle_armature",
            "turn_handle_frictionloss",
        ):
            _validate_non_negative(getattr(self, field_name), field_name)
        _validate_finite(self.floor_z_m, "floor_z_m")

        _validate_vector(
            self.palm_collision_pos_m, "palm_collision_pos_m", positive=False
        )
        _validate_vector(
            self.palm_collision_size_m, "palm_collision_size_m", positive=True
        )
        _validate_vector(
            self.turn_handle_center_pos_m,
            "turn_handle_center_pos_m",
            positive=False,
        )
        for field_name in (
            "palm_friction",
            "finger_friction",
            "floor_friction",
            "cube_friction",
            "turn_handle_friction",
        ):
            _validate_vector(getattr(self, field_name), field_name, positive=False)
            if any(value < 0 for value in getattr(self, field_name)):
                raise ValueError(f"{field_name} values must be non-negative.")
        _validate_vector(
            self.joint_limit_solref,
            "joint_limit_solref",
            length=2,
            positive=True,
        )
        if self.joint_limit_solref[0] < 2.0 * self.physics_timestep_s:
            raise ValueError(
                "joint_limit_solref time constant must be at least twice "
                "physics_timestep_s."
            )
        _validate_solimp(self.joint_limit_solimp, "joint_limit_solimp")
        _validate_vector(
            self.hand_self_collision_solref,
            "hand_self_collision_solref",
            length=2,
            positive=True,
        )
        if (
            self.hand_self_collision_mode != HAND_SELF_COLLISION_DISABLED
            and self.hand_self_collision_solref[0] < 2.0 * self.physics_timestep_s
        ):
            raise ValueError(
                "hand_self_collision_solref time constant must be at least twice "
                "physics_timestep_s."
            )
        _validate_solimp(self.hand_self_collision_solimp)
        for field_name in (
            "hand_collision_contype",
            "hand_collision_conaffinity",
            "cube_collision_contype",
            "cube_collision_conaffinity",
            "floor_collision_contype",
            "floor_collision_conaffinity",
            "turn_handle_collision_contype",
            "turn_handle_collision_conaffinity",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation used for provenance."""

        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }

    @property
    def physics_steps_per_control(self) -> int:
        """Return the exact number of MuJoCo steps in one control interval."""

        return int(round(1.0 / (self.control_hz * self.physics_timestep_s)))

    def with_runtime_overrides(
        self,
        *,
        control_hz: float | None = None,
        target_velocity_limit_rad_s: float | None = None,
        actuator_kp: float | None = None,
        actuator_force_limit: float | None = None,
    ) -> MujocoPhysicsProfile:
        """Return a validated profile with explicit runtime overrides applied."""

        return replace(
            self,
            control_hz=self.control_hz if control_hz is None else float(control_hz),
            target_velocity_limit_rad_s=(
                self.target_velocity_limit_rad_s
                if target_velocity_limit_rad_s is None
                else float(target_velocity_limit_rad_s)
            ),
            actuator_kp=self.actuator_kp if actuator_kp is None else float(actuator_kp),
            actuator_force_limit=(
                self.actuator_force_limit
                if actuator_force_limit is None
                else float(actuator_force_limit)
            ),
        )


class MujocoPhysicsProfileArtifactError(ValueError):
    """Raised when an external MuJoCo profile artifact is not trustworthy."""


def mujoco_physics_profile_from_dict(
    payload: Mapping[str, object],
) -> MujocoPhysicsProfile:
    """Parse one complete, strictly versioned physics-profile mapping."""

    expected = {field.name for field in fields(MujocoPhysicsProfile)}
    supplied = set(payload)
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise MujocoPhysicsProfileArtifactError(
            "MuJoCo physics profile fields do not match the current schema: "
            + "; ".join(details)
        )
    if payload.get("schema_version") != MUJOCO_PHYSICS_PROFILE_SCHEMA_VERSION:
        raise MujocoPhysicsProfileArtifactError(
            "Unsupported MuJoCo physics profile schema_version."
        )

    normalized = dict(payload)
    for field_name, length in _MUJOCO_PROFILE_VECTOR_LENGTHS.items():
        raw = normalized[field_name]
        if not isinstance(raw, list | tuple):
            raise MujocoPhysicsProfileArtifactError(
                f"{field_name} must be a {length}-value JSON array."
            )
        normalized[field_name] = tuple(raw)
    try:
        return MujocoPhysicsProfile(**normalized)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MujocoPhysicsProfileArtifactError(
            f"Invalid MuJoCo physics profile: {exc}"
        ) from exc


def validated_mujoco_parameters_sha256(
    parameters: Mapping[str, object],
) -> str:
    """Hash the exact identified-parameter subset used by validation artifacts."""

    if set(parameters) != set(VALIDATED_MUJOCO_PARAMETER_NAMES):
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo parameters must contain exactly: "
            + ", ".join(VALIDATED_MUJOCO_PARAMETER_NAMES)
        )
    normalized: dict[str, float] = {}
    for name in VALIDATED_MUJOCO_PARAMETER_NAMES:
        raw = parameters[name]
        if isinstance(raw, bool):
            raise MujocoPhysicsProfileArtifactError(f"{name} must be numeric.")
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise MujocoPhysicsProfileArtifactError(f"{name} must be numeric.") from exc
        if not math.isfinite(value) or value < 0.0:
            raise MujocoPhysicsProfileArtifactError(
                f"{name} must be finite and non-negative."
            )
        if name == "actuator_kp" and value <= 0.0:
            raise MujocoPhysicsProfileArtifactError("actuator_kp must be positive.")
        normalized[name] = value
    serialized = json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def mujoco_joint_ranges_from_urdf(
    urdf_path: Path | str,
) -> dict[str, tuple[float, float]]:
    """Load MuJoCo hand joint ranges from the source URDF."""

    limits = load_urdf_joint_limits(urdf_path)
    ranges = {
        name: (limit.lower, limit.upper)
        for name, limit in limits.items()
        if _mujoco_joint_id(name) is not None
    }
    if not ranges:
        raise ValueError(f"URDF does not define any id<N> hand joints: {urdf_path}")
    return dict(sorted(ranges.items(), key=lambda item: _mujoco_joint_id(item[0]) or 0))


def make_mujoco_joint_calibrations(
    *,
    urdf_path: Path | str,
    side: str = "right",
    motor_ids: Sequence[int] = tuple(range(16)),
) -> tuple[JointCalibration, ...]:
    """Create runtime calibrations whose clamps match source-URDF limits."""

    base = make_default_joint_calibrations(side=side, motor_ids=motor_ids)
    ranges = mujoco_joint_ranges_from_urdf(urdf_path)
    missing = tuple(
        f"id{calibration.pose_dxl_id}"
        for calibration in base
        if f"id{calibration.pose_dxl_id}" not in ranges
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"URDF is missing LEAP V1 KAIZEN joint limits for: {missing_text}.")
    return tuple(
        _MujocoJointCalibration(
            joint_name=calibration.joint_name,
            dxl_id=calibration.dxl_id,
            urdf_dxl_id=calibration.urdf_dxl_id,
            zero_tick=calibration.zero_tick,
            ticks_per_radian=calibration.ticks_per_radian,
            direction=calibration.direction,
            lower_rad=ranges[f"id{calibration.pose_dxl_id}"][0],
            upper_rad=ranges[f"id{calibration.pose_dxl_id}"][1],
        )
        for calibration in base
    )


def _mujoco_joint_id(name: str) -> int | None:
    if not name.startswith("id") or not name[2:].isdigit():
        return None
    return int(name[2:])


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _validate_positive(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_vector(
    values: Sequence[float],
    name: str,
    *,
    positive: bool,
    length: int = 3,
) -> None:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    for value in values:
        if positive:
            _validate_positive(value, name)
        else:
            _validate_finite(value, name)


def _validate_solimp(
    values: Sequence[float],
    name: str = "hand_self_collision_solimp",
) -> None:
    _validate_vector(values, name, length=5, positive=True)
    dmin, dmax, _, midpoint, power = values
    if not 0.0 < dmin <= dmax < 1.0:
        raise ValueError(f"{name} dmin/dmax must satisfy 0 < dmin <= dmax < 1.")
    if midpoint > 1.0:
        raise ValueError(f"{name} midpoint must not exceed 1.")
    if power < 1.0:
        raise ValueError(f"{name} power must be at least 1.")


DEFAULT_MUJOCO_PHYSICS_PROFILE: Final[MujocoPhysicsProfile] = MujocoPhysicsProfile()
TELEOP_MUJOCO_PHYSICS_PROFILE: Final[MujocoPhysicsProfile] = replace(
    DEFAULT_MUJOCO_PHYSICS_PROFILE,
    name="ghand-teleop-responsive",
    joint_damping=0.06,
    joint_armature=0.0001,
    actuator_kp=20.0,
    actuator_force_limit=1.0,
)
def load_validated_mujoco_physics_profile(
    path: Path | str,
    *,
    base_profile: MujocoPhysicsProfile = DEFAULT_MUJOCO_PHYSICS_PROFILE,
) -> MujocoPhysicsProfile:
    """Apply an independently validated identification artifact to a base profile.

    Only the two parameters identified by the unloaded Phase-1--3 replay are
    accepted. Task geometry, contact parameters, control rate, and actuator
    force limit always remain those of ``base_profile``.
    """

    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MujocoPhysicsProfileArtifactError(
            f"Failed to read validated MuJoCo profile {artifact_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise MujocoPhysicsProfileArtifactError(
            f"Validated MuJoCo profile must be a JSON object: {artifact_path}"
        )
    schema_version = payload.get("schema_version")
    if schema_version not in {
        MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION,
        _LEGACY_APPROVED_PROFILE_SCHEMA_VERSION,
    }:
        raise MujocoPhysicsProfileArtifactError(
            "Unsupported validated MuJoCo profile schema_version."
        )
    if schema_version == MUJOCO_VALIDATED_PROFILE_SCHEMA_VERSION:
        if payload.get("status") != "validated":
            raise MujocoPhysicsProfileArtifactError(
                "MuJoCo profile artifact has not passed validation."
            )
    else:
        if payload.get("status") != "approved":
            raise MujocoPhysicsProfileArtifactError(
                "Legacy MuJoCo profile artifact has not been explicitly approved."
            )
        approval = payload.get("approval")
        if not isinstance(approval, Mapping):
            raise MujocoPhysicsProfileArtifactError(
                "Legacy validated MuJoCo profile approval metadata is missing."
            )
        for name in ("approved_by", "approved_at_utc"):
            value = approval.get(name)
            if not isinstance(value, str) or not value.strip():
                raise MujocoPhysicsProfileArtifactError(
                    f"Legacy validated MuJoCo profile approval.{name} is missing."
                )

    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile artifact_id is missing."
        )
    source_validation = payload.get("source_validation")
    if not isinstance(source_validation, Mapping):
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile source_validation is missing."
        )
    source_validation_path = source_validation.get("path")
    if (
        not isinstance(source_validation_path, str)
        or not source_validation_path.strip()
    ):
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile source validation path is invalid."
        )
    source_validation_sha = source_validation.get("sha256")
    if not isinstance(source_validation_sha, str) or len(source_validation_sha) != 64:
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile source validation SHA-256 is invalid."
        )
    try:
        int(source_validation_sha, 16)
    except ValueError as exc:
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile source validation SHA-256 is invalid."
        ) from exc

    parameter_names = payload.get("validated_parameter_names")
    if parameter_names != list(VALIDATED_MUJOCO_PARAMETER_NAMES):
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo parameter names do not match the current contract."
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo profile parameters are missing."
        )
    expected_parameters_sha = validated_mujoco_parameters_sha256(parameters)
    if payload.get("parameters_sha256") != expected_parameters_sha:
        raise MujocoPhysicsProfileArtifactError(
            "Validated MuJoCo parameter hash is stale or invalid."
        )

    combined = base_profile.to_dict()
    for name in VALIDATED_MUJOCO_PARAMETER_NAMES:
        combined[name] = float(parameters[name])  # type: ignore[arg-type]
    combined["name"] = f"{base_profile.name}+{artifact_id}"
    return mujoco_physics_profile_from_dict(combined)
