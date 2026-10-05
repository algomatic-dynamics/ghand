"""Adapters from retargeting output into hand-control runtimes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ghand.data.lerobot_schema import (
    ACTION_KEY,
    FeatureDict,
)
from ghand.hand import (
    CurrentLimitTarget,
    JOINT_CURRENT_LIMIT_KEY,
    JOINT_POSITION_KEY,
    JointCalibration,
    JointTarget,
    make_default_joint_calibrations,
    PositionCurrentAction,
    RadianJointTarget,
)
from ghand.control.operator_runtime import (
    DEFAULT_OPERATOR_TARGET_HZ,
    DEFAULT_TARGET_VELOCITY_LIMIT,
    HandOperatorRuntime,
    OperatorControlMode,
    OperatorRuntimeSnapshot,
)
from ghand.hardware import HandController


@dataclass(frozen=True)
class NormalizedToRadianJointTargetAdapter:
    """Convert normalized retargeting output to runtime joint-angle radians."""

    calibrations: tuple[JointCalibration, ...]

    @classmethod
    def from_runtime(cls, runtime: HandOperatorRuntime) -> NormalizedToRadianJointTargetAdapter:
        """Create an adapter from a runtime's active joint calibrations."""

        return cls(runtime.calibrations)

    def __post_init__(self) -> None:
        if not self.calibrations:
            raise ValueError("At least one joint calibration is required.")
        names = [calibration.joint_name for calibration in self.calibrations]
        if len(set(names)) != len(names):
            raise ValueError("Joint calibration names must be unique.")

    def __call__(self, target: JointTarget) -> RadianJointTarget:
        """Convert one normalized JointTarget into a radian target."""

        by_name = {calibration.joint_name: calibration for calibration in self.calibrations}
        missing = tuple(joint_name for joint_name in target.joints if joint_name not in by_name)
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Joint target(s) not configured in runtime: {missing_text}.")
        return RadianJointTarget(
            timestamp_ns=target.timestamp_ns,
            joints={
                joint_name: by_name[joint_name].normalized_to_angle_rad(value)
                for joint_name, value in target.joints.items()
            },
        )


@dataclass(frozen=True)
class PositionCurrentActionVectorAdapter:
    """Convert a flat policy action vector into position and current targets."""

    action_names: tuple[str, ...]

    @classmethod
    def from_features(cls, features: FeatureDict) -> PositionCurrentActionVectorAdapter:
        """Create an adapter from a LeRobot feature dictionary."""

        try:
            raw_names = features[ACTION_KEY]["names"]
        except KeyError as exc:
            raise ValueError(f"features must include {ACTION_KEY!r}.") from exc
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) for name in raw_names
        ):
            raise ValueError(f"{ACTION_KEY!r} feature must define string names.")
        return cls(tuple(raw_names))

    def __post_init__(self) -> None:
        if not self.action_names:
            raise ValueError("At least one action name is required.")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("Action names must be unique.")
        position_names = tuple(
            name for name in self.action_names if _joint_value_key(name) == JOINT_POSITION_KEY
        )
        current_limit_names = tuple(
            name
            for name in self.action_names
            if _joint_value_key(name) == JOINT_CURRENT_LIMIT_KEY
        )
        if not position_names:
            raise ValueError("Action names must include joint position channels.")
        if not current_limit_names:
            raise ValueError("Action names must include current-limit channels.")

    def __call__(
        self,
        values: Sequence[float],
        *,
        timestamp_ns: int,
    ) -> PositionCurrentAction:
        """Convert one flat action vector into a structured action."""

        if len(values) != len(self.action_names):
            raise ValueError(
                f"Expected action vector length {len(self.action_names)}, got {len(values)}."
            )

        positions: dict[str, float] = {}
        current_limits: dict[str, float] = {}
        for name, raw_value in zip(self.action_names, values, strict=True):
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} action value must be finite.")
            value_key = _joint_value_key(name)
            if value_key == JOINT_POSITION_KEY:
                positions[name] = value
            elif value_key == JOINT_CURRENT_LIMIT_KEY:
                current_limits[_replace_joint_value_key(name, JOINT_POSITION_KEY)] = value
            else:
                raise ValueError(f"Unsupported action channel: {name}.")

        return PositionCurrentAction(
            position=RadianJointTarget(timestamp_ns=timestamp_ns, joints=positions),
            current_limit=CurrentLimitTarget(
                timestamp_ns=timestamp_ns,
                currents_ma=current_limits,
            ),
        )


def make_fake_hand_runtime(
    *,
    side: str = "right",
    motor_ids: Sequence[int] = tuple(range(16)),
    dry_run: bool = False,
    target_hz: float = DEFAULT_OPERATOR_TARGET_HZ,
    readback_hz: float | None = None,
    target_velocity_limit: float = DEFAULT_TARGET_VELOCITY_LIMIT,
    control_mode: OperatorControlMode = "position",
    default_current_limit_ma: float | None = None,
    max_current_limit_ma: float | None = None,
) -> HandOperatorRuntime:
    """Create a fake hand runtime wired with the default joint-name mapping."""

    return HandOperatorRuntime(
        HandController.from_fake(motor_ids=motor_ids),
        make_default_joint_calibrations(side=side, motor_ids=motor_ids),
        backend="fake",
        dry_run=dry_run,
        target_hz=target_hz,
        readback_hz=readback_hz,
        target_velocity_limit=target_velocity_limit,
        control_mode=control_mode,
        default_current_limit_ma=default_current_limit_ma,
        max_current_limit_ma=max_current_limit_ma,
    )


def apply_joint_target(
    runtime: HandOperatorRuntime,
    target: JointTarget,
    *,
    now: float | None = None,
) -> OperatorRuntimeSnapshot:
    """Convert and apply a normalized retargeting target to a radian runtime."""

    return apply_normalized_joint_target(runtime, target, now=now)


def apply_normalized_joint_target(
    runtime: HandOperatorRuntime,
    target: JointTarget,
    *,
    now: float | None = None,
) -> OperatorRuntimeSnapshot:
    """Apply a normalized retargeting target to a radian runtime."""

    adapter = NormalizedToRadianJointTargetAdapter.from_runtime(runtime)
    return apply_radian_joint_target(runtime, adapter(target), now=now)


def apply_radian_joint_target(
    runtime: HandOperatorRuntime,
    target: RadianJointTarget,
    *,
    now: float | None = None,
) -> OperatorRuntimeSnapshot:
    """Apply a radian joint target to a runtime and advance one control tick."""

    _validate_runtime_joints(runtime, target)
    runtime.set_targets(target.joints)
    return runtime.step(now=now)


def apply_position_current_action(
    runtime: HandOperatorRuntime,
    action: PositionCurrentAction,
    *,
    now: float | None = None,
) -> OperatorRuntimeSnapshot:
    """Apply a structured position+current action to a runtime.

    Current limits are stored in the runtime snapshot. They are also written to
    hardware when the runtime uses current-based position control.
    """

    _validate_runtime_joints(runtime, action.position)
    _validate_runtime_current_limits(runtime, action.current_limit)
    runtime.set_targets(action.position.joints)
    runtime.set_current_limits(action.current_limit.currents_ma)
    return runtime.step(now=now)


def _validate_runtime_joints(
    runtime: HandOperatorRuntime,
    target: JointTarget | RadianJointTarget,
) -> None:
    runtime_joint_names = set(runtime.joint_names)
    missing = tuple(
        joint_name for joint_name in target.joints if joint_name not in runtime_joint_names
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Joint target(s) not configured in runtime: {missing_text}.")


def _validate_runtime_current_limits(
    runtime: HandOperatorRuntime,
    target: CurrentLimitTarget,
) -> None:
    runtime_joint_names = set(runtime.joint_names)
    missing = tuple(
        joint_name for joint_name in target.currents_ma if joint_name not in runtime_joint_names
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Current-limit target(s) not configured in runtime: {missing_text}.")


def _joint_value_key(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _replace_joint_value_key(name: str, value_key: str) -> str:
    prefix, separator, _key = name.rpartition(".")
    if separator == "":
        raise ValueError(f"Expected a joint action channel with a value suffix: {name}.")
    return f"{prefix}.{value_key}"
