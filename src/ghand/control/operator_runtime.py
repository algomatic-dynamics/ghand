"""Runtime state and safety gates for the operator Web UI."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from ghand.config.hardware import (
    DEFAULT_HAND_PROFILE_ACCELERATION,
    DEFAULT_HAND_PROFILE_VELOCITY,
)
from ghand.hand import (
    JointCalibration,
)
from ghand.hardware import (
    HandController,
    normalize_position_tick,
)
from ghand.safety.filters import clamp


DEFAULT_OPERATOR_TARGET_HZ = 30.0
DEFAULT_OPERATOR_TELEMETRY_HZ = 1.0
DEFAULT_TARGET_VELOCITY_LIMIT = 2.0
DEFAULT_OPERATOR_PROFILE_ACCELERATION = DEFAULT_HAND_PROFILE_ACCELERATION
DEFAULT_OPERATOR_PROFILE_VELOCITY = DEFAULT_HAND_PROFILE_VELOCITY
OperatorControlMode = Literal["position", "current_position"]
OPERATOR_CONTROL_MODE_POSITION: Final[OperatorControlMode] = "position"
OPERATOR_CONTROL_MODE_CURRENT_POSITION: Final[OperatorControlMode] = "current_position"
@dataclass(frozen=True)
class JointRuntimeState:
    """UI-facing state for one joint and its backing motor."""

    joint_name: str
    dxl_id: int
    target: float
    measured: float
    commanded: float
    target_tick: int
    measured_tick: int
    torque_enabled: bool
    current_limit_ma: float | None = None
    present_current_ma: float | None = None
    temperature_celsius: int | None = None
    input_voltage: float | None = None
    hardware_error_status: int | None = None
    urdf_dxl_id: int | None = None

    @property
    def pose_dxl_id(self) -> int:
        """Return the URDF/model joint ID used for display and visualization."""

        return self.dxl_id if self.urdf_dxl_id is None else self.urdf_dxl_id


@dataclass(frozen=True)
class OperatorRuntimeSnapshot:
    """Snapshot of the full operator runtime."""

    backend: str
    connected: bool
    dry_run: bool
    emergency_stop: bool
    last_error: str | None
    joints: tuple[JointRuntimeState, ...]


class HandOperatorRuntime:
    """Owns operator targets, safety gates, and controller read/write calls."""

    def __init__(
        self,
        controller: HandController,
        calibrations: Sequence[JointCalibration],
        *,
        backend: str,
        dry_run: bool,
        target_hz: float = DEFAULT_OPERATOR_TARGET_HZ,
        readback_hz: float | None = None,
        telemetry_hz: float = DEFAULT_OPERATOR_TELEMETRY_HZ,
        target_velocity_limit: float = DEFAULT_TARGET_VELOCITY_LIMIT,
        profile_acceleration: int = DEFAULT_OPERATOR_PROFILE_ACCELERATION,
        profile_velocity: int = DEFAULT_OPERATOR_PROFILE_VELOCITY,
        control_mode: OperatorControlMode = OPERATOR_CONTROL_MODE_POSITION,
        default_current_limit_ma: float | None = None,
        max_current_limit_ma: float | None = None,
        combined_safety_readback: bool = False,
    ) -> None:
        if not calibrations:
            raise ValueError("At least one joint calibration is required.")
        if target_hz <= 0:
            raise ValueError("target_hz must be positive.")
        if readback_hz is not None and readback_hz < 0:
            raise ValueError("readback_hz must be non-negative.")
        if telemetry_hz < 0:
            raise ValueError("telemetry_hz must be non-negative.")
        if target_velocity_limit <= 0:
            raise ValueError("target_velocity_limit must be positive.")
        if profile_acceleration < 0:
            raise ValueError("profile_acceleration must be non-negative.")
        if profile_velocity < 0:  # 0 means no Dynamixel velocity limit.
            raise ValueError("profile_velocity must be non-negative.")
        if control_mode not in {
            OPERATOR_CONTROL_MODE_POSITION,
            OPERATOR_CONTROL_MODE_CURRENT_POSITION,
        }:
            raise ValueError("control_mode must be 'position' or 'current_position'.")
        if max_current_limit_ma is not None:
            max_current_limit_ma = _validate_current_limit_ma(max_current_limit_ma)

        joint_names = [calibration.joint_name for calibration in calibrations]
        dxl_ids = [calibration.dxl_id for calibration in calibrations]
        urdf_dxl_ids = [calibration.pose_dxl_id for calibration in calibrations]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Joint calibration names must be unique.")
        if len(set(dxl_ids)) != len(dxl_ids):
            raise ValueError("Joint calibration motor IDs must be unique.")
        if len(set(urdf_dxl_ids)) != len(urdf_dxl_ids):
            raise ValueError("Joint calibration URDF IDs must be unique.")

        self.controller = controller
        self.calibrations = tuple(calibrations)
        self.backend = backend
        self.target_hz = target_hz
        self.readback_hz = target_hz if readback_hz is None else float(readback_hz)
        self.telemetry_hz = telemetry_hz
        self.target_velocity_limit = target_velocity_limit
        self.profile_acceleration = int(profile_acceleration)
        self.profile_velocity = int(profile_velocity)
        self.control_mode = control_mode
        self.max_current_limit_ma = max_current_limit_ma
        self.combined_safety_readback = bool(combined_safety_readback)
        self.dry_run = dry_run
        self.emergency_stop = False
        self.last_error: str | None = None
        self._torque_enabled = False
        self._last_step_time: float | None = None
        self._last_readback_time: float | None = None
        self._last_telemetry_time: float | None = None
        self._torque_enabled_by_id: dict[int, bool] | None = None
        self._targets = {
            calibration.joint_name: 0.0 for calibration in self.calibrations
        }
        self._commanded = dict(self._targets)
        default_current = (
            None
            if default_current_limit_ma is None
            else self._bounded_current_limit_ma(default_current_limit_ma)
        )
        self._current_limits_ma = {
            calibration.joint_name: default_current for calibration in self.calibrations
        }
        self._goal_currents_dirty = default_current is not None
        self._measured = dict(self._targets)
        self._measured_ticks = {
            calibration.joint_name: calibration.zero_tick
            for calibration in self.calibrations
        }
        self._telemetry: dict[
            str,
            tuple[float | None, int | None, float | None, int | None],
        ] = {
            calibration.joint_name: (None, None, None, None)
            for calibration in self.calibrations
        }

    @property
    def connected(self) -> bool:
        """Return whether the underlying controller is connected."""

        return self.controller.is_connected

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the runtime joint order."""

        return tuple(calibration.joint_name for calibration in self.calibrations)

    @property
    def dxl_ids(self) -> tuple[int, ...]:
        """Return the backing Dynamixel motor IDs."""

        return tuple(calibration.dxl_id for calibration in self.calibrations)

    def connect(self) -> None:
        """Connect and prepare the controller backend."""

        self.controller.connect()
        self.configure()

    def configure(self) -> None:
        """Prepare an already connected bus without reopening the serial port."""

        if not self.dry_run:
            if self.control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION:
                self.controller.configure_current_based_position_control(
                    self.dxl_ids,
                    profile_acceleration=self.profile_acceleration,
                    profile_velocity=self.profile_velocity,
                )
            else:
                self.controller.configure_position_control(
                    self.dxl_ids,
                    profile_acceleration=self.profile_acceleration,
                    profile_velocity=self.profile_velocity,
                )
        self._last_telemetry_time = None
        self._last_readback_time = None
        self._torque_enabled_by_id = None
        self._goal_currents_dirty = any(
            value is not None for value in self._current_limits_ma.values()
        )
        self.last_error = None

    def start(self) -> OperatorRuntimeSnapshot:
        """Connect or resume at the measured pose after an explicit start action."""

        try:
            if not self.connected:
                self.connect()
            self.clear_emergency_stop()
            snapshot = self.sync_motion_state_to_measured()
            if snapshot.last_error is not None:
                raise RuntimeError(snapshot.last_error)
            if not self.dry_run:
                self.controller.write_goal_positions(
                    {joint.dxl_id: joint.measured_tick for joint in snapshot.joints}
                )
            self.set_torque_enabled(True)
        except Exception:
            self.disconnect()
            raise
        return self.snapshot()

    def disconnect(self) -> None:
        """Torque off when possible, then disconnect the controller backend."""

        try:
            if self.connected and not self.dry_run:
                try:
                    self.controller.set_torque_enabled_many(self.dxl_ids, False)
                except (
                    Exception
                ) as exc:  # pragma: no cover - hardware timing dependent.
                    self.last_error = str(exc)
        finally:
            self._torque_enabled = False
            self._torque_enabled_by_id = None
            self._last_readback_time = None
            self._last_telemetry_time = None
            self.controller.disconnect()

    def set_dry_run(self, enabled: bool) -> None:
        """Set whether motor writes are suppressed."""

        if enabled and self.connected and not self.dry_run:
            self.controller.set_torque_enabled_many(self.dxl_ids, False)
            self._torque_enabled = False
        self.dry_run = bool(enabled)

    def set_target(self, joint_name: str, value: float) -> None:
        """Set the latest desired joint angle in radians for one joint."""

        calibration = self._calibration(joint_name)
        if self.emergency_stop:
            self.last_error = (
                "Emergency stop is active; clear it before changing targets."
            )
            return
        self._targets[joint_name] = calibration.clamp_target(value)
        self.last_error = None

    def set_targets(self, targets: Mapping[str, float]) -> None:
        """Set latest desired joint angles in radians for multiple joints."""

        for joint_name, value in targets.items():
            self.set_target(joint_name, value)

    def set_current_limit(self, joint_name: str, current_limit_ma: float) -> None:
        """Set the latest desired current limit in milliamps for one joint."""

        self._calibration(joint_name)
        if self.emergency_stop:
            self.last_error = (
                "Emergency stop is active; clear it before changing current limits."
            )
            return
        bounded = self._bounded_current_limit_ma(current_limit_ma)
        if self._current_limits_ma[joint_name] != bounded:
            self._current_limits_ma[joint_name] = bounded
            self._goal_currents_dirty = True
        self.last_error = None

    def set_current_limits(self, current_limits_ma: Mapping[str, float]) -> None:
        """Set latest desired current limits in milliamps for multiple joints."""

        for joint_name, current_limit_ma in current_limits_ma.items():
            self.set_current_limit(joint_name, current_limit_ma)

    def home(self) -> None:
        """Return all desired targets to the neutral position."""

        if self.emergency_stop:
            self.last_error = "Emergency stop is active; clear it before homing."
            return
        for joint_name in self.joint_names:
            self._targets[joint_name] = 0.0
        self.last_error = None

    def reset_motion_state(self) -> None:
        """Reset target, commanded, and measured state to neutral without writes."""

        for calibration in self.calibrations:
            joint_name = calibration.joint_name
            self._targets[joint_name] = 0.0
            self._commanded[joint_name] = 0.0
            self._measured[joint_name] = 0.0
            self._measured_ticks[joint_name] = calibration.angle_rad_to_tick(0.0)
        self._last_step_time = None
        self.last_error = None

    def sync_motion_state_to_measured(
        self, *, now: float | None = None
    ) -> OperatorRuntimeSnapshot:
        """Align target and commanded state to the latest measured joint positions."""

        now = time.monotonic() if now is None else now
        try:
            if self.connected:
                self._readback(now, force=True)
            for joint_name in self.joint_names:
                measured = self._measured[joint_name]
                self._targets[joint_name] = measured
                self._commanded[joint_name] = measured
            self._last_step_time = None
            self._last_readback_time = now
            self.last_error = None
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through concrete backends.
            self.last_error = str(exc)
        return self.snapshot()

    def set_torque_enabled(self, enabled: bool) -> None:
        """Enable or disable torque through the controller when writes are allowed."""

        self._torque_enabled = bool(enabled)
        if self.connected and not self.dry_run:
            if enabled and self.control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION:
                self._write_goal_currents()
            self.controller.set_torque_enabled_many(self.dxl_ids, enabled)
        self._torque_enabled_by_id = None
        self.last_error = None

    def emergency_stop_now(self) -> None:
        """Latch emergency stop and torque off immediately when possible."""

        self.emergency_stop = True
        self._torque_enabled = False
        self._torque_enabled_by_id = None
        if self.connected and not self.dry_run:
            self.controller.set_torque_enabled_many(self.dxl_ids, False)
        self.last_error = "Emergency stop is active."

    def clear_emergency_stop(self) -> None:
        """Clear the emergency stop latch without enabling torque."""

        self.emergency_stop = False
        self.last_error = None

    def step(self, *, now: float | None = None) -> OperatorRuntimeSnapshot:
        """Advance one UI/control tick and return a fresh runtime snapshot."""

        now = time.monotonic() if now is None else now
        try:
            if self.connected:
                self._write_commanded_targets(now)
                self._step_backend_simulation()
                self._readback(now, force=self.combined_safety_readback)
                if not self.emergency_stop:
                    self.last_error = None
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through concrete backends.
            self.last_error = str(exc)
        return self.snapshot()

    def write_targets(self, *, now: float | None = None) -> OperatorRuntimeSnapshot:
        """Write the latest targets without performing a sensor readback."""

        now = time.monotonic() if now is None else now
        try:
            if self.connected:
                self._write_commanded_targets(now)
                self._step_backend_simulation()
                if not self.emergency_stop:
                    self.last_error = None
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through concrete backends.
            self.last_error = str(exc)
        return self.snapshot()

    def observe(self, *, now: float | None = None) -> OperatorRuntimeSnapshot:
        """Read back sensor state without writing targets or stepping simulation.

        Unlike :meth:`step`, this never advances simulated backends and never
        sends goal positions, so observations stay aligned with the state the
        next action will act on.
        """

        now = time.monotonic() if now is None else now
        try:
            if self.connected:
                self._readback(now, force=True)
                self.last_error = None
        except (
            Exception
        ) as exc:  # pragma: no cover - exercised through concrete backends.
            self.last_error = str(exc)
        return self.snapshot()

    def reset_control_timing(self) -> None:
        """Make the next target write advance by one nominal control period."""

        self._last_step_time = None

    def snapshot(self) -> OperatorRuntimeSnapshot:
        """Return the latest known UI-facing state without touching hardware."""

        states: list[JointRuntimeState] = []
        for calibration in self.calibrations:
            joint_name = calibration.joint_name
            current_ma, temperature, voltage, hardware_error = self._telemetry[
                joint_name
            ]
            states.append(
                JointRuntimeState(
                    joint_name=joint_name,
                    dxl_id=calibration.dxl_id,
                    urdf_dxl_id=calibration.pose_dxl_id,
                    target=self._targets[joint_name],
                    measured=self._measured[joint_name],
                    commanded=self._commanded[joint_name],
                    target_tick=calibration.angle_rad_to_tick(
                        self._targets[joint_name]
                    ),
                    measured_tick=self._measured_ticks[joint_name],
                    torque_enabled=(
                        self._torque_enabled_by_id.get(
                            calibration.dxl_id, self._torque_enabled
                        )
                        if self._torque_enabled_by_id is not None
                        else self._torque_enabled
                    ),
                    current_limit_ma=self._current_limits_ma[joint_name],
                    present_current_ma=current_ma,
                    temperature_celsius=temperature,
                    input_voltage=voltage,
                    hardware_error_status=hardware_error,
                )
            )
        return OperatorRuntimeSnapshot(
            backend=self.backend,
            connected=self.connected,
            dry_run=self.dry_run,
            emergency_stop=self.emergency_stop,
            last_error=self.last_error,
            joints=tuple(states),
        )

    def _write_commanded_targets(self, now: float) -> None:
        if self.dry_run or self.emergency_stop or not self._torque_enabled:
            self._last_step_time = now
            return

        previous = self._last_step_time
        self._last_step_time = now
        dt = 1.0 / self.target_hz if previous is None else max(now - previous, 0.0)
        max_delta = self.target_velocity_limit * dt
        writes: dict[int, int] = {}
        for calibration in self.calibrations:
            joint_name = calibration.joint_name
            current = self._commanded[joint_name]
            target = self._targets[joint_name]
            delta = clamp(target - current, -max_delta, max_delta)
            commanded = calibration.clamp_target(current + delta)
            self._commanded[joint_name] = commanded
            writes[calibration.dxl_id] = calibration.angle_rad_to_tick(commanded)
        if self.control_mode == OPERATOR_CONTROL_MODE_CURRENT_POSITION:
            self._write_goal_currents()
        self.controller.write_goal_positions(writes)

    def _write_goal_currents(self) -> None:
        if not self._goal_currents_dirty:
            return
        writes = {
            calibration.dxl_id: current_limit_ma
            for calibration in self.calibrations
            if (current_limit_ma := self._current_limits_ma[calibration.joint_name])
            is not None
        }
        if writes:
            self.controller.write_goal_currents_ma(writes)
        self._goal_currents_dirty = False

    def _readback(self, now: float, *, force: bool = False) -> None:
        if self.combined_safety_readback and (
            force
            or self._should_read_positions(now)
            or self._should_read_telemetry(now)
        ):
            self._read_combined_safety_state(now)
            return
        if force or self._should_read_positions(now):
            positions = self.controller.read_motor_positions(self.dxl_ids)
            self._last_readback_time = now
            for calibration in self.calibrations:
                position = positions[calibration.dxl_id]
                self._measured_ticks[calibration.joint_name] = position
                self._measured[calibration.joint_name] = calibration.tick_to_angle_rad(
                    position
                )

        if not self._should_read_telemetry(now):
            return

        self._last_telemetry_time = now
        telemetry_by_id = {
            telemetry.dxl_id: telemetry
            for telemetry in self.controller.read_motor_telemetry(self.dxl_ids)
        }
        for calibration in self.calibrations:
            telemetry = telemetry_by_id[calibration.dxl_id]
            self._telemetry[calibration.joint_name] = (
                telemetry.present_current_ma,
                telemetry.temperature_celsius,
                telemetry.input_voltage,
                telemetry.hardware_error_status,
            )

    def _read_combined_safety_state(self, now: float) -> None:
        samples = {
            sample.dxl_id: sample
            for sample in self.controller.read_safety_samples(self.dxl_ids)
        }
        if set(samples) != set(self.dxl_ids):
            raise RuntimeError("Safety readback did not return every configured motor.")
        self._last_readback_time = now
        self._last_telemetry_time = now
        self._torque_enabled_by_id = {
            dxl_id: sample.torque_enabled for dxl_id, sample in samples.items()
        }
        for calibration in self.calibrations:
            sample = samples[calibration.dxl_id]
            position = normalize_position_tick(sample.present_position_tick)
            self._measured_ticks[calibration.joint_name] = position
            self._measured[calibration.joint_name] = calibration.tick_to_angle_rad(
                position
            )
            self._telemetry[calibration.joint_name] = (
                sample.present_current_ma,
                sample.temperature_celsius,
                sample.input_voltage,
                sample.hardware_error_status,
            )

    def _should_read_telemetry(self, now: float) -> bool:
        if self.telemetry_hz <= 0:
            return False
        if self._last_telemetry_time is None:
            return True
        return now - self._last_telemetry_time >= 1.0 / self.telemetry_hz

    def _should_read_positions(self, now: float) -> bool:
        if self.readback_hz <= 0:
            return False
        if self._last_readback_time is None:
            return True
        return now - self._last_readback_time >= 1.0 / self.readback_hz

    def _step_backend_simulation(self) -> None:
        step = getattr(self.controller.bus, "step", None)
        if callable(step):
            step()

    def _calibration(self, joint_name: str) -> JointCalibration:
        for calibration in self.calibrations:
            if calibration.joint_name == joint_name:
                return calibration
        raise KeyError(f"Unknown joint: {joint_name}")

    def _bounded_current_limit_ma(self, value: float) -> float:
        current = _validate_current_limit_ma(value)
        if self.max_current_limit_ma is None:
            return current
        return min(current, self.max_current_limit_ma)


def _validate_current_limit_ma(value: float) -> float:
    current = float(value)
    if not math.isfinite(current):
        raise ValueError("current_limit_ma must be finite.")
    if current < 0.0:
        raise ValueError("current_limit_ma must be non-negative.")
    return current
