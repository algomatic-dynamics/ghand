"""High-level hand facade for scripts and notebooks.

``Hand`` wraps :class:`~ghand.control.HandOperatorRuntime`, so every joint
command still passes through the same limits, pacing, and torque handling as
the operator UI and teleoperation. It does not add a second write path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

from ghand.control.operator_runtime import (
    DEFAULT_OPERATOR_PROFILE_ACCELERATION,
    DEFAULT_OPERATOR_PROFILE_VELOCITY,
    DEFAULT_OPERATOR_TARGET_HZ,
    DEFAULT_TARGET_VELOCITY_LIMIT,
    HandOperatorRuntime,
    OperatorControlMode,
    OperatorRuntimeSnapshot,
)
from ghand.hand import (
    GHAND_RIGHT_MOTOR_IDS,
    HAND_SIDE_PREFIXES,
    JOINT_POSITION_KEY,
    make_real_hand_joint_calibrations,
)
from ghand.hardware import DEFAULT_BAUDRATE, HandController
from ghand.hardware.serial_ports import resolve_serial_port

if TYPE_CHECKING:
    from ghand.viewer import RerunViewer


HOME_POSE: Final[str] = "home"


class Hand:
    """One LEAP V1 KAIZEN hand on a fake, real, or MuJoCo backend.

    Joint positions are radians keyed by short joint names such as
    ``index_mcp_flex``; the full LeRobot channel names are accepted too.
    Every backend uses the packaged motor mapping with the URDF joint limits,
    so targets are clamped to the hand's range. Use the instance as a context
    manager to connect, enable torque, and release torque again when the
    block exits. Pass ``viewer=True`` to any constructor to open the Rerun
    viewer (``viewer`` extra) and log every runtime snapshot the hand takes.
    """

    def __init__(
        self,
        runtime: HandOperatorRuntime,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        viewer: RerunViewer | Any | None = None,
    ) -> None:
        self.runtime = runtime
        self._sleep = sleep
        self._clock = clock
        self.viewer = viewer
        self._last_step_time: float | None = None
        self._full_names = {
            _short_joint_name(joint_name): joint_name for joint_name in runtime.joint_names
        }

    @classmethod
    def fake(
        cls,
        *,
        dry_run: bool = False,
        target_hz: float = DEFAULT_OPERATOR_TARGET_HZ,
        target_velocity_limit: float = DEFAULT_TARGET_VELOCITY_LIMIT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        viewer: bool = False,
    ) -> Hand:
        """Return a hand backed by the in-memory fake bus; nothing leaves the process."""

        runtime = HandOperatorRuntime(
            HandController.from_fake(motor_ids=GHAND_RIGHT_MOTOR_IDS),
            make_real_hand_joint_calibrations(),
            backend="fake",
            dry_run=dry_run,
            target_hz=target_hz,
            target_velocity_limit=target_velocity_limit,
        )
        return cls(runtime, sleep=sleep, clock=clock, viewer=_open_viewer(viewer))

    @classmethod
    def connect(
        cls,
        port: str | None = None,
        *,
        baudrate: int = DEFAULT_BAUDRATE,
        dry_run: bool = False,
        target_hz: float = DEFAULT_OPERATOR_TARGET_HZ,
        target_velocity_limit: float = DEFAULT_TARGET_VELOCITY_LIMIT,
        profile_acceleration: int = DEFAULT_OPERATOR_PROFILE_ACCELERATION,
        profile_velocity: int = DEFAULT_OPERATOR_PROFILE_VELOCITY,
        control_mode: OperatorControlMode = "position",
        default_current_limit_ma: float | None = None,
        max_current_limit_ma: float | None = None,
        viewer: bool = False,
    ) -> Hand:
        """Return a hand on the real Dynamixel bus; ``port=None`` picks the single adapter.

        The serial port is opened by :meth:`start` or by entering the context
        manager, not here.

        Two limits bound how fast the hand moves. ``target_velocity_limit``
        caps how fast the commanded joint angles advance, in rad/s.
        ``profile_velocity`` and ``profile_acceleration`` are the Dynamixel
        Profile Velocity(112) and Profile Acceleration(108) values that each
        servo applies on top of that, in units of 0.229 rev/min and 214.577
        rev/min^2, and 0 removes the servo-side limit. A large
        ``target_velocity_limit`` with both profile values at 0 lets the servos
        move as fast as they can, which is how ``ghand teleop`` runs. Joints
        that cannot keep up with the command then stop arriving together, so
        check a fast multi-joint move for self-collision first.
        """

        device = resolve_serial_port(port)
        runtime = HandOperatorRuntime(
            HandController.from_port(port=device, baudrate=baudrate),
            make_real_hand_joint_calibrations(),
            backend=f"dynamixel:{device}",
            dry_run=dry_run,
            target_hz=target_hz,
            target_velocity_limit=target_velocity_limit,
            profile_acceleration=profile_acceleration,
            profile_velocity=profile_velocity,
            control_mode=control_mode,
            default_current_limit_ma=default_current_limit_ma,
            max_current_limit_ma=max_current_limit_ma,
        )
        return cls(runtime, viewer=_open_viewer(viewer))

    @classmethod
    def mujoco(
        cls,
        scene: str = "turn_handle",
        *,
        viewer: bool = False,
        mujoco_viewer: bool = False,
        dry_run: bool = False,
    ) -> Hand:
        """Return a hand simulated in MuJoCo (requires the ``sim`` extra).

        ``mujoco_viewer`` opens MuJoCo's own passive viewer; ``viewer`` opens
        Rerun like the other constructors.
        """

        from ghand.sim.mujoco import make_mujoco_hand_runtime

        runtime = make_mujoco_hand_runtime(scene=scene, viewer=mujoco_viewer, dry_run=dry_run)
        return cls(runtime, viewer=_open_viewer(viewer))

    def __enter__(self) -> Hand:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the short joint names in the packaged joint order."""

        return tuple(self._full_names)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return the lower and upper joint limits in radians by short joint name."""

        return {
            _short_joint_name(calibration.joint_name): (
                calibration.lower_rad,
                calibration.upper_rad,
            )
            for calibration in self.runtime.calibrations
        }

    def start(self) -> OperatorRuntimeSnapshot:
        """Connect, align targets with the measured pose, and enable torque."""

        if self.viewer is not None:
            self.viewer.start()
        return self._log(self.runtime.start())

    def close(self) -> None:
        """Release torque when possible and disconnect the backend."""

        self.runtime.disconnect()

    def enable_torque(self) -> None:
        """Enable torque on every joint."""

        self.runtime.set_torque_enabled(True)

    def disable_torque(self) -> None:
        """Release torque on every joint while staying connected."""

        self.runtime.set_torque_enabled(False)

    def emergency_stop(self) -> None:
        """Latch the emergency stop and release torque immediately."""

        self.runtime.emergency_stop_now()

    def joint_positions(self) -> dict[str, float]:
        """Read the measured joint positions in radians."""

        snapshot = self._log(self.runtime.observe(now=self._clock()))
        if snapshot.last_error is not None:
            raise RuntimeError(snapshot.last_error)
        return {
            _short_joint_name(joint.joint_name): joint.measured for joint in snapshot.joints
        }

    def set_joint_positions(self, positions: Mapping[str, float]) -> None:
        """Set target joint positions in radians; call :meth:`step` to send them."""

        self.runtime.set_targets(
            {self._full_name(name): float(value) for name, value in positions.items()}
        )
        if self.runtime.last_error is not None:
            raise RuntimeError(self.runtime.last_error)

    def step(self) -> OperatorRuntimeSnapshot:
        """Send the latest targets through the runtime limits and read back."""

        now = self._clock()
        self._last_step_time = now
        snapshot = self._log(self.runtime.step(now=now))
        if snapshot.last_error is not None:
            raise RuntimeError(snapshot.last_error)
        return snapshot

    def move_to(
        self,
        pose: str | Mapping[str, float],
        *,
        seconds: float = 1.0,
    ) -> OperatorRuntimeSnapshot:
        """Interpolate from the current command to ``pose`` over ``seconds``.

        ``pose`` is ``"home"`` (the packaged zero pose) or a mapping of joint
        radians. Joints missing from the mapping keep their current command.
        The runtime's joint velocity limit still applies, so a large move can
        take longer than ``seconds``. Commands go out one control period apart,
        measured from the start of the previous step, so the time a step spends
        on the bus or in the viewer does not stretch the move.
        """

        if seconds < 0.0:
            raise ValueError("seconds must be non-negative.")
        goal = self._resolve_pose(pose)
        start = {
            joint.joint_name: joint.commanded for joint in self.runtime.snapshot().joints
        }
        period = 1.0 / self.runtime.target_hz
        steps = max(1, round(seconds * self.runtime.target_hz))
        snapshot = self.runtime.snapshot()
        for index in range(1, steps + 1):
            alpha = index / steps
            self.runtime.set_targets(
                {
                    joint_name: start[joint_name] + alpha * (value - start[joint_name])
                    for joint_name, value in goal.items()
                }
            )
            self._wait_for_next_tick(period)
            snapshot = self.step()
        return snapshot

    def snapshot(self) -> OperatorRuntimeSnapshot:
        """Return the latest known runtime state without touching the backend."""

        return self.runtime.snapshot()

    def _wait_for_next_tick(self, period: float) -> None:
        """Sleep until one control period after the previous step started."""

        if self._last_step_time is None:
            return
        remaining = self._last_step_time + period - self._clock()
        if remaining > 0.0:
            self._sleep(remaining)

    def _log(self, snapshot: OperatorRuntimeSnapshot) -> OperatorRuntimeSnapshot:
        if self.viewer is not None:
            self.viewer.log_runtime_snapshot(snapshot)
        return snapshot

    def _resolve_pose(self, pose: str | Mapping[str, float]) -> dict[str, float]:
        if isinstance(pose, str):
            if pose != HOME_POSE:
                raise ValueError(f"Unknown pose {pose!r}; use {HOME_POSE!r} or a mapping.")
            return {joint_name: 0.0 for joint_name in self.runtime.joint_names}
        return {self._full_name(name): float(value) for name, value in pose.items()}

    def _full_name(self, name: str) -> str:
        if name in self._full_names:
            return self._full_names[name]
        if name in self.runtime.joint_names:
            return name
        raise KeyError(
            f"Unknown joint {name!r}. Expected one of: {', '.join(self._full_names)}."
        )


def _open_viewer(enabled: bool) -> RerunViewer | None:
    """Create a spawned Rerun viewer for the packaged hand model when requested."""

    if not enabled:
        return None
    try:
        from ghand.viewer import RerunViewer
    except ImportError as exc:
        raise RuntimeError(
            "The Rerun viewer needs the `viewer` extra: pip install 'ghand[viewer]'."
        ) from exc
    from ghand.hand import default_right_hand_urdf_path

    return RerunViewer(spawn=True, urdf_path=default_right_hand_urdf_path())


def _short_joint_name(joint_name: str) -> str:
    """Strip the hand prefix and position key from a LeRobot channel name."""

    parts = joint_name.split(".")
    if len(parts) == 3 and parts[0] in HAND_SIDE_PREFIXES.values() and parts[2] == JOINT_POSITION_KEY:
        return parts[1]
    return joint_name
